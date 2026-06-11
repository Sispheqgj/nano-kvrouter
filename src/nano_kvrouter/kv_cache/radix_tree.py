from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class RadixNode:
    """A single node in the prefix radix tree.

    Each non-root node owns one or more KV cache blocks (identified by
    ``block_ids``).  The list length equals ``ceil(len(key) / block_size)``
    — a node whose key spans multiple physical blocks gets multiple entries.
    ``last_access_time`` drives LRU eviction; ``ref_count`` pins blocks held
    by in-flight requests so they cannot be evicted out from under them.

    ``parent`` is excluded from ``repr`` / ``compare`` to avoid infinite
    recursion when dataclasses walk the structure.
    """

    key: list[int] = field(default_factory=list)
    block_ids: list[str] = field(default_factory=list)
    last_access_time: float = field(default_factory=time.time)
    ref_count: int = 0
    children: dict[int, "RadixNode"] = field(default_factory=dict)
    parent: "RadixNode | None" = field(default=None, repr=False, compare=False)


class RadixTree:
    """SGLang-style prefix tree mapping token sequences to KV block IDs.

    The tree supports three operations central to cache-aware
    scheduling: insert a new prompt (carving fresh blocks or reusing a
    shared prefix), find the longest matching prefix already cached, and
    evict cold blocks under memory pressure.

    Design notes:
        * Edges are labelled with *runs* of tokens (not single tokens),
          so the structure is a true radix tree, not a trie. Inserting a
          sequence that partially overlaps an existing edge triggers an
          edge split.
        * Each node owns ``ceil(len(key) / block_size)`` physical blocks,
          tracked in ``block_ids``.  The primary key in ``_nodes`` is
          ``block_ids[0]`` (the first block of each node).
        * Splits re-distribute block_ids between the shortened child and the
          new mid-node; excess blocks freed from a shortened child are
          returned via the ``free_blocks`` callback.
        * Eviction is LRU among leaves with ``ref_count == 0``.  Evicted
          block_ids are returned to the caller for release to the backing pool.
        * Time stamps are produced by an injected ``clock`` callable so that
          the production simulator can pass ``engine.now`` and keep LRU
          ordering in simulated time.  The default (``time.time``) preserves
          the original wall-clock behaviour used by standalone unit tests.
        * The root node has ``block_ids = []`` and is NOT in ``_nodes``.
    """

    def __init__(
        self,
        clock: Callable[[], float] | None = None,
        block_size: int = 1,
        mint: Callable[[int], list[str]] | None = None,
        free_blocks: Callable[[list[str]], None] | None = None,
    ) -> None:
        """Initialize an empty tree with a sentinel root node.

        Args:
            clock: Optional callable returning the current time as a float.
                Defaults to ``time.time`` (wall-clock seconds) when ``None``.
                Pass ``engine.now`` when running inside
                :class:`~nano_kvrouter.simulator.engine.SimulationEngine`.
            block_size: Number of tokens per physical KV block.  Each new
                node's ``block_ids`` list has length
                ``ceil(len(key) / block_size)``.  Defaults to 1 so standalone
                tests with small token counts remain meaningful.
            mint: Callable ``(n: int) -> list[str]`` that allocates *n* fresh
                block IDs (e.g. ``pool.allocate(n, "gpu")``).  Defaults to a
                local UUID4 generator when ``None``.
            free_blocks: Callable ``(ids: list[str]) -> None`` invoked when a
                node's key is shortened by a split and its excess blocks must
                be released to the backing pool.  Defaults to a no-op.
        """
        self._clock: Callable[[], float] = clock if clock is not None else time.time
        self._block_size: int = max(1, block_size)
        self._mint: Callable[[int], list[str]] = (
            mint if mint is not None else lambda n: [str(uuid.uuid4()) for _ in range(n)]
        )
        self._free_blocks: Callable[[list[str]], None] = (
            free_blocks if free_blocks is not None else lambda ids: None
        )
        self._root = RadixNode(key=[], block_ids=[])
        # Non-root nodes indexed by block_ids[0] for O(1) eviction lookup.
        self._nodes: dict[str, RadixNode] = {}

    def _n_blocks(self, key: list[int]) -> int:
        """Physical blocks needed for a node with the given key length."""
        if not key:
            return 0
        return (len(key) + self._block_size - 1) // self._block_size

    @staticmethod
    def _cp_len(a: list[int], b: list[int]) -> int:
        """Length of the common prefix between two token lists."""
        i = 0
        while i < len(a) and i < len(b) and a[i] == b[i]:
            i += 1
        return i

    def insert(self, token_ids: list[int]) -> list[str]:
        """Insert a token sequence, creating or extending nodes as needed.

        New nodes and split mid-nodes have their ``block_ids`` allocated via
        the ``mint`` callback.  When a split shortens a child node's key, any
        excess blocks are released via the ``free_blocks`` callback.

        Args:
            token_ids: Prompt tokens to insert.  Empty input is a no-op
                that returns the root's (empty) block_ids.

        Returns:
            ``block_ids`` of the deepest node that corresponds to the full
            sequence — the blocks the caller should associate with the cached
            KV state.
        """
        if not token_ids:
            return self._root.block_ids

        node = self._root
        pos = 0
        now = self._clock()

        while pos < len(token_ids):
            first = token_ids[pos]

            if first not in node.children:
                # No edge starts with this token — create a fresh leaf.
                key = token_ids[pos:]
                bids = self._mint(self._n_blocks(key))
                leaf = RadixNode(key=key, block_ids=bids, last_access_time=now, parent=node)
                node.children[first] = leaf
                self._nodes[bids[0]] = leaf
                return bids

            child = node.children[first]
            remaining = token_ids[pos:]
            cp = self._cp_len(remaining, child.key)

            if cp == len(child.key):
                # Fully matched this edge — descend.
                child.last_access_time = now
                pos += cp
                node = child
                continue

            # Partial match → split this edge at position cp.
            old_key = child.key
            old_block_ids = child.block_ids
            bs = self._block_size
            n_old = len(old_block_ids)
            cp_blocks = (cp + bs - 1) // bs                      # blocks owned by prefix
            tail_blocks = (len(old_key) - cp + bs - 1) // bs     # blocks owned by suffix

            # A new block for child's head is needed iff the split-point block is shared
            # between mid and child (i.e., cp_blocks + tail_blocks > n_old means the
            # boundary block is counted in both halves and child needs a fresh one).
            needs_extra = (cp_blocks + tail_blocks > n_old)

            # Step 1: Pre-mint if needed — transactional: tree is unchanged if this raises.
            new_child_head: list[str] = []
            if needs_extra:
                new_child_head = self._mint(1)

            # Step 2: All allocations succeeded; execute the irreversible split.
            # mid inherits the first cp_blocks from old_block_ids (the prefix portion).
            mid_key = old_key[:cp]
            mid_bids = list(old_block_ids[:cp_blocks])
            mid = RadixNode(key=mid_key, block_ids=mid_bids, last_access_time=now, parent=node)

            # child keeps the suffix block_ids; when needs_extra it gets a fresh head block
            # because the boundary block is wholly owned by mid.
            child.key = old_key[cp:]
            child.block_ids = new_child_head + list(old_block_ids[cp_blocks:])
            child.parent = mid
            mid.children[child.key[0]] = child
            node.children[first] = mid

            # Update _nodes: mid takes over old_block_ids[0]; child moves to its new key.
            self._nodes[mid_bids[0]] = mid
            self._nodes[child.block_ids[0]] = child

            tail = remaining[cp:]
            if tail:
                leaf_bids = self._mint(self._n_blocks(tail))
                leaf = RadixNode(key=tail, block_ids=leaf_bids, last_access_time=now, parent=mid)
                self._nodes[leaf_bids[0]] = leaf
                mid.children[tail[0]] = leaf
                return leaf_bids

            # Sequence ends exactly at split point — mid is the answer.
            return mid_bids

        node.last_access_time = now
        return node.block_ids

    def match_prefix(self, token_ids: list[int]) -> tuple[int, list[str]]:
        """Find the longest cached prefix of ``token_ids``.

        Args:
            token_ids: Candidate prompt tokens to look up.

        Returns:
            ``(matched_token_count, block_ids)``.  ``matched_token_count`` is
            the number of leading tokens that align with a complete chain of
            edges in the tree; ``block_ids`` is the list of block IDs for the
            deepest fully-matched node (the root's empty list if nothing
            matches).  The match must end on an edge boundary — a partial-edge
            match is not a cache hit.
        """
        if not token_ids:
            return 0, self._root.block_ids

        node = self._root
        pos = 0
        best_pos = 0
        best_ids: list[str] = self._root.block_ids
        now = self._clock()

        while pos < len(token_ids):
            first = token_ids[pos]
            if first not in node.children:
                break

            child = node.children[first]
            cp = self._cp_len(token_ids[pos:], child.key)

            if cp < len(child.key):
                break

            child.last_access_time = now
            pos += cp
            node = child
            best_pos = pos
            best_ids = child.block_ids

        return best_pos, best_ids

    def evict_lru(self, n_nodes: int) -> list[list[str]]:
        """Evict up to *n_nodes* cold, unreferenced leaf nodes.

        Args:
            n_nodes: Maximum number of leaves to evict.  Fewer may be
                returned if no further candidates exist.

        Returns:
            List of ``block_ids`` lists, one per evicted node, in eviction
            order (oldest first).  Only leaves with ``ref_count == 0`` are
            eligible.  The caller is responsible for releasing these block IDs
            to the backing pool.
        """
        evicted: list[list[str]] = []

        for _ in range(n_nodes):
            candidates = [
                n for n in self._nodes.values()
                if not n.children and n.ref_count == 0
            ]
            if not candidates:
                break

            victim = min(candidates, key=lambda n: n.last_access_time)
            assert victim.parent is not None
            del victim.parent.children[victim.key[0]]
            victim.parent = None
            evicted.append(victim.block_ids)
            del self._nodes[victim.block_ids[0]]

        return evicted

    def evict_lru_with_lengths(self, n_nodes: int) -> list[int]:
        """Like :meth:`evict_lru`, but returns ``len(key)`` instead of block IDs.

        Retained for backward compatibility with tests that verify token-count
        semantics.  New callers should prefer :meth:`evict_lru`.

        Args:
            n_nodes: Maximum number of leaves to evict.

        Returns:
            List of ``len(node.key)`` for each evicted node, in eviction order.
        """
        evicted: list[int] = []

        for _ in range(n_nodes):
            candidates = [
                n for n in self._nodes.values()
                if not n.children and n.ref_count == 0
            ]
            if not candidates:
                break

            victim = min(candidates, key=lambda n: n.last_access_time)
            assert victim.parent is not None
            del victim.parent.children[victim.key[0]]
            victim.parent = None
            evicted.append(len(victim.key))
            del self._nodes[victim.block_ids[0]]

        return evicted

    def match_prefix_path(self, token_ids: list[int]) -> tuple[int, list[str]]:
        """Find the longest cached prefix and return all block_ids on the path.

        Like :meth:`match_prefix` but accumulates block_ids from every node
        on the root-to-deepest-match path, not just the deepest node.

        Args:
            token_ids: Candidate prompt tokens to look up.

        Returns:
            ``(matched_token_count, path_block_ids)`` where
            ``path_block_ids`` is the flat concatenation of ``block_ids``
            for every fully-matched node from root to the deepest match.
            The root (which has no block_ids) is excluded.
        """
        if not token_ids:
            return 0, []

        node = self._root
        pos = 0
        best_pos = 0
        path_block_ids: list[str] = []
        now = self._clock()

        while pos < len(token_ids):
            first = token_ids[pos]
            if first not in node.children:
                break

            child = node.children[first]
            cp = self._cp_len(token_ids[pos:], child.key)

            if cp < len(child.key):
                break

            child.last_access_time = now
            pos += cp
            node = child
            best_pos = pos
            path_block_ids.extend(child.block_ids)

        return best_pos, path_block_ids

    def purge_block_ids(
        self,
        freed_block_ids: list[str],
        tier_of: "Callable[[str], str | None]",
    ) -> int:
        """Remove leaf nodes whose every block_id is no longer allocated.

        M6.fix2 (Important #1): synchronous tree GC. When
        ``CacheManager._demote_block`` truly frees a block (disk full, or
        cpu_blocks==disk_blocks==0), the tree node that owned it becomes
        a *zombie* — still present, still discoverable by
        ``match_prefix*``, but pointing at freed pool slots. Future
        ``lookup`` correctly skips zombies via ``pool.tiers_of``, but
        ``admit``'s fast path historically used ``tree.match_prefix``
        (which has no pool awareness) and short-circuited on zombies.

        This API gives the caller a way to purge those zombie leaves
        immediately after a real free, keeping the tree consistent with
        the pool.

        Only leaves with ``ref_count == 0`` are eligible — non-leaves
        cannot be safely removed (their children depend on the path).

        Args:
            freed_block_ids: Block IDs that have just been released from
                the pool. Used as a hint to narrow the scan to nodes
                that could plausibly contain a freed block.
            tier_of: Callable ``(block_id) -> tier | None`` used to
                detect zombies. Typically ``pool.tier_of``.

        Returns:
            Number of zombie leaf nodes removed from the tree.
        """
        if not freed_block_ids:
            return 0
        freed_set = set(freed_block_ids)

        purged = 0
        # Snapshot ``_nodes.values()`` because we mutate during iteration.
        for node in list(self._nodes.values()):
            if node.children or node.ref_count > 0:
                continue
            if not any(b in freed_set for b in node.block_ids):
                continue
            if any(tier_of(bid) is not None for bid in node.block_ids):
                # Mixed-state node: some blocks still live. Don't touch.
                continue
            # Fully zombie leaf — detach.
            if node.parent is not None:
                # Use first key token (not block_ids[0]) to index parent.children.
                node.parent.children.pop(node.key[0], None)
                node.parent = None
            self._nodes.pop(node.block_ids[0], None)
            purged += 1
        return purged

    def _purge_subtree_alive(
        self,
        subtree_root: "RadixNode",
        tier_of: "Callable[[str], str | None]",
        free_blocks: "Callable[[list[str]], None]",
    ) -> int:
        """Recursively detach all descendants, freeing any alive blocks."""
        purged = 0
        for child in list(subtree_root.children.values()):
            alive = [b for b in child.block_ids if tier_of(b) is not None]
            if alive:
                free_blocks(alive)
            if child.block_ids:
                self._nodes.pop(child.block_ids[0], None)
            child.parent = None
            purged += 1
            purged += self._purge_subtree_alive(child, tier_of, free_blocks)
        subtree_root.children.clear()
        return purged

    def purge_path_with_any_dead(
        self,
        token_ids: list[int],
        tier_of: "Callable[[str], str | None]",
        free_blocks: "Callable[[list[str]], None]",
    ) -> int:
        """Walk the prefix path and purge any node that contains a dead block.

        Called by :meth:`CacheManager.admit` when the fast path detects a
        partial-zombie on the cached path (some blocks alive, some freed).
        :meth:`purge_block_ids` intentionally skips mixed-state nodes; this
        method removes the whole node (and its subtree) even when some blocks
        are still alive, returning those alive blocks to the pool so their
        tier-capacity is not leaked.

        After this call, :meth:`insert` re-creates the path from scratch
        and allocates fresh GPU blocks for every position, satisfying the
        post-admit invariant that the full prompt is resident on GPU.

        Only unreferenced nodes (ref_count == 0) are eligible.

        Args:
            token_ids: Token sequence that selects the path to inspect.
            tier_of: pool.tier_of — None return marks dead blocks.
            free_blocks: pool.free — called with alive block IDs on
                removed nodes so tier capacity is reclaimed.

        Returns:
            Total number of tree nodes removed (node + its descendants).
        """
        if not token_ids:
            return 0

        node = self._root
        pos = 0
        purged = 0

        while pos < len(token_ids):
            first = token_ids[pos]
            if first not in node.children:
                break

            child = node.children[first]
            if child.ref_count > 0:
                break  # pinned — cannot purge

            cp = self._cp_len(token_ids[pos:], child.key)
            if cp < len(child.key):
                break

            if any(tier_of(bid) is None for bid in child.block_ids):
                # Node has at least one dead block — purge it and its subtree.
                alive = [b for b in child.block_ids if tier_of(b) is not None]
                if alive:
                    free_blocks(alive)
                del node.children[first]
                if child.block_ids:
                    self._nodes.pop(child.block_ids[0], None)
                child.parent = None
                purged += 1
                purged += self._purge_subtree_alive(child, tier_of, free_blocks)
                break  # path severed here

            pos += cp
            node = child

        return purged

    def demote_lru(
        self,
        n_nodes: int,
        tier_of: "Callable[[str], str | None] | None" = None,
        is_pinned: "Callable[[str], bool] | None" = None,
    ) -> list[list[str]]:
        """Select LRU leaf nodes for tier demotion without removing them from the tree.

        Like :meth:`evict_lru` in candidate selection (LRU unreferenced
        leaves) but keeps nodes in the tree so future prefix matches can
        still find them (at a transfer cost if their blocks are on a slow
        tier).

        If *tier_of* is supplied, zombie nodes — nodes whose every
        block_id is no longer allocated (``tier_of`` returns ``None``) —
        are silently evicted from the tree during the scan so they cannot
        cause infinite loops in the caller.

        Args:
            n_nodes: Maximum number of live nodes to return.
            tier_of: Optional callable ``(block_id) -> tier | None``
                (e.g. ``pool.tier_of``) used to detect and clean up
                zombie nodes. Pass ``None`` to skip zombie cleanup.
            is_pinned: Optional callable ``(block_id) -> bool`` used by
                BlockTable-aware callers to skip active request blocks.

        Returns:
            List of ``block_ids`` lists, one per selected node. The
            caller is responsible for moving or freeing these blocks.
        """
        from collections.abc import Callable  # local import to avoid top-level cycle

        demoted: list[list[str]] = []

        # Bound iterations to prevent infinite scan when many zombies exist.
        max_scans = len(self._nodes) + n_nodes + 1
        scans = 0

        while len(demoted) < n_nodes and scans < max_scans:
            scans += 1
            candidates = [
                n for n in self._nodes.values()
                if not n.children and n.ref_count == 0
                and not (is_pinned is not None and any(is_pinned(bid) for bid in n.block_ids))
            ]
            if not candidates:
                break

            victim = min(candidates, key=lambda n: n.last_access_time)

            if tier_of is not None:
                # Check for zombie (all blocks freed from pool).
                live = [b for b in victim.block_ids if tier_of(b) is not None]
                if not live:
                    # Evict zombie node: clean it from tree but don't touch pool.
                    assert victim.parent is not None
                    del victim.parent.children[victim.key[0]]
                    victim.parent = None
                    del self._nodes[victim.block_ids[0]]
                    continue  # pick next candidate

            # Live node: return its block_ids for the caller to demote.
            demoted.append(victim.block_ids)
            # Advance last_access_time so repeated calls pick different nodes.
            victim.last_access_time = self._clock() + 1e-9

        return demoted
