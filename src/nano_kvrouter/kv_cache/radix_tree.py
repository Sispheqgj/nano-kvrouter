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
