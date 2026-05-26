from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class RadixNode:
    """A single node in the prefix radix tree.

    Each non-root node owns one KV cache block (identified by
    `block_id`) and stores the token sub-sequence (`key`) that labels
    the edge from its parent. `last_access_time` drives LRU eviction;
    `ref_count` pins blocks held by in-flight requests so they cannot be
    evicted out from under them.

    `parent` is excluded from `repr` / `compare` to avoid infinite
    recursion when dataclasses walk the structure.
    """

    key: list[int] = field(default_factory=list)
    block_id: str = field(default_factory=lambda: str(uuid.uuid4()))
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
        * Eviction is LRU among leaves with `ref_count == 0`. Internal
          nodes are never evicted directly — they go away only when all
          their children are gone (not modelled here; out of scope for
          the simulator's purposes).
        * Time stamps are produced by an injected ``clock`` callable so that
          the production simulator can pass ``engine.now`` and keep LRU
          ordering in simulated time. The default (``time.time``) preserves
          the original wall-clock behaviour used by standalone unit tests.
    """

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        """Initialize an empty tree with a sentinel root node.

        Args:
            clock: Optional callable returning the current time as a float.
                Defaults to ``time.time`` (wall-clock seconds) when ``None``.
                Pass ``engine.now`` when running inside
                :class:`~nano_kvrouter.simulator.engine.SimulationEngine` so
                LRU timestamps are in simulated milliseconds instead of
                wall-clock seconds.
        """
        self._clock: Callable[[], float] = clock if clock is not None else time.time
        self._root = RadixNode(key=[], block_id="root")
        # All non-root nodes indexed by block_id for O(1) eviction lookup.
        self._nodes: dict[str, RadixNode] = {}

    @staticmethod
    def _cp_len(a: list[int], b: list[int]) -> int:
        """Length of the common prefix between two token lists."""
        i = 0
        while i < len(a) and i < len(b) and a[i] == b[i]:
            i += 1
        return i

    def insert(self, token_ids: list[int]) -> str:
        """Insert a token sequence, creating or extending nodes as needed.

        Args:
            token_ids: Prompt tokens to insert. Empty input is a no-op
                that returns the root's id.

        Returns:
            `block_id` of the deepest node that corresponds to the full
            sequence — this is the block the caller should associate
            with the cached KV state.
        """
        if not token_ids:
            return self._root.block_id

        node = self._root
        pos = 0
        now = self._clock()

        while pos < len(token_ids):
            first = token_ids[pos]

            if first not in node.children:
                # No edge starts with this token — create a fresh leaf
                # that owns the entire remaining suffix.
                leaf = RadixNode(key=token_ids[pos:], last_access_time=now, parent=node)
                node.children[first] = leaf
                self._nodes[leaf.block_id] = leaf
                return leaf.block_id

            child = node.children[first]
            remaining = token_ids[pos:]
            cp = self._cp_len(remaining, child.key)

            if cp == len(child.key):
                # Fully matched this edge — descend and keep walking.
                child.last_access_time = now
                pos += cp
                node = child
                continue

            # Partial match → split this edge at position `cp` so that
            # the shared prefix becomes its own intermediate node `mid`
            # and the old child hangs off it as a sibling of the new
            # tail (if any). This is the classic radix-tree split.
            old_key = child.key
            mid = RadixNode(key=old_key[:cp], last_access_time=now, parent=node)
            self._nodes[mid.block_id] = mid

            child.key = old_key[cp:]
            child.parent = mid
            mid.children[child.key[0]] = child
            node.children[first] = mid

            tail = remaining[cp:]
            if tail:
                leaf = RadixNode(key=tail, last_access_time=now, parent=mid)
                self._nodes[leaf.block_id] = leaf
                mid.children[tail[0]] = leaf
                return leaf.block_id

            # Sequence ends exactly at the split point — `mid` itself is
            # the answer; no extra leaf needed.
            return mid.block_id

        # token_ids fully consumed by descending whole edges — the
        # current node is the deepest match.
        node.last_access_time = now
        return node.block_id

    def match_prefix(self, token_ids: list[int]) -> tuple[int, str]:
        """Find the longest cached prefix of `token_ids`.

        Args:
            token_ids: Candidate prompt tokens to look up.

        Returns:
            `(matched_token_count, block_id)`. `matched_token_count` is
            the number of leading tokens that align with a complete
            chain of edges in the tree; `block_id` is the id of the
            deepest fully-matched node (the root's id if nothing
            matches). The match must end on an edge boundary — a
            partial-edge match is not a cache hit because the simulator
            stores one KV block per node.
        """
        if not token_ids:
            return 0, self._root.block_id

        node = self._root
        pos = 0
        best_pos = 0
        best_id = self._root.block_id
        now = self._clock()

        while pos < len(token_ids):
            first = token_ids[pos]
            if first not in node.children:
                break

            child = node.children[first]
            cp = self._cp_len(token_ids[pos:], child.key)

            if cp < len(child.key):
                # The edge is only partially traversed — we cannot
                # claim this block as a hit, so stop at the previous
                # complete-edge node.
                break

            child.last_access_time = now
            pos += cp
            node = child
            best_pos = pos
            best_id = child.block_id

        return best_pos, best_id

    def evict_lru(self, n_blocks: int) -> list[str]:
        """Evict up to `n_blocks` cold, unreferenced leaf blocks.

        Args:
            n_blocks: Maximum number of leaves to evict. Fewer may be
                returned if no further candidates exist.

        Returns:
            `block_id`s of evicted leaves, in eviction order (oldest
            first). Only leaves with `ref_count == 0` are eligible —
            blocks currently held by in-flight requests stay pinned.
        """
        evicted: list[str] = []

        for _ in range(n_blocks):
            # Recompute candidates each iteration: evicting one leaf
            # can promote its parent into a leaf (if it had no other
            # children), so the eligible set shrinks/changes.
            candidates = [
                n for n in self._nodes.values()
                if not n.children and n.ref_count == 0
            ]
            if not candidates:
                break

            victim = min(candidates, key=lambda n: n.last_access_time)
            assert victim.parent is not None  # only the root has no parent, and root is not in _nodes
            del victim.parent.children[victim.key[0]]
            victim.parent = None
            evicted.append(victim.block_id)
            del self._nodes[victim.block_id]

        return evicted

    def evict_lru_with_lengths(self, n_blocks: int) -> list[int]:
        """Like evict_lru, but returns len(key) of each evicted node instead of block_id.

        Used by CacheManager to compute physical block reclamation via
        ceil(key_len / block_size) without a parallel block_id → pool mapping.

        Args:
            n_blocks: Maximum number of leaves to evict.

        Returns:
            List of ``len(node.key)`` for each evicted node, in eviction order.
            Empty if no eligible (unpinned, leaf) candidates remain.
        """
        evicted: list[int] = []

        for _ in range(n_blocks):
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
            del self._nodes[victim.block_id]

        return evicted
