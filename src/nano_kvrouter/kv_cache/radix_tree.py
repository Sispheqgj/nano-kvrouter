import time
import uuid
from dataclasses import dataclass, field


@dataclass
class RadixNode:
    key: list[int] = field(default_factory=list)
    block_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    last_access_time: float = field(default_factory=time.time)
    ref_count: int = 0
    children: dict[int, "RadixNode"] = field(default_factory=dict)
    parent: "RadixNode | None" = field(default=None, repr=False, compare=False)


class RadixTree:
    def __init__(self) -> None:
        self._root = RadixNode(key=[], block_id="root")
        # All non-root nodes indexed by block_id
        self._nodes: dict[str, RadixNode] = {}

    @staticmethod
    def _cp_len(a: list[int], b: list[int]) -> int:
        i = 0
        while i < len(a) and i < len(b) and a[i] == b[i]:
            i += 1
        return i

    def insert(self, token_ids: list[int]) -> str:
        """Insert a token sequence and return its block_id."""
        if not token_ids:
            return self._root.block_id

        node = self._root
        pos = 0
        now = time.time()

        while pos < len(token_ids):
            first = token_ids[pos]

            if first not in node.children:
                leaf = RadixNode(key=token_ids[pos:], last_access_time=now, parent=node)
                node.children[first] = leaf
                self._nodes[leaf.block_id] = leaf
                return leaf.block_id

            child = node.children[first]
            remaining = token_ids[pos:]
            cp = self._cp_len(remaining, child.key)

            if cp == len(child.key):
                # Fully matched this edge — descend
                child.last_access_time = now
                pos += cp
                node = child
                continue

            # Partial match — split edge at cp
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

            # Sequence ends exactly at the split point
            return mid.block_id

        # token_ids fully consumed — node is the deepest match
        node.last_access_time = now
        return node.block_id

    def match_prefix(self, token_ids: list[int]) -> tuple[int, str]:
        """Return (matched_token_count, block_id) for the longest cached prefix."""
        if not token_ids:
            return 0, self._root.block_id

        node = self._root
        pos = 0
        best_pos = 0
        best_id = self._root.block_id
        now = time.time()

        while pos < len(token_ids):
            first = token_ids[pos]
            if first not in node.children:
                break

            child = node.children[first]
            cp = self._cp_len(token_ids[pos:], child.key)

            if cp < len(child.key):
                # Partial edge match — cannot use this node's block
                break

            child.last_access_time = now
            pos += cp
            node = child
            best_pos = pos
            best_id = child.block_id

        return best_pos, best_id

    def evict_lru(self, n_blocks: int) -> list[str]:
        """Evict up to n_blocks leaf nodes with ref_count == 0, oldest first."""
        evicted: list[str] = []

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
            evicted.append(victim.block_id)
            del self._nodes[victim.block_id]

        return evicted
