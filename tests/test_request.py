from __future__ import annotations

import pytest

from nano_kvrouter.config import NanoKVConfig
from nano_kvrouter.request import Request, _hash_prefix, make_request


@pytest.fixture
def cfg() -> NanoKVConfig:
    return NanoKVConfig()


def test_make_request_fields(cfg: NanoKVConfig) -> None:
    tokens = list(range(64))
    req = make_request(tokens, arrival_time=1.5, config=cfg)

    assert isinstance(req, Request)
    assert req.token_ids == tokens
    assert req.arrival_time == 1.5
    assert req.expected_output_len == cfg.workload.avg_output_len
    assert req.slo_ttft == cfg.slo.ttft_target_ms
    assert req.slo_tbt == cfg.slo.tbt_target_ms
    assert req.priority == 0
    assert len(req.request_id) > 0


def test_request_id_unique(cfg: NanoKVConfig) -> None:
    tokens = list(range(32))
    r1 = make_request(tokens, 0.0, cfg)
    r2 = make_request(tokens, 0.0, cfg)
    assert r1.request_id != r2.request_id


def test_prefix_hash_deterministic(cfg: NanoKVConfig) -> None:
    tokens = list(range(64))
    h1 = _hash_prefix(tokens, cfg.model.block_size)
    h2 = _hash_prefix(tokens, cfg.model.block_size)
    assert h1 == h2
    assert len(h1) == 8


def test_shared_prefix_same_hash(cfg: NanoKVConfig) -> None:
    # Both requests share the same first block_size tokens → same prefix_hash.
    block = cfg.model.block_size
    shared = list(range(block))
    tokens_a = shared + [100, 101, 102]
    tokens_b = shared + [200, 201]

    req_a = make_request(tokens_a, 0.0, cfg)
    req_b = make_request(tokens_b, 1.0, cfg)

    assert req_a.prefix_hash == req_b.prefix_hash


def test_different_prefix_different_hash(cfg: NanoKVConfig) -> None:
    block = cfg.model.block_size
    tokens_a = list(range(block)) + [999]
    tokens_b = list(range(1, block + 1)) + [999]

    req_a = make_request(tokens_a, 0.0, cfg)
    req_b = make_request(tokens_b, 0.0, cfg)

    assert req_a.prefix_hash != req_b.prefix_hash


def test_short_token_ids_no_error(cfg: NanoKVConfig) -> None:
    # Fewer tokens than block_size — prefix is the full list.
    tokens = [1, 2, 3]
    req = make_request(tokens, 0.0, cfg)
    assert len(req.prefix_hash) == 8


def test_priority_passthrough(cfg: NanoKVConfig) -> None:
    req = make_request([1, 2, 3], 0.0, cfg, priority=5)
    assert req.priority == 5


def test_token_ids_copied(cfg: NanoKVConfig) -> None:
    tokens = [10, 20, 30]
    req = make_request(tokens, 0.0, cfg)
    tokens.append(99)
    assert req.token_ids == [10, 20, 30]
