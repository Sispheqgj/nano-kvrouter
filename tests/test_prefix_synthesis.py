"""Tests for PrefixSynthesisConfig and PrefixSynthesisModel."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from nano_kvrouter.simulator.prefix_synthesis import PrefixSynthesisConfig, PrefixSynthesisModel


def _make_model(
    num_buckets: int = 8,
    zipf_alpha: float = 1.0,
    p_local: float = 0.0,
    local_window_s: float = 60.0,
    sharing_layers: list | None = None,
    seed: int = 0,
    block_size: int = 16,
    vocab_size: int = 1000,
    initial_prompt_len: int = 512,
) -> PrefixSynthesisModel:
    layers = sharing_layers or [(0.5, 0.5), (0.5, 0.0)]
    cfg = PrefixSynthesisConfig(
        num_buckets=num_buckets,
        zipf_alpha=zipf_alpha,
        p_local=p_local,
        local_window_s=local_window_s,
        sharing_layers=layers,
        seed=seed,
    )
    return PrefixSynthesisModel(
        config=cfg,
        block_size=block_size,
        vocab_size=vocab_size,
        initial_prompt_len=initial_prompt_len,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: aligned prefix length is a multiple of block_size
# ─────────────────────────────────────────────────────────────────────────────

def test_aligned_prefix_len_is_block_multiple():
    """Returned prefix must have length that is a multiple of block_size."""
    model = _make_model(
        block_size=16,
        sharing_layers=[(1.0, 0.5)],  # always 50% ratio
    )
    for prompt_len in [30, 64, 100, 200, 33, 17]:
        prefix = model.assign_prefix_tokens(arrival_ms=0.0, prompt_len=prompt_len)
        assert len(prefix) % 16 == 0, (
            f"prompt_len={prompt_len}: prefix len {len(prefix)} not multiple of 16"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: zero ratio layer always returns empty list
# ─────────────────────────────────────────────────────────────────────────────

def test_aligned_prefix_len_zero_when_layer_ratio_zero():
    """When all sharing layers have ratio 0.0, prefix is always empty."""
    model = _make_model(
        block_size=16,
        sharing_layers=[(0.3, 0.0), (0.4, 0.0), (0.3, 0.0)],
    )
    for _ in range(20):
        prefix = model.assign_prefix_tokens(arrival_ms=0.0, prompt_len=512)
        assert prefix == [], "zero-ratio layers must return empty prefix"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: same bucket via locality gives same prefix
# ─────────────────────────────────────────────────────────────────────────────

def test_same_bucket_gives_same_prefix():
    """With p_local=1.0, second request reuses the same bucket as the first
    (assuming first request is still in the local window), so the first N
    tokens of both prefixes are identical."""
    # Force sharing_layers to always pick ratio 0.5 (layer_idx=0)
    # With a fixed seed and deterministic choices, both requests should pick
    # the same bucket via the local-window path.
    model = _make_model(
        block_size=16,
        p_local=1.0,
        local_window_s=60.0,
        sharing_layers=[(1.0, 0.5)],  # always 50%
        seed=7,
        num_buckets=4,
        initial_prompt_len=256,
    )
    prompt_len = 128
    # First request — populates recent deque
    prefix_a = model.assign_prefix_tokens(arrival_ms=0.0, prompt_len=prompt_len)
    # Second request in the same window — local path must reuse a bucket from recent
    prefix_b = model.assign_prefix_tokens(arrival_ms=1000.0, prompt_len=prompt_len)

    # Both prefixes are non-empty
    assert len(prefix_a) > 0
    assert len(prefix_b) > 0

    # They share the same bucket tokens (same block_size * n tokens from same bucket)
    shared_len = min(len(prefix_a), len(prefix_b))
    assert prefix_a[:shared_len] == prefix_b[:shared_len], (
        "With p_local=1.0, consecutive requests in same window must reuse bucket"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Zipf distribution is top-biased
# ─────────────────────────────────────────────────────────────────────────────

def test_zipf_distribution_top_biased():
    """With p_local=0 and zipf_alpha=2.0, bucket 0 should be sampled far more
    often than bucket num_buckets-1."""
    model = _make_model(
        num_buckets=64,
        zipf_alpha=2.0,
        p_local=0.0,
        sharing_layers=[(1.0, 0.5)],  # always non-zero prefix
        seed=42,
        initial_prompt_len=512,
        block_size=16,
    )

    counts = [0] * 64
    N = 10000
    for i in range(N):
        prefix = model.assign_prefix_tokens(arrival_ms=float(i), prompt_len=128)
        # Determine which bucket this prefix came from by matching first block_size tokens
        for b_idx in range(64):
            if model._bucket_prefixes[b_idx][:16] == prefix[:16]:
                counts[b_idx] += 1
                break

    freq_0 = counts[0] / N
    freq_last = counts[63] / N
    assert freq_0 > 0.1, f"bucket 0 freq {freq_0:.3f} too low for zipf_alpha=2"
    assert freq_0 > freq_last * 10, (
        f"bucket 0 ({freq_0:.3f}) should be >> bucket 63 ({freq_last:.3f}) for alpha=2"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: time locality window — purge works correctly
# ─────────────────────────────────────────────────────────────────────────────

def test_time_locality_window():
    """Bucket X added at t=0 is still in window at t=50s but purged by t=70s."""
    model = _make_model(
        p_local=1.0,
        local_window_s=60.0,
        sharing_layers=[(1.0, 0.5)],
        seed=0,
        block_size=16,
        initial_prompt_len=512,
    )
    # Trigger first request at t=0 ms to populate recent deque
    model.assign_prefix_tokens(arrival_ms=0.0, prompt_len=64)
    assert len(model._recent) == 1, "should have one entry in recent"

    # At 50 s (50000 ms), entry is still inside the 60 s window
    model._purge_expired(50_000.0)
    assert len(model._recent) == 1, "entry at t=0 should still be in window at t=50s"

    # At 70 s (70000 ms), entry is outside the 60 s window → purged
    model._purge_expired(70_000.0)
    assert len(model._recent) == 0, "entry at t=0 should be purged at t=70s"


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: invalid config raises ValueError
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kwargs,match", [
    ({"num_buckets": 0}, "num_buckets"),
    ({"zipf_alpha": 0.0}, "zipf_alpha"),
    ({"p_local": -0.1}, "p_local"),
    ({"local_window_s": 0.0}, "local_window_s"),
    ({"sharing_layers": [(0.5, 0.5), (0.4, 0.0)]}, "sum"),  # sums to 0.9
])
def test_invalid_config_raises(kwargs, match):
    """Invalid PrefixSynthesisConfig fields must raise ValidationError."""
    base = {
        "num_buckets": 8,
        "zipf_alpha": 1.0,
        "p_local": 0.5,
        "local_window_s": 60.0,
        "sharing_layers": [(0.5, 0.5), (0.5, 0.0)],
        "seed": 0,
    }
    base.update(kwargs)
    with pytest.raises((ValueError, ValidationError)):
        PrefixSynthesisConfig(**base)


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: long prompt exceeds initial_prompt_len — bucket lazy extends
# ─────────────────────────────────────────────────────────────────────────────

def test_long_prompt_extends_bucket():
    """critical 1 regression: prompt_len >> initial_prompt_len must not truncate.

    Without lazy extend, a 8192-token prompt with sharing ratio 1.0 would be
    silently capped at initial_prompt_len tokens.
    """
    cfg = PrefixSynthesisConfig(
        num_buckets=4,
        p_local=1.0,
        sharing_layers=[(1.0, 1.0)],  # always full prefix
        seed=0,
    )
    model = PrefixSynthesisModel(cfg, block_size=16, vocab_size=32000, initial_prompt_len=128)

    # Warm up: trigger bucket assignment to populate recent deque
    model.assign_prefix_tokens(0.0, 128)

    # Request with much larger prompt — must NOT be truncated
    large_prompt_len = 8192
    prefix = model.assign_prefix_tokens(100.0, large_prompt_len)

    # With ratio=1.0 and block_size=16: aligned = (8192//16)*16 = 8192
    assert len(prefix) == large_prompt_len, (
        f"Expected prefix len {large_prompt_len}, got {len(prefix)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 8+9+10: critical 2 — invalid ratio / prob rejected by validator
# ─────────────────────────────────────────────────────────────────────────────

def test_invalid_ratio_rejected_above_one():
    """critical 2: sharing ratio > 1.0 must raise ValidationError."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        PrefixSynthesisConfig(sharing_layers=[(1.0, 1.5)])


def test_invalid_ratio_rejected_negative():
    """critical 2: sharing ratio < 0 must raise ValidationError."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        PrefixSynthesisConfig(sharing_layers=[(1.0, -0.1)])


def test_invalid_negative_probability_rejected():
    """critical 2: prob < 0 in sharing_layers must raise ValidationError."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        PrefixSynthesisConfig(sharing_layers=[(0.5, 0.5), (-0.5, 0.5), (1.0, 0.0)])
