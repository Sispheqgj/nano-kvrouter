"""Tests for Bidaw KV-load slot models."""
from __future__ import annotations

import pytest

from nano_kvrouter.simulator.bidaw_load_model import (
    MultiStreamLoadModel,
    SingleSlotLoadModel,
)


def test_single_slot_residual_and_identity() -> None:
    model = SingleSlotLoadModel(["d0"])

    assert model.slot_residuals_ms("d0", now_ms=0.0) == [0.0]
    assert model.in_flight_request_ids("d0") == frozenset()
    assert model.in_flight_disk_blocks("d0") == 0

    model.start_load("d0", "r0", now_ms=10.0, service_ms=30.0, disk_blocks=2)

    assert model.slot_residuals_ms("d0", now_ms=25.0) == [15.0]
    assert model.slot_residuals_ms("d0", now_ms=50.0) == [0.0]
    assert model.in_flight_request_ids("d0") == frozenset({"r0"})
    assert model.in_flight_disk_blocks("d0") == 2

    model.complete_load("d0", "r0", now_ms=40.0)
    assert model.slot_residuals_ms("d0", now_ms=40.0) == [0.0]
    assert model.in_flight_request_ids("d0") == frozenset()


def test_multistream_residual_identity_and_disk_sum() -> None:
    model = MultiStreamLoadModel(["d0"], num_streams=4)

    model.start_load("d0", "r0", now_ms=0.0, service_ms=10.0, disk_blocks=2)
    model.start_load("d0", "r1", now_ms=0.0, service_ms=20.0, disk_blocks=3)
    model.start_load("d0", "r2", now_ms=0.0, service_ms=30.0, disk_blocks=5)

    assert model.slot_residuals_ms("d0", now_ms=5.0) == [5.0, 15.0, 25.0, 0.0]
    assert model.in_flight_request_ids("d0") == frozenset({"r0", "r1", "r2"})
    assert model.in_flight_disk_blocks("d0") == 10


@pytest.mark.parametrize(
    "model",
    [
        SingleSlotLoadModel(["d0"]),
        MultiStreamLoadModel(["d0"], num_streams=2),
    ],
)
def test_start_load_raises_on_duplicate_request_id(
    model: SingleSlotLoadModel | MultiStreamLoadModel,
) -> None:
    model.start_load("d0", "r0", now_ms=0.0, service_ms=10.0, disk_blocks=1)

    with pytest.raises(RuntimeError, match="already in-flight"):
        model.start_load("d0", "r0", now_ms=1.0, service_ms=10.0, disk_blocks=1)


def test_multistream_claims_first_idle_slot_by_index() -> None:
    model = MultiStreamLoadModel(["d0"], num_streams=4)
    model.start_load("d0", "r0", now_ms=0.0, service_ms=100.0, disk_blocks=1)
    model.start_load("d0", "r1", now_ms=0.0, service_ms=200.0, disk_blocks=1)
    model.complete_load("d0", "r0", now_ms=10.0)

    model.start_load("d0", "r2", now_ms=10.0, service_ms=30.0, disk_blocks=1)

    assert model.slot_residuals_ms("d0", now_ms=10.0) == [30.0, 190.0, 0.0, 0.0]


def test_multistream_requires_positive_stream_count() -> None:
    with pytest.raises(ValueError, match="num_streams"):
        MultiStreamLoadModel(["d0"], num_streams=0)
