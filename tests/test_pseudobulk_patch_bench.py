"""Blocked pseudobulk merge against `cnaster`'s, at #487's 6,000-spot sample (#488)."""

from collections.abc import Callable
from typing import Any

import pytest
from cnaster.pseudobulk import merge_pseudobulk_by_index_mix
from port.patch.pseudobulk import (
    merge_pseudobulk_by_index_mix as port_merge_pseudobulk_by_index_mix,
)
from pytest_benchmark.fixture import BenchmarkFixture

from tests.fixtures import pseudobulk_inputs, tiers

GATE = {"n_obs": 300, "n_spots": 400, "n_clones": 3}
STRESS = {"n_obs": 4000, "n_spots": 6000, "n_clones": 4}


def _upstream() -> Callable[..., Any]:
    return merge_pseudobulk_by_index_mix  # type: ignore[no-any-return]


def _blocked() -> Callable[..., Any]:
    return port_merge_pseudobulk_by_index_mix


@pytest.mark.benchmark
@pytest.mark.parametrize("size", tiers(GATE, STRESS))
@pytest.mark.parametrize("arm", [_upstream, _blocked], ids=["cnaster", "blocked"])
def test_merge(
    benchmark: BenchmarkFixture,
    arm: Callable[[], Callable[..., Any]],
    size: dict[str, int],
) -> None:
    """One merge, warm."""
    inputs = pseudobulk_inputs(size["n_obs"], size["n_spots"], size["n_clones"], seed=0)
    function = arm()
    function(**inputs)
    benchmark(function, **inputs)
