"""Field from tables against the fused kernel's `lgamma`, at gate and stress sizes (#433)."""

from collections.abc import Callable
from functools import partial

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from tests.fixtures import SpotCloneField, fused_field_of, spot_clone_field, tiers

GATE = {"n_states": 5, "n_obs": 400, "n_spots": 300, "n_clones": 2}
STRESS = {"n_states": 7, "n_obs": 3000, "n_spots": 5000, "n_clones": 4}


_fused = fused_field_of
_tabulated = partial(fused_field_of, tabulated=True)


@pytest.mark.benchmark
@pytest.mark.parametrize("size", tiers(GATE, STRESS))
@pytest.mark.parametrize("arm", [_fused, _tabulated], ids=["fused", "tabulated"])
def test_field(
    benchmark: BenchmarkFixture,
    arm: Callable[[SpotCloneField, np.ndarray], np.ndarray],
    size: dict[str, int],
) -> None:
    """One field, warm."""
    fixture = spot_clone_field(**size)
    weight = np.ones(fixture.n_spots)
    arm(fixture, weight)
    benchmark(arm, fixture, weight)
