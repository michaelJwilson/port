"""The field from tables against the fused kernel's `lgamma` (#433).

Bitwise equal (`tests/test_hmrf_tabulated_field.py`), so the rows are a
ratio and nothing else. Tables cost `n_states x max count` `lgamma` once per
call, so at the gate size they are overhead; the stress size is where the
`n_obs x n_spots x n_clones` scores dominate.
"""

from collections.abc import Callable

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from tests.fixtures import SpotCloneField, spot_clone_field, tiers

GATE = {"n_states": 5, "n_obs": 400, "n_spots": 300, "n_clones": 2}
STRESS = {"n_states": 7, "n_obs": 3000, "n_spots": 5000, "n_clones": 4}


def _fused(fixture: SpotCloneField, weight: np.ndarray) -> np.ndarray:
    from port.patch.hmrf.fused_field import fused_spot_clone_field

    field: np.ndarray = fused_spot_clone_field(
        fixture.counts_nb,
        fixture.base_nb_mean,
        fixture.counts_bb,
        fixture.total_bb_RD,
        fixture.log_mu,
        fixture.alphas,
        fixture.p_binom,
        fixture.taus,
        fixture.pred,
        weight,
        np.empty((fixture.n_spots, fixture.n_clones)),
    )
    return field


def _tabulated(fixture: SpotCloneField, weight: np.ndarray) -> np.ndarray:
    from port.patch.hmrf.tabulated_field import tabulated_spot_clone_field

    field: np.ndarray = tabulated_spot_clone_field(
        fixture.counts_nb,
        fixture.base_nb_mean,
        fixture.counts_bb,
        fixture.total_bb_RD,
        fixture.log_mu,
        fixture.alphas,
        fixture.p_binom,
        fixture.taus,
        fixture.pred,
        weight,
        np.empty((fixture.n_spots, fixture.n_clones)),
    )
    return field


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


INTEGRAL_GATE = {"n_obs": 400, "n_spots": 300}
INTEGRAL_STRESS = {"n_obs": 3779, "n_spots": 8649}
"""The stress size is the 2x rung's BAF stage, `(3779, 2, 8649)` (#569)."""


@pytest.mark.benchmark
@pytest.mark.parametrize("size", tiers(INTEGRAL_GATE, INTEGRAL_STRESS))
@pytest.mark.parametrize("arm", ["reference", "compiled"])
def test_integral_check(
    benchmark: BenchmarkFixture, arm: str, size: dict[str, int]
) -> None:
    """The table kernel's integer check on a strided `single_X[:, 1, :]` view, warm."""
    from port.patch.hmrf.tabulated_field import _integral, _integral_reference

    counts = np.random.default_rng(0).poisson(2, (size["n_obs"], 2, size["n_spots"]))
    view = counts[:, 1, :]
    check = _integral if arm == "compiled" else _integral_reference
    check(view)
    assert benchmark(check, view)
