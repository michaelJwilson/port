"""Field from tables against the fused kernel's `lgamma`, at gate and stress sizes (#433)."""

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
