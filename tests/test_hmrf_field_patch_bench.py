"""Benchmark: `cnaster`'s spot/clone field against the strided patch (#59 item 1).

Gate per PR; stress (`n_obs` 3,000, 5,000 spots, 7 clones) under `release`.
"""

from collections.abc import Callable

import numpy as np
import pytest
from port.patch.hmrf.field import compute_loglike_spot_assignment_strided
from pytest_benchmark.fixture import BenchmarkFixture

from tests.fixtures import SpotCloneField, spot_clone_field, tiers

GATE = {"n_states": 5, "n_obs": 400, "n_spots": 300, "n_clones": 3}
"""Small enough for the per-PR budget; decides no ratio."""

STRESS = {"n_states": 7, "n_obs": 3000, "n_spots": 5000, "n_clones": 7}
"""1.7 GB across the two channels, where the layout tells."""


def _run_cnaster(fixture: SpotCloneField) -> np.ndarray:
    from cnaster.hmrf import compute_loglike_spot_assignment

    field: np.ndarray = compute_loglike_spot_assignment(
        fixture.n_spots,
        np.ones(fixture.n_spots),
        np.ones(fixture.n_spots),
        np.empty(0),
        False,
        fixture.log_emission_rdr,
        fixture.log_emission_baf,
        fixture.pred,
        fixture.n_obs,
        fixture.n_clones,
    )
    return field


def _run_patch(fixture: SpotCloneField) -> np.ndarray:
    field: np.ndarray = compute_loglike_spot_assignment_strided(
        fixture.n_spots,
        np.ones(fixture.n_spots),
        np.ones(fixture.n_spots),
        np.empty(0),
        False,
        fixture.log_emission_rdr,
        fixture.log_emission_baf,
        fixture.pred,
        fixture.n_obs,
        fixture.n_clones,
    )
    return field


@pytest.mark.benchmark
@pytest.mark.parametrize("size", tiers(GATE, STRESS))
@pytest.mark.parametrize("arm", [_run_cnaster, _run_patch], ids=["cnaster", "patched"])
def test_field(
    benchmark: BenchmarkFixture,
    arm: Callable[[SpotCloneField], np.ndarray],
    size: dict[str, int],
) -> None:
    """Time each arm's field kernel, warmed by one call."""
    fixture = spot_clone_field(**size)
    arm(fixture)
    benchmark(arm, fixture)
