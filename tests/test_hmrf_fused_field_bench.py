"""Two-step emission plus field against the fused field (#59 item 2).

The flop saving is `n_states / n_clones`; the memory claim is asserted in
`test_hmrf_fused_field.py`.
"""

from collections.abc import Callable

import numpy as np
import pytest
from port.patch.hmrf.field import compute_loglike_spot_assignment_strided
from pytest_benchmark.fixture import BenchmarkFixture

from tests.fixtures import SpotCloneField, fused_field_of, spot_clone_field, tiers

GATE = {"n_states": 5, "n_obs": 400, "n_spots": 300, "n_clones": 2}
STRESS = {"n_states": 7, "n_obs": 2000, "n_spots": 2000, "n_clones": 4}
"""`n_clones < n_states`, which is where the flop saving exists at all."""


def _two_step(fixture: SpotCloneField, weight: np.ndarray) -> np.ndarray:
    from cnaster.hmm_nophasing import _dense_bb_logpmf, _dense_nb_logpmf

    # NB `_dense_*_logpmf` indexes `[i, 0]`; the fused kernel takes `(n_states,)` (#278).
    rdr = _dense_nb_logpmf(
        fixture.counts_nb,
        fixture.base_nb_mean,
        fixture.log_mu[:, None],
        fixture.alphas[:, None],
    )
    baf = _dense_bb_logpmf(
        fixture.counts_bb,
        fixture.total_bb_RD,
        fixture.p_binom[:, None],
        fixture.taus[:, None],
    )
    field: np.ndarray = compute_loglike_spot_assignment_strided(
        fixture.n_spots,
        weight,
        weight,
        np.empty(0),
        False,
        rdr,
        baf,
        fixture.pred,
        fixture.n_obs,
        fixture.n_clones,
    )
    return field


@pytest.mark.benchmark
@pytest.mark.parametrize("size", tiers(GATE, STRESS))
@pytest.mark.parametrize("arm", [_two_step, fused_field_of], ids=["two-step", "fused"])
def test_field(
    benchmark: BenchmarkFixture,
    arm: Callable[[SpotCloneField, np.ndarray], np.ndarray],
    size: dict[str, int],
) -> None:
    """Producer plus reordered field, against one pass holding no emission (0.9 GB at
    stress).
    """
    fixture = spot_clone_field(**size)
    weight = np.ones(fixture.n_spots)
    arm(fixture, weight)
    benchmark(arm, fixture, weight)
