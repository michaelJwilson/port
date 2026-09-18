"""What fusing the emission into the field is worth (issue #59 item 2).

Two claims, and only one of them is a ratio.

**The flops.** The two-step scores every `n_states` at every `(bin, spot)`
and the field reads `n_clones` of them, so fusing saves a factor
`n_states / n_clones`. Measured against item 1's reordered field:

| `n_obs` | `n_spots` | states | clones | two-step | fused | ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3,000 | 5,000 | 7 | 4 | 16,046 ms | 7,698 ms | 2.08 |
| 3,000 | 5,000 | 7 | 7 | 16,102 ms | 14,043 ms | 1.15 |
| 10,000 | 2,500 | 7 | 4 | 26,735 ms | 14,449 ms | 1.85 |

`CLAUDE.md` puts a speedup claim at 2x, so **this clears it only where
`n_clones < n_states`**. At `n_clones == n_states` there is nothing to save
and the 1.15x is the materialization alone.

**The memory, which is a capability rather than a ratio.** At `n_states = 7`,
`n_obs = 30,000`, `n_spots = 5,000` the two emission channels are 16.8 GB.
Measured on a machine with 13 GB free: the fused form completed in 67.9 s,
and the two-step allocated its first 8.4 GB channel and was killed by the
kernel on the second. That size is not in this file -- it takes minutes and
needs the memory -- but the arithmetic it rests on is asserted in
`test_hmrf_fused_field.py`.
"""

import numpy as np
import pytest
from port.patch.hmrf_field import compute_loglike_spot_assignment_strided
from port.patch.hmrf_fused_field import fused_spot_clone_field
from pytest_benchmark.fixture import BenchmarkFixture

from tests.fixtures import SpotCloneField, spot_clone_field

GATE = {"n_states": 5, "n_obs": 400, "n_spots": 300, "n_clones": 2}
STRESS = {"n_states": 7, "n_obs": 2000, "n_spots": 2000, "n_clones": 4}
"""`n_clones < n_states`, which is where the flop saving exists at all."""


def _two_step(fixture: SpotCloneField, weight: np.ndarray) -> np.ndarray:
    from cnaster.hmm_nophasing import _dense_bb_logpmf, _dense_nb_logpmf

    rdr = _dense_nb_logpmf(
        fixture.counts_nb, fixture.base_nb_mean, fixture.log_mu, fixture.alphas
    )
    baf = _dense_bb_logpmf(
        fixture.counts_bb, fixture.total_bb_RD, fixture.p_binom, fixture.taus
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


def _fused(fixture: SpotCloneField, weight: np.ndarray) -> np.ndarray:
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
    )
    return field


@pytest.mark.infra
@pytest.mark.benchmark
def test_two_step_gate(benchmark: BenchmarkFixture) -> None:
    """Producer plus reordered field, at gate size."""
    fixture = spot_clone_field(**GATE)
    weight = np.ones(fixture.n_spots)
    _two_step(fixture, weight)
    benchmark(_two_step, fixture, weight)


@pytest.mark.infra
@pytest.mark.benchmark
def test_fused_gate(benchmark: BenchmarkFixture) -> None:
    """One pass, at gate size, for the pair."""
    fixture = spot_clone_field(**GATE)
    weight = np.ones(fixture.n_spots)
    _fused(fixture, weight)
    benchmark(_fused, fixture, weight)


@pytest.mark.infra
@pytest.mark.benchmark
@pytest.mark.release
def test_two_step_stress(benchmark: BenchmarkFixture) -> None:
    """Producer plus field at a size where the emission is 0.9 GB."""
    fixture = spot_clone_field(**STRESS)
    weight = np.ones(fixture.n_spots)
    _two_step(fixture, weight)
    benchmark(_two_step, fixture, weight)


@pytest.mark.infra
@pytest.mark.benchmark
@pytest.mark.release
def test_fused_stress(benchmark: BenchmarkFixture) -> None:
    """The same, fused, holding no emission at all."""
    fixture = spot_clone_field(**STRESS)
    weight = np.ones(fixture.n_spots)
    _fused(fixture, weight)
    benchmark(_fused, fixture, weight)
