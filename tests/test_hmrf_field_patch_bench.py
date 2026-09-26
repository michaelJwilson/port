"""What the layout costs `cnaster`'s spot/clone field, measured.

Issue #59 item 1. The gate pair runs per pull request; the stress sweep
carries `release`, because one stress point allocates 1.7 GB and runs for
seconds.

**A ratio at a gate size decides nothing** (`CLAUDE.md`, Measurement), so the
gate rows exist to catch a regression in either kernel, not to argue the
patch. The argument is the stress pair.

The patch takes `cnaster`'s own layout, so there is no transpose to time and
no producer change to account for. An earlier form transposed the emission
instead: it measured 61.8 ms against this one's 36.1 ms at 3,000 x 5,000 x 7,
because it fixed the contiguity and left the scalar reduction.

Recorded, minimum of three runs at `n_states = 7`:

| `n_obs` | `n_spots` | clones | `cnaster` | patch | ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3,000 | 5,000 | 4 | 69.6 ms | 17.8 ms | 3.90 |
| 3,000 | 5,000 | 7 | 198.7 ms | 36.1 ms | 5.50 |
| 10,000 | 2,500 | 7 | 483.6 ms | 67.4 ms | 7.17 |

**`n_obs = 30,000` at `n_spots = 5,000` does not fit**: the two channels are
16.8 GB, and 33.6 GB with the transposed copies a comparison needs. That is
issue #59 item 2 -- the materialization is a scaling limit rather than an
optimization -- and it is why the large row trades spots for bins.
"""

from collections.abc import Callable

import numpy as np
import pytest
from port.patch.hmrf.field import compute_loglike_spot_assignment_strided
from pytest_benchmark.fixture import BenchmarkFixture

from tests.fixtures import SpotCloneField, spot_clone_field, tiers

GATE = {"n_states": 5, "n_obs": 400, "n_spots": 300, "n_clones": 3}
"""Small enough for the per-pull-request budget; decides no ratio."""

STRESS = {"n_states": 7, "n_obs": 3000, "n_spots": 5000, "n_clones": 7}
"""1.7 GB across the two channels, which is where the layout tells."""


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
    """`cnaster`'s layout against the transposed one, warmed by one call.

    The transpose itself is **not** timed, and that is deliberate rather than
    favourable: at the stress size it costs 606 ms against a 74 ms kernel, so
    doing it per call never pays. The change belongs in
    `compute_emission_probability_nb_betabinom`'s output indexing, and what
    the patched row prices is the kernel a producer emitting that layout
    would feed. Timing the copy would price a patch nobody is proposing.
    """
    fixture = spot_clone_field(**size)
    arm(fixture)
    benchmark(arm, fixture)
