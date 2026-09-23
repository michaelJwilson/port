"""One realization of `tests.realizations`' genome, through `run_cnaster_port` (#291).

`release`: each realization is a whole pipeline run, 35 to 40 s, in its own
process because a run peaks at 4.7 GB. The figure is
`python -m tests.realizations`; these two tests pin what its drawn
realization (4 of 8, seed 12) says, at the planted genome it is drawn for.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from tests.fixtures import CoreInferenceTruth
    from tests.realizations import Fit

    First = tuple[CoreInferenceTruth, Fit]

pytestmark = pytest.mark.release


@pytest.fixture(scope="module")
def first(tmp_path_factory: pytest.TempPathFactory) -> First:
    from tests.realizations import GENOME, chosen, fit_one, planted_genome

    root: Path = tmp_path_factory.mktemp("realizations")
    index = chosen(8, int(GENOME["seed"]))  # type: ignore[call-overload]

    return planted_genome(), fit_one(None, index, root, errors=True)


@pytest.mark.oracle
def test_the_rebuilt_objective_is_at_its_optimum_where_the_pipeline_stopped(
    first: First,
) -> None:
    """The Newton decrement of the `jax` objective at `cnaster`'s fit.

    The covariance is the curvature of an objective rebuilt from
    `run_core_inference`'s captured inputs, not the one `cnaster` optimized.
    If the two were different objectives, the pipeline's point would not be
    an optimum of the rebuilt one and the decrement would say so. Stated
    below 5e-2 in chi-square units, a fifth of a standard error, because the
    pipeline's EM stops at `tol = 1e-3` rather than at the optimum; realized
    2.4e-02 with the shift on.
    """
    _, fit = first

    assert fit.decrement is not None
    assert fit.decrement < 5e-2, f"Newton decrement {fit.decrement:.2e}"


@pytest.mark.bug
def test_the_fit_is_many_standard_errors_from_the_planted_rates(first: First) -> None:
    """With the shift on, `p` is recovered and `mu` still reads low.

    Realization 4, pinned so the neutral state is 1: `mu` 1.340 and 2.525
    against planted 1.5 and 3, at -6.5 and -11.2 standard errors; `p` within
    0.5 sigma in every state. **Not a local optimum**: on realization 1,
    which lands in the same place, the shifted likelihood prefers the fit
    to the planted parameters by 67 nats. Two of eight realizations recover
    `(1.5, 3)` to 0.5 per cent.

    Undiagnosed (#293), and pinned as found: written to fail when every
    unpinned `mu` is within five of its standard errors of truth.
    """
    from tests.realizations import planted_minor, planted_mu

    truth, fit = first
    assert fit.covariance is not None

    sigma = np.sqrt(np.stack([fit.covariance[:, 0, 0], fit.covariance[:, 1, 1]]).T)
    sigma = np.where(sigma > 0.0, sigma, np.nan)
    planted = np.stack([planted_mu(truth), planted_minor(truth)]).T
    bias = (np.stack([fit.mu, fit.p]).T - planted) / sigma

    unpinned = np.isfinite(bias[:, 0])

    assert np.all(np.abs(bias[unpinned, 0]) > 5.0), f"mu bias {bias[:, 0]} sigma"
