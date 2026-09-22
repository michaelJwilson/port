"""One realization of `tests.realizations`' genome, through `run_cnaster_port` (#291).

`release`: each realization is a whole pipeline run, 35 to 40 s, in its own
process because a run peaks at 4.7 GB. The figure is
`python -m tests.realizations`; these two tests pin what its first
realization says, at the planted genome it is drawn for.
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
    from tests.realizations import fit_one, planted_genome

    root: Path = tmp_path_factory.mktemp("realizations")

    return planted_genome(), fit_one(None, 0, root, errors=True)


@pytest.mark.oracle
def test_the_rebuilt_objective_is_at_its_optimum_where_the_pipeline_stopped(
    first: First,
) -> None:
    """The Newton decrement of the `jax` objective at `cnaster`'s fit.

    The covariance is the curvature of an objective rebuilt from
    `run_core_inference`'s captured inputs, not the one `cnaster` optimized.
    If the two were different objectives, the pipeline's point would not be
    an optimum of the rebuilt one and the decrement would say so. Stated
    below 1e-2 in chi-square units, a tenth of a standard error, because the
    pipeline's EM stops at `tol = 1e-3` rather than at the optimum; realized
    3.7e-03.
    """
    _, fit = first

    assert fit.decrement is not None
    assert fit.decrement < 1e-2, f"Newton decrement {fit.decrement:.2e}"


@pytest.mark.bug
def test_the_fit_is_many_standard_errors_from_the_planted_rates(first: First) -> None:
    """Precise and biased: `mu` misses the truth by 31 to 50 of its sigma.

    `mu` unshifted on both sides: the planted one realizes UMIs relative to
    normal coverage, which the fit's `exp(log_mu)` estimates. Realized on
    realization 0: 0.767, 1.056, 2.010 against planted 1, 1.5, 3, at -32.7,
    -50.4 and -30.9 standard errors -- 0.77, 0.70 and 0.67 of the truth, so
    close to one common scale. **The neutral state is the one to read
    first**: it is the whole of clone 0, whose spots are the normal ones, and
    it comes back at 0.77 rather than 1. The planted 0.42 allele fraction
    comes back 0.483, at 54.1 sigma.

    Undiagnosed, and pinned as found: written to fail when every `mu` is
    within five of its standard errors of truth. Seven more realizations are
    in the figure; three of them recover the 0.42.
    """
    from tests.realizations import planted_minor, planted_mu

    truth, fit = first
    assert fit.covariance is not None

    sigma = np.sqrt(np.stack([fit.covariance[:, 0, 0], fit.covariance[:, 1, 1]]).T)
    planted = np.stack([planted_mu(truth), planted_minor(truth)]).T
    bias = (np.stack([fit.mu, fit.p]).T - planted) / sigma

    assert np.all(np.abs(bias[:, 0]) > 5.0), f"mu bias {bias[:, 0]} sigma"
    assert abs(bias[1, 1]) > 5.0, f"p bias of the (1.5, 0.42) state {bias[1, 1]}"
