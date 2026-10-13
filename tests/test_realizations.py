"""One realization (4 of 8, seed 12) of `port.sim.realizations`' genome through
`run_cnaster_port` (#291).

`release`: each realization is a whole run in its own process.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from port.sim.realizations import (
    GENOME,
    chosen,
    fit_one,
    planted_genome,
    planted_minor,
    planted_mu,
)

if TYPE_CHECKING:
    from port.sim.realizations import Fit
    from port.sim.truth import CoreInferenceTruth

    First = tuple[CoreInferenceTruth, Fit]

pytestmark = pytest.mark.release


@pytest.fixture(scope="module")
def first(tmp_path_factory: pytest.TempPathFactory) -> First:
    root: Path = tmp_path_factory.mktemp("realizations")
    index = chosen(8, int(GENOME["seed"]))

    return planted_genome(), fit_one(None, index, root, errors=True)


@pytest.mark.oracle
def test_the_rebuilt_objective_is_at_its_optimum_where_the_pipeline_stopped(
    first: First,
) -> None:
    """The Newton decrement of the `jax` objective at cnaster's fit is below 5e-2 (EM stops at `tol = 1e-3`)."""
    _, fit = first

    assert fit.decrement is not None
    assert fit.decrement < 5e-2, f"Newton decrement {fit.decrement:.2e}"


@pytest.mark.bug
def test_the_fit_is_many_standard_errors_from_the_planted_rates(first: First) -> None:
    """With the shift on, `p` is within 0.5 sigma and `mu` reads low; pinned as found (#293)."""

    truth, fit = first
    assert fit.covariance is not None

    sigma = np.sqrt(np.stack([fit.covariance[:, 0, 0], fit.covariance[:, 1, 1]]).T)
    sigma = np.where(sigma > 0.0, sigma, np.nan)
    planted = np.stack([planted_mu(truth), planted_minor(truth)]).T
    bias = (np.stack([fit.mu, fit.p]).T - planted) / sigma

    unpinned = np.isfinite(bias[:, 0])

    assert np.all(np.abs(bias[unpinned, 0]) > 5.0), f"mu bias {bias[:, 0]} sigma"
