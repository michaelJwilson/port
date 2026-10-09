"""`port.sandbox`'s negative binomial sites at a vanishing mean (#560, #540).

No console script reaches them; judged against `tests.exact_densities`, as the live
sites in `tests.test_nb_bb_edges` are.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

import numpy as np
import port.patch.hmm_nophasing.shifted_emission
import port.patch.hmm_phased.coded_emission  # noqa: F401  -- a site once imported
import pytest
from port.sandbox.np_merge import _emissions
from scipy.special import logsumexp

from tests.exact_densities import nb_logpmf
from tests.test_nb_bb_edges import (
    COUNTS,
    DEGENERATE,
    NORMAL,
    SANDBOX_KERNELS,
    VANISHING,
    _exact,
)


@pytest.mark.oracle
@pytest.mark.parametrize("name", SANDBOX_KERNELS)
@pytest.mark.parametrize(("mean", "alpha"), NORMAL)
def test_each_sandbox_kernel_is_the_negative_binomial(
    name: str, mean: float, alpha: float
) -> None:
    """The 50-digit sums of logs, to 1e-9 relative and 1e-12 absolute."""
    np.testing.assert_allclose(
        SANDBOX_KERNELS[name](COUNTS, mean, alpha),
        _exact(COUNTS, mean, alpha),
        rtol=1e-9,
        atol=1e-12,
    )


@pytest.mark.analytic
@pytest.mark.parametrize("name", SANDBOX_KERNELS)
@pytest.mark.parametrize("mean", [1e-3, VANISHING])
def test_each_sandbox_kernel_sums_to_one_at_a_small_mean(
    name: str, mean: float
) -> None:
    """`logsumexp` over counts 0..200 is 0 to 1e-12: a pmf, not a score of 1 per count."""
    counts = np.arange(201, dtype=np.float64)

    assert abs(logsumexp(SANDBOX_KERNELS[name](counts, mean, 0.5))) < 1e-12


def _bound_kernel(module: str) -> float:
    """The `_nb_logpmf_1d` `module` compiles in by name, on one degenerate bin."""

    log_mu, alpha, exposure, count = DEGENERATE
    out = np.full(1, np.nan)
    importlib.import_module(module)._nb_logpmf_1d(
        np.array([count]), np.array([exposure]), float(np.exp(log_mu)), alpha, out
    )
    return float(out[0])


def _np_merge() -> float:
    log_mu, alpha, exposure, count = DEGENERATE
    X = np.zeros((1, 2, 1))
    X[0, 0, 0] = count
    res = {
        "new_log_mu": [log_mu],
        "new_alphas": [alpha],
        "new_p_binom": [0.5],
        "new_taus": [30.0],
    }
    rdr, _ = _emissions(
        X, np.full((1, 1), exposure), np.zeros((1, 1)), res, np.zeros(1)
    )
    return float(rdr[0, 0, 0])


SANDBOX_SITES: dict[str, Callable[[], float]] = {
    "np_merge": _np_merge,
    "hmm_initialize_backends": lambda: _bound_kernel(
        "port.sandbox.patch.hmm_initialize.backends"
    ),
}
"""The sandbox sites that compile `_nb_logpmf_1d` in by name, out of `LOG_SPACE_SWAPS`' reach."""


@pytest.mark.oracle
@pytest.mark.parametrize("site", sorted(SANDBOX_SITES))
def test_every_sandbox_site_scores_the_degenerate_state_in_log_space(
    site: str,
) -> None:
    """At `log mu = -43.22`, exposure 1000, each sandbox site scores count 1000 as -38,403.9 nats, to 1e-9 (#560)."""
    log_mu, alpha, exposure, count = DEGENERATE
    score = SANDBOX_SITES[site]()

    assert score < -30_000.0
    np.testing.assert_allclose(
        score, nb_logpmf(int(count), exposure * float(np.exp(log_mu)), alpha), rtol=1e-9
    )
