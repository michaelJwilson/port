"""#560: the log-space negative binomial against scipy, against `cnaster` where `cnaster` is right, and at the mean where it is not."""

from __future__ import annotations

import numpy as np
import pytest
from cnaster.hmm_nophasing import _nb_logpmf_1d as upstream
from port.patch.hmm_nophasing import nb_logpmf as patch
from scipy.stats import nbinom

COUNTS = np.array([0, 1, 7, 42, 300, 1000, 2500], dtype=np.float64)


def _scores(kernel, mu: float, alpha: float, exposure: float = 1000.0) -> np.ndarray:  # type: ignore[no-untyped-def]
    out = np.zeros(COUNTS.size)
    kernel(COUNTS, np.full(COUNTS.size, exposure), mu, alpha, out)
    return out


CASES = [
    (mu, alpha)
    for mu in (1e-4, 0.3, 1.0, 2.7)
    for alpha in (1e-3, 0.07, 0.5, 3.0)
    if alpha * 1000.0 * mu >= 1e-4
]
"""Where `alpha * lambda >= 1e-4`: below it scipy forms `p = 1 / (1 + a)` and loses digits itself (8e-8 at 1e-6)."""


@pytest.mark.oracle
@pytest.mark.parametrize(("mu", "alpha"), CASES)
def test_the_patched_kernel_is_scipys_negative_binomial(
    mu: float, alpha: float
) -> None:
    """`nbinom.logpmf(k, 1/alpha, 1/(1 + alpha*lambda))` to 1e-9 relative, 1e-9 absolute."""
    lam = 1000.0 * mu
    exact = nbinom.logpmf(COUNTS, 1.0 / alpha, 1.0 / (1.0 + alpha * lam))

    np.testing.assert_allclose(
        _scores(patch._nb_logpmf_1d, mu, alpha), exact, rtol=1e-9, atol=1e-9
    )


@pytest.mark.patch
@pytest.mark.parametrize("mu", [1e-5, 0.3, 1.0, 2.7])
def test_the_patched_kernel_is_cnasters_where_cnasters_p_is_below_one(
    mu: float,
) -> None:
    """Where `alpha * lambda >= 1e-8`, upstream's kernel to 1e-9 relative: the patch changes only the rounding regime."""

    np.testing.assert_allclose(
        _scores(patch._nb_logpmf_1d, mu, 0.12),
        _scores(upstream, mu, 0.12),
        rtol=1e-9,
        atol=1e-9,
    )


@pytest.mark.bug
def test_cnaster_scores_any_count_at_probability_one_once_p_rounds_to_one() -> None:
    """At `log mu = -43.22` (dev_tree_1s_hard r0) upstream scores 1,000 UMIs at 0; the patch below -30,000."""

    mu, alpha = float(np.exp(-43.22)), 0.1184
    assert (_scores(upstream, mu, alpha) == 0.0).all()
    assert _scores(patch._nb_logpmf_1d, mu, alpha)[COUNTS == 1000][0] < -30_000.0
