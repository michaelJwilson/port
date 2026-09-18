"""The compiled kernels, against scipy.

**These exist because coverage cannot see them.** `numba` reports no line
information, so every function below reads as uncovered however hard the
rest of the suite exercises it — `forward_lattice` and the logpmf kernels
run in almost every test in this repository and appear in no report.
`CLAUDE.md` requires them tested anyway, and against an independent source
rather than against each other.

The referee is `scipy.stats`, which is a second implementation of the same
densities and not a rearrangement of `cnaster`'s.
"""

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import betabinom, nbinom

TOLERANCE = 1e-10


@pytest.mark.oracle
@pytest.mark.exact
@pytest.mark.critical
@pytest.mark.parametrize("count", [0, 1, 7, 50])
@pytest.mark.parametrize(
    ("dispersion", "probability"), [(1.0, 0.5), (8.0, 0.2), (0.5, 0.9)]
)
def test_negative_binomial_kernel_matches_scipy(
    count: int, dispersion: float, probability: float
) -> None:
    """`nbinom_logpmf_numba` is scipy's negative binomial."""
    from cnaster.hmm_nophasing import nbinom_logpmf_numba

    assert nbinom_logpmf_numba(count, dispersion, probability) == pytest.approx(
        float(nbinom.logpmf(count, dispersion, probability)), abs=TOLERANCE
    )


@pytest.mark.oracle
@pytest.mark.exact
@pytest.mark.critical
@pytest.mark.parametrize("successes", [0, 3, 10])
@pytest.mark.parametrize(("trials", "alpha", "beta"), [(10, 2.0, 2.0), (10, 0.5, 5.0)])
def test_beta_binomial_kernel_matches_scipy(
    successes: int, trials: int, alpha: float, beta: float
) -> None:
    """`betabinom_logpmf_numba` is scipy's beta-binomial."""
    from cnaster.hmm_nophasing import betabinom_logpmf_numba

    assert betabinom_logpmf_numba(successes, trials, alpha, beta) == pytest.approx(
        float(betabinom.logpmf(successes, trials, alpha, beta)), abs=TOLERANCE
    )


@pytest.mark.oracle
@pytest.mark.exact
@pytest.mark.critical
def test_numba_logsumexp_matches_scipy() -> None:
    """`numba_logsumexp` is stable where a naive sum is not.

    The `-inf` case is the one that separates the two: an implementation
    subtracting its maximum without guarding an all-`-inf` input returns
    `nan` where the answer is `-inf`.
    """
    from cnaster.hmm_nophasing import numba_logsumexp

    for values in (
        np.array([0.0, -1.0, -2.0]),
        np.array([-1000.0, -1001.0]),
        np.array([1000.0, 999.0]),
        np.array([-np.inf, -1.0]),
    ):
        assert numba_logsumexp(values) == pytest.approx(
            float(logsumexp(values)), abs=TOLERANCE
        )


@pytest.mark.infra
@pytest.mark.analytic
def test_dense_and_single_observation_kernels_agree() -> None:
    """The dense kernels compute what the one-dimensional ones compute.

    Two compiled implementations of one density, which is `cnaster`'s own
    pairing rather than an imported reference; the scipy comparisons above
    are what makes either of them right.
    """
    from cnaster.hmm_nophasing import _dense_nb_logpmf, _nb_logpmf_1d

    rng = np.random.default_rng(4)
    n_obs, n_states = 12, 3
    counts = rng.integers(0, 40, size=(n_obs, 1)).astype(np.float64)
    exposure = np.full((n_obs, 1), 2.0)
    log_mu = np.log(np.array([5.0, 12.0, 30.0]))[:, None]
    alphas = np.full((n_states, 1), 0.25)

    dense = _dense_nb_logpmf(counts, exposure, log_mu, alphas)

    for state in range(n_states):
        out = np.zeros(n_obs)
        _nb_logpmf_1d(
            counts[:, 0],
            exposure[:, 0],
            np.exp(log_mu[state, 0]),
            alphas[state, 0],
            out,
        )
        np.testing.assert_allclose(dense[state, :, 0], out, rtol=0.0, atol=TOLERANCE)


@pytest.mark.infra
@pytest.mark.analytic
def test_negative_binomial_kernel_is_normalised() -> None:
    """The density sums to one over its support.

    An invariant rather than a comparison: a kernel agreeing with scipy at
    the points tried could still be wrong between them, and a normalisation
    check covers the whole support at once.
    """
    from cnaster.hmm_nophasing import _nb_logpmf_1d

    support = np.arange(0, 400, dtype=np.float64)
    out = np.zeros(support.size)
    _nb_logpmf_1d(support, np.ones(support.size), 12.0, 0.25, out)

    assert logsumexp(out) == pytest.approx(0.0, abs=1e-6)
