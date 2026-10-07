"""#561: the beta-binomial at a large concentration, against 50-digit sums of logs.

`a = p tau` and `b = (1 - p) tau`. `cnaster` and `sal` subtract `lgamma`
values near `tau log tau`; `port.patch.hmm_nophasing.bb_logpmf` forms each
rising factorial without that subtraction. The referee is
`tests.exact_densities`, the density written from its definition in
`decimal`. Every port site on the live path is judged here: the kernel the
`LOG_SPACE_SWAPS` rows install, `dense_emission.bb_states` (the HMM under
`--sal`), the M-step gradient and `copy_likelihood`'s decode.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp

from tests.exact_densities import bb_logpmf, digamma_rise

FLOOR = 1e-10
TRIALS = 100
COUNTS = np.array([0, 1, 7, 42, 99, 100], dtype=np.float64)
TAUS = [10.0, 999.0, 5e3, 1e5, 1e8, 1e12, 1e16]
SHARES = [0.3, 1e-12]


def _shapes(p: float, tau: float) -> tuple[float, float]:
    return max(p * tau, FLOOR), max((1.0 - p) * tau, FLOOR)


def _exact(p: float, tau: float) -> np.ndarray:
    a, b = _shapes(p, tau)
    return np.array([bb_logpmf(int(k), TRIALS, a, b) for k in COUNTS])


def _kernel(p: float, tau: float) -> np.ndarray:
    from port.patch.hmm_nophasing.bb_logpmf import _bb_logpmf_1d

    out = np.full(COUNTS.size, np.nan)
    _bb_logpmf_1d(COUNTS, np.full(COUNTS.size, float(TRIALS)), p, tau, out)
    return out


@pytest.mark.oracle
@pytest.mark.parametrize("p", SHARES)
@pytest.mark.parametrize("tau", TAUS)
def test_the_kernel_is_the_beta_binomial_at_every_concentration(
    tau: float, p: float
) -> None:
    """The 50-digit sums of logs, to 1e-11 absolute; measured worst 8e-13, at 1e16.

    `cnaster`'s kernel is 5e-3 nats off at 1e12 and 1e2 at 1e16.
    """
    np.testing.assert_allclose(_kernel(p, tau), _exact(p, tau), rtol=0, atol=1e-11)


@pytest.mark.analytic
@pytest.mark.parametrize("tau", [1e8, 1e12, 1e16])
def test_the_kernel_is_a_pmf_at_a_large_concentration(tau: float) -> None:
    """Over `k = 0..100` the pmf sums to 1 within 1e-10, where `cnaster`'s sums to `e^132`."""
    from port.patch.hmm_nophasing.bb_logpmf import _bb_logpmf_1d

    k = np.arange(TRIALS + 1, dtype=np.float64)
    out = np.full(k.size, np.nan)
    _bb_logpmf_1d(k, np.full(k.size, float(TRIALS)), 0.3, tau, out)

    assert abs(logsumexp(out)) < 1e-10


@pytest.mark.patch
@pytest.mark.parametrize("p", [*SHARES, 0.0, 1.0, -0.2, 1.3])
@pytest.mark.parametrize("tau", [10.0, 999.0, 5e3, 1e5])
def test_the_kernel_is_cnasters_where_cnaster_is_exact(tau: float, p: float) -> None:
    """Up to `tau = 1e5`, where `cnaster`'s own loss is 1.3e-10, its kernel to 1e-9 absolute.

    Including the floored `p` at and beyond 0 and 1, and `k > n`, which both
    score 0.
    """
    from cnaster.hmm_nophasing import _bb_logpmf_1d as upstream
    from port.patch.hmm_nophasing.bb_logpmf import _bb_logpmf_1d

    counts = np.append(COUNTS, TRIALS + 1.0)
    trials = np.full(counts.size, float(TRIALS))
    ours, theirs = np.full(counts.size, np.nan), np.full(counts.size, np.nan)
    _bb_logpmf_1d(counts, trials, p, tau, ours)
    upstream(counts, trials, p, tau, theirs)

    np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-9)
    assert ours[-1] == 0.0


@pytest.mark.patch
def test_the_dense_kernel_is_the_per_state_kernel() -> None:
    """`_dense_bb_logpmf` is `_bb_logpmf_1d` per state and spot, bitwise, as upstream's is."""
    from port.patch.hmm_nophasing.bb_logpmf import _bb_logpmf_1d, _dense_bb_logpmf

    rng = np.random.default_rng(5)
    trials = rng.integers(0, 60, (40, 3)).astype(np.float64)
    successes = np.floor(trials * rng.uniform(0, 1, trials.shape))
    p_binom = np.array([[0.2], [0.5], [0.9]])
    taus = np.array([[30.0], [4e4], [1e13]])

    dense = _dense_bb_logpmf(successes, trials, p_binom, taus)

    for state in range(3):
        for spot in range(3):
            out = np.zeros(40)
            _bb_logpmf_1d(
                successes[:, spot],
                trials[:, spot],
                p_binom[state, 0],
                taus[state, 0],
                out,
            )
            np.testing.assert_array_equal(dense[state, :, spot], out)


@pytest.mark.oracle
@pytest.mark.parametrize("tau", [5e3, 1e12])
def test_the_sal_emission_scores_a_large_concentration_exactly(tau: float) -> None:
    """`dense_emission.bb_states`, the HMM's beta-binomial under `--sal`: 1e-9 at 5e3, 1e-11 at 1e12.

    Below `STABLE_TAU` it is sal's table (1.9e-10 at 1e5); at and above, port's
    kernel.
    """
    from port.patch.hmm_nophasing.dense_emission import STABLE_TAU, bb_states

    scores = bb_states(
        COUNTS, np.full(COUNTS.size, float(TRIALS)), np.array([0.3]), np.array([tau])
    )
    tolerance = 1e-11 if tau >= STABLE_TAU else 1e-9

    np.testing.assert_allclose(scores[0], _exact(0.3, tau), rtol=0, atol=tolerance)


@pytest.mark.oracle
@pytest.mark.parametrize("tau", [5e3, 1e12])
def test_the_copy_decode_scores_a_large_concentration_exactly(tau: float) -> None:
    """`copy_likelihood.pseudobulk_log_pmf`'s allele channel, no depth: the sums of logs to 1e-11."""
    from port.extensions.copy_likelihood import Pseudobulk, pseudobulk_log_pmf

    zeros = np.zeros(COUNTS.size)
    bulk = Pseudobulk(
        counts_nb=zeros,
        base_nb_mean=zeros,
        counts_bb=COUNTS,
        total_bb_RD=np.full(COUNTS.size, float(TRIALS)),
        normal_log_lambda=zeros,
        dispersion=0.1,
        taus=tau,
    )
    scores = pseudobulk_log_pmf(
        np.array(0.0), np.array(0.3), bulk, np.arange(COUNTS.size)
    )

    np.testing.assert_allclose(scores, _exact(0.3, tau), rtol=0, atol=1e-11)


@pytest.mark.oracle
@pytest.mark.parametrize("m", [0, 1, 7, 100, 2500])
@pytest.mark.parametrize("x", [1e-10, 0.4, 999.0, 1e3, 3e4, 1e8, 1e12, 1e16])
def test_the_digamma_rise_is_the_sum_of_reciprocals(x: float, m: int) -> None:
    """`psi(x + m) - psi(x)` against `sum_{j < m} 1 / (x + j)` at 50 digits, to 1e-12 relative."""
    from port.patch.hmm_nophasing.bb_logpmf import digamma_rise as ours

    np.testing.assert_allclose(
        float(ours(np.array(x), np.array(float(m)))),
        digamma_rise(x, m),
        rtol=1e-12,
        atol=0,
    )


@pytest.mark.oracle
@pytest.mark.parametrize("tau", [50.0, 5e3, 1e8, 1e12])
def test_the_closed_form_gradient_is_the_exact_derivative(tau: float) -> None:
    """`bb_partials` against `d ell / d a = sum 1 / (a + j) - sum 1 / (a + b + j)`, to 1e-9 relative, 1e-11 absolute.

    At 1e12 `d ell / d log tau` is of order `n^2 / tau`; differencing
    `digamma` there was off by 6e-3.
    """
    from port.patch.hmm_nophasing.gradient import bb_partials

    p = 0.3
    a, b = _shapes(p, tau)
    d_p, d_log_tau = bb_partials(
        COUNTS, np.full(COUNTS.size, float(TRIALS)), np.array(p), np.array(tau)
    )

    joint = [digamma_rise(a + b, TRIALS) for _ in COUNTS]
    d_a = np.array([digamma_rise(a, int(k)) for k in COUNTS]) - joint
    d_b = np.array([digamma_rise(b, TRIALS - int(k)) for k in COUNTS]) - joint

    np.testing.assert_allclose(d_p, tau * (d_a - d_b), rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(d_log_tau, a * d_a + b * d_b, rtol=1e-9, atol=1e-11)


# --- the dependencies' kernels --------------------------------------------


@pytest.mark.bug
def test_cnasters_beta_binomial_is_not_a_pmf_at_a_large_concentration() -> None:
    """At `tau = 1e16`, `p = 0.3`, `n = 100` `cnaster`'s pmf sums to about `e^132`, not 1.

    `lgamma(n - k + b) - lgamma(n + a + b)` cancels values near 3.6e17,
    whose spacing is 64. `cnaster` passes `bounds=None` to its M step, so
    `log tau` is unbounded; fits so far stop at 6e4 to 8e4.
    """
    from cnaster.hmm_nophasing import _bb_logpmf_1d

    k = np.arange(TRIALS + 1, dtype=np.float64)
    out = np.full(k.size, np.nan)
    _bb_logpmf_1d(k, np.full(k.size, float(TRIALS)), 0.3, 1e16, out)

    assert abs(logsumexp(out)) > 1.0


@pytest.mark.bug
def test_sals_beta_binomial_is_not_a_pmf_at_a_large_concentration() -> None:
    """`sal`'s dense beta-binomial at `tau = 1e16`, `p = 0.3`, `n = 100` sums to about `e^-60`, not 1.

    The same `lgamma` cancellation; `bb_states` scores such a state with
    port's kernel instead.
    """
    from sal.emissions import BetaBinomialEmission
    from sal.emissions.dense import Order, log_emission

    k = np.arange(TRIALS + 1, dtype=np.float64)
    family = BetaBinomialEmission(
        alpha=np.array([0.3e16]), beta=np.array([0.7e16]), trials=np.ones(1)
    )
    scores = log_emission(
        family, k, np.full((k.size, 1), float(TRIALS)), order=Order.FAMILY
    )

    assert abs(logsumexp(scores[0])) > 1.0


def _binomial_exact(p: float) -> np.ndarray:
    """The binomial log pmf from its definition, in `decimal` at 50 digits."""
    from decimal import Decimal, getcontext
    from math import comb

    getcontext().prec = 50
    share = Decimal(p)
    return np.array(
        [
            float(
                Decimal(comb(TRIALS, int(k))).ln()
                + int(k) * share.ln()
                + (TRIALS - int(k)) * (1 - share).ln()
            )
            for k in COUNTS
        ]
    )


@pytest.mark.oracle
@pytest.mark.parametrize("p", [0.3, 0.5, 1e-3])
def test_the_kernel_at_an_infinite_concentration_is_the_binomial(p: float) -> None:
    """`tau = inf` scored NaN (T- #617); it is the binomial, to 1e-11 absolute."""
    np.testing.assert_allclose(
        _kernel(p, np.inf), _binomial_exact(p), rtol=0, atol=1e-11
    )


@pytest.mark.analytic
def test_the_limit_is_continuous_in_the_concentration() -> None:
    """At 1e16 the beta-binomial is the binomial to within `n^2 / tau`."""
    np.testing.assert_allclose(_kernel(0.3, 1e16), _kernel(0.3, np.inf), atol=1e-11)


@pytest.mark.analytic
@pytest.mark.parametrize(("p", "count"), [(0.0, 1.0), (1.0, 0.0)])
def test_the_limit_at_a_share_of_zero_or_one_excludes_the_other_allele(
    p: float, count: float
) -> None:
    from port.patch.hmm_nophasing.bb_logpmf import binomial_logpmf

    assert binomial_logpmf(count, 1.0, p) == -np.inf
    assert binomial_logpmf(1.0 - count, 1.0, p) == 0.0
