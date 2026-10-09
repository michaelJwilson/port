"""`cnaster`'s two beta-binomial implementations against each other and `scipy` (#205, #9).

`hmm_nophasing.betabinom_logpmf_numba` (HMM) and `hmm_emission.betabinom_logpmf`
(M step); also pins both kernels' invalid-input guards returning certainty.
"""

import numpy as np
import pytest
import scipy.stats
from cnaster.hmm_emission import betabinom_logpmf, betabinom_logpmf_zp
from cnaster.hmm_nophasing import betabinom_logpmf_numba, nbinom_logpmf_numba

IMPLEMENTATION_TOLERANCE = 1.0e-12
"""HMM-vs-M-step gap allowed: reassociation (realized 4.1e-13)."""

SCIPY_TOLERANCE = 1.0e-11
"""How far either may sit from `scipy` at integer counts."""


def _draws(size: int = 500) -> tuple[np.ndarray, ...]:
    """Return counts, totals and shapes within `hmm_nophasing.get_bounds`' regime."""
    generator = np.random.default_rng(0)

    totals = generator.integers(1, 200, size).astype(np.float64)
    successes = generator.integers(0, 1 + totals.astype(int)).astype(np.float64)

    return (
        successes,
        totals,
        generator.uniform(0.2, 40.0, size),
        generator.uniform(0.2, 40.0, size),
    )


@pytest.mark.patch
def test_the_hmm_and_the_m_step_score_the_same_density() -> None:
    """The HMM's and M step's beta-binomials agree within `IMPLEMENTATION_TOLERANCE` (#205, #9)."""

    successes, totals, alpha, beta = _draws()

    scored = np.array(
        [
            betabinom_logpmf_numba(successes[i], totals[i], alpha[i], beta[i])
            for i in range(successes.size)
        ]
    )
    optimized = betabinom_logpmf(
        successes, totals, alpha, beta, betabinom_logpmf_zp(successes, totals)
    )

    realized = float(np.abs(scored - optimized).max())
    assert realized < IMPLEMENTATION_TOLERANCE, f"the two differ by {realized:.3g}"


@pytest.mark.oracle
def test_both_agree_with_scipy_at_integer_counts() -> None:
    """Both agree with `scipy.stats.betabinom` within `SCIPY_TOLERANCE` at integer counts."""

    successes, totals, alpha, beta = _draws()
    expected = scipy.stats.betabinom.logpmf(successes, totals, alpha, beta)

    scored = np.array(
        [
            betabinom_logpmf_numba(successes[i], totals[i], alpha[i], beta[i])
            for i in range(successes.size)
        ]
    )
    optimized = betabinom_logpmf(
        successes, totals, alpha, beta, betabinom_logpmf_zp(successes, totals)
    )

    assert float(np.abs(scored - expected).max()) < SCIPY_TOLERANCE
    assert float(np.abs(optimized - expected).max()) < SCIPY_TOLERANCE


@pytest.mark.warning
@pytest.mark.parametrize(
    ("label", "successes", "totals", "alpha", "beta"),
    [
        ("more successes than trials", 5.0, 3.0, 2.0, 2.0),
        ("a negative count", -1.0, 3.0, 2.0, 2.0),
        ("a non-positive shape", 1.0, 3.0, 0.0, 2.0),
    ],
)
def test_an_impossible_observation_scores_as_certain(
    label: str, successes: float, totals: float, alpha: float, beta: float
) -> None:
    """The HMM's guard returns 0.0 (certainty) where the M step and `scipy` give `-inf`."""

    guarded = betabinom_logpmf_numba(successes, totals, alpha, beta)

    assert guarded == 0.0, f"{label}: the guard no longer returns zero"

    counts = np.array([successes])
    trials = np.array([totals])

    optimized = betabinom_logpmf(
        counts,
        trials,
        np.array([alpha]),
        np.array([beta]),
        betabinom_logpmf_zp(counts, trials),
    )[0]

    assert optimized < guarded, (
        f"{label}: the M step's form scored {optimized}, the HMM's {guarded}"
    )


@pytest.mark.warning
def test_the_negative_binomial_guard_has_the_same_direction() -> None:
    """`nbinom_logpmf_numba` returns 0.0 when a tiny exposure underflows `p` to 1 (#30)."""

    alpha, exposure, mu = 1.0e-6, 1.0e-12, 1.0
    rate = exposure * mu

    assert 1.0 + alpha * rate == 1.0, "the fixture no longer underflows"

    probability = 1.0 / (1.0 + alpha * rate)

    assert probability >= 1.0
    assert nbinom_logpmf_numba(3.0, 1.0 / alpha, probability) == 0.0
