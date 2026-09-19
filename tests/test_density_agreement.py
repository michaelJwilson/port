"""`cnaster`'s beta-binomial, written three times, against each other (#205, #9).

**Step 2 of #205's plan needed a correction before it could start.** The
ticket counted "nine density routines for two distributions" and read that
as duplication. Six of them are not: `_dense_nb_logpmf` calls
`_nb_logpmf_1d` calls `nbinom_logpmf_numba`, and the beta-binomial side has
the same three-rung shape hierarchy -- scalar, vector, tensor -- which is
how a `numba` kernel is written for three call shapes, not a routine written
three times.

What is genuinely written twice is the **beta-binomial itself**:
`hmm_nophasing.betabinom_logpmf_numba`, which the HMM scores with, and
`hmm_emission.betabinom_logpmf`, which the M step optimizes. The third,
`hmm_phased._switch_betabinom_1d`, is the same density at the swapped
allele and is pinned as such in `tests/test_emission_entry_points.py`.

So the question step 2 rests on is whether **the density the M step
maximizes is the density the HMM scores**. It is, to 4.1e-13 -- and the
guards are where they part.
"""

import numpy as np
import pytest
import scipy.stats

IMPLEMENTATION_TOLERANCE = 1.0e-12
"""How far the HMM's beta-binomial may sit from the M step's.

Realized 4.1e-13 over 500 draws with `n` to 200 and both shapes to 40. The
two group the same seven log-gammas differently -- the M step carries the
binomial coefficient in a `zero_point` computed outside the kernel -- so the
gap is reassociation.
"""

SCIPY_TOLERANCE = 1.0e-11
"""How far either may sit from `scipy` at integer counts."""


def _draws(size: int = 500) -> tuple[np.ndarray, ...]:
    """Counts, totals and shapes in the regime a fit stays inside.

    `alphas` and `taus` are bounded away from zero by
    `hmm_nophasing.get_bounds` (`min_alpha=1e-6`, `min_tau=1e-4`, both in log
    space), so a comparison drawn from an unbounded prior would be measuring
    a regime no fit reaches.
    """
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
    """What the M step maximizes is what the HMM scores.

    The precondition for #205's step 2: two implementations cannot be made
    one until they are shown to be one. Also the open half of #9 -- neither
    is designated the referee, so this says they agree without saying which
    would be wrong.
    """
    from cnaster.hmm_emission import betabinom_logpmf, betabinom_logpmf_zp
    from cnaster.hmm_nophasing import betabinom_logpmf_numba

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
    """`scipy.stats.betabinom` is the outside referee, on its own domain.

    On **integer** counts. Both `cnaster` forms evaluate the log-gamma
    expression, which is finite at a fractional count; `scipy`'s pmf is zero
    there, so a comparison drawn off the integers would report an infinite
    disagreement about a value no observation takes.
    """
    from cnaster.hmm_emission import betabinom_logpmf, betabinom_logpmf_zp
    from cnaster.hmm_nophasing import betabinom_logpmf_numba

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
    """The guard returns `0.0`, which is a log-probability of **one**.

    `betabinom_logpmf_numba` refuses an invalid argument by returning zero,
    and zero is not a refusal in log space -- it is certainty. `scipy` and
    the M step's form both give `-inf` on the same inputs, so the HMM scores
    an impossible observation as the **most** likely one it could see.

    `warning` rather than `bug` because none of the three is reachable from
    a fit: `get_bounds` floors the shapes at `1e-6` in log space, and a B
    allele count above its total is a malformed input rather than a decision.
    What is pinned is the **direction** of the guard -- a defensive branch
    whose failure mode is maximal likelihood is the wrong way round, and the
    same shape appears in the negative binomial, where `p >= 1.0` returns
    zero for a small dispersion at a tiny exposure.
    """
    from cnaster.hmm_emission import betabinom_logpmf, betabinom_logpmf_zp
    from cnaster.hmm_nophasing import betabinom_logpmf_numba

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
    """`p >= 1.0` returns zero, and a tiny exposure is how `p` gets there.

    `p = 1 / (1 + alpha * exposure * mu)`, so an exposure small enough that
    `alpha * lambda` underflows the addition gives `p == 1.0` exactly and the
    guard fires. The bin then scores `0.0` -- certainty -- rather than the
    value the continuous form would give.

    Pinned with the arithmetic rather than through a fit, because a fit that
    reaches it is #30's and not this file's.
    """
    from cnaster.hmm_nophasing import nbinom_logpmf_numba

    alpha, exposure, mu = 1.0e-6, 1.0e-12, 1.0
    rate = exposure * mu

    assert 1.0 + alpha * rate == 1.0, "the fixture no longer underflows"

    probability = 1.0 / (1.0 + alpha * rate)

    assert probability >= 1.0
    assert nbinom_logpmf_numba(3.0, 1.0 / alpha, probability) == 0.0
