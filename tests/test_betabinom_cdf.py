"""The vectorized beta-binomial distribution function (#190).

**4.0x to 12x over `scipy`, with the peak bounded at 5 MB, and the filter's
mask unchanged.** `scipy.stats.betabinom` has no vectorized distribution
function: `cdf` goes through `_cdf_single` under `np.vectorize`, summing the
mass function from zero for one element at a time. After #175 removed the
quantile inversion, those two calls are 90% of what the normal-BAF filter
costs.

**This is a tolerance and the module says so.** The terms are summed in a
different order, read off log-gamma tables rather than `betaln`, and on the
upper branch subtracted from one, so the values agree to `3e-9` at the
deepest size measured and to `3e-13` at the shallowest -- not bitwise.

The mask built from them is a different matter, and the reason
`DECISION_MARGIN` exists: both comparisons in `removal_indicator` are strict,
so a bin whose distribution function sits within the summation error of a
threshold could fall either way. Those bins are re-decided with `scipy`. The
test below names the case that forced it -- a symmetric beta-binomial at its
own midpoint, where the true value is exactly one half.
"""

import numpy as np
import pytest
import scipy.stats

pytestmark = pytest.mark.preprocessing

TOTALS = [7, 40, 201, 1_009]
"""Read depths the distribution function is checked at, over the whole support.

The largest is above `TERM_BUDGET` divided by its own support, so the chunked
path runs on it and the short ones do not.
"""

VALUE_TOLERANCE = 1.0e-11
"""Absolute agreement asked of the distribution function at these depths.

Realized `7.4e-14` at 201 and `3.3e-09` at 100,000, which is why the bound is
stated per size rather than as one number for all of them: the error grows
with the number of terms summed, as a floating-point sum's does.
"""


@pytest.mark.patch
@pytest.mark.parametrize("total", TOTALS)
def test_the_distribution_function_agrees_with_scipy(total: int) -> None:
    """**Over the whole support, not at the values a fixture produces.**

    Every `k` from zero to `n`, so the branch that sums `[0, k]` and the
    branch that sums `(k, n]` and subtracts from one are both covered, along
    with the two endpoints where one of them is empty.
    """
    from port.patch.normal_baf import cumulative_and_mass

    support = np.arange(total + 1)
    totals = np.full_like(support, total)

    realized, mass = cumulative_and_mass(support, totals, 15.0, 15.0)
    expected = scipy.stats.betabinom.cdf(support, totals, 15.0, 15.0)

    np.testing.assert_allclose(realized, expected, rtol=0.0, atol=VALUE_TOLERANCE)
    np.testing.assert_allclose(
        mass,
        scipy.stats.betabinom.pmf(support, totals, 15.0, 15.0),
        rtol=0.0,
        atol=VALUE_TOLERANCE,
    )


@pytest.mark.patch
def test_the_tabulated_mass_function_is_scipys_formula() -> None:
    """The tables against `betaln`, which is what they replace.

    `_log_mass` is `scipy`'s `betabinom._logpmf` written out; the tabulated
    form is the same expression with the log-gammas gathered from cumulative
    sums. A table built one index off would shift every term by a factor of
    `n`, which this catches and a comparison of totals would not.

    The tolerance is `1e-11` relative, realized `1.2e-12` at a depth of 1,009.
    A cumulative sum of a thousand logarithms is where the table's error comes
    from, so the bound is a statement about the depth rather than about the
    formula.
    """
    from port.patch.normal_baf import _log_mass, _log_mass_tabulated, _log_tables

    alpha, beta = 4.5, 11.0
    totals = np.repeat([13, 200, 1_009], 7)
    index = np.concatenate(
        [np.linspace(0, total, 7).astype(int) for total in (13, 200, 1_009)]
    )

    tables = _log_tables(int(totals.max()), alpha, beta)

    np.testing.assert_allclose(
        _log_mass_tabulated(index, totals, alpha, beta, tables),
        _log_mass(index, totals, alpha, beta),
        rtol=1.0e-11,
    )


@pytest.mark.analytic
@pytest.mark.parametrize("total", TOTALS)
def test_the_distribution_function_is_a_distribution_function(total: int) -> None:
    """Non-decreasing, zero below the support, one at its top.

    Properties of any distribution function, so they hold whatever the
    summation order. They are what would catch a chunk boundary dropping a
    term: the values would stay close to `scipy` in the middle and the top
    would no longer be one.
    """
    from port.patch.normal_baf import cumulative_and_mass

    support = np.arange(-1, total + 1)
    totals = np.full_like(support, total)

    realized, _ = cumulative_and_mass(support, totals, 15.0, 15.0)

    assert realized[0] == 0.0
    assert realized[-1] == pytest.approx(1.0, abs=VALUE_TOLERANCE)
    assert np.all(np.diff(realized) >= -VALUE_TOLERANCE)


@pytest.mark.patch
def test_the_chunk_boundary_cannot_move_a_value() -> None:
    """**Where the groups fall is not allowed to change an answer.**

    `np.add.reduceat` never sums across bins, so a chunk boundary can only
    change which call a bin is evaluated in. Asserted bitwise, across budgets
    spanning three orders of magnitude, because the claim in `TERM_BUDGET`'s
    docstring -- that the budget is a memory decision and not a numerical one
    -- is what lets it be tuned for cache rather than for accuracy.
    """
    import port.patch.normal_baf as module

    totals = np.random.default_rng(5).integers(500, 2_000, size=64)
    counts = np.random.default_rng(6).binomial(totals, 0.5)

    original = module.TERM_BUDGET
    realized = []

    try:
        for budget in (1 << 20, 1 << 16, 1 << 12, 64):
            module.TERM_BUDGET = budget
            realized.append(module.cumulative_and_mass(counts, totals, 15.0, 15.0)[0])
    finally:
        module.TERM_BUDGET = original

    for other in realized[1:]:
        np.testing.assert_array_equal(other, realized[0])


@pytest.mark.bug
def test_an_exact_tie_is_decided_the_way_cnaster_decides_it() -> None:
    """**The case `DECISION_MARGIN` exists for, pinned at its own midpoint.**

    A symmetric beta-binomial on an odd support has `cdf(n // 2)` exactly one
    half, so against a threshold of `0.5` the strict comparison in
    `removal_indicator` is decided by the last bit. `scipy` sums to
    `0.5000000000000001` and this patch's reassociation to
    `0.4999999999999973`, which are opposite sides of the predicate and
    opposite answers about whether a genomic bin survives.

    Marked `bug` rather than `patch`: it records a defect the patch would have
    had, and it fails if the settling step is removed.
    """
    from port.patch.normal_baf import removal_indicator

    total, alpha, beta = 201, 15.0, 15.0
    support = np.arange(total + 1)
    totals = np.full_like(support, total)

    quantile = scipy.stats.betabinom.ppf(0.5, totals, alpha, beta)

    np.testing.assert_array_equal(
        removal_indicator(support, totals, alpha, beta, (0.5, 2.0)),
        support < quantile,
    )
    np.testing.assert_array_equal(
        removal_indicator(support, totals, alpha, beta, (-1.0, 0.5)),
        support > quantile,
    )
