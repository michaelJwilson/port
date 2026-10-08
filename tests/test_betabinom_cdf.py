"""The vectorized beta-binomial distribution function against scipy's `betabinom.cdf`
(#190).

Values agree to a per-depth tolerance; the filter's mask is unchanged, with bins within
`DECISION_MARGIN` of a threshold re-decided by scipy.
"""

import numpy as np
import pytest
import scipy.stats

pytestmark = pytest.mark.preprocessing

TOTALS = [7, 40, 201, 1_009]
"""Read depths checked over the whole support; the largest takes the chunked path."""

VALUE_TOLERANCE = 1.0e-11
"""Absolute tolerance per depth; the error grows with the number of terms summed."""


@pytest.mark.patch
@pytest.mark.parametrize("total", TOTALS)
def test_the_distribution_function_agrees_with_scipy(total: int) -> None:
    """Every `k` in `[0, n]` agrees with scipy, covering both summation branches."""
    from port.patch.normal_spot import cumulative_and_mass

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
    """The log-gamma tables agree with `betaln` to 1e-11 relative."""
    from port.patch.normal_spot import _log_mass, _log_mass_tabulated, _log_tables

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
    """Non-decreasing, zero below the support, one at its top."""
    from port.patch.normal_spot import cumulative_and_mass

    support = np.arange(-1, total + 1)
    totals = np.full_like(support, total)

    realized, _ = cumulative_and_mass(support, totals, 15.0, 15.0)

    assert realized[0] == 0.0
    assert realized[-1] == pytest.approx(1.0, abs=VALUE_TOLERANCE)
    assert np.all(np.diff(realized) >= -VALUE_TOLERANCE)


@pytest.mark.patch
def test_the_chunk_boundary_cannot_move_a_value() -> None:
    """Results are bitwise identical across `TERM_BUDGET`s spanning three orders of
    magnitude.
    """
    import port.patch.normal_spot as module

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
    """At a symmetric beta-binomial's midpoint (exactly 0.5), the mask matches scipy's
    decision.
    """
    from port.patch.normal_spot import removal_indicator

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
