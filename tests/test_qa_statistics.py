"""`port.qa.statistics` against values a known sample fixes (T- #673 G1).

Each referee is a property of the statistic, not of the code: the order
statistics of `0..100`, the nominal coverage of a 95% interval, and a peak
resident set no smaller than what the block allocated and touched.
"""

from __future__ import annotations

import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest
from port.qa.statistics import (
    bars,
    bootstrap_interval,
    chi_square_pvalue,
    measured,
    ranks,
    resample_weights,
)


@pytest.mark.analytic
def test_bars_are_the_median_and_its_distances_to_the_deciles() -> None:
    """On `0..100` the median is 50 and the 10% and 90% quantiles 10 and 90, exactly."""
    median, errors = bars(pd.Series(np.arange(101, dtype=float)))

    assert median == 50.0
    assert errors == [[40.0], [40.0]]


@pytest.mark.analytic
def test_ranks_count_from_the_lowest_and_ties_share_the_lower() -> None:
    """Competition ranking: 1, 2, 2, 4; a missing value is not ranked."""
    found = ranks({"a": 0.1, "b": 0.5, "c": 0.5, "d": 2.0, "e": float("nan")})

    assert found == {"a": 1, "b": 2, "c": 2, "d": 4}


@pytest.mark.oracle
def test_the_chi_square_pvalue_is_scipys_over_the_kept_bins() -> None:
    """Statistic and p-value equal `scipy.stats.chisquare` over the bins at 5 or more, to 1e-12.

    The fourth bin expects 2 and is dropped, so a version that kept it would
    read a statistic 50 higher (`(12 - 2)**2 / 2`) on one more degree of freedom.
    """
    from scipy.stats import chisquare

    observed = np.array([18.0, 31.0, 51.0, 12.0])
    expected = np.array([20.0, 30.0, 50.0, 2.0])

    chi, pvalue = chi_square_pvalue(observed, expected)
    referee = chisquare(observed[:3], expected[:3])

    assert chi == pytest.approx(referee.statistic, rel=1e-12)
    assert pvalue == pytest.approx(referee.pvalue, rel=1e-12)


@pytest.mark.analytic
def test_the_bootstrap_interval_covers_the_mean_at_its_nominal_rate() -> None:
    """A 95% percentile interval for a normal mean covers it at 0.95, within 0.035.

    400 samples of 200 standard normals, one member each, 1,000 resamples per
    sample. 0.035 is 3 binomial standard errors at 400 trials (0.011 each);
    the percentile interval's shortfall at n = 200, about 0.003, sits inside
    it. The seeded draw reads 0.940.
    """
    rng = np.random.default_rng(673)
    n_members, trials = 200, 400
    covered = 0
    for _ in range(trials):
        sums = rng.standard_normal(n_members)
        weights = resample_weights(n_members, 1000, rng)
        low, high = bootstrap_interval(sums, np.ones(n_members), weights)
        covered += bool(low <= 0.0 <= high)

    assert abs(covered / trials - 0.95) <= 0.035, covered / trials


@pytest.mark.analytic
def test_measured_reports_at_least_the_wall_and_the_memory_spent() -> None:
    """A 0.25 GiB array written in the block, and a 0.05 s sleep, bound the readings below."""
    with measured() as cost:
        held = np.ones(2**25)  # NB 2**25 float64: 0.25 GiB, every page written
        time.sleep(0.05)

    assert held.nbytes == 2**28
    assert cost.wall_s >= 0.05
    assert cost.peak_gb >= 0.25


@pytest.mark.analytic
def test_measured_reads_a_childs_peak_for_children() -> None:
    """A child that writes 0.25 GiB leaves a waited-for peak at least that large."""
    script = "import numpy as np; np.ones(2**25)"
    with measured(children=True) as cost:
        subprocess.run([sys.executable, "-c", script], check=True)

    assert cost.peak_gb >= 0.25
