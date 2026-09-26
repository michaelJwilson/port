"""The quantile inversion `normal_baf_bin_filter` does not need (#174).

Three claims, and they are different claims:

1. the substitution is an **identity** for a discrete law, checked over a grid
   rather than on the instance it is used on;
2. the patched filter returns what `cnaster`'s returns, **bitwise**, on the dev
   fixture;
3. what it returns is still right -- the planted imbalanced bins and no others,
   which is the claim `tests/test_run_cnaster_stages.py` makes of `cnaster`'s
   own filter and which a patch has to keep.

The third is what makes this more than an agreement between two
implementations: two functions computing the same wrong mask would satisfy the
second.
"""

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
import scipy.stats

from tests.fixtures import CoreInferenceTruth, balanced_clone
from tests.run_config import PlantedInstance
from tests.tmp_inputs import read_to_bins, written_config

pytestmark = pytest.mark.preprocessing

SHIPPED_CONFIDENCE = (0.01, 0.99)
"""`zenodo_sim_config.yaml`'s `quality.normal_allele_specific_confidence`.

`tests/run_config.py` widens it to `(0.0, 1.0)`, under which the filter removes
nothing and the comparison would be between two empty masks.
"""


@pytest.fixture(scope="module")
def binned_instance(
    planted_instance: PlantedInstance,
) -> Iterator[tuple[CoreInferenceTruth, Any, Any, np.ndarray]]:
    """The prep chain's table and counts, and the planted normal spots.

    The filter takes what the binner produced, so the input is the pipeline's
    rather than one this module built: a hand-made table would not exercise the
    renumbering, which is most of what the function does after the test.
    """
    truth, _, written, config_path = planted_instance

    with written_config(config_path):
        chain = read_to_bins(written)
        yield (
            truth,
            chain.table,
            chain.bins,
            np.flatnonzero(truth.labels == balanced_clone(truth)),
        )


def _arguments(
    table: Any, binned: Any, index_normal: np.ndarray
) -> tuple[Any, Any, Any, Any, float, float, np.ndarray, None]:
    """Fresh copies, because the filter writes through its own table (#89)."""
    return (
        table.copy(),
        binned.X.copy(),
        binned.base_nb_mean.copy(),
        binned.total_bb_RD.copy(),
        1.0,
        0.0,
        index_normal,
        None,
    )


@pytest.mark.analytic
@pytest.mark.parametrize("quantile", [0.01, 0.05, 0.5, 0.95, 0.99])
def test_the_quantile_test_is_a_distribution_function_comparison(
    quantile: float,
) -> None:
    """**`x < ppf(q)` is `cdf(x) < q`, and `x > ppf(q)` is `cdf(x-1) >= q`.**

    The identity the patch rests on, and it is a property of any discrete law
    rather than of this one: `ppf(q) = min{k : cdf(k) >= q}`, so `cdf(x) < q`
    says exactly that `x` is below that minimum, and `cdf(x-1) < q` says
    exactly that `x` is at most it.

    Checked over the whole support at three trial counts rather than at the
    values the filter happens to produce, because an identity that held only
    where it is used would be a coincidence. Five quantiles, including the two
    that ship and the median, where `ppf` lands on the mode and an off-by-one
    would be invisible at the tails.
    """
    from port.patch.normal_spot import removal_indicator

    alpha, beta = 15.0, 15.0

    for total in (7, 40, 201):
        support = np.arange(total + 1)
        totals = np.full_like(support, total)

        lower = support < scipy.stats.betabinom.ppf(quantile, totals, alpha, beta)
        upper = support > scipy.stats.betabinom.ppf(quantile, totals, alpha, beta)

        by_cdf_lower = removal_indicator(support, totals, alpha, beta, (quantile, 2.0))
        by_cdf_upper = removal_indicator(support, totals, alpha, beta, (-1.0, quantile))

        np.testing.assert_array_equal(by_cdf_lower, lower)
        np.testing.assert_array_equal(by_cdf_upper, upper)


@pytest.mark.patch
@pytest.mark.parametrize("interval", [(0.0, 1.0), (0.01, 1.0), (0.0, 0.99)])
def test_a_closed_end_removes_nothing_on_its_side_as_cnaster(
    interval: tuple[float, float],
) -> None:
    """`cnaster`'s `ppf` mask, bitwise, where a tail saturates (#332).

    A normal pool diluted by an LOH clone reads BAF 0.62 over 29,000 reads,
    against a fitted `alpha = beta = 500`: the upper tail is below `1e-50`,
    and `cdf(x - 1)` rounds to exactly 1.0. At `hi = 1` the distribution-
    function form removed that bin and `ppf(1) = n` keeps it; the whole-run
    LOH fixture lost 6 bins to it and crashed in `run_cnaster`'s gene writer.
    The mirrored bins, BAF 0.38 and 0.02, saturate the lower tail.
    """
    from port.patch.normal_spot import removal_indicator

    totals = np.array([3_000.0, 29_000.0, 29_000.0, 29_000.0, 29_000.0])
    counts = np.array([2_950.0, 17_980.0, 11_020.0, 580.0, 14_500.0])

    for alpha, beta in ((15.0, 15.0), (500.0, 500.0)):
        lo, hi = interval
        expected = (counts < scipy.stats.betabinom.ppf(lo, totals, alpha, beta)) | (
            counts > scipy.stats.betabinom.ppf(hi, totals, alpha, beta)
        )

        np.testing.assert_array_equal(
            removal_indicator(counts, totals, alpha, beta, interval), expected
        )


@pytest.mark.patch
def test_the_patched_filter_returns_what_cnasters_returns(
    binned_instance: tuple[CoreInferenceTruth, Any, Any, np.ndarray],
) -> None:
    """Every array and the renumbered column, bitwise.

    The counts are integers and the mask is boolean, so there is no arithmetic
    here for a tolerance to absorb: the two either drop the same bins and
    relabel the survivors the same way, or they do not.

    The `bin_id` column is compared as well as the arrays. It carries the
    renumbering, which is where an off-by-one in the survivor map would land,
    and the arrays alone would not see it -- they are sliced by the same
    `index_remaining` either way.
    """
    from cnaster.normal_spot import normal_baf_bin_filter as upstream
    from port.patch.normal_spot import normal_baf_bin_filter as patched

    _, table, binned, index_normal = binned_instance

    reference_table, reference = upstream(
        *_arguments(table, binned, index_normal),
        confidence_interval=SHIPPED_CONFIDENCE,
    )
    patched_table, realized = patched(
        *_arguments(table, binned, index_normal),
        confidence_interval=SHIPPED_CONFIDENCE,
    )

    np.testing.assert_array_equal(
        np.asarray(realized.lengths), np.asarray(reference.lengths)
    )
    np.testing.assert_array_equal(realized.X, reference.X)
    np.testing.assert_array_equal(realized.total_bb_RD, reference.total_bb_RD)
    np.testing.assert_array_equal(
        np.asarray(realized.base_nb_mean), np.asarray(reference.base_nb_mean)
    )
    assert patched_table.bin_id.equals(reference_table.bin_id)


@pytest.mark.end2end
def test_the_patched_filter_removes_the_planted_imbalanced_bins(
    binned_instance: tuple[CoreInferenceTruth, Any, Any, np.ndarray],
) -> None:
    """**The patch keeps the claim, not just the output (#160).**

    `tests/test_run_cnaster_stages.py` establishes that `cnaster`'s filter
    removes exactly the bins the planted normal clone carries an event in. Two
    implementations agreeing says nothing about whether either is right, so the
    patch is put to the same referee: the planted states.

    Realized: eight bins removed, eight planted, no bin either way -- and the
    surviving segmentation is the planted `[10, 21, 9]` less the removals
    charged to each chromosome.
    """
    from port.patch.normal_spot import normal_baf_bin_filter as patched

    truth, table, binned, index_normal = binned_instance

    filtered, counts = patched(
        *_arguments(table, binned, index_normal),
        confidence_interval=SHIPPED_CONFIDENCE,
    )

    dropped = filtered.bin_id.isna() & table.bin_id.notna()
    removed = np.unique(table.loc[dropped, "bin_id"].to_numpy().astype(int))

    planted_imbalanced = np.flatnonzero(truth.states[balanced_clone(truth)] != 0)

    np.testing.assert_array_equal(removed, planted_imbalanced)

    chromosome_of_bin = np.repeat(
        np.arange(truth.lengths.size), np.asarray(truth.lengths)
    )
    expected = np.asarray(truth.lengths) - np.bincount(
        chromosome_of_bin[removed], minlength=truth.lengths.size
    )

    np.testing.assert_array_equal(np.asarray(counts.lengths), expected)
