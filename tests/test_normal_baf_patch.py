"""The quantile inversion `normal_baf_bin_filter` does not need (#174).

Referees: the identity over a grid; `cnaster`'s filter, bitwise; and the planted
imbalanced bins.
"""

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
import scipy.stats
from cnaster.normal_spot import normal_baf_bin_filter as upstream
from port.patch.normal_spot import normal_baf_bin_filter as patched
from port.patch.normal_spot import removal_indicator
from port.sim.inputs import read_to_bins, written_config
from port.sim.run_config import PlantedInstance
from port.sim.truth import CoreInferenceTruth, balanced_clone

pytestmark = pytest.mark.preprocessing

SHIPPED_CONFIDENCE = (0.01, 0.99)
"""`zenodo_sim_config.yaml`'s confidence; `run_config.py`'s `(0.0, 1.0)` would make both masks empty."""


@pytest.fixture(scope="module")
def binned_instance(
    planted_instance: PlantedInstance,
) -> Iterator[tuple[CoreInferenceTruth, Any, Any, np.ndarray]]:
    """The prep chain's table and counts, and the planted normal spots."""
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
    """`x < ppf(q)` iff `cdf(x) < q` and `x > ppf(q)` iff `cdf(x-1) >= q`, over the whole support at three trial counts."""

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
    """`cnaster`'s `ppf` mask, bitwise, where a tail saturates (#332)."""

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
    """The patched filter returns `cnaster`'s arrays and renumbered `bin_id`, bitwise."""

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
    """The patched filter removes exactly the eight planted imbalanced bins (#160)."""

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
