"""The preprocessing patches' cost against what they replace, per component (#190).

Results are recorded in #190.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
import scipy.stats
from cnaster.reference import get_reference_genes as cnaster_reference_genes
from cnaster.spatial import best_equal_partition as cnaster_partition
from cnaster.spatial import (
    construct_multislice_lattice_adjacency as cnaster_adjacency,
)
from port.patch.normal_spot import cumulative_and_mass
from port.patch.reference import get_reference_genes as patched_reference_genes
from port.patch.spatial import best_equal_partition as patched_partition
from port.patch.spatial import (
    construct_multislice_lattice_adjacency as patched_adjacency,
)
from pytest_benchmark.fixture import BenchmarkFixture

from tests.adapters import square_coords
from tests.fixtures import tiers

pytestmark = pytest.mark.preprocessing

GATE_LATTICE = (25, 40)
STRESS_LATTICE = (100, 100)
"""1,000 spots and 10,000: the dev instance, and twice a slide's width."""

GATE_DEPTH = (2_000, 400)
STRESS_DEPTH = (20_000, 400)
"""`(reads per bin, bins)` for the distribution function; depth grows with the slide."""


@pytest.fixture(scope="module")
def hgtable(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """A synthetic reference gene table of a given size, written once per size."""
    root = tmp_path_factory.mktemp("hgtable")
    written: dict[int, Any] = {}

    def of_size(rows: int) -> Any:
        if rows not in written:
            path = root / f"hgtable_{rows}.tsv"
            generator = np.random.default_rng(3)

            pd.DataFrame(
                {
                    "name": [f"tx_{index}" for index in range(rows)],
                    "name2": [f"gene_{index}" for index in range(rows)],
                    "chrom": [f"chr{1 + (index % 24)}" for index in range(rows)],
                    "cdsStart": generator.integers(0, 250_000_000, rows),
                    "cdsEnd": generator.integers(0, 250_000_000, rows),
                }
            ).to_csv(path, sep="\t", index=False)
            written[rows] = path

        return written[rows]

    return of_size


def _bins(depth: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    """B-allele counts and totals for `count` bins at roughly `depth` reads."""
    generator = np.random.default_rng(11)
    totals = generator.integers(depth // 2, depth, size=count)

    return generator.binomial(totals, 0.5), totals


def _scipy_cumulative_and_mass(
    counts: np.ndarray, totals: np.ndarray, alpha: float, beta: float
) -> tuple[np.ndarray, np.ndarray]:
    """`betabinom.cdf` twice, as `removal_indicator` called it."""
    return (
        scipy.stats.betabinom.cdf(counts, totals, alpha, beta),
        scipy.stats.betabinom.cdf(counts - 1, totals, alpha, beta),
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("lattice", tiers(GATE_LATTICE, STRESS_LATTICE))
@pytest.mark.parametrize(
    "arm",
    [cnaster_adjacency, patched_adjacency],
    ids=["cnaster-dense", "patched-sparse"],
)
def test_adjacency(
    benchmark: BenchmarkFixture,
    arm: Callable[..., object],
    lattice: tuple[int, int],
) -> None:
    """Dense block diagonal against sparse, at 1,000 and 10,000 spots."""
    coords = square_coords(*lattice).astype(float)
    sample_ids = np.zeros(len(coords), dtype=int)

    benchmark(arm, sample_ids, [0], coords, None, 1, 1, 1)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("lattice", "n_trials"),
    [
        pytest.param(GATE_LATTICE, 200, id="gate"),
        pytest.param(STRESS_LATTICE, 1_000, id="stress", marks=pytest.mark.release),
    ],
)
@pytest.mark.parametrize(
    "arm",
    [cnaster_partition, patched_partition],
    ids=["cnaster-index-lists", "patched-batched"],
)
def test_partition(
    benchmark: BenchmarkFixture,
    arm: Callable[..., object],
    lattice: tuple[int, int],
    n_trials: int,
) -> None:
    """Per-trial index lists against a summed-area table batched across trials."""
    coords = square_coords(*lattice).astype(float)

    benchmark(arm, coords, 3, 3, n_trials=n_trials)


@pytest.mark.benchmark
@pytest.mark.parametrize("depth", tiers(GATE_DEPTH, STRESS_DEPTH))
@pytest.mark.parametrize(
    "arm",
    [_scipy_cumulative_and_mass, cumulative_and_mass],
    ids=["scipy", "vectorized"],
)
def test_distribution_function(
    benchmark: BenchmarkFixture,
    arm: Callable[..., object],
    depth: tuple[int, int],
) -> None:
    """Two scipy calls against one sweep for both, at gate and 20,000-read depths."""
    counts, totals = _bins(*depth)

    benchmark(arm, counts, totals, 15.0, 15.0)


@pytest.mark.benchmark
@pytest.mark.parametrize("rows", tiers(1_213, 250_000))
@pytest.mark.parametrize(
    "arm",
    [cnaster_reference_genes, patched_reference_genes],
    ids=["cnaster-pandas", "patched-polars"],
)
def test_reference_read(
    benchmark: BenchmarkFixture,
    hgtable: Any,
    arm: Callable[[str], object],
    rows: int,
) -> None:
    """`pd.read_csv` against `pl.read_csv` via Arrow, at 1,213 and 250,000 transcripts, warm (#185)."""
    benchmark(arm, str(hgtable(rows)))
