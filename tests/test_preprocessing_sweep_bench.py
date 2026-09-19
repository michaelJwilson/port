"""What the preprocessing patches cost against what they replace (#190).

**21,368 ms to 1,007 ms on the whole chain from the files to
`run_core_inference`, 21.2x, at 2,500 spots and 400 bins.** Per stage, timed
best of three with the allocation pass taken separately:

| stage | `cnaster` | patched | | peak |
| --- | ---: | ---: | ---: | --- |
| `normal_baf_bin_filter` | 19,926 ms | 249 ms | **80.0x** | 46 MB, unchanged |
| `multislice_adjacency` | 72 ms | 4.8 ms | **15.1x** | 100.4 -> 1.4 MB |
| `form_gene_snp_table` | 126 ms | 10.5 ms | **12.1x** | 0.5 MB |
| `best_equal_partition` | 156 ms | 18.2 ms | **8.6x** | 0.2 MB |
| `assign_initial_blocks` | 583 ms | 275 ms | 2.1x | 25 MB |
| `load_input_data` (#186) | 283 ms | 228 ms | 1.2x | 159 MB |
| unpatched remainder | 221 ms | 221 ms | 1.0x | 57 MB |

The chain's peak is unchanged at 159 MB because it is set by the loader's
dense return, which `sparse_counts` removes and which nothing downstream
accepts yet (#186).

The rows here are the components rather than the chain, because a chain
measurement needs a written instance and a benchmark that builds one is
timing the fixture. The synthetic sizes below are chosen to bracket what the
chain measurement saw.

**Peak is not a column.** `pytest-benchmark` measures time, and interleaving
`tracemalloc` with it distorts both. The two allocation claims are in their
own modules' docstrings: 100.4 MB to 1.4 MB for the adjacency at 2,500 spots,
1,601 MB to 5.5 MB at 10,000, and 0.6 MB to 4.6 MB for the distribution
function at a slide's read depth -- the one place a patch here allocates more
than what it replaces.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest
import scipy.stats
from pytest_benchmark.fixture import BenchmarkFixture

pytestmark = pytest.mark.preprocessing

GATE_LATTICE = (25, 40)
STRESS_LATTICE = (100, 100)
"""1,000 spots and 10,000: the dev instance, and twice a slide's width."""

GATE_DEPTH = (2_000, 400)
STRESS_DEPTH = (20_000, 400)
"""`(reads per bin, bins)` for the distribution function.

A bin pools its B-allele counts across every normal spot, so the depth grows
with the slide and the bin count does not. The stress row is where the
summation is long enough for the tabulated log-gammas to matter.
"""


@pytest.fixture(scope="module")
def hgtable(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """A reference gene table of a given size, written once per size.

    Synthetic rather than the fixture's own, because the claim is about a
    human reference -- 250,000 transcripts against the dev instance's 1,213 --
    and no fixture here carries one.
    """
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


def _lattice(rows: int, columns: int) -> np.ndarray:
    """Spot coordinates on a rectangular lattice, as a slide carries them."""
    x_grid, y_grid = np.meshgrid(np.arange(rows), np.arange(columns), indexing="ij")

    return np.stack([x_grid.ravel(), y_grid.ravel()], axis=1).astype(float)


def _bins(depth: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    """B-allele counts and totals for `count` bins at roughly `depth` reads."""
    generator = np.random.default_rng(11)
    totals = generator.integers(depth // 2, depth, size=count)

    return generator.binomial(totals, 0.5), totals


@pytest.mark.benchmark
def test_cnasters_adjacency_at_the_gate_size(benchmark: BenchmarkFixture) -> None:
    """The dense block diagonal: 14.9 ms at 1,000 spots."""
    from cnaster.spatial import construct_multislice_lattice_adjacency

    coords = _lattice(*GATE_LATTICE)
    sample_ids = np.zeros(len(coords), dtype=int)

    benchmark(
        lambda: construct_multislice_lattice_adjacency(
            sample_ids, [0], coords, None, 1, 1, 1
        )
    )


@pytest.mark.benchmark
def test_the_sparse_adjacency_at_the_gate_size(benchmark: BenchmarkFixture) -> None:
    """Sparse throughout: 2.49 ms, so **6.0x** at 1,000 spots."""
    from port.patch.spatial import construct_multislice_lattice_adjacency

    coords = _lattice(*GATE_LATTICE)
    sample_ids = np.zeros(len(coords), dtype=int)

    benchmark(
        lambda: construct_multislice_lattice_adjacency(
            sample_ids, [0], coords, None, 1, 1, 1
        )
    )


@pytest.mark.benchmark
def test_cnasters_partition_at_the_gate_size(benchmark: BenchmarkFixture) -> None:
    """Index lists per trial, at 1,000 spots and 200 trials: 24.2 ms."""
    from cnaster.spatial import best_equal_partition

    coords = _lattice(*GATE_LATTICE)

    benchmark(lambda: best_equal_partition(coords, 3, 3, n_trials=200))


@pytest.mark.benchmark
def test_the_batched_partition_at_the_gate_size(benchmark: BenchmarkFixture) -> None:
    """Summed-area table, batched across trials: 3.90 ms, **6.2x**."""
    from port.patch.spatial import best_equal_partition

    coords = _lattice(*GATE_LATTICE)

    benchmark(lambda: best_equal_partition(coords, 3, 3, n_trials=200))


@pytest.mark.benchmark
def test_scipys_distribution_function_at_the_gate_depth(
    benchmark: BenchmarkFixture,
) -> None:
    """`betabinom.cdf` twice, as `removal_indicator` called it: 53.9 ms."""
    counts, totals = _bins(*GATE_DEPTH)

    benchmark(
        lambda: (
            scipy.stats.betabinom.cdf(counts, totals, 15.0, 15.0),
            scipy.stats.betabinom.cdf(counts - 1, totals, 15.0, 15.0),
        )
    )


@pytest.mark.benchmark
def test_the_vectorized_distribution_function_at_the_gate_depth(
    benchmark: BenchmarkFixture,
) -> None:
    """One sweep for both: 7.36 ms, **7.3x**."""
    from port.patch.normal_baf import cumulative_and_mass

    counts, totals = _bins(*GATE_DEPTH)

    benchmark(lambda: cumulative_and_mass(counts, totals, 15.0, 15.0))


@pytest.mark.benchmark
@pytest.mark.release
def test_cnasters_adjacency_at_the_stress_size(benchmark: BenchmarkFixture) -> None:
    """1,250 ms at 10,000 spots, and 1,601 MB to hold 0.3 MB of graph."""
    from cnaster.spatial import construct_multislice_lattice_adjacency

    coords = _lattice(*STRESS_LATTICE)
    sample_ids = np.zeros(len(coords), dtype=int)

    benchmark(
        lambda: construct_multislice_lattice_adjacency(
            sample_ids, [0], coords, None, 1, 1, 1
        )
    )


@pytest.mark.benchmark
@pytest.mark.release
def test_the_sparse_adjacency_at_the_stress_size(benchmark: BenchmarkFixture) -> None:
    """15.9 ms and 5.5 MB: **78x**, and 289x less allocated.

    The ratio grows with the spot count because what is removed is quadratic
    in it and what is left -- the k-d tree query -- is not.
    """
    from port.patch.spatial import construct_multislice_lattice_adjacency

    coords = _lattice(*STRESS_LATTICE)
    sample_ids = np.zeros(len(coords), dtype=int)

    benchmark(
        lambda: construct_multislice_lattice_adjacency(
            sample_ids, [0], coords, None, 1, 1, 1
        )
    )


@pytest.mark.benchmark
@pytest.mark.release
def test_cnasters_partition_at_the_stress_size(benchmark: BenchmarkFixture) -> None:
    """304 ms at 10,000 spots and 1,000 trials."""
    from cnaster.spatial import best_equal_partition

    coords = _lattice(*STRESS_LATTICE)

    benchmark(lambda: best_equal_partition(coords, 3, 3, n_trials=1_000))


@pytest.mark.benchmark
@pytest.mark.release
def test_the_batched_partition_at_the_stress_size(benchmark: BenchmarkFixture) -> None:
    """18.6 ms: **16x**, and 17.2 ms at 2,500 spots -- the same figure.

    The cost stops depending on the spot count, which is the claim: a trial
    reads `x_part * y_part` corners of a table built once.
    """
    from port.patch.spatial import best_equal_partition

    coords = _lattice(*STRESS_LATTICE)

    benchmark(lambda: best_equal_partition(coords, 3, 3, n_trials=1_000))


@pytest.mark.benchmark
@pytest.mark.release
def test_scipys_distribution_function_at_the_stress_depth(
    benchmark: BenchmarkFixture,
) -> None:
    """404.6 ms at 20,000 reads per bin, summing one element at a time."""
    counts, totals = _bins(*STRESS_DEPTH)

    benchmark(
        lambda: (
            scipy.stats.betabinom.cdf(counts, totals, 15.0, 15.0),
            scipy.stats.betabinom.cdf(counts - 1, totals, 15.0, 15.0),
        )
    )


@pytest.mark.benchmark
@pytest.mark.release
def test_the_vectorized_distribution_function_at_the_stress_depth(
    benchmark: BenchmarkFixture,
) -> None:
    """73.1 ms: **5.5x**, and 4.9x again at 100,000 reads per bin.

    Where the claim is made. Both forms are linear in the terms summed, so the
    ratio is what a term costs: four gathers against two `betaln` calls.
    """
    from port.patch.normal_baf import cumulative_and_mass

    counts, totals = _bins(*STRESS_DEPTH)

    benchmark(lambda: cumulative_and_mass(counts, totals, 15.0, 15.0))


@pytest.mark.benchmark
def test_cnasters_reference_read_at_the_gate_size(
    benchmark: BenchmarkFixture, hgtable: Any
) -> None:
    """`pd.read_csv` at 1,213 transcripts: 6.31 ms (#185)."""
    from cnaster.reference import get_reference_genes

    benchmark(lambda: get_reference_genes(str(hgtable(1_213))))


@pytest.mark.benchmark
def test_the_polars_reference_read_at_the_gate_size(
    benchmark: BenchmarkFixture, hgtable: Any
) -> None:
    """`pl.read_csv`, handed back through Arrow: 4.03 ms, **0.98x** (#185).

    **No faster than `pandas` at this size, and that is the finding.** The
    gate size has 1,213 transcripts: a multi-threaded parser has nothing to
    divide and the Arrow conversion's fixed cost is the whole of the read.
    The claim is at the stress size below, and it is about memory.
    """
    from port.patch.reference import get_reference_genes

    benchmark(lambda: get_reference_genes(str(hgtable(1_213))))


@pytest.mark.benchmark
@pytest.mark.release
def test_cnasters_reference_read_at_the_stress_size(
    benchmark: BenchmarkFixture, hgtable: Any
) -> None:
    """430.7 ms and 49.5 MB at 250,000 transcripts, a human reference's size."""
    from cnaster.reference import get_reference_genes

    benchmark(lambda: get_reference_genes(str(hgtable(250_000))))


@pytest.mark.benchmark
@pytest.mark.release
def test_the_polars_reference_read_at_the_stress_size(
    benchmark: BenchmarkFixture, hgtable: Any
) -> None:
    """60.5 ms and 15.5 MB: **7.1x**, and **3.2x less allocated**.

    Where the claim is made, and the claim is the second column. Reading the
    frame back column by column through `numpy` is 42.8 ms and 23.3 MB --
    **faster by 1.41x** and heavier by 1.50x -- so `pyarrow` is a memory
    patch and a time cost, not a speedup, and `CLAUDE.md`'s 2x bar is
    therefore not the rule that decides it. The evidence that does is the
    bitwise test: `tests/test_reference_patch.py` compares the frame,
    its index, its column order and its dtypes against `cnaster`'s.

    Warm, best of five, and both routes measured in the same pass -- the
    first read of a 250,000-row file is the page cache, not the parser.
    """
    from port.patch.reference import get_reference_genes

    benchmark(lambda: get_reference_genes(str(hgtable(250_000))))
