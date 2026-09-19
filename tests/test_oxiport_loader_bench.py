"""What the `oxiport` kernels cost against the expressions they replace (#184).

**1.8x and 0.97x at the stress size, where the bar is 2x.** The row sums
read 2.21x under `pytest-benchmark` and 1.75x under a `perf_counter` loop over
the same two calls, which is a claim at the width of its own measurement
rather than a claim above the bar; the gene counts are slower than what they
replace. #184 records that rather than landing it.

The number a kernel is worth is the one against the form the loader actually
runs, and #167 had already replaced both expressions before this was written:

| 5,000 spots, 20,000 SNPs, 2% | row sums | gene counts |
| --- | --- | --- |
| `cnaster` | 256 ms, 864 MB | 9.0 ms, 26 MB |
| the sparse intermediate | 43.4 ms, 80 MB | -- |
| **#167's form** | **1.59 ms, 0.2 MB** | **3.75 ms** |
| the kernel | 0.72 ms | 3.87 ms |

Against `cnaster` the kernels read as 389x and 2.3x; against what replaced it,
2.2x and 0.97x. The first pair is #167's win and not this one's, which is why
all three forms are benchmarked and not two.

**Why there was nothing left to win.** `A.sum(axis=1)` is a `scipy` matvec
against a ones vector and `getnnz(axis=0)` is a `bincount` over the column
indices -- both already one C pass over the stored values, which is what the
kernels are. Rust buys the loop it was already getting. What `cnaster` paid
was the intermediate, and #167 had stopped paying it.

`rayon` does not change this: the kernels run their parallel path at both
sizes on 4 cores, and the row sums' figure **is** the parallel one.

These stay as a benchmark because the negative result is the deliverable. A
later kernel proposed for this path has a reference to beat that is measured
rather than assumed.
"""

from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp
from pytest_benchmark.fixture import BenchmarkFixture

pytestmark = pytest.mark.preprocessing

GATE_SHAPE = (2_500, 782, 0.5)
"""`(rows, columns, density)` at the dev instance's scale."""

STRESS_SHAPE = (5_000, 20_000, 0.02)
"""A slide's spots against a chromosome's SNPs, at a slide's density.

The Measurement rule puts the claim here, and the claim it carried was the one
that failed: 2.21x at this size against 2.15x at the gate size, so the ratio is
flat in the problem and no larger instance moves it clear of its own error.
"""


def _csr(rows: int, columns: int, density: float, seed: int) -> Any:
    """A CSR matrix with the index types the kernels take."""
    matrix = sp.random(
        rows,
        columns,
        density=density,
        format="csr",
        random_state=seed,
        dtype=np.float64,
    )
    matrix.indices = matrix.indices.astype(np.int64)
    matrix.indptr = matrix.indptr.astype(np.int64)

    return matrix


def _pair(shape: tuple[int, int, float]) -> tuple[Any, Any]:
    """The two allele matrices, at one shape."""
    return _csr(*shape, seed=1), _csr(*shape, seed=2)


def _row_sums_rust(a_matrix: Any, b_matrix: Any) -> np.ndarray:
    """The kernel, called as `port.patch.input_data` calls it."""
    from port import oxiport

    return np.asarray(
        oxiport.csr_pair_row_sums(
            a_matrix.data, a_matrix.indptr, b_matrix.data, b_matrix.indptr
        )
    )


def _counts_rust(matrix: Any) -> np.ndarray:
    """The kernel, called as `port.patch.input_data` calls it."""
    from port import oxiport

    return np.asarray(
        oxiport.csr_positive_per_column(matrix.data, matrix.indices, matrix.shape[1])
    )


@pytest.mark.benchmark
def test_cnasters_allele_row_sums_at_the_gate_size(benchmark: BenchmarkFixture) -> None:
    """`(A + B).todense().sum(axis=1)`, the loader's form: 20.8 ms, 39 MB."""
    a_matrix, b_matrix = _pair(GATE_SHAPE)

    benchmark(lambda: np.asarray((a_matrix + b_matrix).todense().sum(axis=1)).ravel())


@pytest.mark.benchmark
def test_the_patched_allele_row_sums_at_the_gate_size(
    benchmark: BenchmarkFixture,
) -> None:
    """`A.sum(axis=1) + B.sum(axis=1)`, what #167 landed: 0.812 ms, 0.1 MB.

    The reference the kernel is held to. It forms no intermediate at all, so
    there is no allocation left for a port to remove.
    """
    a_matrix, b_matrix = _pair(GATE_SHAPE)

    benchmark(
        lambda: np.asarray(a_matrix.sum(axis=1)).ravel()
        + np.asarray(b_matrix.sum(axis=1)).ravel()
    )


@pytest.mark.benchmark
def test_the_kernel_allele_row_sums_at_the_gate_size(
    benchmark: BenchmarkFixture,
) -> None:
    """`csr_pair_row_sums`: 0.378 ms against 0.812, so **2.15x**."""
    a_matrix, b_matrix = _pair(GATE_SHAPE)

    benchmark(lambda: _row_sums_rust(a_matrix, b_matrix))


@pytest.mark.benchmark
def test_cnasters_gene_counts_at_the_gate_size(benchmark: BenchmarkFixture) -> None:
    """`np.sum(X > 0, axis=0)`, the loader's form: 4.03 ms, 12.8 MB."""
    matrix = _csr(*GATE_SHAPE, seed=3)

    benchmark(lambda: np.asarray(np.sum(matrix > 0, axis=0)).ravel())


@pytest.mark.benchmark
def test_the_patched_gene_counts_at_the_gate_size(benchmark: BenchmarkFixture) -> None:
    """`getnnz(axis=0)`, what #167 landed: 1.200 ms.

    It counts stored entries rather than positive ones, which is the same
    answer only where no stored entry is zero; the patch keeps the explicit
    count as the fallback and this benchmarks the fast path it takes.
    """
    matrix = _csr(*GATE_SHAPE, seed=3)

    benchmark(lambda: np.asarray(matrix.getnnz(axis=0)).ravel())


@pytest.mark.benchmark
def test_the_kernel_gene_counts_at_the_gate_size(benchmark: BenchmarkFixture) -> None:
    """`csr_positive_per_column`: 0.983 ms against 1.200, so **1.22x**."""
    matrix = _csr(*GATE_SHAPE, seed=3)

    benchmark(lambda: _counts_rust(matrix))


@pytest.mark.benchmark
@pytest.mark.release
def test_cnasters_allele_row_sums_at_the_stress_size(
    benchmark: BenchmarkFixture,
) -> None:
    """The densification at a slide's scale: 279 ms, 864 MB."""
    a_matrix, b_matrix = _pair(STRESS_SHAPE)

    benchmark(lambda: np.asarray((a_matrix + b_matrix).todense().sum(axis=1)).ravel())


@pytest.mark.benchmark
@pytest.mark.release
def test_the_sparse_allele_row_sums_at_the_stress_size(
    benchmark: BenchmarkFixture,
) -> None:
    """The sparse add, which neither reference runs: 47.0 ms, 80 MB.

    Kept because it is the step between the two: it shows that dropping the
    densification is worth 5.9x and that dropping the sum matrix as well is
    worth a further 30x, so the 389x `cnaster`-to-kernel ratio is almost
    entirely #167's.
    """
    a_matrix, b_matrix = _pair(STRESS_SHAPE)

    benchmark(lambda: np.asarray((a_matrix + b_matrix).sum(axis=1)).ravel())


@pytest.mark.benchmark
@pytest.mark.release
def test_the_patched_allele_row_sums_at_the_stress_size(
    benchmark: BenchmarkFixture,
) -> None:
    """The reference at a slide's scale: 1.586 ms and 0.2 MB."""
    a_matrix, b_matrix = _pair(STRESS_SHAPE)

    benchmark(
        lambda: np.asarray(a_matrix.sum(axis=1)).ravel()
        + np.asarray(b_matrix.sum(axis=1)).ravel()
    )


@pytest.mark.benchmark
@pytest.mark.release
def test_the_kernel_allele_row_sums_at_the_stress_size(
    benchmark: BenchmarkFixture,
) -> None:
    """0.716 ms against 1.586: **2.21x** here, 1.75x under a `perf_counter`
    loop over the same pair. The bar is 2x and the two readings straddle it,
    which is the result: a win this small is not distinguishable from its
    harness."""
    a_matrix, b_matrix = _pair(STRESS_SHAPE)

    benchmark(lambda: _row_sums_rust(a_matrix, b_matrix))


@pytest.mark.benchmark
@pytest.mark.release
def test_cnasters_gene_counts_at_the_stress_size(benchmark: BenchmarkFixture) -> None:
    """`np.sum(X > 0, axis=0)` at a slide's scale: 8.92 ms, 26 MB."""
    matrix = _csr(*STRESS_SHAPE, seed=3)

    benchmark(lambda: np.asarray(np.sum(matrix > 0, axis=0)).ravel())


@pytest.mark.benchmark
@pytest.mark.release
def test_the_patched_gene_counts_at_the_stress_size(
    benchmark: BenchmarkFixture,
) -> None:
    """`getnnz(axis=0)` at a slide's scale: 3.752 ms."""
    matrix = _csr(*STRESS_SHAPE, seed=3)

    benchmark(lambda: np.asarray(matrix.getnnz(axis=0)).ravel())


@pytest.mark.benchmark
@pytest.mark.release
def test_the_kernel_gene_counts_at_the_stress_size(benchmark: BenchmarkFixture) -> None:
    """3.87 ms against 3.75: **0.97x**, so the kernel is the slower of the two.

    The gate size's 1.23x does not survive the column count. 20,000 columns is
    a private accumulator per worker that no longer fits in cache, where 782
    does, and the reduction over them is `n_workers * n_cols` additions that
    `bincount` never performs.
    """
    matrix = _csr(*STRESS_SHAPE, seed=3)

    benchmark(lambda: _counts_rust(matrix))
