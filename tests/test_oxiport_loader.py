"""The `oxiport` kernels against the expressions they replace (#184).

Two kernels, and the same three questions of each: does it compute what the
`numpy` expression computes, does the loader built on it return what the loader
built on `numpy` returns, and is what comes out still right against the truth
the fixture planted.

The third is what makes this more than a pair of implementations agreeing. A
backend swap is exactly the change that can be wrong in the same way on both
sides of a comparison, so the planted counts are the referee for the one that
counts.

**Exactness, and where it stops.** `csr_positive_per_column` counts integers
and is asserted bitwise. `csr_pair_row_sums` adds floating-point values in a
different association from `scipy`'s -- per row across two matrices rather than
over a summed matrix -- so it is asserted to `1e-9` relative, and the realized
difference on the shapes below is `2e-12`. The loader's use of it is a
comparison against an integer UMI floor, where nothing at that scale can change
which side a spot falls on.
"""

from collections.abc import Iterator
from inspect import signature
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp

from tests.fixtures import CoreInferenceTruth, core_inference_truth
from tests.run_config import write_run_cnaster_config
from tests.tmp_inputs import WrittenInputs, write_tmp_inputs
from tests.unsegment import unsegment

pytestmark = pytest.mark.preprocessing

ROW_SUM_TOLERANCE = 1.0e-9
"""Relative agreement asked of the row sums. Realized 2e-12 at 2,500 x 782.

The two differ in association and not in arithmetic, so the gap is the
floating-point sum's own reordering. It is stated as a tolerance rather than as
`assert_array_equal` because an exact claim would be a claim about association
order, which neither implementation promises.
"""

SHAPES = [(40, 7, 0.9), (1_000, 79, 0.99), (2_500, 782, 0.5)]
"""`(rows, columns, density)` the kernels are put through.

The first sits below `PARALLEL_THRESHOLD` and the others above it, so both the
sequential and the parallel path are exercised; the densities bracket the
fixture's near-dense SNP matrices and a slide's sparse ones.
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


@pytest.mark.backend
@pytest.mark.parametrize(("rows", "columns", "density"), SHAPES)
def test_the_kernel_sums_the_rows_scipy_sums(
    rows: int, columns: int, density: float
) -> None:
    """**`csr_pair_row_sums` is `(A + B).sum(axis=1)`, without the `A + B`.**

    `scipy` builds the sum matrix and reduces it away; the kernel folds each
    row's stored values in both matrices and never forms it. The answer is the
    same, which is the whole claim -- what differs is a third CSR with up to
    `nnz(A) + nnz(B)` entries that one of them allocates.
    """
    from port import oxiport

    a_matrix = _csr(rows, columns, density, seed=1)
    b_matrix = _csr(rows, columns, density, seed=2)

    expected = np.asarray((a_matrix + b_matrix).sum(axis=1)).ravel()
    realized = np.asarray(
        oxiport.csr_pair_row_sums(
            a_matrix.data, a_matrix.indptr, b_matrix.data, b_matrix.indptr
        )
    )

    assert realized.shape == (rows,)
    np.testing.assert_allclose(realized, expected, rtol=ROW_SUM_TOLERANCE, atol=0.0)


@pytest.mark.backend
@pytest.mark.parametrize(("rows", "columns", "density"), SHAPES)
def test_the_kernel_counts_the_spots_numpy_counts(
    rows: int, columns: int, density: float
) -> None:
    """**`csr_positive_per_column` is `np.sum(X > 0, axis=0)`, bitwise.**

    `numpy` materializes a second matrix of the full shape to count its
    entries. The kernel counts the stored values that are positive, which is
    the same set: a stored zero is not greater than zero either way.

    Integers, so exact. An off-by-one in the scatter would show here and
    nowhere else -- the loader compares the count against a floor, and a floor
    hides a small error until the gene at the boundary is the one that matters.
    """
    from port import oxiport

    matrix = _csr(rows, columns, density, seed=3)

    expected = np.asarray(np.sum(matrix > 0, axis=0)).ravel()
    realized = np.asarray(
        oxiport.csr_positive_per_column(matrix.data, matrix.indices, columns)
    )

    np.testing.assert_array_equal(realized, expected)


@pytest.mark.backend
def test_the_kernels_refuse_a_matrix_they_cannot_read() -> None:
    """A malformed index is a message, not a segfault.

    The kernels index borrowed slices with values a caller supplies, so the
    validation pass is what stands between a wrong `indptr` and reading past
    the buffer. It runs before the fold on purpose: an error raised inside a
    `rayon` fold would have to be carried back across a thread boundary and
    the released GIL.
    """
    from port import oxiport

    matrix = _csr(8, 4, 0.5, seed=4)

    with pytest.raises(ValueError, match="disagree on the row count"):
        oxiport.csr_pair_row_sums(
            matrix.data, matrix.indptr, matrix.data, matrix.indptr[:-1]
        )

    past_the_end = matrix.indptr.copy()
    past_the_end[-1] = matrix.data.size + 1

    with pytest.raises(ValueError, match="not inside its data"):
        oxiport.csr_pair_row_sums(matrix.data, past_the_end, matrix.data, past_the_end)

    with pytest.raises(ValueError, match="outside a matrix"):
        oxiport.csr_positive_per_column(matrix.data, matrix.indices, 1)


@pytest.fixture(scope="module")
def planted_instance(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[CoreInferenceTruth, Any, WrittenInputs, Path]:
    """The dev instance, planted and written once for the module."""
    truth = core_inference_truth(
        n_clones=2, n_states=3, lattice=(25, 40), n_obs=40, n_segments=3, seed=11
    )
    pre_image = unsegment(truth, flip_every=0)
    root: Path = tmp_path_factory.mktemp("oxiport_loader")
    written = write_tmp_inputs(truth, pre_image, root)

    return truth, pre_image, written, write_run_cnaster_config(written, truth)


@pytest.fixture(scope="module")
def installed(
    planted_instance: tuple[CoreInferenceTruth, Any, WrittenInputs, Path],
) -> Iterator[Any]:
    """That instance's configuration, installed for the module."""
    from cnaster.config import YAMLConfig, get_global_config, set_global_config

    previous = get_global_config()
    set_global_config(None)
    set_global_config(YAMLConfig.from_file(planted_instance[3]))
    try:
        yield get_global_config()
    finally:
        set_global_config(None)
        set_global_config(previous)


@pytest.mark.patch
def test_the_rust_backend_loads_what_the_numpy_backend_loads(installed: Any) -> None:
    """Every returned field, across the backend swap.

    The two reductions the backend chooses between decide which spots survive
    the UMI floor and which genes survive the expression floor, so a difference
    in either would show as a different **selection** rather than as a
    different number -- which is why the barcodes and the gene index are
    compared and not only the counts.
    """
    from port.patch.input_data import load_input_data

    with_numpy = load_input_data(installed)
    with_rust = load_input_data(installed, backend="rust")

    np.testing.assert_array_equal(
        np.asarray(with_rust.adata.layers["count"]),
        np.asarray(with_numpy.adata.layers["count"]),
    )
    np.testing.assert_array_equal(
        with_rust.cell_snp_Aallele, with_numpy.cell_snp_Aallele
    )
    np.testing.assert_array_equal(
        with_rust.cell_snp_Ballele, with_numpy.cell_snp_Ballele
    )
    np.testing.assert_array_equal(
        np.asarray(with_rust.barcodes), np.asarray(with_numpy.barcodes)
    )
    np.testing.assert_array_equal(
        np.asarray(with_rust.adata.var.index), np.asarray(with_numpy.adata.var.index)
    )
    np.testing.assert_array_equal(with_rust.unique_snp_ids, with_numpy.unique_snp_ids)


@pytest.mark.end2end
def test_the_rust_backend_returns_the_planted_counts(
    planted_instance: tuple[CoreInferenceTruth, Any, WrittenInputs, Path],
    installed: Any,
) -> None:
    """**What the Rust backend loads is what the fixture planted (#184).**

    Two implementations agreeing says nothing about whether either is right, so
    the backend is put to the referee the loader is already held to: the
    planted per-gene counts, aligned by name and barcode, and the gene set the
    planted counts put above the expression floor.

    The second is the one the kernels can break. `csr_positive_per_column`
    decides which genes survive, so an error there is a different gene set with
    entirely correct counts in it.
    """
    from port.patch.input_data import load_input_data

    _, pre_image, written, _ = planted_instance

    loaded = load_input_data(installed, backend="rust")

    planted = np.asarray(pre_image.adata.layers["count"])
    names = np.asarray(pre_image.adata.var.index)
    planted_genes = {str(name): column for column, name in enumerate(names)}
    spot_of = {str(barcode): spot for spot, barcode in enumerate(written.barcodes)}

    rows = np.array([spot_of[str(barcode)] for barcode in loaded.barcodes])
    columns = np.array(
        [planted_genes[str(name)] for name in np.asarray(loaded.adata.var.index)]
    )

    np.testing.assert_array_equal(
        np.asarray(loaded.adata.layers["count"]), planted[np.ix_(rows, columns)]
    )

    fraction = signature(load_input_data).parameters["min_percent_expressed_spots"]
    surviving = planted[rows]
    floor = float(fraction.default) * surviving.shape[0]
    expected = {str(name) for name in names[(surviving > 0).sum(axis=0) >= floor]}

    assert {str(name) for name in loaded.adata.var.index} == expected


@pytest.mark.smoke
def test_an_unknown_backend_is_refused_by_name(installed: Any) -> None:
    """A typo is an error rather than a silent fall back to `numpy`.

    Which is the one failure a benchmark cannot see: a misspelled backend that
    quietly ran the other implementation would read as a Rust backend that
    bought nothing.
    """
    from port.patch.input_data import load_input_data

    with pytest.raises(ValueError, match="unknown backend"):
        load_input_data(installed, backend="rusty")
