"""The loader's `sparse_counts=True` path against its dense return and the planted counts (#186)."""

from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp
from port.sim.run_config import PlantedInstance

pytestmark = pytest.mark.preprocessing

SHAPES = [(60, 25, 0.4), (400, 137, 0.05)]
"""`(rows, columns, density)` the column helpers are put through."""


def _counts(rows: int, columns: int, density: float, seed: int) -> np.ndarray:
    """Return a dense integer count matrix with a spread gene filters react to."""
    generator = np.random.default_rng(seed)
    dense = np.zeros((rows, columns), dtype=np.int64)
    drawn = generator.random((rows, columns)) < density
    dense[drawn] = generator.integers(1, 5_000, size=int(drawn.sum()))

    return dense


@pytest.mark.patch
@pytest.mark.parametrize(("rows", "columns", "density"), SHAPES)
def test_the_gene_totals_agree_across_the_container(
    rows: int, columns: int, density: float
) -> None:
    """`_gene_umis` returns the same per-gene totals from sparse and dense."""
    from port.patch.io import _gene_umis

    dense = _counts(rows, columns, density, seed=3)

    np.testing.assert_array_equal(_gene_umis(sp.csr_matrix(dense)), _gene_umis(dense))


@pytest.mark.patch
@pytest.mark.parametrize(("rows", "columns", "density"), SHAPES)
def test_scaling_a_column_truncates_the_same_way_in_both_forms(
    rows: int, columns: int, density: float
) -> None:
    """`_scaled_columns` truncates entry for entry alike in both forms, keeping the int dtype."""
    from port.patch.io import _scaled_columns

    dense = _counts(rows, columns, density, seed=5)
    generator = np.random.default_rng(7)

    factors = np.ones(columns, dtype=float)
    factors[generator.choice(columns, columns // 4, replace=False)] = 1.0 / 3.0
    factors[generator.choice(columns, columns // 4, replace=False)] = 1.0 / 7.0
    factors[generator.choice(columns, columns // 8, replace=False)] = 0.0

    scaled_sparse = _scaled_columns(sp.csr_matrix(dense), factors)
    scaled_dense = _scaled_columns(dense.copy(), factors)

    assert sp.issparse(scaled_sparse)
    assert scaled_sparse.dtype == dense.dtype
    np.testing.assert_array_equal(scaled_sparse.toarray(), scaled_dense)


@pytest.mark.patch
def test_zeroing_a_column_removes_it_from_the_stored_values() -> None:
    """Zeroing a column removes it from the sparse `.data`, in place."""
    from port.patch.io import _gene_umis, _scaled_columns

    dense = _counts(80, 12, 0.5, seed=11)
    sparse = sp.csr_matrix(dense)

    factors = np.ones(12, dtype=float)
    factors[[2, 7]] = 0.0

    stored = sparse.nnz
    scaled = _scaled_columns(sparse, factors)

    # NB in place: the caller assigns the result back over its input
    assert scaled is sparse
    assert scaled.nnz == stored - int((dense[:, [2, 7]] > 0).sum())
    assert _gene_umis(scaled)[[2, 7]].tolist() == [0.0, 0.0]
    np.testing.assert_array_equal(
        scaled.getnnz(axis=0)[[2, 7]], np.zeros(2, dtype=np.int64)
    )


@pytest.fixture(scope="module")
def both_returns(gate_config: Any) -> tuple[Any, Any]:
    """Run the loader once dense and once sparse on one instance."""
    from port.patch.io import load_input_data

    return load_input_data(gate_config), load_input_data(
        gate_config, sparse_counts=True
    )


@pytest.mark.patch
def test_the_sparse_return_carries_every_field_the_dense_one_does(
    both_returns: tuple[Any, Any],
) -> None:
    """Every sparse field equals the dense return's, including genes and barcodes."""
    dense, sparse = both_returns

    assert sp.issparse(sparse.cell_snp_Aallele)
    assert sp.issparse(sparse.adata.layers["count"])
    assert sp.issparse(sparse.exp_counts)

    np.testing.assert_array_equal(
        sparse.adata.layers["count"].toarray(),
        np.asarray(dense.adata.layers["count"]),
    )
    np.testing.assert_array_equal(
        sparse.cell_snp_Aallele.toarray(), dense.cell_snp_Aallele
    )
    np.testing.assert_array_equal(
        sparse.cell_snp_Ballele.toarray(), dense.cell_snp_Ballele
    )
    np.testing.assert_array_equal(
        sparse.exp_counts.toarray(), dense.exp_counts.sparse.to_dense().to_numpy()
    )
    np.testing.assert_array_equal(
        np.asarray(sparse.adata.var.index), np.asarray(dense.adata.var.index)
    )
    np.testing.assert_array_equal(
        np.asarray(sparse.barcodes), np.asarray(dense.barcodes)
    )
    np.testing.assert_array_equal(sparse.unique_snp_ids, dense.unique_snp_ids)


@pytest.mark.end2end
def test_the_sparse_loader_returns_the_planted_counts(
    planted_instance: PlantedInstance,
    both_returns: tuple[Any, Any],
) -> None:
    """The sparse return equals the planted counts, integer dtype, aligned by name."""
    _, pre_image, written, _ = planted_instance
    _, sparse = both_returns

    planted = np.asarray(pre_image.adata.layers["count"])
    names = np.asarray(pre_image.adata.var.index)
    planted_genes = {str(name): column for column, name in enumerate(names)}
    spot_of = {str(barcode): spot for spot, barcode in enumerate(written.barcodes)}

    rows = np.array([spot_of[str(barcode)] for barcode in sparse.barcodes])
    columns = np.array(
        [planted_genes[str(name)] for name in np.asarray(sparse.adata.var.index)]
    )

    loaded = sparse.adata.layers["count"].toarray()

    assert loaded.dtype == planted.dtype
    np.testing.assert_array_equal(loaded, planted[np.ix_(rows, columns)])
