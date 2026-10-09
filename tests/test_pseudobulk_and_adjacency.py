"""Pseudobulk aggregation and the spatial graph, against definitions and invariants
(#12, #13).
"""

import numpy as np
import pytest
from cnaster.adjacency import multislice_adjacency
from cnaster.pseudobulk import merge_pseudobulk_by_index_mix


@pytest.mark.oracle
@pytest.mark.critical
@pytest.mark.parametrize("n_clones", [1, 3])
def test_pseudobulk_sums_the_spots_of_each_clone(n_clones: int) -> None:
    """A clone's column is the sum over the spots assigned to it."""

    rng = np.random.default_rng(6)
    n_obs, n_spots = 9, 12
    single_X = rng.integers(0, 30, size=(n_obs, 2, n_spots)).astype(float)
    base = rng.random((n_obs, n_spots))
    total = rng.random((n_obs, n_spots))

    assignment = rng.integers(0, n_clones, size=n_spots)
    clone_index = [np.where(assignment == c)[0] for c in range(n_clones)]

    X, merged_base, merged_total, _ = merge_pseudobulk_by_index_mix(
        single_X, base, total, clone_index
    )

    assert X.shape == (n_obs, 2, n_clones)
    for clone, idx in enumerate(clone_index):
        np.testing.assert_allclose(X[:, :, clone], single_X[:, :, idx].sum(axis=-1))
        np.testing.assert_allclose(merged_base[:, clone], base[:, idx].sum(axis=-1))
        np.testing.assert_allclose(merged_total[:, clone], total[:, idx].sum(axis=-1))


@pytest.mark.analytic
def test_pseudobulk_conserves_the_total_over_a_partition() -> None:
    """Summing the clones returns the sum over every spot."""

    rng = np.random.default_rng(8)
    n_obs, n_spots, n_clones = 6, 15, 4
    single_X = rng.integers(0, 20, size=(n_obs, 2, n_spots)).astype(float)
    base = rng.random((n_obs, n_spots))
    total = rng.random((n_obs, n_spots))
    assignment = rng.integers(0, n_clones, size=n_spots)

    X, _, _, _ = merge_pseudobulk_by_index_mix(
        single_X, base, total, [np.where(assignment == c)[0] for c in range(n_clones)]
    )

    np.testing.assert_allclose(X.sum(axis=-1), single_X.sum(axis=-1))


def square_grid(side: int, offset: float = 0.0) -> np.ndarray:
    """A `side x side` lattice of unit spacing."""
    return np.array(
        [[x + offset, y + offset] for x in range(side) for y in range(side)],
        dtype=float,
    )


@pytest.mark.analytic
def test_adjacency_is_symmetric_and_has_no_self_edges() -> None:
    """Neighbourhood is symmetric with no self-edges."""

    coords = square_grid(5)
    adjacency, _ = multislice_adjacency(
        coords, np.zeros(len(coords), dtype=int), lattice_type="square"
    )

    assert (adjacency - adjacency.T).nnz == 0
    assert adjacency.diagonal().sum() == 0


@pytest.mark.analytic
def test_adjacency_does_not_join_slices_by_default() -> None:
    """Two distant slices share no edge unless one is supplied."""

    first = square_grid(4)
    coords = np.vstack([first, square_grid(4, offset=100.0)])
    sample_ids = np.array([0] * len(first) + [1] * len(first))

    adjacency, _ = multislice_adjacency(coords, sample_ids, lattice_type="square")

    assert adjacency[: len(first), len(first) :].nnz == 0


@pytest.mark.smoke
def test_lattice_type_sets_the_coordination_number() -> None:
    """`square` gives four neighbours and `triangular` six."""

    coords = square_grid(6)
    sample_ids = np.zeros(len(coords), dtype=int)

    square, _ = multislice_adjacency(coords, sample_ids, lattice_type="square")
    triangular, _ = multislice_adjacency(coords, sample_ids, lattice_type="triangular")
    explicit, _ = multislice_adjacency(coords, sample_ids, n_nearest=4)

    assert triangular.nnz > square.nnz
    assert (square - explicit).nnz == 0
