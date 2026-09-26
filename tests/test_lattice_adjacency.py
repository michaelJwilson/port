"""The lattice adjacency and its guard (#417).

The sparsity pattern is refereed by brute force -- every pair at unit
Euclidean distance in the lattice's own embedding -- and the weights by the
rule's two properties: exactly 1 between interior spots, above 1 wherever an
end is on the boundary. The guard is refereed by the three graphs it must
refuse, `cnaster`'s own among them.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp


def _square(rows: int, cols: int) -> np.ndarray:
    r, c = np.unravel_index(np.arange(rows * cols), (rows, cols))
    return np.stack([r, c], axis=1)


def _triangular(rows: int, cols: int) -> np.ndarray:
    """Visium array positions: `col` even on even rows, odd on odd rows."""
    r, k = np.unravel_index(np.arange(rows * cols), (rows, cols))
    return np.stack([r, 2 * k + r % 2], axis=1)


def _brute_force(coords: np.ndarray, kind: str) -> np.ndarray:
    """Square: distance 1 and sqrt(2) (Moore); triangular: 1 in its hexagonal embedding."""
    points = coords.astype(np.float64)
    if kind == "triangular":
        points = np.stack([points[:, 0] * np.sqrt(3.0) / 2.0, points[:, 1] / 2.0], 1)
    distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    pattern: np.ndarray = np.isclose(distance, 1.0, atol=1e-9)
    if kind == "square":
        pattern |= np.isclose(distance, np.sqrt(2.0), atol=1e-9)
    return pattern


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("kind", "build", "shape"),
    [
        ("square", _square, (7, 5)),
        ("square", _square, (1, 6)),
        ("triangular", _triangular, (6, 7)),
        ("triangular", _triangular, (3, 3)),
    ],
)
def test_the_neighbours_are_the_brute_force_pairs(
    kind: str, build: object, shape: tuple[int, int]
) -> None:
    from port.patch.spatial import lattice_adjacency, lattice_kind

    coords = build(*shape)  # type: ignore[operator]

    assert lattice_kind(coords) == kind
    adjacency = lattice_adjacency(coords)
    np.testing.assert_array_equal(adjacency.toarray() > 0, _brute_force(coords, kind))


@pytest.mark.analytic
@pytest.mark.parametrize(
    ("kind", "build"), [("square", _square), ("triangular", _triangular)]
)
def test_interior_edges_weigh_one_and_boundary_edges_more(
    kind: str, build: object
) -> None:
    """Symmetric, no self loops, 1 in the interior, above 1 at the boundary."""
    from port.patch.spatial import COORDINATION, lattice_adjacency

    adjacency = lattice_adjacency(build(8, 9))  # type: ignore[operator]
    z = COORDINATION[kind]
    coo = adjacency.tocoo()
    degree = np.bincount(coo.row, minlength=adjacency.shape[0])
    interior = (degree[coo.row] == z) & (degree[coo.col] == z)

    assert abs(adjacency - adjacency.T).max() == 0.0
    assert not adjacency.diagonal().any()
    assert np.all(coo.data[interior] == 1.0)
    assert np.all(coo.data[~interior] > 1.0)
    assert degree.max() == z

    # NB a spot whose neighbours are all interior feels exactly `z`; one next
    #    to the boundary shares a reinforced edge and feels a little more; a
    #    boundary spot is lifted above its own neighbour count.
    weighted = np.asarray(adjacency.sum(axis=1)).ravel()
    deep = np.array(
        [
            degree[i] == z and np.all(degree[adjacency.indices[s:e]] == z)
            for i, (s, e) in enumerate(
                zip(adjacency.indptr[:-1], adjacency.indptr[1:], strict=True)
            )
        ]
    )
    np.testing.assert_allclose(weighted[deep], z, rtol=0, atol=1e-12)
    assert np.all(weighted[(degree == z) & ~deep] > z)
    assert np.all(weighted[degree < z] > degree[degree < z])


@pytest.mark.analytic
def test_the_guard_refuses_cnasters_directed_graph() -> None:
    """`cnaster`'s eight nearest neighbours on a square grid: not symmetric."""
    from cnaster.spatial import construct_lattice_adjacency
    from port.patch.spatial import AdjacencyError, validate_adjacency

    _, directed = construct_lattice_adjacency(
        _square(12, 12).astype(float), unit_xsquared=1, unit_ysquared=1
    )

    with pytest.raises(AdjacencyError, match="not symmetric"):
        validate_adjacency(directed)


@pytest.mark.analytic
def test_the_guard_refuses_a_self_loop_and_an_unreinforced_boundary() -> None:
    from port.patch.spatial import (
        AdjacencyError,
        lattice_adjacency,
        validate_adjacency,
    )

    adjacency = lattice_adjacency(_square(4, 4))
    validate_adjacency(adjacency, 8)

    looped = adjacency.tolil()
    looped[0, 0] = 1.0
    with pytest.raises(AdjacencyError, match="self loop"):
        validate_adjacency(looped.tocsr(), 8)

    flat = adjacency.copy()
    flat.data[:] = 1.0
    with pytest.raises(AdjacencyError, match="reinforced"):
        validate_adjacency(flat, 8)


@pytest.mark.analytic
def test_positions_that_are_no_lattice_are_refused() -> None:
    from port.patch.spatial import AdjacencyError, lattice_kind

    with pytest.raises(AdjacencyError, match="integer"):
        lattice_kind(np.array([[0.0, 0.5], [1.0, 0.0]]))

    with pytest.raises(AdjacencyError, match="neither"):
        lattice_kind(np.array([[0, 0], [5, 7], [11, 3]]))


@pytest.mark.analytic
def test_slices_are_assembled_block_diagonal_and_validated() -> None:
    """Two slices, `cnaster`'s order and its identity pooling matrix."""
    from port.patch.spatial import (
        lattice_adjacency,
        lattice_multislice_adjacency,
    )

    first, second = _square(3, 4), _square(5, 2)
    coords = np.concatenate([first, second])
    sample_ids = np.repeat([0, 1], [len(first), len(second)])

    adjacency, smooth = lattice_multislice_adjacency(
        sample_ids, ["A", "B"], coords, None, maxspots_pooling=1
    )

    expected = sp.block_diag([lattice_adjacency(first), lattice_adjacency(second)])
    np.testing.assert_array_equal(adjacency.toarray(), expected.toarray())
    np.testing.assert_array_equal(smooth.toarray(), np.eye(len(coords), dtype=np.int8))


@pytest.mark.patch
def test_the_interior_is_cnasters_graph_on_a_square_grid() -> None:
    """Away from the boundary, the Moore lattice's rows are `cnaster`'s kNN rows.

    `construct_lattice_adjacency` takes eight nearest neighbours; on a unit
    square grid those are the four axis and four diagonal spots, so its
    interior rows and the lattice's agree entry for entry. At the boundary
    the kNN reaches farther to make up eight, which is the directed part this
    replaces -- so the comparison is over rows two or more spots in.
    """
    from cnaster.spatial import construct_lattice_adjacency
    from port.patch.spatial import lattice_adjacency

    side = 12
    coords = _square(side, side)
    _, directed = construct_lattice_adjacency(
        coords.astype(float), unit_xsquared=1, unit_ysquared=1
    )
    ours = lattice_adjacency(coords)
    rows, cols = coords[:, 0], coords[:, 1]
    inner = (rows >= 2) & (rows < side - 2) & (cols >= 2) & (cols < side - 2)

    # NB `cnaster`'s own rows, not the union: a boundary spot's kNN reaches two
    #    spots in to make up eight, so the union adds those edges to inner rows
    #    too, and they are exactly the directed part this replaces.
    theirs = (directed.toarray() > 0)[inner]
    np.testing.assert_array_equal((ours.toarray() > 0)[inner], theirs)
    np.testing.assert_array_equal(ours.toarray()[inner][theirs], 1.0)
