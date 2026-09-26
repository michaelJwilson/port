"""The spatial adjacencies and their guard (#417).

Two constructions over three neighbourhoods (square 4, Moore 8, triangular
6). `knn` -- the default -- is `cnaster`'s directed construction with `k`
the neighbourhood's coordination; on a square grid with `k = 8` it is
`cnaster`'s graph entry for entry. `lattice` is the neighbourhood's offsets,
symmetric, boundary edges reinforced. Patterns are refereed by brute force
(pairs at the neighbourhood's distances), weights by the reinforcement
rule's properties, and the guard by the graphs it must refuse.
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


def _brute_force(coords: np.ndarray, neighbourhood: str) -> np.ndarray:
    """Pairs at distance 1 -- and sqrt(2) for Moore -- in the lattice's embedding."""
    points = coords.astype(np.float64)
    if neighbourhood == "triangular":
        points = np.stack([points[:, 0] * np.sqrt(3.0) / 2.0, points[:, 1] / 2.0], 1)
    distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    pattern: np.ndarray = np.isclose(distance, 1.0, atol=1e-9)
    if neighbourhood == "moore":
        pattern |= np.isclose(distance, np.sqrt(2.0), atol=1e-9)
    return pattern


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("kind", "neighbourhood", "build", "shape"),
    [
        ("square", "square", _square, (7, 5)),
        ("square", "moore", _square, (7, 5)),
        ("square", "moore", _square, (1, 6)),
        ("triangular", "triangular", _triangular, (6, 7)),
        ("triangular", "triangular", _triangular, (3, 3)),
    ],
)
def test_the_lattice_neighbours_are_the_brute_force_pairs(
    kind: str, neighbourhood: str, build: object, shape: tuple[int, int]
) -> None:
    from port.extensions.adjacency import lattice_adjacency, lattice_kind

    coords = build(*shape)  # type: ignore[operator]

    assert lattice_kind(coords) == kind
    adjacency = lattice_adjacency(coords, neighbourhood)  # type: ignore[arg-type]
    np.testing.assert_array_equal(
        adjacency.toarray() > 0, _brute_force(coords, neighbourhood)
    )


@pytest.mark.analytic
@pytest.mark.parametrize(
    ("neighbourhood", "build"),
    [("square", _square), ("moore", _square), ("triangular", _triangular)],
)
def test_interior_edges_weigh_one_and_boundary_edges_more(
    neighbourhood: str, build: object
) -> None:
    """Symmetric, no self loops, 1 in the interior, above 1 at the boundary."""
    from port.extensions.adjacency import COORDINATION, lattice_adjacency

    adjacency = lattice_adjacency(build(8, 9), neighbourhood)  # type: ignore[operator, arg-type]
    z = COORDINATION[neighbourhood]
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
    from port.extensions.adjacency import AdjacencyError, validate_adjacency

    _, directed = construct_lattice_adjacency(
        _square(12, 12).astype(float), unit_xsquared=1, unit_ysquared=1
    )

    with pytest.raises(AdjacencyError, match="not symmetric"):
        validate_adjacency(directed)


@pytest.mark.analytic
def test_the_guard_refuses_a_self_loop_and_an_unreinforced_boundary() -> None:
    from port.extensions.adjacency import (
        AdjacencyError,
        lattice_adjacency,
        validate_adjacency,
    )

    adjacency = lattice_adjacency(_square(4, 4), "moore")
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
    from port.extensions.adjacency import AdjacencyError, lattice_kind

    with pytest.raises(AdjacencyError, match="integer"):
        lattice_kind(np.array([[0.0, 0.5], [1.0, 0.0]]))

    with pytest.raises(AdjacencyError, match="neither"):
        lattice_kind(np.array([[0, 0], [5, 7], [11, 3]]))


@pytest.mark.analytic
def test_slices_are_assembled_block_diagonal_and_validated() -> None:
    """Two slices, `cnaster`'s order and its identity pooling matrix."""
    from port.extensions.adjacency import knn_adjacency
    from port.patch.spatial import lattice_multislice_adjacency

    first, second = _square(4, 4), _square(5, 3)
    coords = np.concatenate([first, second])
    sample_ids = np.repeat([0, 1], [len(first), len(second)])

    adjacency, smooth = lattice_multislice_adjacency(
        sample_ids, ["A", "B"], coords, None, maxspots_pooling=1
    )

    # NB the default construction, `knn`, with the default Moore neighbourhood.
    expected = sp.block_diag(
        [knn_adjacency(first, "moore"), knn_adjacency(second, "moore")]
    )
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
    from port.extensions.adjacency import lattice_adjacency

    side = 12
    coords = _square(side, side)
    _, directed = construct_lattice_adjacency(
        coords.astype(float), unit_xsquared=1, unit_ysquared=1
    )
    ours = lattice_adjacency(coords, "moore")
    rows, cols = coords[:, 0], coords[:, 1]
    inner = (rows >= 2) & (rows < side - 2) & (cols >= 2) & (cols < side - 2)

    # NB `cnaster`'s own rows, not the union: a boundary spot's kNN reaches two
    #    spots in to make up eight, so the union adds those edges to inner rows
    #    too, and they are exactly the directed part this replaces.
    theirs = (directed.toarray() > 0)[inner]
    np.testing.assert_array_equal((ours.toarray() > 0)[inner], theirs)
    np.testing.assert_array_equal(ours.toarray()[inner][theirs], 1.0)


@pytest.mark.patch
def test_knn_moore_is_cnasters_graph_on_a_square_grid() -> None:
    """The default construction: `cnaster`'s eight nearest neighbours, entry for entry."""
    from cnaster.spatial import construct_lattice_adjacency
    from port.extensions.adjacency import knn_adjacency, validate_adjacency

    coords = _square(40, 40)
    _, theirs = construct_lattice_adjacency(
        coords.astype(float), unit_xsquared=1, unit_ysquared=1
    )
    ours = knn_adjacency(coords, "moore")

    assert (ours != theirs).nnz == 0
    validate_adjacency(ours, 8, construction="knn")


@pytest.mark.analytic
@pytest.mark.parametrize(
    ("neighbourhood", "build", "shape", "realized"),
    [
        ("moore", _square, (40, 40), 0.987),
        ("moore", _square, (12, 10), 0.942),
        ("moore", _square, (4, 4), 0.781),
        ("square", _square, (4, 5), 0.775),
        ("triangular", _triangular, (40, 40), 0.975),
        ("triangular", _triangular, (4, 4), 0.792),
    ],
)
def test_knn_is_symmetric_away_from_the_boundary(
    neighbourhood: str, build: object, shape: tuple[int, int], realized: float
) -> None:
    """Every interior spot's edges are reciprocated; the asymmetry is at the edge.

    A spot two or more rows from every edge has `k` neighbours at the
    lattice spacing, each of which names it back, so its row is symmetric.
    The share of reciprocated edges is pinned at its realized value, which
    is what `RECIPROCATED` (0.6) is set under.
    """
    from port.extensions.adjacency import RECIPROCATED, knn_adjacency

    coords = build(*shape)  # type: ignore[operator]
    adjacency = knn_adjacency(coords, neighbourhood)  # type: ignore[arg-type]
    degree = np.asarray((adjacency > 0).sum(axis=1)).ravel()
    reciprocated = np.asarray(adjacency.multiply(adjacency.T).astype(bool).sum(axis=1))
    full = reciprocated.ravel() == degree

    rows = coords[:, 0]
    cols = np.arange(coords.shape[0]) % shape[1]
    inner = (rows >= 2) & (rows < shape[0] - 2) & (cols >= 2) & (cols < shape[1] - 2)
    share = adjacency.multiply(adjacency.T).nnz / adjacency.nnz

    assert full[inner].all()
    assert share == pytest.approx(realized, abs=5e-4)
    assert share >= RECIPROCATED


@pytest.mark.analytic
def test_the_knn_guard_refuses_what_is_not_a_knn_lattice() -> None:
    """A self loop, a wrong row count, a weight other than 1, a mostly one-way graph."""
    from port.extensions.adjacency import (
        AdjacencyError,
        knn_adjacency,
        validate_adjacency,
    )

    adjacency = knn_adjacency(_square(12, 12), "moore")
    validate_adjacency(adjacency, 8, construction="knn")

    looped = adjacency.tolil()
    looped[5, 5] = 1.0
    with pytest.raises(AdjacencyError, match="self loop"):
        validate_adjacency(looped.tocsr(), 8, construction="knn")

    with pytest.raises(AdjacencyError, match="neighbours, not the k"):
        validate_adjacency(adjacency, 4, construction="knn")

    heavy = adjacency.copy()
    heavy.data[:] = 2.0
    with pytest.raises(AdjacencyError, match="unit weights"):
        validate_adjacency(heavy, 8, construction="knn")

    # NB eight random targets per spot: a directed graph almost nowhere reciprocated.
    rng = np.random.default_rng(0)
    n = 144
    targets = np.array(
        [rng.choice(np.delete(np.arange(n), i), 8, False) for i in range(n)]
    )
    one_way = sp.csr_matrix(
        (np.ones(n * 8), (np.repeat(np.arange(n), 8), targets.ravel())), shape=(n, n)
    )
    with pytest.raises(AdjacencyError, match="reciprocated"):
        validate_adjacency(one_way, 8, construction="knn")


@pytest.mark.patch
def test_the_swap_reproduces_cnasters_multislice_adjacency_on_square_grids() -> None:
    """The run's default: `cnaster`'s builder's return, bitwise, for square slices.

    Two slices of different shapes through `cnaster`'s
    `construct_multislice_lattice_adjacency` and through the swap, at the
    fixtures' isotropic units: the same adjacency entry for entry, and the
    same identity pooling matrix.
    """
    from cnaster.spatial import construct_multislice_lattice_adjacency
    from port.patch.spatial import lattice_multislice_adjacency

    first, second = _square(12, 10), _square(9, 14)
    coords = np.concatenate([first, second]).astype(float)
    sample_ids = np.repeat([0, 1], [len(first), len(second)])

    theirs = construct_multislice_lattice_adjacency(
        sample_ids, ["A", "B"], coords, None, 1, unit_xsquared=1, unit_ysquared=1
    )
    ours = lattice_multislice_adjacency(
        sample_ids, ["A", "B"], coords, None, 1, unit_xsquared=1, unit_ysquared=1
    )

    assert (ours.adjacency_mat != theirs.adjacency_mat).nnz == 0
    assert (ours.smooth_mat != theirs.smooth_mat).nnz == 0
