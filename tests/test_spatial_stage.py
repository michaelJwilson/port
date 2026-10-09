"""`cnaster.spatial` adjacency, equal partition and refinement, against the planted lattice and bands (#160)."""

from typing import Any

import numpy as np
import pytest
from cnaster.spatial import (
    best_equal_partition,
    construct_lattice_adjacency,
    construct_multislice_lattice_adjacency,
    initialize_rdr_clone_refininement,
)
from port.sim.truth import CoreInferenceTruth, core_inference_truth

pytestmark = pytest.mark.preprocessing

LATTICE = (25, 40)
"""A thousand spots, as the other stage modules use."""

COORDINATION_NUMBER = 8
"""`construct_lattice_adjacency`'s default `k`: every spot's out-degree."""

ISOTROPIC = {"unit_xsquared": 1, "unit_ysquared": 1}
"""`run_config.py`'s isotropic units, under which k-nearest equals the 8 lattice neighbours."""


def _coordinates(truth: CoreInferenceTruth) -> np.ndarray:
    """Return the planted lattice coordinates in `(row, column)` order."""
    rows, columns = truth.lattice

    return np.stack(
        [
            np.repeat(np.arange(rows), columns),
            np.tile(np.arange(columns), rows),
        ],
        axis=1,
    ).astype(float)


def _moore_neighbourhood(spot: int, rows: int, columns: int) -> set[int]:
    """The eight lattice neighbours of a spot, clipped at the boundary."""
    row, column = divmod(spot, columns)

    return {
        (row + dr) * columns + (column + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if (dr or dc) and 0 <= row + dr < rows and 0 <= column + dc < columns
    }


@pytest.fixture(scope="module")
def adjacency(planted: CoreInferenceTruth) -> tuple[np.ndarray, np.ndarray]:
    """Run `construct_multislice_lattice_adjacency` once on the one planted slice."""

    result = construct_multislice_lattice_adjacency(
        np.zeros(planted.n_spots, dtype=int),
        ["S1"],
        _coordinates(planted),
        None,
        1,
        **ISOTROPIC,
    )

    return (
        np.asarray(result.adjacency_mat.toarray()),
        np.asarray(result.smooth_mat.toarray()),
    )


@pytest.mark.end2end
def test_the_adjacency_is_the_planted_lattice_away_from_its_edges(
    planted: CoreInferenceTruth, adjacency: tuple[np.ndarray, np.ndarray]
) -> None:
    """Every interior spot's neighbours are its eight lattice neighbours (#160)."""
    matrix, _ = adjacency
    rows, columns = planted.lattice

    interior = [
        spot
        for spot in range(planted.n_spots)
        if 0 < spot // columns < rows - 1 and 0 < spot % columns < columns - 1
    ]

    assert len(interior) == (rows - 2) * (columns - 2)

    for spot in interior:
        assert set(np.flatnonzero(matrix[spot]).tolist()) == _moore_neighbourhood(
            spot, rows, columns
        ), f"spot {spot} is not adjacent to its lattice neighbours"


@pytest.mark.warning
def test_the_boundary_reaches_further_and_the_adjacency_is_not_symmetric(
    planted: CoreInferenceTruth, adjacency: tuple[np.ndarray, np.ndarray]
) -> None:
    """Every spot gets 8 neighbours, so edges reach past the lattice; 284 edges one-way."""
    matrix, _ = adjacency
    rows, columns = planted.lattice

    degrees = (matrix > 0).sum(axis=1)

    np.testing.assert_array_equal(degrees, COORDINATION_NUMBER)
    assert len(_moore_neighbourhood(0, rows, columns)) == 3

    one_way = int(((matrix > 0) != (matrix.T > 0)).sum())

    assert one_way == 284, (
        f"{one_way} ordered edges hold one way only, against the 284 this "
        "lattice gave when the asymmetry was first measured"
    )


@pytest.mark.warning
def test_the_pooling_matrix_is_the_identity_whatever_is_asked_for(
    planted: CoreInferenceTruth, adjacency: tuple[np.ndarray, np.ndarray]
) -> None:
    """`maxspots_pooling` is discarded: the smoothing matrix is the identity."""

    _, smooth = adjacency

    np.testing.assert_array_equal(smooth, np.eye(planted.n_spots))

    pooled, _ = construct_lattice_adjacency(
        _coordinates(planted), maxspots_pooling=7, **ISOTROPIC
    )

    np.testing.assert_array_equal(np.asarray(pooled.toarray()), np.eye(planted.n_spots))


@pytest.mark.end2end
def test_the_equal_partition_recovers_the_planted_bands(
    planted: CoreInferenceTruth,
) -> None:
    """`best_equal_partition` returns the planted bands exactly (#160)."""

    # NB equal bands: #298's normal clone would make the planted bands unequal
    planted = core_inference_truth(
        n_clones=2,
        n_states=3,
        lattice=LATTICE,
        n_obs=40,
        n_segments=3,
        seed=11,
        normal_clone=False,
    )

    index, _ = best_equal_partition(
        _coordinates(planted), planted.n_clones, 1, n_trials=50
    )

    recovered = np.empty(planted.n_spots, dtype=np.int64)
    for clone, spots in enumerate(index):
        recovered[spots] = clone

    agreement = max(
        float(np.mean(recovered == planted.labels)),
        float(np.mean(recovered == planted.n_clones - 1 - planted.labels)),
    )

    assert agreement == 1.0
    assert sum(len(spots) for spots in index) == planted.n_spots


@pytest.mark.end2end
def test_the_read_depth_refinement_splits_clones_without_mixing_them(
    planted: CoreInferenceTruth,
) -> None:
    """Each refined clone lies inside one planted clone, allowed its own two (#160)."""

    config = _refinement_config(n_clones_rdr=2)

    assignment, allowed, total = initialize_rdr_clone_refininement(
        merged_baf_assignment=planted.labels,
        coords=_coordinates(planted),
        single_total_bb_RD=planted.total_bb_RD,
        n_obs=planted.n_obs,
        config=config,
    )

    assert total == planted.n_clones * 2
    assert allowed.shape == (planted.n_spots, total)

    for clone in range(total):
        planted_of = set(planted.labels[assignment == clone].tolist())

        assert len(planted_of) == 1, (
            f"refined clone {clone} draws from planted clones {planted_of}"
        )

    np.testing.assert_array_equal(allowed.sum(axis=1), 2)

    for spot in range(planted.n_spots):
        block = planted.labels[spot] * 2

        np.testing.assert_array_equal(np.flatnonzero(allowed[spot]), [block, block + 1])


def _refinement_config(*, n_clones_rdr: int) -> Any:
    """Return a stub with the two configuration values the refinement reads."""

    class Hmrf:
        pass

    class Hmm:
        pass

    class Config:
        pass

    hmrf, hmm, config = Hmrf(), Hmm(), Config()
    hmrf.n_clones_rdr = n_clones_rdr  # type: ignore[attr-defined]
    hmm.gmm_random_state = 0  # type: ignore[attr-defined]
    config.hmrf = hmrf  # type: ignore[attr-defined]
    config.hmm = hmm  # type: ignore[attr-defined]

    return config
