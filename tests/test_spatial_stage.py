"""`cnaster.spatial`, the stage that decides which spots are neighbours (#160).

Three functions `run_cnaster` calls inline and nothing here had refereed: the
multi-slice adjacency the Potts prior is written over, the equal partition the
BAF clones start from, and the refinement that splits them with the read
depth. 156 statements missed under the judged guard, of which these reach 118.

The fixture plants the lattice and the clone bands, so what a neighbourhood
and a partition **should** be is known before anything runs. That is what
makes the adjacency claims below judgements rather than descriptions: a test
that read the neighbour count back off the matrix would pass whatever the
construction did.
"""

from typing import Any

import numpy as np
import pytest

from tests.fixtures import CoreInferenceTruth, core_inference_truth
from tests.run_config import PlantedInstance

pytestmark = pytest.mark.preprocessing

LATTICE = (25, 40)
"""A thousand spots, as the other stage modules use."""

COORDINATION_NUMBER = 8
"""`construct_lattice_adjacency`'s default, and what it queries the tree for.

Restated because every claim below is about it: the construction is
`k`-nearest-neighbour with `k` fixed, not a distance rule, so this is the
out-degree of **every** spot rather than an upper bound on it.
"""

ISOTROPIC = {"unit_xsquared": 1, "unit_ysquared": 1}
"""What `tests/run_config.py` sets, against the shipped 9 and 3.

The shipped pair scales `x` and `y` differently, for Visium's hexagonal
packing. The fixture's lattice is square, so the isotropic pair is the one
under which "the eight nearest spots" and "the eight lattice neighbours" are
the same set and the comparison below has an answer.
"""


def _coordinates(truth: CoreInferenceTruth) -> np.ndarray:
    """The lattice the fixture planted, in `(row, column)` order.

    `tests/tmp_inputs.py` writes the lattice row as `x`, so a spot's index is
    `row * columns + column` and the geometry is recoverable from the index
    alone -- which is what lets a neighbourhood be predicted rather than read.
    """
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
def planted(planted_instance: PlantedInstance) -> CoreInferenceTruth:
    """The session's gate instance."""
    return planted_instance[0]


@pytest.fixture(scope="module")
def adjacency(planted: CoreInferenceTruth) -> tuple[np.ndarray, np.ndarray]:
    """`construct_multislice_lattice_adjacency` on one slice, run once.

    One slice, because that is what the fixture writes and what the block
    diagonal degenerates to. What the multi-slice path adds is the block
    structure and the inter-slice term, and neither can be judged against a
    fixture with one slice -- stated rather than skipped quietly.
    """
    from cnaster.spatial import construct_multislice_lattice_adjacency

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
    """**Every interior spot's neighbours are its eight lattice neighbours (#160).**

    The Potts prior is written over this matrix, so what counts as a neighbour
    decides what the spatial term rewards. The fixture plants a square lattice
    and lays its clones in bands over it, so the neighbourhood is known: the
    eight spots surrounding it.

    Asserted as set equality over all 874 interior spots rather than as a
    degree, because a `k`-nearest-neighbour query returns eight neighbours
    whatever the geometry -- the content of the claim is **which** eight.
    """
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
    """**A fixed coordination number makes the edge spots reach past the lattice.**

    `construct_lattice_adjacency` queries a KD-tree for `coordination_num + 1`
    neighbours and drops the self, so **every** spot gets eight however few it
    has. A corner has three lattice neighbours and is given eight, the extra
    five coming from the second ring.

    Two consequences, and both are modelling statements rather than bugs:

    * the neighbourhood is not the lattice's at the boundary, so the prior
      couples spots the geometry does not;
    * the matrix is **asymmetric** -- 284 of its 8,000 ordered edges hold one
      way only -- so a Potts energy summed over it weights those pairs once
      rather than twice, and which of the two spots carries the edge depends on
      the `k`-nearest-neighbour tie-breaking rather than on the model.

    Pinned rather than asserted away: the numbers are this lattice's, and what
    they say is that the neighbourhood is a construction choice nothing states.
    """
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
    """**`maxspots_pooling` is accepted and discarded.**

    `construct_lattice_adjacency` takes it, never reads it, and returns
    `scipy.sparse.identity` with `logger.warning("Assuming identity smooth
    mat.")` beside it. `run_cnaster` passes 1 with `# DEPRECATE` on the line,
    so the call agrees with the behaviour today -- but the parameter is in the
    signature, and a caller that asked for pooling would get none and be told
    only in a log line.

    The matrix is what every later stage smooths its BAF profiles with, so
    "no pooling" is a decision the configuration appears to expose and does
    not.
    """
    from cnaster.spatial import construct_lattice_adjacency

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
    """**The partition that minimizes size variance is the planted one (#160).**

    `best_equal_partition` draws `n_trials` rectangular partitions and keeps
    the one whose group sizes vary least. The fixture lays its clones in
    horizontal bands, so a partition cut along that axis **is** the planted
    labelling, and the strongest available claim is that it comes back exactly
    rather than approximately.

    `run_cnaster` calls it with `x_part, y_part = 3, 3` -- nine groups
    regardless of `n_clones` -- and 10,000 trials, which is 4.3 s on this
    fixture at 0.43 ms a trial. Fifty trials are used here: the claim is what
    the partition is, not how long the search takes, and the search's cost is
    a separate measurement.
    """
    from cnaster.spatial import best_equal_partition

    # NB equal bands (`normal_clone=False`): the claim is that the equal
    #    partition *is* the planted one, which holds only where the planted
    #    bands are equal. #298's normal clone takes 30 per cent of the rows.
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
    """**Every refined clone lies inside one planted clone (#160).**

    `initialize_rdr_clone_refininement` takes the BAF-only assignment and
    splits each of its clones into `n_clones_rdr` spatial pieces, which is
    where a run stops being able to recover from a bad BAF partition: a
    refinement that moved a spot across clones would be correcting the earlier
    stage, and this one cannot.

    Given the planted labels as the BAF assignment, the claim is therefore that
    the refinement **refines**: four clones out of two, each a subset of one
    planted clone, and each spot allowed exactly the two clones its own BAF
    clone was split into. `allowed_clones` is the mask the solver searches
    under, so a spot allowed a clone outside its block could be assigned across
    the boundary later even though the initialization was clean.

    The splits are uneven -- 455 against 65, and 415 against 65 -- which is the
    initializer's own floor at work: `initialize_rectangular_clones` accepts
    any split giving each clone more than 20 per cent of an equal share, so a
    seven-to-one split is within specification.
    """
    from cnaster.spatial import initialize_rdr_clone_refininement

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
    """The two configuration values the refinement reads, and nothing else.

    A stub rather than a written YAML: the function takes `config` and touches
    `hmrf.n_clones_rdr` and `hmm.gmm_random_state`, so a full configuration
    would say the test depends on fields it does not.
    """

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
