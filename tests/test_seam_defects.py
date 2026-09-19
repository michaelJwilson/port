"""The three defects #206 pins before the seam moves: #45, #58, #81.

**A refactor that does not pin these first carries them into the rewrite**,
and then there is no way to tell a defect that was always there from one the
refactor introduced. #58 is already pinned by `tests/test_external_field.py`;
this module is the other two, and the half of #45 that was not.

| defect | what is pinned here |
| --- | --- |
| #45 | the sweep writes the caller's labelling in place |
| #81 | a clone under 200 spots is dissolved into random neighbours, and nothing in the pipeline or the configuration can move the floor |

`tests/test_icm_interface.py` pins the other half of #45 -- that two runs of
one problem give two answers, because the queue is shuffled with the legacy
global RNG.

**These are `bug` and `warning` markers, so none of them counts as
validation.** They say what is wrong rather than what works, which is the
whole of their job: each is written to fail when `cnaster` fixes it.
"""

import inspect

import numpy as np
import pytest
from scipy.sparse import csr_matrix

FLOOR = 200
"""`icm_sweep_deque`'s `min_clone_spots` default, and the subject of #81."""

SEED = 4_211


def _lattice(side: int) -> csr_matrix:
    """A four-neighbour lattice, which is the graph the pipeline builds."""
    n_spots = side * side
    rows, cols, data = [], [], []

    for spot in range(n_spots):
        row, column = divmod(spot, side)

        for delta_row, delta_column in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbour_row, neighbour_column = row + delta_row, column + delta_column

            if 0 <= neighbour_row < side and 0 <= neighbour_column < side:
                rows.append(spot)
                cols.append(neighbour_row * side + neighbour_column)
                data.append(1.0)

    return csr_matrix((data, (rows, cols)), shape=(n_spots, n_spots))


def _bands(side: int, sizes: tuple[int, ...]) -> np.ndarray:
    """A labelling in contiguous row bands of the given spot counts."""
    labels = np.zeros(side * side, dtype=np.int64)
    start = 0

    for clone, size in enumerate(sizes):
        labels[start : start + size] = clone
        start += size

    return labels


def _field(labels: np.ndarray, n_clones: int, margin: float = 40.0) -> np.ndarray:
    """A unary cost that prefers the labelling it is given, by `margin`.

    Built rather than fitted, as `tests/test_cold_inference.py` argues: what
    is being asked is what the solver does to a labelling it is already
    pointed at, and a field from a fit would confuse that with the fit.
    """
    field = np.full((labels.size, n_clones), -margin)
    field[np.arange(labels.size), labels] = 0.0

    return field


@pytest.mark.bug
def test_the_sweep_writes_its_callers_labelling_in_place() -> None:
    """#45's second half: the argument is the output.

    `icm_sweep_deque` returns `(niter, cost)` and communicates the labelling
    by mutating `new_assignment`. A caller that kept a reference to the array
    it passed -- as `pipeline_clone_assignment` does, through
    `copy.copy(prev_assignment)` -- has its own copy rewritten under it.

    `copy.copy` is what makes that survivable today, and it is one line away
    from not being: the defect is that the contract is undocumented, not that
    the current call site is wrong.
    """
    from cnaster.icm import icm_sweep_deque

    side = 12
    labels = _bands(side, (72, 72))
    graph = _lattice(side)

    passed = labels.copy()
    scrambled = (passed + 1) % 2

    before = scrambled.copy()

    # NPY002 is the finding, not the violation: the sweep shuffles its queue
    # with the legacy global RNG, so a `Generator` cannot reach it (#45).
    np.random.seed(SEED)  # noqa: NPY002

    icm_sweep_deque(
        single_llf=_field(labels, 2),
        adj_indptr=graph.indptr,
        adj_indices=graph.indices,
        adj_weights=graph.data,
        new_assignment=scrambled,
        spatial_weight=1.0,
        posterior=None,
        min_clone_spots=0,
    )

    assert not np.array_equal(scrambled, before), (
        "the sweep returned without writing its caller's array"
    )
    assert np.array_equal(scrambled, passed), (
        "the sweep wrote the caller's array, but not to the planted labelling"
    )


@pytest.mark.bug
def test_a_clone_under_the_floor_is_dissolved_into_random_neighbours() -> None:
    """#81: below 200 spots a clone stops existing, whatever the field says.

    The field here prefers the planted labelling by 40 nats per spot, and the
    small clone is the labelling the data supports. It is still scattered --
    `np.random.choice` over the clones that clear the floor -- because the
    enforcement runs after the sweep and does not consult the cost it is
    undoing.

    The floor is the reason `tests/fixtures.py` cannot plant ten clones below
    2,000 spots, which is a constraint on every fixture in this repository
    rather than a property of any one of them.
    """
    from cnaster.icm import icm_sweep_deque

    side = 24
    small = 100
    labels = _bands(side, (side * side - small, small))
    graph = _lattice(side)

    assert np.bincount(labels).min() == small < FLOOR

    solved = labels.copy()

    # NPY002 is the finding, not the violation (#45).
    np.random.seed(SEED)  # noqa: NPY002

    icm_sweep_deque(
        single_llf=_field(labels, 2),
        adj_indptr=graph.indptr,
        adj_indices=graph.indices,
        adj_weights=graph.data,
        new_assignment=solved,
        spatial_weight=1.0,
        posterior=None,
    )

    assert (solved == 1).sum() == 0, (
        f"{(solved == 1).sum()} spots survived in the clone under the floor"
    )


@pytest.mark.warning
def test_the_same_clone_survives_once_it_clears_the_floor() -> None:
    """The other side of #81, so the pin above is about the floor and not the field.

    Same graph, same field, same solver, one more spot than the floor asks
    for. The clone is kept. Without this the test above would pass for a
    solver that simply never recovers a small clone.
    """
    from cnaster.icm import icm_sweep_deque

    side = 24
    small = FLOOR + 1
    labels = _bands(side, (side * side - small, small))
    graph = _lattice(side)

    solved = labels.copy()

    # NPY002 is the finding, not the violation (#45).
    np.random.seed(SEED)  # noqa: NPY002

    icm_sweep_deque(
        single_llf=_field(labels, 2),
        adj_indptr=graph.indptr,
        adj_indices=graph.indices,
        adj_weights=graph.data,
        new_assignment=solved,
        spatial_weight=1.0,
        posterior=None,
    )

    assert (solved == 1).sum() >= FLOOR


@pytest.mark.bug
def test_the_floor_is_not_reachable_from_the_pipeline() -> None:
    """#81's second half: the threshold exists and nothing can set it.

    `min_clone_spots` is a default on `icm_sweep_deque` and
    `pipeline_clone_assignment` does not pass it, so no configuration a user
    writes reaches it. The key that *looks* like it should --
    `hmrf.min_spots_per_clone` -- goes to `merge_by_minspots`, a different
    stage with a different threshold, which is why the collision is worth
    pinning rather than describing.
    """
    from cnaster import hmrf, icm

    signature = inspect.signature(icm.icm_sweep_deque)

    assert signature.parameters["min_clone_spots"].default == FLOOR

    source = inspect.getsource(hmrf.pipeline_clone_assignment)
    live = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )

    assert "min_clone_spots" not in live, "the call site now sets the floor"
    assert "min_spots_per_clone" not in live, "the config key now reaches the solver"
