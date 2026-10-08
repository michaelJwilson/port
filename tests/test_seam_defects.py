"""Defects #45 and #81 pinned before the seam moves (#206); #58 is
`test_external_field.py`'s.

`bug` and `warning` markers: each fails when `cnaster` fixes it.
"""

import inspect

import numpy as np
import pytest

from tests.adapters import lattice_adjacency

FLOOR = 200
"""`icm_sweep_deque`'s `min_clone_spots` default, and the subject of #81."""

SEED = 4_211


def _bands(side: int, sizes: tuple[int, ...]) -> np.ndarray:
    """A labelling in contiguous row bands of the given spot counts."""
    labels = np.zeros(side * side, dtype=np.int64)
    start = 0

    for clone, size in enumerate(sizes):
        labels[start : start + size] = clone
        start += size

    return labels


def _field(labels: np.ndarray, n_clones: int, margin: float = 40.0) -> np.ndarray:
    """A unary cost preferring the given labelling by `margin`, built rather than fitted."""
    field = np.full((labels.size, n_clones), -margin)
    field[np.arange(labels.size), labels] = 0.0

    return field


@pytest.mark.bug
def test_the_sweep_writes_its_callers_labelling_in_place() -> None:
    """`icm_sweep_deque` rewrites the caller's labelling in place (#45)."""
    from cnaster.icm import icm_sweep_deque

    side = 12
    labels = _bands(side, (72, 72))
    graph = lattice_adjacency((side, side))

    passed = labels.copy()
    scrambled = (passed + 1) % 2

    before = scrambled.copy()

    # NPY002 is the finding: the sweep shuffles with the legacy global RNG (#45).
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
    """A clone under 200 spots is dissolved though the field prefers it by 40 nats per spot
    (#81).
    """
    from cnaster.icm import icm_sweep_deque

    side = 24
    small = 100
    labels = _bands(side, (side * side - small, small))
    graph = lattice_adjacency((side, side))

    assert np.bincount(labels).min() == small < FLOOR

    solved = labels.copy()

    # NPY002 is the finding: the sweep shuffles with the legacy global RNG (#45).
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
    """One spot above the floor the same clone survives (#81 control)."""
    from cnaster.icm import icm_sweep_deque

    side = 24
    small = FLOOR + 1
    labels = _bands(side, (side * side - small, small))
    graph = lattice_adjacency((side, side))

    solved = labels.copy()

    # NPY002 is the finding: the sweep shuffles with the legacy global RNG (#45).
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
    """No pipeline argument or configuration key reaches `min_clone_spots` (#81)."""
    from cnaster import hmrf, icm

    signature = inspect.signature(icm.icm_sweep_deque)

    assert signature.parameters["min_clone_spots"].default == FLOOR

    source = inspect.getsource(hmrf.pipeline_clone_assignment)
    live = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )

    assert "min_clone_spots" not in live, "the call site now sets the floor"
    assert "min_spots_per_clone" not in live, "the config key now reaches the solver"
