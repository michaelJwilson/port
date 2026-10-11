"""Per-function rows for the spatial layer: grid partitions, rectangular initial clones, the spot graph.

Inputs: the staged coordinates and sample ids (`05_baf/construct_df_clone_label`'s
and `get_sample_list`'s), the RDR initialization's replayed input; else a
synthetic layout small enough to enumerate.
"""

from __future__ import annotations

import io
import itertools
import logging
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp
from cnamaste.spatial import (
    admits_assignment,
    banded,
    best_equal_partition,
    construct_lattice_adjacency,
    construct_multislice_lattice_adjacency,
    initialize_clones,
    initialize_rdr_clone_refininement,
    initialize_rectangular_clones,
    log_sparse_matrix_stats,
    rectangle_partition,
)

from audit.fn import Replay, Row, run, table, unchanged


def coords_of(ctx: Any) -> dict[str, Any]:
    return {"coords": np.asarray(ctx.sim.stored("05_baf/construct_df_clone_label/in/args/1")),
            "sample_ids": np.asarray(ctx.sim.stored("00_inputs/get_sample_list/out/1")),
            "initial": ctx.sim.stored("03_phasing/initialize_clones/out")}


def scattered(n: int = 60, seed: int = 8) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0, 10, size=(n, 2))


# --- oracle --------------------------------------------------------------------


def _rectangles_are_quadrants(_: Any) -> None:
    coords = np.array([(x, y) for x in range(4) for y in range(4)], dtype=float)
    index, labels = rectangle_partition(coords, 2, 2)
    want = (coords[:, 0] >= 2).astype(int) * 2 + (coords[:, 1] >= 2).astype(int)  # NB right=True: the edge at 1.5 (x in [0, 3])
    assert np.array_equal(labels, want) and all(np.array_equal(i, np.where(want == c)[0]) for c, i in enumerate(index))


def _admits_equals_brute(_: Any) -> None:
    rng = np.random.default_rng(13)
    for _ in range(40):
        sizes = rng.integers(0, 12, size=rng.integers(1, 6))
        k, floor = int(rng.integers(1, 4)), float(rng.uniform(0, 10))
        brute = any(all(sum(s for s, c in zip(sizes, assign, strict=True) if c == j) > floor for j in range(k))
                    for assign in itertools.product(range(k), repeat=len(sizes)))
        assert admits_assignment(sizes, k, floor) == brute, (sizes.tolist(), k, floor)


def _knn_eight_nearest(_: Any) -> None:
    coords = scattered()
    adjacency = construct_lattice_adjacency(coords, unit_xsquared=4, unit_ysquared=1)
    scaled = coords * np.array([2.0, 1.0])
    d = np.sum((scaled[:, None, :] - scaled[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(d, np.inf)
    want = np.sort(np.argsort(d, axis=1)[:, :8], axis=1)
    a = adjacency.tocsr()
    found = np.sort(a.indices.reshape(-1, 8), axis=1)
    assert np.array_equal(found, want), "each spot's 8 nearest neighbours in the scaled metric"


def _multislice_block_diagonal(_: Any) -> None:
    coords = np.vstack([scattered(30, 1), scattered(20, 2)])
    ids = np.repeat([0, 1], [30, 20])
    adjacency = unchanged(construct_multislice_lattice_adjacency, ids, ["a", "b"], coords, None, unit_xsquared=1, unit_ysquared=1)
    a = adjacency.toarray()
    assert not a[:30, 30:].any() and not a[30:, :30].any(), "no edge crosses slices"
    for idx in (slice(0, 30), slice(30, 50)):
        one = construct_lattice_adjacency(coords[idx], unit_xsquared=1, unit_ysquared=1)
        assert np.array_equal(a[idx, idx], one.toarray()), "each slice's block is its own lattice adjacency"


def _equal_partition_least_variance(_: Any) -> None:
    grid = np.array([(x, y) for x in range(6) for y in range(6)], dtype=float)
    for trials in (50, 200):
        index, labels = best_equal_partition(grid, 2, 2, n_trials=trials)
        sizes = [[len(i) for i in rectangle_partition(grid, 2, 2, random_state=t)[0]] for t in range(trials)]
        first = int(np.argmin([np.var(z) for z in sizes]))  # NB exhaustive over the trials; the first least variance wins
        want_index, want_labels = rectangle_partition(grid, 2, 2, random_state=first)
        assert np.array_equal(labels, want_labels) and all(np.array_equal(a, b) for a, b in zip(index, want_index, strict=True)), trials
    assert first == 78 and [len(i) for i in index] == [9, 9, 9, 9], "200 trials reach the global optimum: four 9-spot parts, variance 0"
    quadrant = (grid[:, 0] >= 3).astype(int) * 2 + (grid[:, 1] >= 3).astype(int)
    assert np.array_equal(labels, quadrant), "the 9-spot parts of a 6x6 grid are its quadrants"
    index, _ = best_equal_partition(np.array([(x, y) for x in range(6) for y in range(4)], dtype=float), 3, 2, n_trials=50)
    assert [len(i) for i in index] == [4] * 6, "6x4 into 3x2: six 4-spot parts"


def _bands_by_hand(_: Any) -> None:
    coords = np.array([(5, 0), (1, 2), (1, 1), (0, 3), (0, 3), (9, 1)], dtype=float)
    index, labels = banded(coords, 2)
    # NB 4 distinct values on each axis: x, the first; order by x, then y, then index: 3, 4, 2, 1, 0, 5
    assert labels.tolist() == [1, 1, 0, 0, 0, 1] and [i.tolist() for i in index] == [[2, 3, 4], [0, 1, 5]]
    line = np.column_stack([np.zeros(10), np.arange(10)[::-1]])  # NB y has the distinct values
    assert banded(line, 3)[1].tolist() == [2, 2, 2, 1, 1, 1, 0, 0, 0, 0], "bands of 4, 3, 3 along y: i * 3 // 10"


ORACLE: list[Row] = table(
    "oracle",
    ("spatial:rectangle_partition", "synthetic: a 4x4 grid", lambda c: None, _rectangles_are_quadrants, "2x2 parts are the four quadrants"),
    ("spatial:admits_assignment", "synthetic: 40 random block sets", lambda c: None, _admits_equals_brute, "equals the brute force over every block-to-clone map"),
    ("spatial:best_equal_partition", "synthetic: 6x6 and 6x4 grids", lambda c: None, _equal_partition_least_variance,
     "the first least-variance partition over the trials (exhaustive); at 200 trials the 6x6 grid's quadrants, variance 0, the global optimum"),
    ("spatial:banded", "synthetic: 6 spots with ties, 10 on a line", lambda c: None, _bands_by_hand, "hand-computed bands: the axis with more distinct values, ties by the other axis then index"),
    ("spatial:construct_lattice_adjacency", "synthetic: 60 scattered spots", lambda c: None, _knn_eight_nearest, "each spot's out-edges are its 8 nearest neighbours in the scaled metric"),
    ("spatial:construct_multislice_lattice_adjacency", "synthetic: two slices, 30 and 20 spots", lambda c: None, _multislice_block_diagonal, "block diagonal of the per-slice adjacencies; no input mutation"),
)


# --- invariants ---------------------------------------------------------------------


def _initial_disjoint_rectangles(d: dict[str, Any]) -> None:
    index, coords = d["initial"], d["coords"]
    every = np.sort(np.concatenate(index))
    assert np.array_equal(every, np.arange(len(coords)))
    boxes = [(coords[i, 0].min(), coords[i, 0].max(), coords[i, 1].min(), coords[i, 1].max()) for i in index if len(i)]
    for (a, b) in itertools.combinations(boxes, 2):
        assert a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2], "two initial clones' bounding boxes overlap"


def _rectangular_above_floor(d: dict[str, Any]) -> None:
    coords = d["coords"][d["coords"][:, 0] < np.median(d["coords"][:, 0])]
    for k in (1, 2, 3, 4):
        index, labels = initialize_rectangular_clones(coords, k, random_state=k)
        assert np.array_equal(np.sort(np.concatenate(index)), np.arange(len(coords))) and np.array_equal(np.unique(labels), np.arange(k))
        assert k == 1 or min(len(i) for i in index) > 0.2 * len(coords) / k, "every clone above the 20% floor"
        again = initialize_rectangular_clones(coords, k, random_state=k)[1]
        assert np.array_equal(labels, again), "seeded: the same labels twice"


def _bands_equal_above_floor(_: Any) -> None:
    rng = np.random.default_rng(5)
    for n, k in ((7, 7), (30, 4), (61, 5)):
        coords = rng.integers(0, 6, size=(n, 2)).astype(float)
        index, labels = unchanged(banded, coords, k)
        sizes = np.bincount(labels, minlength=k)
        assert np.array_equal(np.sort(np.concatenate(index)), np.arange(n)) and sizes.max() - sizes.min() <= 1, "an equal-count partition"
        assert sizes.min() > 0.2 * n / k, "every band passes initialize_rectangular_clones' 20% floor"
        axis = int(np.argmax([np.unique(coords[:, a]).size for a in (0, 1)]))
        key = [(coords[i, axis], coords[i, 1 - axis], i) for i in range(n)]
        for c in range(k - 1):
            assert max(key[i] for i in index[c]) < min(key[i] for i in index[c + 1]), "bands are contiguous in the sort order"


def _sparse_stats_logged(_: Any) -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger("cnamaste.spatial")
    logger.addHandler(handler)
    try:
        log_sparse_matrix_stats(sp.csr_matrix(np.array([[0, 2, 2], [1, 0, 0], [3, 0, 0]], dtype=float)), "m")
    finally:
        logger.removeHandler(handler)
    text = stream.getvalue()
    assert "Median edge weight: 2.00" in text and "1 neighbors\t66.7%" in text and "2 neighbors\t33.3%" in text


def _rdr_init(ctx: Any) -> dict[str, Any]:
    given = ctx("08_rdr/initialize_rdr_clone_refininement/in")
    return {"kwargs": given["kwargs"], "out": ctx.sim.stored("08_rdr/initialize_rdr_clone_refininement/out")}


def _rdr_refine_matches_run(d: dict[str, Any]) -> None:
    labels, allowed, total = unchanged(initialize_rdr_clone_refininement, **d["kwargs"])
    want_labels, want_allowed, want_total = d["out"]
    assert np.array_equal(labels, want_labels) and np.array_equal(allowed, want_allowed) and total == want_total
    baf = np.asarray(d["kwargs"]["merged_baf_assignment"])
    for c in np.unique(baf):
        assert (np.unique(allowed[baf == c], axis=0).shape[0] == 1), "spots of one BAF clone share their allowed RDR clones"
    assert np.all(allowed[np.arange(len(labels)), labels]), "every spot starts in a clone it is allowed"


INVARIANT: list[Row] = table(
    "invariant",
    ("spatial:banded", "synthetic: 7, 30 and 61 spots on a 6x6 lattice", lambda c: None, _bands_equal_above_floor,
     "sizes within one of n/k, each above 20% of n/k, contiguous in the (axis, other axis, index) order; no input mutation"),
    ("spatial:initialize_clones", "03_phasing/initialize_clones out, staged coordinates", coords_of, _initial_disjoint_rectangles, "the initial clones partition the spots into disjoint rectangles"),
    ("spatial:initialize_rectangular_clones", "staged coordinates, the left half", coords_of, _rectangular_above_floor, "a partition into k non-empty clones above 20% of n/k, reproducible by seed"),
    ("spatial:log_sparse_matrix_stats", "synthetic: a 3x3 sparse matrix", lambda c: None, _sparse_stats_logged, "logs the median weight and each degree's fraction"),
    ("spatial:initialize_rdr_clone_refininement", Replay("08_rdr/initialize_rdr_clone_refininement in"), _rdr_init, _rdr_refine_matches_run,
     "recomputed from its recorded input it returns the run's split; every spot starts in an allowed clone; no input mutation"),
)


# --- captured -------------------------------------------------------------------------


def _initialize_matches_run(d: dict[str, Any]) -> None:
    found = initialize_clones(d["coords"], d["sample_ids"], 2, 2)
    assert len(found) == len(d["initial"]) and all(np.array_equal(a, b) for a, b in zip(found, d["initial"], strict=True))


CAPTURED: list[Row] = table(
    "captured",
    ("spatial:initialize_clones", "staged coordinates and sample ids, npart_phasing 2", coords_of, _initialize_matches_run, "2x2 per slice: the run's initial clones"),
    ("spatial:rectangle_partition", "staged coordinates and sample ids, npart_phasing 2", coords_of, _initialize_matches_run, "the partition initialize_clones takes per slice is the run's"),
)


@pytest.mark.parametrize("row", ORACLE, ids=[r.id for r in ORACLE])
def test_oracle_agrees(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """Quadrants, brute-force feasibility, brute-force nearest neighbours."""
    run(row, ctx, request)


@pytest.mark.parametrize("row", INVARIANT, ids=[r.id for r in INVARIANT])
def test_invariants_hold(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """Partitions, floors, reproducibility, no input mutation."""
    run(row, ctx, request)


@pytest.mark.parametrize("row", CAPTURED, ids=[r.id for r in CAPTURED])
def test_matches_run(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """The run's partition, recomputed from the staged coordinates."""
    run(row, ctx, request)
