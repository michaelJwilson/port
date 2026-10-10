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


def _rectangles(_: Any) -> None:
    coords = np.array([(x, y) for x in range(4) for y in range(4)], dtype=float)
    index, labels = rectangle_partition(coords, 2, 2)
    want = (coords[:, 0] >= 2).astype(int) * 2 + (coords[:, 1] >= 2).astype(int)  # NB right=True: the edge at 1.5 (x in [0, 3])
    assert np.array_equal(labels, want) and all(np.array_equal(i, np.where(want == c)[0]) for c, i in enumerate(index))


def _admits(_: Any) -> None:
    rng = np.random.default_rng(13)
    for _ in range(40):
        sizes = rng.integers(0, 12, size=rng.integers(1, 6))
        k, floor = int(rng.integers(1, 4)), float(rng.uniform(0, 10))
        brute = any(all(sum(s for s, c in zip(sizes, assign, strict=True) if c == j) > floor for j in range(k))
                    for assign in itertools.product(range(k), repeat=len(sizes)))
        assert admits_assignment(sizes, k, floor) == brute, (sizes.tolist(), k, floor)


def _knn(_: Any) -> None:
    coords = scattered()
    adjacency = construct_lattice_adjacency(coords, unit_xsquared=4, unit_ysquared=1)
    scaled = coords * np.array([2.0, 1.0])
    d = np.sum((scaled[:, None, :] - scaled[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(d, np.inf)
    knn = np.zeros(d.shape, dtype=bool)
    np.put_along_axis(knn, np.argsort(d, axis=1)[:, :8], True, axis=1)
    a = adjacency.toarray()
    assert np.array_equal(a != 0, knn | knn.T), "each spot's 8 nearest neighbours in the scaled metric, and each spot it is among"
    assert np.all(a[a != 0] == 1)


def _multislice(_: Any) -> None:
    coords = np.vstack([scattered(30, 1), scattered(20, 2)])
    ids = np.repeat([0, 1], [30, 20])
    adjacency = unchanged(construct_multislice_lattice_adjacency, ids, ["a", "b"], coords, None, unit_xsquared=1, unit_ysquared=1)
    a = adjacency.toarray()
    assert not a[:30, 30:].any() and not a[30:, :30].any(), "no edge crosses slices"
    for idx in (slice(0, 30), slice(30, 50)):
        one = construct_lattice_adjacency(coords[idx], unit_xsquared=1, unit_ysquared=1)
        assert np.array_equal(a[idx, idx], one.toarray()), "each slice's block is its own lattice adjacency"


ORACLE: list[Row] = table(
    "oracle",
    ("spatial:rectangle_partition", "synthetic: a 4x4 grid", lambda c: None, _rectangles, "2x2 parts are the four quadrants"),
    ("spatial:admits_assignment", "synthetic: 40 random block sets", lambda c: None, _admits, "equals the brute force over every block-to-clone map"),
    ("spatial:construct_lattice_adjacency", "synthetic: 60 scattered spots", lambda c: None, _knn, "the union of each spot's 8 nearest neighbours in the scaled metric, symmetric, unit weights"),
    ("spatial:construct_multislice_lattice_adjacency", "synthetic: two slices, 30 and 20 spots", lambda c: None, _multislice, "block diagonal of the per-slice adjacencies; no input mutation"),
)


# --- invariants ---------------------------------------------------------------------


def _initial_partition(d: dict[str, Any]) -> None:
    index, coords = d["initial"], d["coords"]
    every = np.sort(np.concatenate(index))
    assert np.array_equal(every, np.arange(len(coords)))
    boxes = [(coords[i, 0].min(), coords[i, 0].max(), coords[i, 1].min(), coords[i, 1].max()) for i in index if len(i)]
    for (a, b) in itertools.combinations(boxes, 2):
        assert a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2], "two initial clones' bounding boxes overlap"


def _rectangular(d: dict[str, Any]) -> None:
    coords = d["coords"][d["coords"][:, 0] < np.median(d["coords"][:, 0])]
    for k in (1, 2, 3, 4):
        index, labels = initialize_rectangular_clones(coords, k, random_state=k)
        assert np.array_equal(np.sort(np.concatenate(index)), np.arange(len(coords))) and np.array_equal(np.unique(labels), np.arange(k))
        assert k == 1 or min(len(i) for i in index) > 0.2 * len(coords) / k, "every clone above the 20% floor"
        again = initialize_rectangular_clones(coords, k, random_state=k)[1]
        assert np.array_equal(labels, again), "seeded: the same labels twice"


def _sparse_stats(_: Any) -> None:
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


def _rdr_refine(d: dict[str, Any]) -> None:
    labels, allowed, total = unchanged(initialize_rdr_clone_refininement, **d["kwargs"])
    want_labels, want_allowed, want_total = d["out"]
    assert np.array_equal(labels, want_labels) and np.array_equal(allowed, want_allowed) and total == want_total
    baf = np.asarray(d["kwargs"]["merged_baf_assignment"])
    for c in np.unique(baf):
        assert (np.unique(allowed[baf == c], axis=0).shape[0] == 1), "spots of one BAF clone share their allowed RDR clones"
    assert np.all(allowed[np.arange(len(labels)), labels]), "every spot starts in a clone it is allowed"


INVARIANT: list[Row] = table(
    "invariant",
    ("spatial:initialize_clones", "03_phasing/initialize_clones out, staged coordinates", coords_of, _initial_partition, "the initial clones partition the spots into disjoint rectangles"),
    ("spatial:initialize_rectangular_clones", "staged coordinates, the left half", coords_of, _rectangular, "a partition into k non-empty clones above 20% of n/k, reproducible by seed"),
    ("spatial:log_sparse_matrix_stats", "synthetic: a 3x3 sparse matrix", lambda c: None, _sparse_stats, "logs the median weight and each degree's fraction"),
    ("spatial:initialize_rdr_clone_refininement", Replay("08_rdr/initialize_rdr_clone_refininement in"), _rdr_init, _rdr_refine,
     "recomputed from its recorded input it returns the run's split; every spot starts in an allowed clone; no input mutation"),
)


# --- captured -------------------------------------------------------------------------


def _initialize(d: dict[str, Any]) -> None:
    found = initialize_clones(d["coords"], d["sample_ids"], 2, 2)
    assert len(found) == len(d["initial"]) and all(np.array_equal(a, b) for a, b in zip(found, d["initial"], strict=True))


CAPTURED: list[Row] = table(
    "captured",
    ("spatial:initialize_clones", "staged coordinates and sample ids, npart_phasing 2", coords_of, _initialize, "2x2 per slice: the run's initial clones"),
    ("spatial:rectangle_partition", "staged coordinates and sample ids, npart_phasing 2", coords_of, _initialize, "the partition initialize_clones takes per slice is the run's"),
)


@pytest.mark.parametrize("row", ORACLE, ids=[r.id for r in ORACLE])
def test_oracle(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """Quadrants, brute-force feasibility, brute-force nearest neighbours."""
    run(row, ctx, request)


@pytest.mark.parametrize("row", INVARIANT, ids=[r.id for r in INVARIANT])
def test_invariant(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """Partitions, floors, reproducibility, no input mutation."""
    run(row, ctx, request)


@pytest.mark.parametrize("row", CAPTURED, ids=[r.id for r in CAPTURED])
def test_captured(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """The run's partition, recomputed from the staged coordinates."""
    run(row, ctx, request)
