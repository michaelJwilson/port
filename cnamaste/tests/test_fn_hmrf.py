"""Per-function rows for the clone field: the spot field, ICM, merges, pseudobulks, clone stacks, re-indexing.

Inputs: the fitted runs' stored results and inputs (`audit.fn.fitted`), restricted
to a 300-spot spatial window where a brute force is per spot; else a small
synthetic lattice with a closed-form energy. Every lattice here is symmetric,
so a directed edge pair is one undirected edge of weight one.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.stats
from cnamaste.cna_hmrf_result import CloneAssignment, CnaHMRFResult, HMMParams, HMMProfile
from cnamaste.hmm_nophasing import hmm_nophasing
from cnamaste.hmrf import compute_loglike_spot_assignment, merge_by_minspots, pipeline_clone_assignment, reindex_clones, run_core_inference
from cnamaste.hmrf_utils import cast_csr, clone_stack_obs, get_clone_assignment, get_clone_indices
from cnamaste.icm import icm_sweep_deque, merge_assignment, unpack_adjacency
from cnamaste.pseudobulk import merge_pseudobulk_by_index_mix

from audit.fn import Replay, Row, fitted, run, table, unchanged

BAF, RDR = "05_baf", "08_rdr"
WINDOW = Replay("05_baf/run_core_inference in and out, the 300 spots nearest spot 0")


def lattice_graph(nx: int, ny: int) -> sp.csr_matrix:
    """The 4-neighbour grid, symmetric, unit weights."""
    n = nx * ny
    rows, cols = [], []
    for x in range(nx):
        for y in range(ny):
            i = x * ny + y
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if 0 <= x + dx < nx and 0 <= y + dy < ny:
                    rows.append(i)
                    cols.append((x + dx) * ny + y + dy)
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def potts(llf: np.ndarray, adjacency: sp.csr_matrix, labels: np.ndarray, weight: float) -> float:
    """sum_i llf[i, l_i] + weight * sum over stored edges with equal labels of w / 2: an edge stored both ways counts w,
    one stored one way w / 2 (`icm.calc_assignment_cost`)."""
    a = adjacency.tocoo()
    return float(llf[np.arange(len(labels)), labels].sum() + weight * np.sum(a.data * (labels[a.row] == labels[a.col])) / 2.0)


def field_case(seed: int = 5) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    adjacency = lattice_graph(8, 8)
    llf = rng.normal(scale=2.0, size=(64, 3))
    return {"llf": llf, "adjacency": adjacency, "labels": rng.integers(0, 3, size=64), "weight": 0.7}


def one_way_case(seed: int = 6) -> dict[str, Any]:
    """`field_case`'s lattice with a tenth of its stored directions dropped: an asymmetric graph, as kNN gives."""
    f = field_case(seed)
    a = f["adjacency"].tocoo()
    keep = np.random.default_rng(seed).random(a.nnz) >= 0.1
    f["adjacency"] = sp.csr_matrix((a.data[keep], (a.row[keep], a.col[keep])), shape=a.shape)
    assert (f["adjacency"] != f["adjacency"].T).nnz > 0
    return f


def window(ctx: Any) -> dict[str, Any]:
    def make() -> dict[str, Any]:
        f = fitted(ctx, BAF)
        coords = np.asarray(ctx.sim.stored("05_baf/construct_df_clone_label/in/args/1"))
        idx = np.sort(np.argsort(np.sum((coords - coords[0]) ** 2, axis=1))[:300])
        res = f["res"]
        return {"idx": idx, "X": f["single_X"][:, :, idx], "base": f["single_base_nb_mean"][:, idx], "total": f["single_total_bb_RD"][:, idx],
                "adjacency": f["adjacency_mat"].tocsr()[idx][:, idx].tocsr(),
                "res": res, "pred": np.argmax(np.asarray(res["log_gamma"]), axis=0), "labels": np.asarray(res["new_assignment"])[idx]}
    return ctx.once("hmrf/window", make)


# --- oracle --------------------------------------------------------------------


def _spot_field_by_hand(_: Any) -> None:
    rng = np.random.default_rng(2)
    n_states, n_obs, n_spots, n_clones = 3, 4, 5, 2
    rdr, baf = rng.normal(size=(n_states, n_obs, n_spots)), rng.normal(size=(n_states, n_obs, n_spots))
    pred = rng.integers(0, n_states, size=(n_obs, n_clones))
    nb, bb = rng.integers(1, 5, size=n_spots), rng.integers(1, 5, size=n_spots)
    found = compute_loglike_spot_assignment(n_spots, nb, bb, np.empty(0), False, rdr, baf, pred, n_obs, n_clones)
    want = np.zeros((n_spots, n_clones))
    for s in range(n_spots):
        for c in range(n_clones):
            want[s, c] = (bb[s] / nb[s]) * rdr[pred[:, c], np.arange(n_obs), s].sum() + baf[pred[:, c], np.arange(n_obs), s].sum()
    assert np.allclose(found, want)
    concatenated = compute_loglike_spot_assignment(n_spots, nb, bb, np.empty(0), False, rdr, baf, pred.T.ravel(), n_obs, n_clones)
    assert np.allclose(concatenated, want), "a clone-concatenated pred reads as the (bins, clones) one"


def _assignment_field_bb(w: dict[str, Any]) -> None:
    res, n_obs = w["res"], w["X"].shape[0]
    labels, llf, total = unchanged(pipeline_clone_assignment, w["X"], w["base"], w["total"], res, w["pred"], w["adjacency"], w["labels"],
                                   np.zeros(w["idx"].size, dtype=int), 1.0, hmmclass=hmm_nophasing)
    p, taus = np.asarray(res["new_p_binom"]), np.asarray(res["new_taus"])
    n_clones = w["pred"].size // n_obs
    want = np.zeros((w["idx"].size, n_clones))
    for c in range(n_clones):
        state = w["pred"][c * n_obs:(c + 1) * n_obs]
        b = scipy.stats.betabinom.logpmf(w["X"][:, 1, :], w["total"], (p[state, 0] * taus[state, 0])[:, None], ((1 - p[state, 0]) * taus[state, 0])[:, None])
        want[:, c] = b.sum(axis=0)
    assert np.allclose(llf, want, atol=1e-6), "the spot field is the BB log-likelihood of each spot under each clone's MAP path (RDR zero)"


def _assignment_objective_potts(w: dict[str, Any]) -> None:
    labels, llf, total = unchanged(pipeline_clone_assignment, w["X"], w["base"], w["total"], w["res"], w["pred"], w["adjacency"], w["labels"],
                                   np.zeros(w["idx"].size, dtype=int), 1.0, hmmclass=hmm_nophasing)
    assert np.isclose(total, potts(llf, w["adjacency"], labels, 1.0)), "the returned objective is the Potts energy of the returned labels"


def _icm_gains_energy(f: dict[str, Any]) -> None:
    a, labels = f["adjacency"], f["labels"].copy()
    before = potts(f["llf"], a, labels, f["weight"])
    np.random.seed(0)  # noqa: NPY002
    _, cost = icm_sweep_deque(f["llf"], a.indptr, a.indices, a.data, labels, f["weight"], None, min_clone_spots=0)
    after = potts(f["llf"], a, labels, f["weight"])
    assert np.isclose(cost, after - before), "the reported cost is the Potts energy gained"
    assert after >= before
    # NB the field of the energy above: half of each out-edge and half of each in-edge
    half = ((a + a.T) / 2.0).tocsr()
    for i in range(len(labels)):
        row = slice(half.indptr[i], half.indptr[i + 1])
        local = f["llf"][i] + f["weight"] * np.bincount(labels[half.indices[row]], weights=half.data[row], minlength=3)
        assert np.isclose(local[labels[i]], local.max()), f"spot {i} is not at its conditional maximum"


def _merge_cost_is_potts(f: dict[str, Any]) -> None:
    spots, neighbours, weights = unpack_adjacency(cast_csr(f["adjacency"]))
    cost, best, (u, v) = merge_assignment(f["llf"], spots, neighbours, weights, f["labels"], f["weight"])
    assert np.isclose(cost, potts(f["llf"], f["adjacency"], f["labels"], f["weight"])), "the current cost is the Potts energy"
    a = f["adjacency"].tocoo()
    touching = {(int(x), int(y)) for x, y in zip(f["labels"][a.row], f["labels"][a.col], strict=True) if x != y}
    assert (u, v) in touching and best > -np.inf, "the proposed pair shares a boundary"


def _merge_gain_is_potts(f: dict[str, Any]) -> None:
    spots, neighbours, weights = unpack_adjacency(cast_csr(f["adjacency"]))
    cost, best, (u, v) = merge_assignment(f["llf"], spots, neighbours, weights, f["labels"], f["weight"])
    merged = np.where(f["labels"] == u, v, f["labels"])
    assert np.isclose(best, potts(f["llf"], f["adjacency"], merged, f["weight"])), "the best merge's cost is the merged labels' Potts energy"


def _pseudobulk_sums_clone_spots(_: Any) -> None:
    x = np.array([[[1, 2, 3, 4], [0, 1, 0, 1]], [[5, 6, 7, 8], [2, 0, 2, 0]]])  # NB 2 bins x 4 spots
    base, total = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]), np.array([[1, 1, 2, 2], [3, 3, 4, 4]])
    px, pb, pt, tumor = unchanged(merge_pseudobulk_by_index_mix, x, base, total, [np.array([0, 2]), np.array([1, 3])])
    assert px.tolist() == [[[4, 6], [0, 2]], [[12, 14], [4, 0]]] and pt.tolist() == [[3, 3], [7, 7]], "by hand: spots {0, 2} and {1, 3}"
    assert np.allclose(pb, [[0.4, 0.6], [1.2, 1.4]], rtol=1e-12, atol=0) and tumor is None
    rng = np.random.default_rng(4)
    x, base, total = rng.integers(0, 9, size=(5, 2, 8)), rng.random((5, 8)), rng.integers(0, 9, size=(5, 8))
    index = [np.array([0, 3, 4]), np.array([], dtype=int), np.array([1, 2, 5, 6, 7])]
    px, pb, pt, tumor = unchanged(merge_pseudobulk_by_index_mix, x, base, total, index)
    for k, idx in enumerate(index):
        assert np.array_equal(px[:, :, k], x[:, :, idx].sum(axis=-1)) and np.allclose(pb[:, k], base[:, idx].sum(axis=1)) and np.array_equal(pt[:, k], total[:, idx].sum(axis=1))
    assert tumor is None


def _pseudobulk_tumor_threshold(_: Any) -> None:
    x = np.array([[[1, 2, 3, 4, 5], [1, 0, 1, 0, 1]], [[6, 7, 8, 9, 10], [0, 1, 0, 1, 0]]])  # NB 2 bins x 5 spots
    base, total = x[:, 0, :] * 0.5, np.array([[2, 2, 2, 2, 2], [1, 1, 1, 1, 1]])
    prop = np.array([0.9, 0.5, 0.7, 0.2, 0.4])  # NB spot 1 sits at the threshold; clone 1 has no spot above it
    px, pb, pt, tumor = unchanged(merge_pseudobulk_by_index_mix, x, base, total, [np.array([0, 1, 2]), np.array([3, 4])], prop, 0.5)
    assert px.tolist() == [[[4, 0], [2, 0]], [[14, 0], [0, 0]]] and pt.tolist() == [[4, 0], [2, 0]], "by hand: spots 0 and 2 kept; clone 1 empty"
    assert pb.tolist() == [[2.0, 0.0], [7.0, 0.0]] and np.allclose(tumor, [0.8, 0.0], rtol=1e-12, atol=0), "tumor: the kept spots' mean, 0 where none"


def _stack_tiles_clones(_: Any) -> None:
    rng = np.random.default_rng(6)
    x, base, total = rng.random((4, 2, 3)), rng.random((4, 3)), rng.random((4, 3))
    sx, sb, st, sl, stm, stp = clone_stack_obs(x, base, total, np.array([1, 3]), np.arange(4.0), np.array([0.2, 0.5, 0.9]))
    for c in range(3):
        assert np.array_equal(sx[c * 4:(c + 1) * 4, :, 0], x[:, :, c]) and np.array_equal(sb[c * 4:(c + 1) * 4, 0], base[:, c]) and np.array_equal(st[c * 4:(c + 1) * 4, 0], total[:, c])
    assert sl.tolist() == [1, 3] * 3 and stm.tolist() == list(range(4)) * 3 and stp.ravel().tolist() == [0.2] * 4 + [0.5] * 4 + [0.9] * 4


def _unpack_is_coo(_: Any) -> None:
    a = sp.random(7, 7, density=0.4, random_state=3, format="csr")
    spots, neighbours, weights = unpack_adjacency(cast_csr(a))
    coo = a.tocoo()
    assert np.array_equal(spots, coo.row) and np.array_equal(neighbours, coo.col) and np.allclose(weights, coo.data)


ORACLE: list[Row] = table(
    "oracle",
    ("hmrf:compute_loglike_spot_assignment", "synthetic: 5 spots, 2 clones", lambda c: None, _spot_field_by_hand, "the field is w_s * RDR + BAF summed over each clone's states, w_s = valid BAF / valid RDR bins"),
    ("hmrf:pipeline_clone_assignment", WINDOW, window, _assignment_field_bb, "the field is scipy's BB log-likelihood under each clone's MAP path"),
    ("hmrf:pipeline_clone_assignment", WINDOW, window, _assignment_objective_potts, "the returned objective is the Potts energy (w / 2 per stored edge)",
     "Ticket#918: the returned objective counts each aligned stored edge with row < column once, unweighted (hmrf.py:275), not w / 2 per stored edge"),
    ("icm:icm_sweep_deque", "synthetic: 8x8 grid, 3 clones", lambda c: field_case(), _icm_gains_energy, "the reported cost is the Potts energy gained, and every spot ends at its conditional maximum"),
    ("icm:merge_assignment", "synthetic: 8x8 grid, 3 clones", lambda c: field_case(), _merge_cost_is_potts, "the current cost is the Potts energy (unary + weight x aligned edges)"),
    ("icm:merge_assignment", "synthetic: 8x8 grid, 3 clones", lambda c: field_case(), _merge_gain_is_potts, "the best merge's cost is the merged labels' Potts energy"),
    ("icm:icm_sweep_deque", "synthetic: 8x8 grid, a tenth of directions dropped", lambda c: one_way_case(), _icm_gains_energy,
     "on an asymmetric graph: the reported cost is the energy gained (w / 2 per stored edge), each spot at its conditional maximum",
     "Ticket#918: the ICM field reads out-edges at w, not the energy's w / 2 per stored edge, on an asymmetric adjacency"),
    ("icm:merge_assignment", "synthetic: 8x8 grid, a tenth of directions dropped", lambda c: one_way_case(), _merge_cost_is_potts,
     "on an asymmetric graph: the current cost is the energy (w / 2 per stored edge)"),
    ("icm:merge_assignment", "synthetic: 8x8 grid, a tenth of directions dropped", lambda c: one_way_case(), _merge_gain_is_potts,
     "on an asymmetric graph: the best merge's cost is the merged labels' energy"),
    ("pseudobulk:merge_pseudobulk_by_index_mix", "synthetic: 2 bins x 4 spots by hand; 8 spots, 3 clones, one empty", lambda c: None, _pseudobulk_sums_clone_spots,
     "each clone's pseudobulk is its spots' sum; an empty clone is zero; no input mutation"),
    ("pseudobulk:merge_pseudobulk_by_index_mix", "synthetic: 2 bins x 5 spots, tumor proportions, by hand", lambda c: None, _pseudobulk_tumor_threshold,
     "with tumor proportions: sums over spots strictly above the threshold, tumor the kept mean; a clone with none is zero; no input mutation"),
    ("hmrf_utils:clone_stack_obs", "synthetic: 4 bins, 3 clones", lambda c: None, _stack_tiles_clones, "clone c's bins are rows c*n..(c+1)*n; lengths and switch probabilities tile, tumor proportions repeat"),
    ("hmrf_utils:cast_csr", "synthetic: a random 7x7 sparse matrix", lambda c: None, _unpack_is_coo, "row lists unpack to the matrix's COO triplets"),
    ("icm:unpack_adjacency", "synthetic: a random 7x7 sparse matrix", lambda c: None, _unpack_is_coo, "(spot, neighbour, weight) are scipy's COO row, col, data"),
)


# --- invariants ---------------------------------------------------------------------


def _indices(ctx: Any) -> np.ndarray:
    return np.asarray(ctx.sim.stored(f"{RDR}/reindex_clones/in/args/0")["new_assignment"])


def _index_round_trip(labels: np.ndarray) -> None:
    index = get_clone_indices(labels, np.unique(labels))
    assert sum(len(i) for i in index) == labels.size and all(np.all(labels[i] == c) for c, i in zip(np.unique(labels), index, strict=True))
    assert np.array_equal(get_clone_assignment(np.zeros((labels.size, 2)), index), labels)


def _reindex_permutes_labels(res: Any) -> None:
    out, posterior = reindex_clones(res.copy(deep=True))
    before, after = np.asarray(res["new_assignment"]), np.asarray(out["new_assignment"])
    old_of = {int(n): int(o) for o, n in zip(before, after, strict=True)}
    assert len(old_of) == np.unique(before).size and sorted(old_of) == list(range(len(old_of))), "a permutation of 0..M-1"
    for new, old in old_of.items():
        assert np.array_equal(np.asarray(out["pred_cnv"])[:, new], np.asarray(res["pred_cnv"])[:, old]), "pred_cnv columns follow the labels"
        assert np.array_equal(np.asarray(out["log_gamma"])[:, :, new], np.asarray(res["log_gamma"])[:, :, old])
    assert posterior is None and np.array_equal(out["new_p_binom"], res["new_p_binom"])
    sizes = np.bincount(after)
    assert np.all(np.diff(sizes[1:]) >= 0), "clones after the normal one ascend by spot count"


def _reindex_keeps_input(res: Any) -> None:
    unchanged(reindex_clones, res.copy(deep=True))


def _minspots_above_floor(_: Any) -> None:
    labels = np.repeat([0, 1, 2, 3], [50, 5, 30, 15])
    n_obs, n_states = 4, 3
    gamma = np.log(np.full((n_states, 4 * n_obs), 1 / n_states))
    pred = np.tile(np.arange(4), 4) % n_states
    params = HMMParams(*(np.zeros((n_states, 1)) for _ in range(6)))
    res = CnaHMRFResult(params, None, HMMProfile(gamma, pred), 0.0, n_states, CloneAssignment(new_assignment=labels))
    total = np.ones((n_obs, labels.size))
    groups, merged = merge_by_minspots(labels.copy(), res.copy(deep=True), total, min_spots_thresholds=20)
    assert sorted(c for g in groups for c in g) == [0, 1, 2, 3] and [g[0] for g in groups] == [0, 2]
    out = np.asarray(merged["new_assignment"])
    assert np.array_equal(np.unique(out), [0, 1]) and np.bincount(out).min() >= 20
    assert np.array_equal(np.asarray(merged["pred_cnv"]), np.concatenate([pred[0:4], pred[8:12]])), "the representatives' paths, in group order"
    assert np.isnan(merged["total_llf"])


def _core_out(ctx: Any) -> list[Any]:
    return [ctx.sim.stored(f"{r}/run_core_inference/out") for r in (BAF, RDR)] + [ctx.sim.internal(r) for r in (BAF, RDR)]


def _core_labels_reindexed(found: list[Any]) -> None:
    baf, rdr, baf_internal, rdr_internal = found
    for res, internal in ((baf, baf_internal), (rdr, rdr_internal)):
        labels = np.asarray(res["new_assignment"])
        assert np.array_equal(np.unique(labels), np.arange(labels.max() + 1))
        assert np.array_equal(labels, np.unique(internal["raw"], return_inverse=True)[1]), "the result's labels are the last ICM labels, re-indexed"
        assert np.array_equal(np.asarray(res["prev_assignment"]), internal["prev"])
    assert np.asarray(baf["log_gamma"]).shape[1] == np.asarray(baf["pred_cnv"]).size and np.asarray(rdr["log_gamma"]).ndim == 3


def _field_uses_posterior(_: Any) -> None:
    body = ast.parse(textwrap.dedent(inspect.getsource(run_core_inference)))
    hard = [n for n in ast.walk(body) if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "pred" for t in n.targets)
            and "argmax" in ast.unparse(n.value)]
    assert not hard, "the clone field is scored under each clone's argmax states"


INVARIANT: list[Row] = table(
    "invariant",
    ("hmrf_utils:get_clone_indices", "08_rdr/reindex_clones in: the merged labels", _indices, _index_round_trip, "indices partition the spots by label"),
    ("hmrf_utils:get_clone_assignment", "08_rdr/reindex_clones in: the merged labels", _indices, _index_round_trip, "inverts get_clone_indices"),
    ("hmrf:reindex_clones", "08_rdr/reindex_clones in", lambda c: c.sim.stored(f"{RDR}/reindex_clones/in/args/0"), _reindex_permutes_labels,
     "a label permutation; paths and posteriors move with it; the rest by size"),
    ("hmrf:reindex_clones", "08_rdr/reindex_clones in", lambda c: c.sim.stored(f"{RDR}/reindex_clones/in/args/0"), _reindex_keeps_input, "the result it is handed is not rewritten",
     "new: reindex_clones rewrites the result it is handed through CnaHMRFResult's shallow copy (labels, pred_cnv, log_gamma); run_cnamaste's merge reaches res_combine the same way"),
    ("hmrf:merge_by_minspots", "synthetic: 4 clones of 50/5/30/15 spots", lambda c: None, _minspots_above_floor, "groups partition the clones; survivors >= the floor; representatives' paths kept"),
    ("hmrf:run_core_inference", "05_baf, 08_rdr run_core_inference out + /internal", _core_out, _core_labels_reindexed, "labels 0..M-1, the last ICM labels re-indexed; prev the last input labels"),
    ("hmrf:run_core_inference", "source", lambda c: None, _field_uses_posterior, "the clone field is scored under each clone's posterior, not its argmax",
     "Ticket#869: Clone assignment scores spots under each clone's argmax states (nodepotential=max): uncertain bins are charged as certain"),
)


# --- captured -------------------------------------------------------------------------


def _merges_captured(ctx: Any) -> dict[str, Any]:
    return {"internal": ctx.sim.internal(RDR), "res": ctx.sim.stored(f"{RDR}/run_core_inference/out")}


def _merges_map_labels(found: dict[str, Any]) -> None:
    internal = found["internal"]
    m = np.arange(int(internal["icm"].max()) + 1)
    for u, v in internal["merges"]:
        m[m == u] = v
    assert np.array_equal(m[internal["icm"]], internal["raw"]), "the ICM labels through the recorded merges are the raw labels"


CAPTURED: list[Row] = table(
    "captured",
    ("icm:merge_assignment", "/internal/08_rdr: the last iteration's ICM labels and merges", _merges_captured, _merges_map_labels, "the recorded merges map the ICM labels to the raw labels the run kept"),
)


@pytest.mark.parametrize("row", ORACLE, ids=[r.id for r in ORACLE])
def test_oracle_agrees(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """Brute-force energies, sums and the scipy field."""
    run(row, ctx, request)


@pytest.mark.parametrize("row", INVARIANT, ids=[r.id for r in INVARIANT])
def test_invariants_hold(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """Partitions, permutations, floors, no input mutation."""
    run(row, ctx, request)


@pytest.mark.parametrize("row", CAPTURED, ids=[r.id for r in CAPTURED])
def test_matches_run(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """The run's recorded internals agree with each other."""
    run(row, ctx, request)
