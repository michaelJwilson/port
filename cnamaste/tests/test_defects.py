"""One row per issue against cnamaste's code: a minimal reproduction, and the assertion of correct behaviour.

A row whose defect cnamaste still has is a strict xfail, `Ticket#N: <title>`, so
a fix turns it into XPASS, a failure, and the marker has to go. A row cnamaste
has fixed (Ticket#105, Ticket#692) is a plain passing regression. A row that
does not reproduce on the inputs used here says so and passes. A planned departure is a
strict xfail asserting the planned behaviour; the runtime goals are
`test_runtime.py`'s.

Inputs: the staged CalicoST easy run where it holds the case, else a few-element
synthetic input built in the row. Issues that concern no code or data
cnamaste holds (Ticket#593, Ticket#614, Ticket#788, Ticket#598, Ticket#803)
and Ticket#161 (`cnaster.wolff`, which `run_cnamaste` does not reach, so
cnamaste does not copy it) have no row.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import io
import json
import textwrap
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import yaml
from scipy.special import logsumexp
from scipy.stats import binom, nbinom, poisson

from audit.segments import DROPPED
from cnamaste.config import YAMLConfig, set_global_config
from cnamaste.count_encoder import CountEncoder
from cnamaste.hmm import compute_copy_state_posterior, pipeline_baum_welch
from cnamaste.hmm_emission import Weighted_BetaBinom
from cnamaste.hmm_initialize import gmm_init
from cnamaste.hmm_nophasing import _bb_logpmf_1d, _nb_logpmf_1d, betabinom_logpmf_numba, get_log_transmat, hmm_nophasing, nbinom_logpmf_numba
from cnamaste.hmm_phased import hmm_phased
from cnamaste.hmrf import compute_loglike_spot_assignment, merge_by_minspots, reindex_clones, run_core_inference
from cnamaste.hmrf import logsumexp as hmrf_logsumexp
from cnamaste.icm import icm_sweep_deque, merge_assignment
from cnamaste.icm import logsumexp as icm_logsumexp
from cnamaste.integer_copy import hill_climbing_integer_copynumber_fixdiploid_milp, hill_climbing_integer_copynumber_oneclone
from cnamaste.io import get_aggregated_barcodes, get_sample_list, get_spaceranger_counts, load_input_data
from cnamaste.normal_spot import determine_normal_candidates, filter_normal_diffexp
from cnamaste.omics import assign_initial_blocks, summarize_counts_for_bins
from cnamaste.recomb import assign_centiMorgans, compute_numbat_phase_switch_prob
from cnamaste.spatial import banded, construct_lattice_adjacency, initialize_rectangular_clones
from cnamaste.utils import top_hat_sum

PROJECT = Path(__file__).resolve().parents[1]
PACKAGE = PROJECT / "python" / "cnamaste"


class Row(NamedTuple):
    issue: str
    function: str
    check: Callable[..., None]
    status: str
    """`xfail`, `regression` (fixed in cnamaste), `not reproduced` (holds here), `departure` (a planned behaviour), `skip`."""
    reason: str
    raises: type[BaseException] | tuple[type[BaseException], ...] | None = None


ROWS: list[Row] = []


def row(issue: str, function: str, status: str, reason: str, raises: Any = None) -> Callable[[Callable[..., None]], Callable[..., None]]:
    def add(f: Callable[..., None]) -> Callable[..., None]:
        ROWS.append(Row(issue, function, f, status, reason, raises))
        return f
    return add


def source(path: str) -> ast.Module:
    return ast.parse((PACKAGE / path).read_text())


def function_node(tree: ast.Module, name: str) -> ast.FunctionDef:
    found = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    return found[-1]  # NB the last definition is the live one (icm.py defines icm_sweep_deque four times)


def configured(document: dict[str, Any]) -> Any:
    config = YAMLConfig(document)
    set_global_config(config)
    return config


def run_config(sim: Any) -> Any:
    """The staged run's own configuration, installed as cnamaste's global one."""
    return configured(yaml.safe_load(sim.config["yaml"]))


def lattice(nx: int, ny: int) -> np.ndarray:
    return np.array([(x, y) for x in range(nx) for y in range(ny)], dtype=float)


def returns_within(seconds: float, call: Callable[[], Any]) -> Any:
    found: list[Any] = []
    thread = threading.Thread(target=lambda: found.append(call()), daemon=True)
    thread.start()
    thread.join(seconds)
    assert found, f"no return within {seconds} s"
    return found[0]


# --- loading -------------------------------------------------------------------


@row("Ticket#88", "io.get_spaceranger_counts", "xfail", "Ticket#88: get_spaceranger_counts assumes a sparse .h5ad and raises AttributeError on a dense one", AttributeError)
def _(tmp_path: Path, **_: Any) -> None:
    configured({"visium": {"filtered_feature_name": "filtered_feature_bc_matrix"}})
    adata = anndata.AnnData(np.ones((3, 2)), obs=pd.DataFrame(index=["a", "b", "c"]), var=pd.DataFrame(index=["g0", "g1"]))
    adata.write_h5ad(tmp_path / "filtered_feature_bc_matrix.h5ad")

    get_spaceranger_counts(str(tmp_path))


@row("Ticket#176", "io.load_input_data", "xfail", "Ticket#176: both filter-file branches of load_input_data crash on a plausible input (a Path gene file)", TypeError)
def _(replayed: Any, **_: Any) -> None:
    config = replayed.staged.config
    load_input_data(config, filter_gene_file=Path(config.references.filtergenelist_file), filter_range_file=None)


@row("Ticket#182", "io.load_input_data", "xfail", "Ticket#182: min_percent_expressed_spots is a fraction under a name that says percent")
def _(**_: Any) -> None:
    body = ast.unparse(function_node(source("io.py"), "load_input_data"))
    assert "min_percent_expressed_spots / 100" in body, "compared as a fraction of spots (io.py:855)"


@row("Ticket#446", "io.get_aggregated_barcodes", "xfail", "Ticket#446: get_aggregated_barcodes gives every barcode sample_id None with 2+ slices, so each slice loads 0 spots")
def _(tmp_path: Path, **_: Any) -> None:
    (tmp_path / "barcodes.txt").write_text("AAAC-1_s1\nAAAG-1_s1\nAAAC-1_s2\n")
    found = get_aggregated_barcodes(str(tmp_path / "barcodes.txt"), known_sample_id=None)
    assert list(found["sample_id"]) == ["s1", "s1", "s2"]


@row("Ticket#878", "io.get_sample_list", "xfail", "Ticket#878: get_sample_list lists a sample twice on interleaved rows A, B, A, so code 0 holds no spot")
def _(**_: Any) -> None:
    adata = anndata.AnnData(np.zeros((3, 1)), obs=pd.DataFrame({"sample": ["A", "B", "A"]}, index=list("xyz")))
    names, ids = get_sample_list(adata)
    assert names == ["A", "B"] and ids.tolist() == [0, 1, 0], f"samples {names}, codes {ids.tolist()}"


# --- genes, blocks, bins ---------------------------------------------------------


def tiny_blocks() -> tuple[pd.DataFrame, Any, np.ndarray, np.ndarray, np.ndarray]:
    """Three genes and four SNPs on two contigs, two spots: `assign_initial_blocks`' inputs."""
    frame = pd.DataFrame({
        "CHR": [1, 1, 1, 1, 2, 2, 2], "START": [100, 150, 300, 350, 100, 120, 500], "END": [200, 151, 400, 351, 600, 121, 501],
        "snp_id": [None, "1_150", None, "1_350", None, "2_120", "2_500"], "gene": ["g0", "g0", "g1", "g1", "g2", "g2", "g2"],
        "is_interval": [True, False, True, False, True, False, False],
    })
    adata = anndata.AnnData(np.ones((2, 3)), obs=pd.DataFrame(index=["s0", "s1"]), var=pd.DataFrame(index=["g0", "g1", "g2"]))
    adata.layers["count"] = sp.csr_matrix(np.full((2, 3), 5))
    ids = np.array(["1_150", "1_350", "2_120", "2_500"])
    a, b = np.full((2, 4), 30), np.full((2, 4), 30)
    return frame, adata, a, b, ids


@row("Ticket#189", "omics.assign_initial_blocks", "xfail", "Ticket#189: assign_initial_blocks writes by positional column and is not re-entrant", IndexError)
def _(**_: Any) -> None:
    frame, adata, a, b, ids = tiny_blocks()
    once = assign_initial_blocks(frame, adata, a, b, ids, initial_min_umi=1)
    again = assign_initial_blocks(frame, adata, a, b, ids, initial_min_umi=1)
    assert again["block_id"].equals(once["block_id"])


@row("Ticket#191", "omics.summarize_blocks", "xfail", "Ticket#191: summarize_blocks is 39% of assign_initial_blocks and returns nothing")
def _(**_: Any) -> None:
    caller = function_node(source("omics.py"), "assign_initial_blocks")
    discarded = [n for n in ast.walk(caller) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                 and getattr(n.value.func, "id", None) == "summarize_blocks"]
    assert not discarded, f"{len(discarded)} calls whose result is dropped (omics.py:621, :731): work done only to log"


@row("Ticket#84", "omics.summarize_counts_for_bins", "xfail", "Ticket#84: summarize_counts_for_bins filters block ids with `is not None`, which a default DataFrame breaks", ValueError)
def _(**_: Any) -> None:
    frame = pd.DataFrame({"CHR": [1, 1, 1], "START": [1, 2, 3], "END": [2, 3, 4], "gene": ["g0", "g0", "g0"],
                          "block_id": [np.nan, 0.0, 0.0], "bin_id": [0, 0, 0]})
    adata = anndata.AnnData(np.ones((2, 1)), var=pd.DataFrame(index=["g0"]))
    adata.layers["count"] = np.ones((2, 1))
    x, depth = np.ones((1, 2, 2), dtype=int), np.ones((1, 2), dtype=int)
    summarize_counts_for_bins(frame, adata, x, depth, np.ones(1), nu=1.0, logphase_shift=-2.0, geneticmap_file=None)


@row("Ticket#466", "omics.assign_initial_blocks / normal_baf_bin_filter", "not reproduced",
     "Ticket#466 (audit): on easy no block spans two contigs and every contig keeps a bin")
def _(sim: Any, **_: Any) -> None:
    frame = sim.stored("02_blocks/assign_initial_blocks/out")
    assert (frame.groupby("block_id")["CHR"].nunique() == 1).all()
    assert np.all(np.asarray(sim.stored("06_normal/normal_baf_bin_filter/out/1/lengths")) > 0)


@row("Ticket#871", "omics.binned_gene_snp", "xfail",
     "Ticket#871: binned_gene_snp joins Python sets, so INCLUDED_GENES' order follows PYTHONHASHSEED")
def _(replayed: Any, sim: Any, **_: Any) -> None:
    table = replayed.run("07_rebin/binned_gene_snp")
    frame = sim.stored("07_rebin/create_bin_ranges/out")
    genes = frame[frame["is_interval"] & frame["bin_id"].notna()]
    order = {g: i for i, g in enumerate(genes["gene"])}
    for joined in table["INCLUDED_GENES"]:
        listed = joined.split(",")
        assert listed == sorted(listed, key=order.__getitem__), "a bin's genes out of genomic order"


@row("new: duplicated gene", "run_cnamaste gene-level writer", "xfail",
     "Ticket#105 (the gene-level writer), new: a gene name the reference carries twice is written 2^C times by the per-clone joins")
def _(sim: Any, **_: Any) -> None:
    genelevel = pd.read_csv(io.BytesIO(sim.file("cnv_genelevel.tsv")), sep="\t", index_col=0)
    assert genelevel.index.is_unique, f"{genelevel.index.duplicated().sum()} repeated rows (LINC01505 on easy: 16 rows)"


@row("Ticket#105", "run_cnamaste gene-level writer", "regression", "Ticket#105: fixed in cnamaste PR1; a removed bin's genes leave cnv_genelevel.tsv")
def _(sim: Any, lineage: Any, **_: Any) -> None:
    genelevel = pd.read_csv(io.BytesIO(sim.file("cnv_genelevel.tsv")), sep="\t", index_col=0)
    names = sim.stored("02_blocks/assign_initial_blocks/out")["gene"].to_numpy()[lineage.genes.row]
    removed = lineage.levels["kept_bins"].label == DROPPED
    assert removed.sum() > 0 and not (set(names[removed]) - set(names[~removed])) & set(genelevel.index)


# --- phasing ---------------------------------------------------------------------


@row("Ticket#122", "phasing.initial_phase_given_partition", "not reproduced", "Ticket#122: on easy the phase vote flips 79 of 3,052 blocks")
def _(sim: Any, **_: Any) -> None:
    assert np.unique(sim.stored("03_phasing/initial_phase_given_partition/out/1")).size == 2


@row("Ticket#449", "recomb.compute_numbat_phase_switch_prob", "xfail",
     "Ticket#449: phase-switch law puts cM into Haldane (100x recombination)")
def _(**_: Any) -> None:
    configured({"phasing": {"min_prob": 1e-12}})
    p = compute_numbat_phase_switch_prob(np.array([0.0, 1.0]), [(1, 0), (1, 1)], nu=1.0, min_prob=1e-12)
    assert np.isclose(p[0], (1 - np.exp(-2 * 0.01)) / 2), f"1 cM switches at {p[0]:.3f}, Haldane's is 0.0099"


@row("Ticket#20", "recomb.assign_centiMorgans", "xfail", "Ticket#20: assign_centiMorgans sorts its caller's list in place")
def _(**_: Any) -> None:
    given = [(1, 300), (1, 100)]
    assign_centiMorgans(given, pd.DataFrame({"chrom": [1, 1], "pos": [0, 1000], "pos_cm": [0.0, 1.0]}))
    assert given == [(1, 300), (1, 100)]


@row("Ticket#172", "recomb.get_sitewise_transmat", "not reproduced",
    "Ticket#172: on easy (min_prob 0.01) 32% of bins are at log 1/2; 67% sit at the min_prob floor, end-to-start gaps")
def _(sim: Any, **_: Any) -> None:
    log_switch = np.asarray(sim.stored("04_bins/get_sitewise_transmat/out"))
    saturated = np.mean(np.isclose(log_switch, np.log(0.5)))
    assert saturated < 0.5, f"{saturated:.3f} of easy's bins carry no phase information"


# --- normal spots and the expression filter ---------------------------------------


@row("Ticket#165", "normal_spot.filter_normal_diffexp", "xfail", "Ticket#165: filter_normal_diffexp splits INCLUDED_GENES on the wrong separator")
def _(**_: Any) -> None:
    counts = pd.DataFrame({"g0": [3, 4, 5, 6], "g1": [1, 1, 1, 1]}, index=list("abcd"))
    table = pd.DataFrame({"INCLUDED_GENES": ["g0,g1"]})
    out = filter_normal_diffexp(counts, table, np.array([True, True, False, False]), use_kmeans=False)
    assert np.array_equal(out[0], counts.sum(axis=1).to_numpy()), "a two-gene bin keeps none of its UMIs"


@row("Ticket#177", "run_cnamaste", "xfail", "Ticket#177: run_cnaster discards filter_normal_diffexp's result (run_cnamaste.py:969 -> 1031)")
def _(**_: Any) -> None:
    body = function_node(source("scripts/run_cnamaste.py"), "run_cnamaste")
    statements = [s for s in ast.walk(body) if isinstance(s, (ast.Assign, ast.Expr, ast.If))]
    lines = sorted({n.lineno for n in ast.walk(body) if isinstance(n, ast.Name) and n.id == "copy_single_X_rdr" and isinstance(n.ctx, ast.Load)})
    stored = [n.lineno for n in ast.walk(body) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "filter_normal_diffexp"]
    rebound = [n.lineno for n in ast.walk(body) if isinstance(n, ast.Name) and n.id == "copy_single_X_rdr" and isinstance(n.ctx, ast.Store) and n.lineno > stored[0]]
    del statements
    assert any(stored[0] < line < rebound[0] for line in lines), "the filter's return is overwritten before any read"


@row("Ticket#440", "normal_spot.filter_normal_diffexp / run_cnamaste", "xfail",
     "Ticket#440: differential-expression filter: fix the separator, reconnect it at gene level (Ticket#165, Ticket#177)")
def _(replayed: Any, **_: Any) -> None:
    removed = replayed.value("06_normal/normal_baf_bin_filter/out/1/X")[:, 0, :].sum() - replayed.value("06_normal/filter_normal_diffexp/out").sum()
    fitted = replayed.value("08_rdr/run_core_inference/in")["args"][0][:, 0, :].sum()
    unfiltered = replayed.value("07_rebin/determine_normal_baseline/out/1").sum()
    assert removed > 0
    assert fitted <= unfiltered - removed, f"the RDR fit holds the {removed:.0f} UMIs the filter removed"


@row("Ticket#479", "normal_spot.determine_normal_candidates", "xfail",
     "Ticket#479: setting preprocessing.normalidx_file crashes the run (determine_normal_candidates returns None)")
def _(**_: Any) -> None:
    config = configured({"preprocessing": {"normalidx_file": "normal.txt", "tumorprop_file": None}})
    found = determine_normal_candidates(config, {"new_assignment": np.zeros(4, dtype=int)}, np.full((1, 3), 0.5),
                                        np.ones((3, 2, 4)), np.ones((3, 4)), None)
    assert isinstance(found, np.ndarray) and found.dtype == bool


@row("Ticket#320", "normal_spot.determine_normal_candidates", "xfail",
     "Ticket#320: determine_normal_candidates admits whole tumor clones, inflating the RDR baseline")
def _(**_: Any) -> None:
    config = configured({"preprocessing": {"normalidx_file": None, "tumorprop_file": None}})
    rng = np.random.default_rng(320)
    n_bins, normal, tumour = 40, 100, 100
    depth = np.full((n_bins, normal + tumour), 20.0)
    depth[: n_bins // 2, normal:] *= 2  # NB a balanced gain: BAF 1/2, as the normal clone's, read depth doubled on half the genome
    rdr = rng.poisson(depth).astype(float)
    candidate = determine_normal_candidates(config, {"new_assignment": np.zeros(normal + tumour, dtype=int)}, np.full((1, n_bins), 0.5),
                                            np.zeros((n_bins, 2, normal + tumour)), rdr, None)
    assert candidate[normal:].mean() < 0.1, f"{candidate[normal:].mean():.2f} of the balanced-gain spots are normal candidates"


# --- the spatial graph and the clone solver ---------------------------------------


@row("Ticket#180", "spatial.construct_lattice_adjacency", "xfail",
     "Ticket#180: the spot adjacency is k-nearest-neighbour, asymmetric, and discards maxspots_pooling")
def _(**_: Any) -> None:
    smooth, adjacency = construct_lattice_adjacency(lattice(5, 5), maxspots_pooling=7, unit_xsquared=1, unit_ysquared=1)
    assert (adjacency != adjacency.T).nnz == 0, "a one-way edge"
    assert smooth.nnz > smooth.shape[0], "maxspots_pooling = 7 pools nothing"


@row("Ticket#417", "spatial.construct_lattice_adjacency", "xfail",
     "Ticket#417: cnaster's adjacency is directed, unreinforced at boundaries, and has no guard")
def _(**_: Any) -> None:
    _, adjacency = construct_lattice_adjacency(lattice(5, 5), unit_xsquared=1, unit_ysquared=1)
    assert np.diff(adjacency.tocsr().indptr)[0] == 3, "a square lattice's corner has 3 king-move neighbours, not 8"


@row("Ticket#692", "spatial.initialize_rectangular_clones", "regression", "Ticket#692: fixed in cnamaste PR1; the dev blocks return, banded")
def _(**_: Any) -> None:
    coords = np.load(PROJECT / "tests" / "data" / "rectangular_hang.npz")["coords"]
    index, labels = returns_within(20, lambda: initialize_rectangular_clones(coords, 4))
    assert sorted(np.concatenate(index).tolist()) == list(range(len(coords)))


@row("Ticket#304", "spatial.initialize_rectangular_clones", "departure",
     "planned departure: rectangle redraws before banding, Ticket#304 / Ticket#692, planned none")
def _(**_: Any) -> None:
    coords = np.load(PROJECT / "tests" / "data" / "rectangular_hang.npz")["coords"]
    _, labels = returns_within(20, lambda: initialize_rectangular_clones(coords, 4))
    assert not np.array_equal(labels, banded(coords, 4)[1]), "banded at once; the planned behaviour redraws the boundaries first (one redraw passes)"


@row("Ticket#45", "icm.icm_sweep_deque", "xfail", "Ticket#45: icm_sweep_deque is not reproducible, and mutates its caller's array")
def _(**_: Any) -> None:
    rng = np.random.default_rng(45)
    llf = rng.normal(size=(9, 2))
    adjacency = sp.csr_matrix(np.ones((9, 9)) - np.eye(9))
    given = np.zeros(9, dtype=np.int64)
    before = given.copy()
    icm_sweep_deque(llf, adjacency.indptr, adjacency.indices, adjacency.data, given, 0.1, None, min_clone_spots=0)
    assert np.array_equal(given, before), "the caller's assignment was rewritten in place"


@row("Ticket#879", "icm.icm_sweep_deque (empty clone)", "xfail",
     "Ticket#879 (Ticket#81): the floor is skipped once any clone is empty: clone_counts.min() counts the empty clone as 0")
def _(**_: Any) -> None:
    np.random.seed(879)
    llf = np.full((30, 3), -50.0)
    llf[:25, 0] = llf[25:, 1] = 0.0  # NB clone 1 holds 5 spots, under the floor of 10; clone 2 holds none
    adjacency = sp.csr_matrix((30, 30))
    labels = np.argmax(llf, axis=1).astype(np.int64)
    icm_sweep_deque(llf, adjacency.indptr, adjacency.indices, adjacency.data, labels, 0.0, None, min_clone_spots=10)
    live = np.bincount(labels)[np.bincount(labels) > 0]
    assert live.size == 1 or live.min() >= 10, f"live clone sizes {live.tolist()} under a floor of 10"


@row("Ticket#879", "hmrf.reindex_clones", "xfail",
     "Ticket#879: reindex_clones checks new_p_binom's column count alone; a per-clone new_log_mu is permuted while new_p_binom is not")
def _(**_: Any) -> None:
    res = {"new_assignment": np.array([0, 0, 1, 1, 1]), "pred_cnv": np.array([0, 0, 1, 1, 0, 1]), "new_log_mu": np.zeros((2, 2)),
           "new_alphas": np.ones((2, 1)), "new_p_binom": np.array([[0.5], [0.2]]), "new_taus": np.ones((2, 1))}
    with pytest.raises(AssertionError):
        reindex_clones(res)


@row("Ticket#81", "hmrf.run_core_inference", "xfail", "Ticket#81: any clone below 200 spots is merged away, and run_core_inference does not expose the threshold")
def _(**_: Any) -> None:
    assert "min_clone_spots" in inspect.signature(run_core_inference).parameters


@row("Ticket#468", "hmrf.pipeline_clone_assignment", "xfail", "Ticket#468: cnaster ignores hmrf.min_spots_per_clone: the ICM floor stays hard-coded at 200")
def _(**_: Any) -> None:
    caller = function_node(source("hmrf.py"), "pipeline_clone_assignment")
    calls = [n for n in ast.walk(caller) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "icm_sweep_deque"]
    assert calls and all(any(k.arg == "min_clone_spots" for k in c.keywords) for c in calls)


@row("Ticket#483", "icm.merge_assignment", "xfail", "Ticket#483: cnaster's merge gain is halved")
def _(**_: Any) -> None:
    llf = np.zeros((2, 2))
    spots, neighbours, weights = np.array([0, 1]), np.array([1, 0]), np.array([1.0, 1.0])
    cost, best, pair = merge_assignment(llf, spots, neighbours, weights, np.array([0, 1]), 1.0)
    assert np.isclose(best - cost, 1.0), f"merging two spots joined by one unit edge gains {best - cost}, not 1"


@row("Ticket#58", "hmrf.compute_loglike_spot_assignment", "xfail", "Ticket#58: the external field the HMM hands ICM is not the one the paper defines")
def _(**_: Any) -> None:
    rdr, baf = -np.ones((1, 2, 1)), -np.ones((1, 2, 1))
    field = compute_loglike_spot_assignment(1, np.array([1]), np.array([2]), np.empty(0), False, rdr, baf,
                                            np.zeros(2, dtype=np.int64), 2, 1, smooth_indices=np.array([0]), smooth_indptr=np.array([0, 1]))
    assert np.isclose(field[0, 0], -4.0), f"the field weighs read depth by valid-bin counts: {field[0, 0]}, the paper's -4"


@row("new: shallow copy", "hmrf.merge_by_minspots", "xfail",
     "Ticket#45 (mutates its caller's array), new: merge_by_minspots rewrites the result it is handed through CnaHMRFResult's shallow copy")
def _(sim: Any, **_: Any) -> None:
    res = sim.stored("08_rdr/run_core_inference/out")
    before = np.asarray(res["new_assignment"]).copy()
    counts = np.sort(np.bincount(before))
    # NB a floor between the two smallest clones: the smallest merges, the rest stand
    merge_by_minspots(before.copy(), res, np.ones((1, before.size)), min_spots_thresholds=int(counts[1]), min_umicount_thresholds=0)
    assert np.array_equal(np.asarray(res["new_assignment"]), before), "the caller's result now carries the merged labels"


@row("new: stale clone_lengths", "hmrf.run_core_inference", "xfail",
     "Ticket#267 (the clone axis), new: clone_lengths is computed once (hmrf.py:564) and is stale after a clone drops out")
def _(**_: Any) -> None:
    body = function_node(source("hmrf.py"), "run_core_inference")
    loop = next(n for n in ast.walk(body) if isinstance(n, ast.While))
    assigned = [n for n in ast.walk(loop) if isinstance(n, ast.Name) and n.id == "clone_lengths" and isinstance(n.ctx, ast.Store)]
    assert assigned, "clone_lengths is not recomputed in the outer loop, where the clone stack shrinks"


@row("Ticket#135", "hmm_nophasing.hmm_nophasing._run_optimization_pipeline", "xfail",
     "Ticket#135: the E step accepts a tumour proportion and never reads it, while the M step fits under one")
def _(**_: Any) -> None:
    body = function_node(source("hmm_nophasing.py"), "_run_optimization_pipeline")
    reads = [n for n in ast.walk(body) if isinstance(n, ast.Name) and n.id == "tumor_prop" and isinstance(n.ctx, ast.Load)]
    assert reads, "tumor_prop is a parameter and nothing reads it"


@row("Ticket#146", "hmm_nophasing (Baum-Welch)", "xfail", "Ticket#146: cnaster's M step never updates the transition matrix, which the paper specifies")
def _(sim: Any, **_: Any) -> None:
    res = sim.result("08_rdr/reindex_clones/out/0")
    fixed = get_log_transmat(np.asarray(res["new_log_transmat"]).shape[0], float(yaml.safe_load(sim.config["yaml"])["hmm"]["t"]))
    assert not np.array_equal(np.asarray(res["new_log_transmat"]), fixed), "the fitted transitions are the configured t"


@row("Ticket#80", "hmrf.run_core_inference", "xfail", "Ticket#80: run_core_inference's default hmmclass raises on any instance with more than one spot")
def _(**_: Any) -> None:
    assert inspect.signature(run_core_inference).parameters["hmmclass"].default is hmm_nophasing


@row("Ticket#269", "hmm_phased.compute_emission_probability_nb_betabinom_coded", "xfail",
     "Ticket#269: hmm_phased's emission raises IndexError on the parameter shape the fit returns", IndexError)
def _(sim: Any, **_: Any) -> None:
    run_config(sim)
    counts, totals = np.array([[1, 2], [3, 4], [1, 2]]), np.array([[5, 5], [6, 6], [5, 5]])
    encoder = CountEncoder(counts, totals)
    one = np.ones((2, 1))
    hmm_phased.compute_emission_probability_nb_betabinom_coded(encoder, encoder, np.zeros((2, 1)), 0.1 * one, 0.5 * one, 10 * one)


@row("Ticket#143", "hmm.pipeline_baum_welch", "xfail", "Ticket#143: pipeline_baum_welch cannot initialize itself: gmm_init is called with five of eight positional arguments", TypeError)
def _(sim: Any, **_: Any) -> None:
    run_config(sim)

    x = np.ones((6, 2, 1))
    pipeline_baum_welch(None, x, np.array([6]), 2, np.ones((6, 1)), 2 * np.ones((6, 1)), np.zeros(6), hmmclass=hmm_nophasing, max_iter=1)


@row("Ticket#876", "icm.logsumexp / hmrf.logsumexp", "xfail", "Ticket#876 (Ticket#413): icm's and hmrf's logsumexp return NaN, not -inf, on an all -inf row")
def _(**_: Any) -> None:
    for f in (icm_logsumexp, hmrf_logsumexp):
        assert np.isclose(f(np.array([-1.0, 2.0, 0.5])), logsumexp([-1.0, 2.0, 0.5]))
        assert np.isneginf(f(np.full(3, -np.inf))), f"{f(np.full(3, -np.inf))}; scipy's is -inf"


@row("Ticket#876", "hmm.compute_copy_state_posterior / hmm_nophasing.get_state_posteriors", "xfail",
     "Ticket#876 (Ticket#413): an all -inf posterior column normalizes to NaN: the zero check tests sum(log_gamma) == 0, not logsumexp == -inf")
def _(**_: Any) -> None:
    with pytest.raises(RuntimeError):
        compute_copy_state_posterior(np.array([[0.0, -np.inf], [-1.0, -np.inf]]), np.zeros((2, 2)))
    emission = np.array([[0.0, -np.inf, 0.0], [-1.0, -np.inf, -1.0]])[:, :, None]
    with pytest.raises(RuntimeError):
        hmm_nophasing().get_state_posteriors(np.array([3]), get_log_transmat(2, 0.9), np.log(np.full(2, 0.5)), emission, np.zeros(3))


@row("Ticket#30", "hmm_emission.Weighted_BetaBinom (via get_em_solver_params)", "not reproduced",
     "Ticket#30: at the shipped em_ftol (1e-4) this one-state beta-binomial stops within 1e-3 nats of the tight optimum")
def _(sim: Any, **_: Any) -> None:
    run_config(sim)
    rng = np.random.default_rng(30)
    n = rng.integers(20, 60, size=2400)
    k = rng.binomial(n, rng.beta(0.3 * 50, 0.7 * 50, size=n.size))
    model = Weighted_BetaBinom(k, np.ones(n.size), weights=np.ones(n.size), exposure=n)
    shipped = model.fit(maxiter=1500, ftol=1e-4, disp=0)
    tight = model.fit(maxiter=15000, ftol=1e-15, disp=0)
    shortfall = model.nloglikeobs(shipped.params) - model.nloglikeobs(tight.params)
    assert shortfall < 1e-3, f"the shipped ftol stops {shortfall:.3f} nats short"


@row("Ticket#9", "hmm_nophasing emissions, dense and coded", "not reproduced", "Ticket#9: the dense and coded emissions agree here, to 1e-12")
def _(sim: Any, **_: Any) -> None:
    run_config(sim)
    x = np.array([[[3], [1]], [[5], [2]], [[3], [1]]], dtype=float)
    base, depth = np.array([[4.0], [6.0], [4.0]]), np.array([[3], [5], [3]])
    params = (np.log(np.array([[0.8], [1.3]])), np.array([[0.1], [0.2]]), np.array([[0.3], [0.5]]), np.array([[20.0], [40.0]]))
    rdr, baf = hmm_nophasing.compute_emission_probability_nb_betabinom(x, base, params[0], params[1], depth, params[2], params[3])
    nb, bb = CountEncoder(x[:, 0, :], base), CountEncoder(x[:, 1, :], depth)
    coded = hmm_nophasing().compute_emission_probability_nb_betabinom_coded(nb, bb, *params)
    assert np.allclose(np.reshape(coded[0], rdr.shape), rdr, atol=1e-12) and np.allclose(np.reshape(coded[1], baf.shape), baf, atol=1e-12)


@row("Ticket#267", "hmrf.run_core_inference (fitted parameters)", "xfail",
     "Ticket#267: the second axis of log_mu, p_binom, alphas and taus is vestigial in the fit and contradictory in its consumers")
def _(sim: Any, **_: Any) -> None:
    res = sim.result("08_rdr/reindex_clones/out/0")
    assert np.asarray(res["new_log_mu"]).ndim == 1, f"shape {np.asarray(res['new_log_mu']).shape}: one column for {np.unique(res['new_assignment']).size} clones"


@row("Ticket#293", "hmrf.run_core_inference (mu scale)", "not reproduced", "Ticket#293: on easy the normal clone's fitted mu is within 5% of 1")
def _(sim: Any, **_: Any) -> None:
    res = sim.result("08_rdr/reindex_clones/out/0")
    mu = np.exp(np.asarray(res["new_log_mu"])[np.asarray(res["pred_cnv"])[:, 0], 0])
    assert abs(np.median(mu) - 1) < 0.05


# --- emission kernels ------------------------------------------------------------


@row("Ticket#560", "hmm_nophasing.nbinom_logpmf_numba", "xfail",
     "Ticket#560: nbinom_logpmf_numba scores any count at probability 1 when p rounds to 1: Baum-Welch finds a degenerate state")
def _(**_: Any) -> None:
    assert nbinom_logpmf_numba(1000.0, 10.0, 1.0) < -100, "1,000 UMIs at mean 0 score log P = 0"


@row("Ticket#561", "hmm_nophasing.betabinom_logpmf_numba", "xfail", "Ticket#561: beta-binomial log-pmf loses precision at large concentration tau")
def _(**_: Any) -> None:
    tau, p, n = 1e16, 0.3, 20
    values = [betabinom_logpmf_numba(float(k), float(n), p * tau, (1 - p) * tau) for k in range(n + 1)]
    assert abs(logsumexp(values)) < 1e-6, f"the pmf sums to e^{logsumexp(values):.1f}"


@row("Ticket#560", "hmm_nophasing._nb_logpmf_1d", "xfail",
     "Ticket#560 (Ticket#877): the 1-d NB kernel scores every count at probability 1 once p rounds to 1 (alpha 1e-17, mu 10)")
def _(**_: Any) -> None:
    k = np.array([0.0, 1.0, 5.0, 20.0])
    out = np.empty(k.size)
    _nb_logpmf_1d(k, np.ones(k.size), 10.0, 1e-17, out)
    # NB alpha is floored at 1e-10 (r = 1e10): within about 1e-7 nats of Poisson(10) for k <= 20
    assert np.allclose(out, poisson.logpmf(k, 10.0), atol=1e-6), f"{out} against Poisson's {poisson.logpmf(k, 10.0)}"


@row("Ticket#561", "hmm_nophasing._bb_logpmf_1d", "xfail", "Ticket#561 (Ticket#877): the 1-d BB kernel loses precision at large concentration tau")
def _(**_: Any) -> None:
    n, p = 20, 0.3
    k = np.arange(n + 1, dtype=float)
    out = np.empty(k.size)
    _bb_logpmf_1d(k, np.full(k.size, float(n)), p, 1e16, out)
    # NB at tau = 1e16 the beta-binomial is the binomial to about n^2 / tau = 4e-14
    assert np.allclose(out, binom.logpmf(k, n, p), atol=1e-9), f"max |diff| {np.max(np.abs(out - binom.logpmf(k, n, p))):.3e}"


@row("Ticket#241", "hmm_nophasing.nbinom_logpmf_numba", "xfail", "Ticket#241: parameter_terms_only=True is what includes the data term, so the flag does the opposite of its name")
def _(**_: Any) -> None:
    k, r, p = 7.0, 3.0, 0.4
    assert np.isclose(nbinom_logpmf_numba(k, r, p, parameter_terms_only=False), nbinom.logpmf(k, r, p)), "False drops log k!, the data term"


@row("Ticket#240", "hmm_nophasing.nbinom_logpmf_numba", "xfail",
     "Ticket#240: the NB log-pmf makes three lgamma calls per element and two of them are recomputation: 3.51x measured")
def _(**_: Any) -> None:
    body = textwrap.dedent(inspect.getsource(nbinom_logpmf_numba.py_func))
    calls = body.count("lgamma(")
    assert calls <= 1, f"{calls} lgamma calls per element: lgamma(r) is per state and lgamma(k + 1) per datum"


@row("Ticket#244", "hmm_nophasing (end to end)", "skip",
     "Ticket#244: 8.6e-13 in the emission becomes 3.2e-3 in the fitted parameters; needs two whole runs, not reproduced at a small size")
def _(**_: Any) -> None:
    pytest.skip("needs two whole runs with libm's and scipy's lgamma")


@row("Ticket#799", "count_encoder.CountEncoder", "xfail", "Ticket#799: CountEncoder: an index gather in place of the one-hot CSR mapping")
def _(sim: Any, **_: Any) -> None:
    run_config(sim)
    encoder = CountEncoder(np.array([[1], [2], [1]]), np.array([[3], [3], [3]]))
    mapper = encoder.mapping_matrices[0]
    per_entry = (mapper.data.nbytes + mapper.indices.nbytes) / mapper.shape[0] if sp.issparse(mapper) else np.asarray(mapper).itemsize
    assert per_entry <= 4, f"{per_entry:.0f} bytes and a sparse matmul per decoded entry, where an index is 4"


# --- integer copies and outputs --------------------------------------------------


@row("Ticket#136", "run_cnamaste (integer copies)", "xfail",
     "Ticket#136: run_cnaster logs that it normalized log_mu, and does not; the integer copy decoder reads the unnormalized value (run_cnamaste.py:1381)")
def _(sim: Any, **_: Any) -> None:
    given = sim.stored("09_outputs/integer_copy_0/in")
    log_mu, base, pred = (np.asarray(given["args"][i]) for i in (0, 1, 3))
    lam = base / base.sum()
    assert np.isclose(np.sum(lam * np.exp(log_mu[pred])), 1.0, atol=1e-9), "the decoder is handed log_mu unnormalized"


@row("Ticket#181", "integer_copy hill climbers", "xfail", "Ticket#181: the two integer decoders report different ploidies for the same profile")
def _(sim: Any, **_: Any) -> None:
    run_config(sim)
    pairs = np.array([(1, 1), (2, 1), (3, 1), (2, 2)])
    args = (np.log(pairs.sum(axis=1) / 2.0), np.ones(100), pairs[:, 1] / pairs.sum(axis=1), np.repeat(np.arange(4), 25))
    milp, one = hill_climbing_integer_copynumber_fixdiploid_milp(*args), hill_climbing_integer_copynumber_oneclone(*args)
    assert np.array_equal(milp[0], one[0]), "the precondition: one profile"
    assert milp[2] == one[2], f"best ploidy {milp[2]} (milp) against {one[2]} (oneclone) for one profile"


@row("Ticket#115", "utils.top_hat_sum", "xfail", "Ticket#115: top_hat_sum returns its input unchanged for a one-dimensional array")
def _(**_: Any) -> None:
    assert np.array_equal(np.ravel(top_hat_sum(np.arange(5.0), 3)), [1.0, 3.0, 6.0, 9.0, 7.0])


@row("Ticket#113", "he.get_he_image", "xfail", "Ticket#113: three defects in the plotting and H&E paths: he.py needs pyarrow, which cnamaste does not declare")
def _(**_: Any) -> None:
    assert importlib.util.find_spec("pyarrow") is not None, "polars' to_pandas needs pyarrow (he.py:125)"


@row("Ticket#311", "he.get_he_image", "xfail", "Ticket#311: get_he_image(num_labels=4) labels the brightest pixel 5, an extra initial clone")
def _(**_: Any) -> None:
    body = ast.unparse(function_node(source("he.py"), "get_he_image"))
    assert "np.digitize(cropped_gray, bins=bins[1:-1])" in body or "right=True" in body, \
        "np.digitize on the 0th-100th percentile edges puts the maximum past the last edge: num_labels + 1 labels"


# --- planned departures ---------------------------------------------------------


@row("Ticket#348 floor", "icm.icm_sweep_deque", "departure",
     "planned departure: refinement mask plus floor merge, smallest-first to the best field (sal FloorPolicy), Ticket#348, planned D5 PR #841")
def _(**_: Any) -> None:
    body = ast.unparse(function_node(source("icm.py"), "icm_sweep_deque"))
    assert "np.random.choice(valid_for_spot)" not in body, "a spot of a clone under the floor goes to a random eligible clone"


@row("Ticket#348 start", "hmm_initialize.gmm_init", "not reproduced",
     "planned departure: distinct GMM start, Ticket#348 / Ticket#143, planned D5 PR #841; on a two-valued input gmm_init already returns 5 distinct states")
def _(sim: Any, **_: Any) -> None:
    run_config(sim)

    rng = np.random.default_rng(348)
    n = 200
    depth = rng.integers(40, 60, size=(n, 1))
    b = rng.binomial(depth, np.where(np.arange(n) < n // 2, 0.5, 0.2)[:, None])
    x = np.stack([np.zeros((n, 1)), b], axis=1)
    _, p, *_ = gmm_init(5, x, np.zeros((n, 1)), depth, "sp", np.array([n]), get_log_transmat(5, 0.99), np.zeros(n), random_state=0)
    assert np.unique(np.round(np.asarray(p), 6)).size == 5, "two of five starting states coincide on two-valued data"


@row("Ticket#425", "hmrf.pipeline_clone_assignment", "departure", "planned departure: sal coded emission, Ticket#425, planned none")
def _(**_: Any) -> None:
    caller = function_node(source("hmrf.py"), "pipeline_clone_assignment")
    called = {getattr(n.func, "attr", "") for n in ast.walk(caller) if isinstance(n, ast.Call)}
    assert "compute_emission_probability_nb_betabinom_coded" in called, "the field's emission is dense over every spot and bin"


@row("Ticket#518", "run_cnamaste outputs", "departure", "planned departure: integer clones merged at 0.99 agreement, Ticket#817 / Ticket#518, planned K2")
def _(sim: Any, **_: Any) -> None:
    assert "clone_labels_integer.tsv" in json.loads(sim.config["files"])


@row("Ticket#836 F1-F3", "run_cnamaste outputs", "departure", "planned departure: combined.png, Ticket#836 F1-F3, planned F1-F3")
def _(**_: Any) -> None:
    written = [p for p in PACKAGE.rglob("*.py") if "combined.png" in p.read_text()]
    assert written, "no cnamaste code writes combined.png (the staged run stubs plots; with them on, none writes it either)"


@row("Ticket#881", "oxicnamaste (Rust crate)", "departure", "planned departure: cnamaste builds and binds its own Rust crate, oxicnamaste, planned Ticket#881")
def _(**_: Any) -> None:
    assert importlib.util.find_spec("cnamaste.oxicnamaste") is not None, "no compiled extension for a kernel to land in"


@row("Ticket#882", "run_cnamaste (integer copies)", "departure",
     "planned departure: integer copies decoded by the fit's likelihood (lattice decode), Ticket#882 / Ticket#362, planned Ticket#882")
def _(**_: Any) -> None:
    body = ast.unparse(function_node(source("scripts/run_cnamaste.py"), "run_cnamaste"))
    assert "hill_climbing_" not in body, "integer copies come from the L1 hill climbs, which decode CalicoST easy's (2, 2) gains as (1, 1)"


@row("Ticket#276", "hmm_nophasing coded emission (logmu shift)", "departure",
     "planned departure: the per-clone library normalizer log Z_c applied in the read-depth emission, Ticket#276, planned Ticket#884")
def _(**_: Any) -> None:
    assert "logmu_shifts are not currently supported" not in (PACKAGE / "hmm_nophasing.py").read_text(), "compute_logmu_shifts' value is discarded"
    calls = [n for n in ast.walk(function_node(source("hmm.py"), "pipeline_baum_welch")) if isinstance(n, ast.Call)]
    assert any(k.arg == "new_log_mu_shift" for c in calls for k in c.keywords), "the fit records no per-clone shift"


@row("Ticket#299", "hmrf.run_core_inference (neutral pin)", "departure",
     "planned departure: the shifted fit's neutral state pinned to mu = 1, the shifts moved with it, Ticket#293 / Ticket#299, planned Ticket#884")
def _(**_: Any) -> None:
    body = function_node(source("hmrf.py"), "run_core_inference")
    stored = [n for n in ast.walk(body) if isinstance(n, ast.Attribute) and n.attr == "new_log_mu_shift" and isinstance(n.ctx, ast.Store)]
    assert stored, "the shifted likelihood is flat along mu -> c mu, and nothing fixes the scale"


@row("Ticket#547", "hmm_initialize.gmm_init (lattice start)", "departure",
     "planned departure: the read-depth start from the integer (A, B) lattice at the best tumour fraction and depth scale, Ticket#540 / Ticket#547, planned Ticket#885")
def _(**_: Any) -> None:
    body = ast.unparse(function_node(source("hmm_initialize.py"), "gmm_init"))
    assert "lattice_start" in body, "the read-depth start is a mixture over the observed ratios, not the integer lattice"


MARKS = {"xfail": True, "departure": True}


@pytest.mark.parametrize("entry", ROWS, ids=[f"{r.issue} {r.function}" for r in ROWS])
def test_defect(sim_hash: str, entry: Row, request: pytest.FixtureRequest, tmp_path: Path) -> None:
    """One issue: its reproduction, asserting the correct behaviour."""
    if MARKS.get(entry.status):
        request.applymarker(pytest.mark.xfail(strict=True, reason=entry.reason, raises=entry.raises))
    wanted = inspect.signature(entry.check).parameters
    fixtures = {name: request.getfixturevalue(name) for name in ("sim", "replayed", "lineage") if name in wanted}
    entry.check(tmp_path=tmp_path, **fixtures)
