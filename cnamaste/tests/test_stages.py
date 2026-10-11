"""Each recorded stage of `run_cnamaste`: replayed, its bookkeeping, and its science against the planted truth.

- **Replay:** every recorded call but `run_core_inference`'s two (116 s and
  214 s on easy; stored whole instead) is called again on its recorded
  input with the generator states it found. Its return must hash to the
  recorded one: bitwise, except `binned_gene_snp`'s comma-joined gene and
  SNP lists, whose order follows the process's string hash seed
  (`PYTHONHASHSEED`): those compare as sets.
- **Bookkeeping:** `BOOKKEEPING`, one row per predicate a stage's output
  must satisfy whatever the data: ordering, conservation, density of labels,
  shapes that agree.
- **Science:** `SCIENCE`, one row per recovery of the planted truth, with a
  floor set at the value this file records less a stated margin. A known
  defect is a strict xfail citing its issue, so a fix fails the row until
  the marker goes.
- **Config criteria:** `CRITERIA`, one row per threshold or setting the
  staged `/config` declares, read at test time and checked on quantities
  recomputed from the staged inputs. A row's stated exceptions are counted;
  a failure outside them by a named mechanism is a strict xfail citing its
  ticket, or `new:`.
"""

from __future__ import annotations

import ast
import copy
import inspect
import io
import json
import random
import textwrap
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
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import LocalOutlierFactor

import cnamaste.hmm_nophasing as nophasing
from audit.capture import digest
from audit.criteria import FRAMES, Bins, RuleBroken, Units, bins_of, minor_baf, normal_baf_outside
from audit.scoring import NEUTRAL, matched, overlap, phase_free, swapped
from audit.segments import DROPPED
from cnamaste.config import set_global_config
from cnamaste.hmm_nophasing import get_log_transmat
from cnamaste.hmm_utils import get_em_solver_params

REPLAY_SKIPPED = {"run_core_inference"}


def test_a_stage_replays_bitwise(sim: Any, replayed: Any, stage: str) -> None:
    """The stage's function, on its recorded input, returns what the run's call returned, bitwise (sha256)."""
    if stage.split("/")[1] in REPLAY_SKIPPED:
        pytest.skip("run_core_inference: not replayed (100-230 s); its outputs are checked by BOOKKEEPING")
    assert replayed.matches(stage), f"{stage}: the replay differs from the run"
    if "write_tsv" in stage:
        assert replayed.outs[stage + "#file"] == sim.file(sim.h5[stage].attrs["file"])


# --- bookkeeping -------------------------------------------------------------

BOOKKEEPING: dict[str, Callable[[Any], None]] = {}


def row(table: dict[str, Callable[[Any], None]], name: str) -> Callable[[Callable[[Any], None]], Callable[[Any], None]]:
    def add(f: Callable[[Any], None]) -> Callable[[Any], None]:
        table[name] = f
        return f
    return add


class Run:
    """Accessors over one staged run, through the replay where a value is not stored."""

    def __init__(self, sim: Any, replayed: Any) -> None:
        self.sim, self.replayed = sim, replayed

    def __call__(self, path: str) -> Any:
        try:
            return self.sim.stored(path)
        except LookupError:
            return self.replayed.value(path)

    @property
    def config(self) -> Any:
        return yaml.safe_load(self.sim.config["yaml"])

    def frame(self, stage: str) -> pd.DataFrame:
        return self(f"{stage}/out")


@pytest.fixture(scope="session")
def run(sim: Any, replayed: Any) -> Run:
    return Run(sim, replayed)


def _sorted(frame: pd.DataFrame) -> None:
    key = frame[["CHR", "START"]].to_numpy()
    assert np.all((np.diff(key[:, 0]) > 0) | ((np.diff(key[:, 0]) == 0) & (np.diff(key[:, 1]) >= 0)))


@row(BOOKKEEPING, "genes: the table is sorted by (CHR, START)")
def _(run: Run) -> None:
    _sorted(run.frame("01_genes/form_gene_snp_table"))


@row(BOOKKEEPING, "genes: every SNP row names a gene, and a gene row names itself")
def _(run: Run) -> None:
    table = run.frame("01_genes/form_gene_snp_table")
    assert table["gene"].notna().all()
    assert table.loc[table["is_interval"], "snp_id"].isna().all()


def _ids_run(frame: pd.DataFrame, key: str) -> np.ndarray:
    ids = frame[key].dropna().to_numpy(dtype=np.int64)
    assert np.all(np.diff(ids) >= 0), f"{key} is not non-decreasing in row order"
    assert np.array_equal(np.unique(ids), np.arange(ids.max() + 1)), f"{key} is not dense"
    contig = frame.loc[frame[key].notna(), "CHR"].to_numpy()
    first = pd.Series(contig).groupby(ids).nunique()
    assert (first == 1).all(), f"a {key} spans two contigs"
    return ids


@row(BOOKKEEPING, "blocks: block_id runs 0..K-1 in row order, one contig each")
def _(run: Run) -> None:
    _ids_run(run.frame("02_blocks/assign_initial_blocks"), "block_id")


@row(BOOKKEEPING, "blocks: sum(lengths) == K == rows of X")
def _(run: Run) -> None:
    counts = run("02_blocks/summarize_counts_for_blocks/out")
    k = run.frame("02_blocks/assign_initial_blocks")["block_id"].nunique()
    assert counts.lengths.sum() == k == counts.X.shape[0] == counts.total_bb_RD.shape[0]


@row(BOOKKEEPING, "blocks: B counts never exceed the A+B depth, and base_nb_mean is 0 before the normal baseline")
def _(run: Run) -> None:
    counts = run("02_blocks/summarize_counts_for_blocks/out")
    assert np.all(counts.X[:, 1, :] <= counts.total_bb_RD) and np.all(counts.X >= 0)
    assert not np.any(counts.base_nb_mean)


@row(BOOKKEEPING, "transmat: one log switch per segment at each level, each <= log(1/2)")
def _(run: Run) -> None:
    for stage, counts in (("02_blocks", "02_blocks/summarize_counts_for_blocks"), ("04_bins", "04_bins/summarize_counts_for_bins"),
                          ("06_normal", "06_normal/normal_baf_bin_filter"), ("07_rebin", "07_rebin/summarize_counts_for_bins")):
        log_switch = np.asarray(run(f"{stage}/get_sitewise_transmat/out"))
        x = run(f"{counts}/out/1/X") if stage == "06_normal" else run(f"{counts}/out/X")
        assert log_switch.shape == (x.shape[0],), stage
        assert np.all(log_switch <= np.log(0.5) + 1e-12), stage


@row(BOOKKEEPING, "phasing: initial clones partition the spots once each")
def _(run: Run) -> None:
    index = run("03_phasing/initialize_clones/out")
    every = np.sort(np.concatenate([np.asarray(i) for i in index]))
    assert np.array_equal(every, np.arange(every.size))


@row(BOOKKEEPING, "phasing: sum(refined_lengths) == blocks; phase_indicator one 0/1 per block")
def _(run: Run) -> None:
    _, phase, refined = run("03_phasing/initial_phase_given_partition/out")
    k = run("02_blocks/summarize_counts_for_blocks/out").X.shape[0]
    assert np.sum(refined) == k and np.asarray(phase).shape == (k,)
    assert set(np.unique(phase)) <= {0, 1}


@row(BOOKKEEPING, "bins: bin_id runs 0..K-1 in row order, one contig each; sum(lengths) == K == rows of X")
def _(run: Run) -> None:
    ids = _ids_run(run.frame("04_bins/create_bin_ranges"), "bin_id")
    counts = run("04_bins/summarize_counts_for_bins/out")
    assert counts.lengths.sum() == ids.max() + 1 == counts.X.shape[0]


@row(BOOKKEEPING, "bins: every bin is a union of whole blocks")
def _(run: Run) -> None:
    frame = run.frame("04_bins/create_bin_ranges")
    assert (frame.groupby("block_id")["bin_id"].nunique() == 1).all()


@row(BOOKKEEPING, "bins: totals conserved from blocks to bins (UMIs, A+B depth)")
def _(run: Run) -> None:
    blocks, bins = run("02_blocks/summarize_counts_for_blocks/out"), run("04_bins/summarize_counts_for_bins/out")
    assert blocks.X[:, 0].sum() == bins.X[:, 0].sum()
    assert blocks.total_bb_RD.sum() == bins.total_bb_RD.sum()


@row(BOOKKEEPING, "bins: the phased B count is B or A+B-B per block, summed")
def _(run: Run) -> None:
    blocks, bins = run("02_blocks/summarize_counts_for_blocks/out"), run("04_bins/summarize_counts_for_bins/out")
    phase = np.asarray(run("03_phasing/initial_phase_given_partition/out")[1]).astype(bool)
    frame = run.frame("04_bins/create_bin_ranges")
    of = frame.groupby("block_id")["bin_id"].first().to_numpy(dtype=np.int64)
    phased = np.where(phase[:, None], blocks.X[:, 1, :], blocks.total_bb_RD - blocks.X[:, 1, :])
    summed = np.zeros_like(bins.X[:, 1, :])
    np.add.at(summed, of, phased)
    assert np.array_equal(summed, bins.X[:, 1, :])


@row(BOOKKEEPING, "adjacency: 8 out-edges per spot, unit weights, no self edges; smooth_mat the identity")
def _(run: Run) -> None:
    adjacency, smooth = run("04_bins/construct_multislice_lattice_adjacency/out")
    a = adjacency.tocsr()
    # NB symmetric it is not: a kNN rule (Ticket#180, test_defects.py)
    assert np.all(np.diff(a.indptr) == 8) and np.all(a.data == 1) and a.diagonal().sum() == 0
    assert (smooth != sp.identity(a.shape[0])).nnz == 0


def _labels_dense(labels: np.ndarray) -> None:
    assert np.array_equal(np.unique(labels), np.arange(labels.max() + 1)), "labels are not 0..M-1, or a clone is empty"


@row(BOOKKEEPING, "baf fit: labels 0..M-1, no empty clone; pred_cnv == argmax log_gamma; posteriors normalized")
def _(run: Run) -> None:
    res = run("05_baf/run_core_inference/out")
    labels = np.asarray(res["new_assignment"])
    _labels_dense(labels)
    log_gamma = np.asarray(res["log_gamma"])
    assert np.array_equal(np.asarray(res["pred_cnv"]), np.argmax(log_gamma, axis=0))

    assert np.allclose(logsumexp(log_gamma, axis=0), 0.0, atol=1e-8)
    n_bins = run("04_bins/summarize_counts_for_bins/out").X.shape[0]
    assert log_gamma.shape[1] == n_bins * (labels.max() + 1)


@row(BOOKKEEPING, "baf merge: every clone >= min_spots_per_clone, labels 0..M-1, groups partition the fitted clones")
def _(run: Run) -> None:
    groups, merged = run("05_baf/merge_by_minspots/out")
    labels = np.asarray(merged["new_assignment"])
    _labels_dense(labels)
    assert np.bincount(labels).min() >= run.config["hmrf"]["min_spots_per_clone"]
    fitted = np.asarray(run("05_baf/run_core_inference/out")["new_assignment"])
    assert sorted(c for g in groups for c in g) == list(range(fitted.max() + 1))


@row(BOOKKEEPING, "baf labels: baf_clone_labels.tsv is construct_df_clone_label's frame, one row per spot")
def _(run: Run) -> None:
    frame = run("05_baf/construct_df_clone_label/out")
    merged = np.asarray(run("05_baf/merge_by_minspots/out")[1]["new_assignment"])
    barcodes = np.asarray(run("00_inputs/load_input_data/out/1")).astype(str)
    assert len(frame) == barcodes.size
    assert np.array_equal(frame.loc[barcodes, "clone_label"].to_numpy(), merged)


@row(BOOKKEEPING, "normal candidates: all inside one BAF clone")
def _(run: Run) -> None:
    candidate = np.asarray(run("06_normal/determine_normal_candidates/out"))
    merged = np.asarray(run("05_baf/merge_by_minspots/out")[1]["new_assignment"])
    assert candidate.any() and np.unique(merged[candidate]).size == 1


@row(BOOKKEEPING, "normal filter: kept rows are the bins' rows, re-ranked 0..K'-1; sum(lengths) == K'")
def _(run: Run) -> None:
    before = run.frame("04_bins/create_bin_ranges")
    after, counts = run("06_normal/normal_baf_bin_filter/out")
    kept = np.unique(before.loc[after["bin_id"].notna(), "bin_id"].to_numpy(dtype=np.int64))
    assert np.array_equal(counts.X, run("04_bins/summarize_counts_for_bins/out/X")[kept])
    _ids_run(after, "bin_id")
    assert counts.lengths.sum() == kept.size


@row(BOOKKEEPING, "rebin: bin_id runs 0..K-1 in row order; sum(lengths) == K == rows of X; removed genes stay removed")
def _(run: Run) -> None:
    frame = run.frame("07_rebin/create_bin_ranges")
    ids = _ids_run(frame, "bin_id")
    counts = run("07_rebin/summarize_counts_for_bins/out")
    assert counts.lengths.sum() == ids.max() + 1 == counts.X.shape[0]
    kept = run("06_normal/normal_baf_bin_filter/out/0")["bin_id"].notna()
    assert frame.loc[~kept.to_numpy(), "bin_id"].isna().all()


@row(BOOKKEEPING, "baseline: rdr_normal sums to 1, 0 exactly where the normal count < min_normal_count_perbin")
def _(run: Run) -> None:
    rdr_normal = np.asarray(run("07_rebin/determine_normal_baseline/out/0"))
    rdr = run("07_rebin/determine_normal_baseline/in/args/0")
    candidate = np.asarray(run("06_normal/determine_normal_candidates/out"))
    low = rdr[:, candidate].sum(axis=1) < run.config["quality"]["min_normal_count_perbin"]
    assert np.isclose(rdr_normal.sum(), 1.0) and np.array_equal(rdr_normal == 0, low)


@row(BOOKKEEPING, "baseline: base_nb_mean is rank one, rdr_normal times each spot's kept UMIs")
def _(run: Run) -> None:
    rdr_normal, rdr, base = (run(f"07_rebin/determine_normal_baseline/out/{k}") for k in range(3))
    assert np.allclose(base.sum(axis=0), rdr.sum(axis=0)) and np.allclose(base.sum(axis=1) / base.sum(), rdr_normal)


@row(BOOKKEEPING, "rdr init: each initial RDR clone lies inside one merged BAF clone")
def _(run: Run) -> None:
    init, allowed, total = run("08_rdr/initialize_rdr_clone_refininement/out")
    merged = np.asarray(run("05_baf/merge_by_minspots/out")[1]["new_assignment"])
    for c in range(total):
        assert np.unique(merged[np.asarray(init) == c]).size == 1
    assert np.all(np.asarray(allowed)[np.arange(len(init)), np.asarray(init)])


@row(BOOKKEEPING, "rdr fit: labels 0..M-1; pred_cnv == argmax log_gamma per clone; parameter columns shared")
def _(run: Run) -> None:
    res = run("08_rdr/run_core_inference/out")
    labels = np.asarray(res["new_assignment"])
    _labels_dense(labels)
    log_gamma = np.asarray(res["log_gamma"])
    assert np.array_equal(np.asarray(res["pred_cnv"]), np.argmax(log_gamma, axis=0))
    assert log_gamma.shape[2] == labels.max() + 1
    n_states = run.config["hmm"]["n_states"]
    assert np.asarray(res["new_log_mu"]).shape == (n_states, 1) and np.asarray(res["new_p_binom"]).shape == (n_states, 1)


@row(BOOKKEEPING, "rdr merge: every clone >= min_spots_per_clone, labels 0..M-1")
def _(run: Run) -> None:
    _, merged = run("08_rdr/merge_by_minspots/out")
    labels = np.asarray(merged["new_assignment"])
    _labels_dense(labels)
    assert np.bincount(labels).min() >= run.config["hmrf"]["min_spots_per_clone"]


@row(BOOKKEEPING, "reindex: a permutation; clone 0 has the least BAF deviation from 1/2")
def _(run: Run) -> None:
    before = np.asarray(run("08_rdr/merge_by_minspots/out")[1]["new_assignment"])
    res = run("08_rdr/reindex_clones/out")[0]
    after = np.asarray(res["new_assignment"])
    assert adjusted_rand_score(before, after) == 1.0 and np.unique(after).size == np.unique(before).size
    p = np.asarray(res["new_p_binom"])[:, 0][np.asarray(res["pred_cnv"])]
    deviation = np.maximum(np.abs(p - 0.5) - 0.05, 0).sum(axis=0)
    assert np.argmin(deviation) == 0


@row(BOOKKEEPING, "integer copies: pairs nonnegative and A+B <= 6; (1, 1) among the states under fixed diploid")
def _(run: Run) -> None:
    for stage in (s for s in run.sim.stages if "integer_copy" in s):
        copies = np.asarray(run(f"{stage}/out/0"))
        assert copies.shape[1] == 2 and np.all(copies >= 0) and np.all(copies.sum(axis=1) <= 6)
        assert np.any(np.all(copies == 1, axis=1)), stage


@row(BOOKKEEPING, "outputs: cnv_seglevel.tsv has one row per re-binned bin; clone_labels.tsv one per spot")
def _(run: Run) -> None:
    seglevel = pd.read_csv(pd.io.common.BytesIO(run.sim.file("cnv_seglevel.tsv")), sep="\t")
    assert len(seglevel) == run("07_rebin/summarize_counts_for_bins/out/X").shape[0]
    labels = pd.read_csv(pd.io.common.BytesIO(run.sim.file("clone_labels.tsv")), sep="\t")
    assert len(labels) == np.asarray(run("00_inputs/load_input_data/out/1")).size


@pytest.mark.parametrize("name", list(BOOKKEEPING))
def test_bookkeeping(run: Run, name: str) -> None:
    """One predicate a stage's output satisfies on any data."""
    BOOKKEEPING[name](run)


# --- science against the planted truth --------------------------------------


def fitted_at_truth(run: Run, truth: Any, labels: np.ndarray) -> np.ndarray:
    """`labels` (in the run's spot order) read at the truth's barcodes; -1 where the run dropped a spot."""
    barcodes = np.asarray(run("00_inputs/load_input_data/out/1")).astype(str)
    return pd.Series(labels, index=barcodes).reindex(truth.barcodes, fill_value=-1).to_numpy()


def final_labels(run: Run) -> np.ndarray:
    return np.asarray(run("08_rdr/reindex_clones/out/0")["new_assignment"])


def copy_scores(run: Run, truth: Any) -> dict[str, float]:
    """score_sample's copy rows: planted pairs at each seglevel midpoint, per matched clone."""
    seglevel = pd.read_csv(pd.io.common.BytesIO(run.sim.file("cnv_seglevel.tsv")), sep="\t")
    fitted = fitted_at_truth(run, truth, final_labels(run))
    scored = fitted >= 0
    clone_of = matched(overlap(truth.labels[scored], fitted[scored], truth.n_clones, int(fitted.max()) + 1))
    middle = ((seglevel["START"] + seglevel["END"]) // 2).to_numpy()
    planted = truth.copies_at(seglevel["CHR"].astype(str).to_numpy(), middle)
    covered = planted[:, 0, 0] >= 0
    t, ab = [], []
    for clone, fit in clone_of.items():
        t.append(planted[covered, clone, 0] * 1_000 + planted[covered, clone, 1])
        ab.append(seglevel[f"clone{fit} A"].to_numpy()[covered] * 1_000 + seglevel[f"clone{fit} B"].to_numpy()[covered])
    t_all, ab_all = np.concatenate(t), np.concatenate(ab)
    altered = t_all != NEUTRAL
    either = (t_all == ab_all) | (t_all == swapped(ab_all))
    return {
        "copy_ari_pf": float(adjusted_rand_score(phase_free(t_all), phase_free(ab_all))),
        "exact_altered_minor": float(np.mean(either[altered])),
    }


def baf_ari(run: Run, truth: Any) -> float:
    merged = np.asarray(run("05_baf/merge_by_minspots/out")[1]["new_assignment"])
    fitted = fitted_at_truth(run, truth, merged)
    return float(adjusted_rand_score(truth.labels[fitted >= 0], fitted[fitted >= 0]))


def rdr_ari(run: Run, truth: Any) -> float:
    fitted = fitted_at_truth(run, truth, final_labels(run))
    return float(adjusted_rand_score(truth.labels[fitted >= 0], fitted[fitted >= 0]))


def normal_purity(run: Run, truth: Any) -> float:
    candidate = np.asarray(run("06_normal/determine_normal_candidates/out"))
    planted = fitted_at_truth(run, truth, np.arange(candidate.size))
    labels = np.full(candidate.size, -1)
    labels[planted[planted >= 0]] = truth.labels[planted >= 0]
    return float(np.mean(labels[candidate] == 0))


def state_recovery(run: Run, truth: Any) -> dict[str, float]:
    """|fitted - planted| per clone-bin, mean: BAF p against B/(A+B), mu against (A+B)/2, at matched clones."""
    res = run("08_rdr/reindex_clones/out/0")
    seglevel = pd.read_csv(pd.io.common.BytesIO(run.sim.file("cnv_seglevel.tsv")), sep="\t")
    fitted = fitted_at_truth(run, truth, final_labels(run))
    scored = fitted >= 0
    clone_of = matched(overlap(truth.labels[scored], fitted[scored], truth.n_clones, int(fitted.max()) + 1))
    middle = ((seglevel["START"] + seglevel["END"]) // 2).to_numpy()
    planted = truth.copies_at(seglevel["CHR"].astype(str).to_numpy(), middle)
    covered = planted[:, 0, 0] >= 0
    pred = np.asarray(res["pred_cnv"])
    p = np.asarray(res["new_p_binom"])[:, 0]
    mu = np.exp(np.asarray(res["new_log_mu"])[:, 0])
    dp, dmu = [], []
    for clone, fit in clone_of.items():
        a, b = planted[covered, clone, 0], planted[covered, clone, 1]
        state = pred[covered, fit]
        minor = np.minimum(p[state], 1 - p[state])
        dp.append(np.abs(minor - np.minimum(a, b) / np.maximum(a + b, 1)))
        dmu.append(np.abs(mu[state] - (a + b) / 2))
    return {"p_error": float(np.mean(np.concatenate(dp))), "mu_error": float(np.mean(np.concatenate(dmu)))}


SCIENCE: dict[str, tuple[Callable[[Run, Any], float], str, float, Any]] = {
    "BAF clone ARI (recorded 0.7225)": (baf_ari, ">=", 0.7025, None),
    "RDR clone ARI (recorded 0.9694, the ledger's)": (rdr_ari, ">=", 0.9494, None),
    "copy ARI, phase-free (recorded 0.8216, the ledger's)": (lambda r, t: copy_scores(r, t)["copy_ari_pf"], ">=", 0.8016, None),
    "exact altered, phase-free (recorded 0.3794, the ledger's)": (lambda r, t: copy_scores(r, t)["exact_altered_minor"], ">=", 0.3594, None),
    "normal-candidate purity (recorded 1.0000)": (normal_purity, ">=", 0.98, None),
    "BAF p recovery, mean |minor p - planted| (recorded 0.0037)": (lambda r, t: state_recovery(r, t)["p_error"], "<=", 0.0237, None),
    "mu recovery, mean |mu - (A+B)/2| (recorded 0.0664)": (lambda r, t: state_recovery(r, t)["mu_error"], "<=", 0.0864, None),
}
"""Name -> (score, direction, bound, xfail reason or None). Each bound is the recorded value less (or plus) `MARGIN`.
The planted defects these scores would show on other fixtures (Ticket#320's impure normal candidates) are `test_defects.py`'s."""

MARGIN = 0.02


@pytest.mark.parametrize("name", list(SCIENCE))
def test_science(run: Run, truth: Any, name: str, request: pytest.FixtureRequest) -> None:
    """A recovery of the planted truth, against its floor (or ceiling)."""
    score, direction, bound, defect = SCIENCE[name]
    if defect is not None:
        request.applymarker(pytest.mark.xfail(strict=True, reason=defect))
    value = score(run, truth)
    print(f"{name}: {value:.4f}")
    assert value >= bound if direction == ">=" else value <= bound


# --- config criteria -----------------------------------------------------------


class Check(NamedTuple):
    ok: Any
    """Per item: the criterion holds."""
    allowed: Any = False
    """Per item: a stated exception."""
    known: Any = False
    """Per item: a failure by the mechanism the row's xfail names."""


def flags(*values: Any) -> np.ndarray:
    return np.asarray(values, dtype=bool)


def phase_breaks(u: Units) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each BAF break (a phase-segment start inside a contig): the largest minor-BAF step over clones, and the size of
    the segment it closes."""
    res, _, refined = u.replayed.value("03_phasing/initial_phase_given_partition/out")
    lengths = np.asarray(u.sim.stored("02_blocks/summarize_counts_for_blocks/out/lengths"))
    minor = minor_baf(res, u.setting("hmm.n_states"), int(lengths.sum()))
    starts = np.cumsum(refined)[:-1]
    at_baf = ~np.isin(starts, np.cumsum(lengths))
    steps = np.abs(minor[:, starts] - minor[:, starts - 1]).max(axis=0)
    return steps[at_baf], np.asarray(refined)[:-1][at_baf], starts[at_baf]


def first_bins(u: Units) -> Bins:
    _, _, refined = u.replayed.value("03_phasing/initial_phase_given_partition/out")
    return bins_of(u, "bins", "blocks", np.asarray(refined), normal=False)


def rebins(u: Units) -> Bins:
    return bins_of(u, "rebinned", "kept_bins", np.asarray(u.sim.stored("06_normal/normal_baf_bin_filter/out/1/lengths")), normal=True)


def floor(binning: Callable[[Units], Bins], key: str) -> Callable[[Units, Any], Check]:
    def measure(u: Units, value: Any) -> Check:
        b = binning(u)
        return Check(b.floors[key][0] >= value, b.alone, b.backed)
    return measure


def length(binning: Callable[[Units], Bins]) -> Callable[[Units, Any], Check]:
    def measure(u: Units, value: Any) -> Check:
        b = binning(u)
        return Check(b.span < value, (b.stop - b.first == 1) | b.head_short, b.ends_run)
    return measure


def raw_inputs(u: Units) -> tuple[Any, np.ndarray, np.ndarray]:
    """The committed sample as the loader reads it: the AnnData, and A and B (spots, SNPs) in its barcode order."""
    sample = u.replayed.staged.root / "inputs" / Path(u.sim.config["sample"]).name
    adata = anndata.read_h5ad(sample / "filtered_feature_bc_matrix.h5ad")
    barcodes = (sample / "barcodes.txt").read_text().split()
    order = pd.Index(barcodes).get_indexer(adata.obs_names)
    assert np.all(order >= 0)
    a, b = (sp.load_npz(sample / f"cell_snp_{x}allele.npz").tocsr()[order] for x in "AB")
    return adata, a, b


def spots_kept(u: Units, value: Any, *, snp: bool) -> Check:
    adata, a, b = raw_inputs(u)
    total = np.asarray((a.sum(axis=1) + b.sum(axis=1)) if snp else adata.X.sum(axis=1)).ravel()
    kept = np.isin(adata.obs_names, np.asarray(u.sim.stored("00_inputs/load_input_data/out/1")).astype(str))
    return Check((total >= value) == kept)


def genes_kept(u: Units, value: Any) -> Check:
    adata, _, _ = raw_inputs(u)
    out = u.replayed.value("00_inputs/load_input_data/out/2")
    spots = np.isin(adata.obs_names, out.obs_names)
    expressed = np.asarray((adata.X[spots] > 0).sum(axis=0)).ravel() >= value * spots.sum()
    listed = np.isin(adata.var_names, pd.read_csv(u.setting("references.filtergenelist_file"), header=None).iloc[:, 0])
    return Check((expressed & ~listed) == np.isin(adata.var_names, out.var_names))


def outliers_zeroed(u: Units, value: Any) -> Check:
    """The loader's own gene sums (its `X`, which the zeroing leaves alone) put through the same LocalOutlierFactor."""
    out = u.replayed.value("00_inputs/load_input_data/out/2")
    sums = np.asarray(out.X.sum(axis=0)).reshape(-1, 1)
    outlier = LocalOutlierFactor(n_neighbors=200).fit_predict(sums) == -1 if value else np.zeros(sums.shape[0], dtype=bool)
    zeroed = (np.asarray(out.layers["count"].sum(axis=0)).ravel() == 0) & (sums.ravel() > 0)
    print(f"genes zeroed as outliers: {int(zeroed.sum())}")
    return Check(outlier == zeroed)


def reads(function: Any, name: str) -> bool:
    """`function` reads `name` outside an f-string."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    inside = {id(n) for f in ast.walk(tree) if isinstance(f, ast.JoinedStr) for n in ast.walk(f)}
    return any(isinstance(n, ast.Name) and n.id == name and id(n) not in inside for n in ast.walk(tree))


def kept_bin_ids(u: Units) -> np.ndarray:
    bins, kept = u.lineage.levels["bins"], u.lineage.levels["kept_bins"]
    return np.unique(bins.label[kept.label != DROPPED])


def fits(u: Units) -> list[dict[str, Any]]:
    return [u.sim.result(f"{run}/run_core_inference/out") for run in ("05_baf", "08_rdr")]


def reproduces_under_another_global_seed(u: Units, stage: str) -> bool:
    """`stage` on its captured input, the global generators set elsewhere: its config seed alone fixes its output."""
    given = u.replayed._rewritten(copy.deepcopy(u.replayed.value(f"{stage}/in")))
    set_global_config(u.replayed.staged.config)
    np.random.seed(20_251_010)  # noqa: NPY002
    random.seed(20_251_010)
    out = u.sim.function(stage)(*given["args"], **given["kwargs"])
    return bool(digest(out) == u.sim.digest_of(f"{stage}/out"))


def header(u: Units, name: str) -> list[str]:
    return list(pd.read_csv(io.BytesIO(u.sim.file(name)), sep="\t", comment="#", nrows=0).columns)


def level_extents(u: Units, level: str) -> Check:
    path, key = FRAMES[level]
    frame = u.sim.stored(path)
    g = frame[frame[key].notna()].groupby(key).agg(CHR=("CHR", "first"), START=("START", "min"), END=("END", "max"))
    same = g["CHR"].to_numpy()[1:] == g["CHR"].to_numpy()[:-1]
    ordered = (np.diff(g["CHR"].to_numpy()) > 0) | (same & (g["END"].to_numpy()[:-1] <= g["START"].to_numpy()[1:]))
    return Check(ordered)


class Criterion(NamedTuple):
    stage: str
    level: str
    key: str | None
    quantity: str
    op: str
    exceptions: str
    measure: Callable[[Units, Any], Check]
    departs: str | None = None
    """A strict xfail on `RuleBroken`: the ticket, or `new:`, and the mechanism."""


BACKED = "new: greedy_binning_nobreak backs off one unit when a bin meeting its floors reaches max_binlength, leaving it under a floor (omics.py:66-73)"
TAIL = "new: a short run tail merged into the bin before it takes that bin past max_binlength (omics.py:85-98)"
CAPTURE = "the capture holds no fit trace: needs scripts/capture_cnamaste.py's fit-trace extension (scratchpad/s872), re-captured"


def traced(u: Units, key: str, *, split: bool = False) -> list[Any]:
    """`/internal/<run>`'s per-iteration `key` for both HMRF fits, flat or per run; a skip where the capture holds none."""
    found = [u.sim.internal(run).get(key) for run in ("05_baf", "08_rdr")]
    if any(f is None for f in found):
        pytest.skip(CAPTURE)
    return found if split else [x for f in found for x in f]


def outer_stop(u: Units, value: Any) -> Check:
    ari = traced(u, "ari", split=True)
    longest = u.setting("hmrf.max_iter_outer") + 1
    return Check(flags(*(len(a) == longest or max(a[:-2], default=0.0) >= value for a in ari)))


def llf_rises(u: Units) -> Check:
    ok = []
    for llf in traced(u, "total_llf", split=True):
        before = np.asarray(llf[:-2])
        ok += list(np.diff(before) >= -1e-6 * np.abs(before[1:]))
    return Check(flags(*ok))

CRITERIA = [
    Criterion("02_blocks", "blocks", "quality.phasing_min_snp_umis", "SNP-covering UMI", ">=", "a contig's last block",
              lambda u, v: Check(u.per_unit("blocks", "snp_umi") >= v, u.lineage.levels["blocks"].boundary)),
    Criterion("03_phasing", "phase segments", "phasing.baf_change_threshold", "largest minor-BAF step at a BAF break", ">=", "none",
              lambda u, v: Check(phase_breaks(u)[0] >= v)),
    Criterion("03_phasing", "phase segments", "phasing.min_new_segment_size", "blocks in the segment a BAF break closes", ">", "none",
              lambda u, v: Check(phase_breaks(u)[1] > v)),
    *(Criterion(stage, level, key, quantity, ">=", "its run's only bin", floor(binning, key), departs)
      for stage, level, binning in (("04_bins", "bins", first_bins), ("07_rebin", "re-binned", rebins))
      for key, quantity, departs in (
          ("quality.secondary_min_umi", "total UMI", None),
          ("quality.secondary_min_snp_umi", "SNP-covering UMI", BACKED),
          *((("quality.secondary_min_normal_umi", "normal-candidate UMI", BACKED),) if level == "re-binned" else ()))),
    Criterion("04_bins", "bins", "quality.max_binlength", "summed block span", "<", "one block, or its head under a floor",
              length(first_bins), TAIL),
    Criterion("07_rebin", "re-binned", "quality.max_binlength", "summed kept-bin span", "<", "one unit, or its head under a floor",
              length(rebins)),
    Criterion("06_normal", "kept bins", "quality.normal_allele_specific_confidence", "pooled normal-spot B count", "in interval",
              "none", lambda u, v: Check(~normal_baf_outside(u)[kept_bin_ids(u)])),
    Criterion("00_inputs", "spots", "quality.spot_min_snp_umis", "raw total UMI, kept iff >= (io.py:657-660)", "==", "none",
              lambda u, v: spots_kept(u, v, snp=False)),
    Criterion("00_inputs", "spots", "quality.spot_min_snp_umis", "raw SNP-covering UMI, kept iff >= (io.py:666-670)", "==", "none",
              lambda u, v: spots_kept(u, v, snp=True)),
    Criterion("00_inputs", "genes", "quality.min_percent_expressed_spots", "spots expressing, a fraction (Ticket#182), kept iff >= and "
              "not in references.filtergenelist_file (io.py:697-732)", "==", "none", genes_kept),
    Criterion("00_inputs", "genes", "quality.local_outlier_filter", "LocalOutlierFactor(200) on gene sums, zeroed iff outlier "
              "(io.py:781-836)", "==", "none", outliers_zeroed),
    Criterion("05_baf, 08_rdr", "fit", "hmm.tol", "the Baum-Welch stop reads tol", "==", "none",
              lambda u, v: Check(flags(reads(nophasing.hmm_nophasing._run_optimization_pipeline, "tol")), known=True),
              "new: hmm.tol only enters a log line (hmm_nophasing.py:1120); scipy.optimize.minimize stops on its literal "
              "gtol 1e-5 or max_iter (hmm_nophasing.py:1005-1010)"),
    Criterion("05_baf, 08_rdr", "fit", "hmm.max_iter", "each M-step optimizer converged, or ran max_iter iterations", "==", "none",
              lambda u, v: Check(flags(*(o["success"] or o["nit"] >= v for o in traced(u, "optimizer"))))),
    Criterion("05_baf, 08_rdr", "fit", "hmrf.ari_tolerance", "ARI of consecutive assignments reaches it before the merge rounds "
              "(hmrf.py:706-720), or the loop ran max_iter_outer + 1 iterations", ">=", "none", lambda u, v: outer_stop(u, v)),
    Criterion("05_baf, 08_rdr", "M step", "hmm.em_ftol", "the M-step optimizer's ftol is hmm.em_ftol", "==", "none",
              lambda u, v: Check(flags(reads(nophasing.hmm_nophasing._run_optimization_pipeline, "get_em_solver_params")), known=True),
              "new: the M step passes the literal ftol 1e-6 to BFGS, which does not read ftol (hmm_nophasing.py:1005-1033)"),
    Criterion("05_baf, 08_rdr", "M step", "hmm.em_ftol", "the last M step's final relative objective change", "<=", "none",
              lambda u, v: Check(flags(*(o[-1]["last_relative_change"] <= float(v) for o in traced(u, "optimizer", split=True))))),
    Criterion("05_baf, 08_rdr", "fit", None, "total_llf non-decreasing over the outer iterations before the merge rounds, "
              "to 1e-6 relative (Ticket#30, Ticket#146)", ">=", "none", lambda u, v: llf_rises(u)),
    Criterion("06_normal", "M step", "hmm.solver", "em_xtol and em_xrtol reach the solver (hmm_utils.py:23-40)", "==", "none",
              lambda u, v: (set_global_config(u.replayed.staged.config),
                            Check(flags(*(k in get_em_solver_params() for k in ("xtol", "xrtol"))), known=True))[1],
              "new: get_em_solver_params reads maxiter, ftol and disp for L-BFGS-B; hmm.em_xtol and hmm.em_xrtol are unread"),
    Criterion("05_baf, 08_rdr", "fit", "hmm.n_states", "rows of log_mu and p_binom; phasing log_gamma 2x", "==", "none",
              lambda u, v: Check(flags(*(np.asarray(f[k]).shape[0] == v for f in fits(u) for k in ("new_log_mu", "new_p_binom")),
                                       np.asarray(u.replayed.value("03_phasing/initial_phase_given_partition/out/0/log_gamma")).shape[0] == 2 * v))),
    Criterion("05_baf, 08_rdr", "fit", "hmm.t", "transmat: diagonal log t, off-diagonal log (1-t)/(K-1), never updated (Ticket#146)",
              "==", "none", lambda u, v: Check(flags(*(np.allclose(f["new_log_transmat"], get_log_transmat(u.setting("hmm.n_states"), v))
                                                         for f in fits(u))))),
    Criterion("05_baf, 08_rdr", "fit", None, "exp(log_mu) <= max_rdr, the hmm_nophasing default 5.0 (hmm_nophasing.py:808); "
              "hmm_initialize's is inf", "<=", "none", lambda u, v: Check(np.concatenate([np.exp(f["new_log_mu"]).ravel() for f in fits(u)]) <= 5.0)),
    Criterion("05_baf, 08_rdr", "fit", None, "alphas in [1e-6, 1e3], taus in [1e-4, 5e3] (hmm_nophasing.py:510-557; not passed: "
              "bounds=None, hmm_nophasing.py:1033)", "in", "none",
              lambda u, v: Check(np.concatenate([((f["new_alphas"] >= 1e-6) & (f["new_alphas"] <= 1e3)).ravel() for f in fits(u)]
                                                + [((f["new_taus"] >= 1e-4) & (f["new_taus"] <= 5e3)).ravel() for f in fits(u)]))),
    Criterion("03_phasing", "clones", "hmrf.n_clones", "initial BAF clones", "==", "none",
              lambda u, v: Check(flags(len(u.sim.stored("03_phasing/initialize_clones/out")) == v), known=True),
              "new: hmrf.n_clones only names the output directory (utils.py:245); the BAF round starts from "
              "phasing.npart_phasing^2 rectangles (run_cnamaste.py:250-256, 276)"),
    Criterion("08_rdr", "clones", "hmrf.n_clones_rdr", "initial RDR clones == merged BAF clones x n_clones_rdr", "==", "none",
              lambda u, v: Check(flags(np.unique(u.sim.stored("08_rdr/initialize_rdr_clone_refininement/out/0")).size
                                       == len(u.sim.stored("05_baf/merge_by_minspots/out/0")) * v))),
    Criterion("08_rdr", "clones", None, "final clones <= the last iteration's start less its Potts merges, less empties, "
              "less min-spot merges", "<=", "none", lambda u, v: clone_drops(u)),
    Criterion("08_rdr", "clones", "hmrf.min_spots_per_clone", "spots per final clone (effective floor 200: Ticket#81, Ticket#468)",
              ">=", "none", lambda u, v: Check(np.bincount(np.asarray(u.sim.result("08_rdr/reindex_clones/out/0")["new_assignment"])) >= v)),
    Criterion("05_baf, 08_rdr", "clones", "hmrf.spatial_weight", "the HMRF fits' spatial_weight", "==", "none",
              lambda u, v: Check(flags(*(u.replayed.value(f"{run}/run_core_inference/in")["kwargs"]["spatial_weight"] == v
                                         for run in ("05_baf", "08_rdr"))))),
    Criterion("09_outputs", "integer copies", "int_copy_num.max_total_copy", "A+B, and each allele <= max_allele_copy", "<=",
              "none", lambda u, v: Check(flags(False), known=True),
              "Ticket#788: max_total_copy and max_allele_copy are not in the config; the code fixes 6 and 5 (integer_copy.py:106-107)"),
    Criterion("09_outputs", "integer copies", "int_copy_num.ploidy", "cnv{medfix}_{genelevel,seglevel,perstate}.tsv per ploidy pass "
              "(run_cnamaste.py:1316-1579)", "==", "none", lambda u, v: ploidy_files(u, v),
              "new: the configured ploidy pass writes no files: its clones yield no state copies and the pass `continue`s "
              "(run_cnamaste.py:1508-1510)"),
    Criterion("09_outputs", "integer copies", "int_copy_num.nonbalance_bafdist", "the (1, 1) state's |p - 1/2| (binds above 0.5 never)",
              "<=", "none", lambda u, v: Check(normal_state(u)[0] <= v)),
    Criterion("09_outputs", "integer copies", "int_copy_num.nondiploid_rdrdist", "the (1, 1) state's |mu - 1|", "<=", "none",
              lambda u, v: Check(normal_state(u)[1] <= v)),
    Criterion("09_outputs", "clone_labels.tsv", None, "barcodes unique, sample_id one of the sample sheet's", "==", "none",
              lambda u, v: clone_label_samples(u),
              "new: construct_df_clone_label reads sample_id after the barcode's last underscore (io.py:50); a one-slice "
              "run's barcodes carry no sample suffix, so `spot_N` gives N"),
    Criterion("09_outputs", "cnv_genelevel.tsv", None, "gene rows unique", "==", "none",
              lambda u, v: Check(~pd.read_csv(io.BytesIO(u.sim.file("cnv_genelevel.tsv")), sep="\t", index_col=0).index.duplicated(),
                                 known=True),
              "Ticket#105, new: a gene the reference carries twice is written 2^C times (test_defects.py)"),
    Criterion("04_bins", "SNPs", None, "each snp_id once, in one block and one bin", "==", "none",
              lambda u, v: Check(flags(u.sim.stored(FRAMES["bins"][0])["snp_id"].dropna().is_unique))),
    Criterion("02_blocks..07_rebin", "every level", None, "units sorted by (CHR, START), none overlapping", "==", "none",
              lambda u, v: Check(np.concatenate([level_extents(u, level).ok for level in FRAMES]))),
    Criterion("06_normal", "binned_gene_snp", None, "gene and SNP lists in genomic order, whatever PYTHONHASHSEED", "==", "none",
              lambda u, v: binned_order(u),
              "Ticket#871: binned_gene_snp joins Python sets, so its lists follow PYTHONHASHSEED (omics.py:242-280)"),
    Criterion("09_outputs", "cnv tables", None, "clone{c} columns unique and the same in seglevel, genelevel and perstate", "==", "none",
              lambda u, v: clone_columns(u)),
    Criterion("02_blocks..07_rebin", "transmat", "phasing.min_prob", "log switch >= log(min_prob) (recomb.py:60-66, Ticket#172)",
              ">=", "none", lambda u, v: Check(np.concatenate([np.asarray(u.sim.stored(f"{s}/get_sitewise_transmat/out"))
                                                               for s in ("02_blocks", "04_bins", "06_normal", "07_rebin")]) >= np.log(v) - 1e-12)),
    Criterion("03_phasing", "seeds", "hmm.gmm_random_state", "initialize_clones and the phasing fit reproduce under another global "
              "generator state", "==", "none",
              lambda u, v: Check(flags(*(reproduces_under_another_global_seed(u, f"03_phasing/{s}")
                                         for s in ("initialize_clones", "initial_phase_given_partition"))))),
    Criterion("09_outputs", "paths", "paths.output_dir", "every written table under paths.output_dir", "==", "none",
              lambda u, v: Check(flags(*(str(u.sim.stored(f"{s}/in")["args"][0]).startswith(v)
                                         for s in json.loads(u.sim.config["files"]).values())))),
    Criterion("06_normal", "paths", "paths.perf_path", "a fit's perf record lands at paths.perf_path", "==", "none",
              lambda u, v: (u.replayed.run("06_normal/normal_baf_bin_filter"),
                            Check(flags(Path(u.replayed.staged.config.paths.perf_path).exists()), known=True))[1],
              "new: flush_perf appends to the literal `cnamaste.perf` in the working directory (hmm_emission.py:130)"),
]
"""(stage, level, `/config` key, quantity, comparison, stated exceptions): each read from `/config` at test time and
checked on quantities recomputed from the staged inputs (`audit.criteria`), never the stage's own counts."""


def clone_drops(u: Units) -> Check:
    internal = u.sim.internal("08_rdr")
    start, merges = np.unique(internal["prev"]).size, len(internal["merges"])
    raw = np.unique(internal["raw"]).size
    groups = len(u.sim.stored("08_rdr/merge_by_minspots/out/0"))
    final = np.unique(u.sim.result("08_rdr/reindex_clones/out/0")["new_assignment"]).size
    print(f"RDR clones: {start} at the last iteration, {merges} Potts merges, {raw} after empties, {groups} after min spots, {final} final")
    return Check(flags(raw <= start - merges, final == groups <= raw, final <= start))


def ploidy_files(u: Units, value: Any) -> Check:
    written = set(json.loads(u.sim.config["files"]))
    medfix = ["", *str(value).split(",")]
    return Check(flags(*(f"cnv{m}_{k}.tsv" in written for m in medfix for k in ("genelevel", "seglevel", "perstate"))),
                 known=flags(*(m != "" for m in medfix for _ in range(3))))


def normal_state(u: Units) -> tuple[np.ndarray, np.ndarray]:
    """Per final clone: the (1, 1) state's |p - 1/2| and |mu - 1|, from the integer decoder's inputs."""
    baf, rdr = [], []
    for stage in (s for s in u.sim.stages if "integer_copy" in s):
        log_mu, _, p, _ = u.replayed.value(f"{stage}/in")["args"]
        copies = np.asarray(u.sim.stored(f"{stage}/out/0"))
        normal = np.all(copies == 1, axis=1)
        baf.append(np.abs(np.asarray(p)[normal] - 0.5))
        rdr.append(np.abs(np.exp(np.asarray(log_mu)[normal]) - 1))
    return np.concatenate(baf), np.concatenate(rdr)


def clone_label_samples(u: Units) -> Check:
    labels = pd.read_csv(io.BytesIO(u.sim.file("clone_labels.tsv")), sep="\t", comment="#")
    sheet = np.asarray(u.replayed.value("00_inputs/get_sample_list/out/0")).astype(str)
    return Check(flags(labels["barcode"].is_unique) & np.isin(labels["sample_id"].astype(str), sheet),
                 known=~np.isin(labels["sample_id"].astype(str), sheet))


def binned_order(u: Units) -> Check:
    table = u.replayed.value("06_normal/binned_gene_snp/out")
    frame = u.sim.stored(FRAMES["kept_bins"][0])
    ok = []
    for column, key in (("INCLUDED_SNP_IDS", "snp_id"), ("INCLUDED_GENES", "gene")):
        order = {name: i for i, name in enumerate(frame[key].dropna().drop_duplicates())}
        ok += [names == sorted(names, key=order.__getitem__) for names in (str(x).split(",") for x in table[column]) if len(names) > 1]
    return Check(flags(*ok), known=True)


def clone_columns(u: Units) -> Check:
    found = [sorted({c.split(" ")[0] for c in header(u, f"cnv_{k}.tsv") if c.startswith("clone")})
             for k in ("seglevel", "genelevel", "perstate")]
    return Check(flags(found[0] == found[1] == found[2], len(set(found[0])) == len(found[0])))


@pytest.mark.parametrize("row", CRITERIA, ids=[f"{c.level}: {c.key or c.quantity}" for c in CRITERIA])
def test_a_config_criterion_holds(units: Units, request: pytest.FixtureRequest, row: Criterion) -> None:
    """The row's quantity, against the run's own `/config` value, holds for every item but its stated exceptions."""
    if row.departs:
        request.applymarker(pytest.mark.xfail(strict=True, reason=row.departs, raises=RuleBroken))
    value = units.setting(row.key) if row.key and not row.key.startswith("int_copy_num.max_") else None
    check = row.measure(units, value)
    ok, allowed, known = (np.broadcast_to(np.asarray(x, dtype=bool), np.shape(check.ok)) for x in check)
    outside = ~ok & ~allowed
    print(f"{row.level}: {row.quantity} {row.op} {row.key}={value}: {ok.size} checked, "
          f"{int((~ok & allowed).sum())} stated exceptions, {int(outside.sum())} outside them")
    assert not np.any(outside & ~known), f"{int((outside & ~known).sum())} items fail outside the stated exceptions"
    if np.any(outside):
        raise RuleBroken(f"{int(outside.sum())} items fail by the named mechanism")
