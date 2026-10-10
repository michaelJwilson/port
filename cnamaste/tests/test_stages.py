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
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score

from audit.scoring import NEUTRAL, matched, overlap, phase_free, swapped

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
        import yaml

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
    # NB symmetric it is not: a kNN rule (#180, test_defects.py)
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
    from scipy.special import logsumexp

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
The planted defects these scores would show on other fixtures (#320's impure normal candidates) are `test_defects.py`'s."""

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
