"""Index lookups through the run's merges, re-indexings and filters, each checked two ways.

Every array the run carries is indexed by something an earlier step chose:
a spot by its row, a segment by its rank among the survivors of a filter, a
clone by the order a merge or `reindex_clones` left, a state by `k` or
`k + C` for its phase, a gene's call by its bin. A lookup that silently
reads the wrong row produces a plausible number; each test here follows one
lookup through the recorded stages and checks it against an independent
route to the same answer.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
import pytest

from audit.scoring import matched, overlap
from audit.segments import DROPPED
from cnamaste.hmm_nophasing import get_log_transmat, hmm_nophasing


def table(sim: Any, name: str, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(sim.file(name)), sep="\t", **kwargs)


def test_barcode_to_row(sim: Any, truth: Any) -> None:
    """Row i of every per-spot array is barcode i: its coordinates are the truth file's and `clone_labels.tsv`'s."""
    barcodes = np.asarray(sim.stored("00_inputs/load_input_data/out/1")).astype(str)
    coords = np.asarray(sim.stored("00_inputs/load_input_data/out/0"))
    written = table(sim, "clone_labels.tsv", comment="#", index_col="barcode")
    assert np.array_equal(written.loc[barcodes, ["x", "y"]].to_numpy(), coords)
    order = pd.Index(truth.barcodes).get_indexer(barcodes)
    assert np.all(order >= 0), "a run barcode the truth does not hold"


LEVELS = {
    "blocks": "02_blocks/summarize_counts_for_blocks/out/X",
    "bins": "04_bins/summarize_counts_for_bins/out/X",
    "kept_bins": "06_normal/normal_baf_bin_filter/out/1/X",
    "rebinned": "07_rebin/summarize_counts_for_bins/out/X",
}


@pytest.mark.parametrize("level", list(LEVELS))
def test_segment_label_to_count_row(replayed: Any, lineage: Any, gene_counts: np.ndarray, level: str) -> None:
    """Row k of the level's UMI counts is the sum of the genes the lineage labels k: `aggregate` of the input counts."""
    expected = lineage.levels[level].aggregate(gene_counts)
    assert np.array_equal(replayed.value(LEVELS[level])[:, 0, :], expected)


def test_bin_removal_re_ranks_the_survivors(sim: Any) -> None:
    """A kept bin's new id is its rank among the kept, in the old order; a removed bin has none."""
    before = sim.stored("04_bins/create_bin_ranges/out")["bin_id"].to_numpy(dtype=np.int64)
    after = sim.stored("06_normal/normal_baf_bin_filter/out/0")["bin_id"]
    kept = np.unique(before[after.notna().to_numpy()])
    rank = np.full(before.max() + 1, -1)
    rank[kept] = np.arange(kept.size)
    assert np.array_equal(rank[before], after.fillna(-1).to_numpy(dtype=np.int64))


@pytest.mark.parametrize("run", ["05_baf", "08_rdr"])
def test_the_empty_clone_reindex_is_dense_and_order_keeping(sim: Any, run: str) -> None:
    """The re-indexing after a clone empties maps the surviving raw labels, in order, to 0..M-1."""
    internal = sim.internal(run)
    mapping = {int(k): int(v) for k, v in internal["re_indexing"].items()}
    assert list(mapping.values()) == list(range(len(mapping)))
    assert list(mapping) == sorted(mapping) == sorted(np.unique(internal["raw"]).tolist())
    print(f"{run}: raw labels {sorted(mapping)}, merges {internal['merges']}, iterations {internal['iterations']}")


def posterior_states(sim: Any, replayed: Any, clone: int) -> tuple[np.ndarray, np.ndarray]:
    """The final fit's states for `clone`, recomputed: its pseudobulk from the spot counts, decoded with the final parameters."""
    res = sim.result("08_rdr/reindex_clones/out/0")
    given = replayed.value("08_rdr/run_core_inference/in")
    single_x, lengths, base, depth = (given["args"][i] for i in range(4))
    spots = np.asarray(res["new_assignment"]) == clone
    x = single_x[:, :, spots].sum(axis=2, keepdims=True)
    b = base[:, spots].sum(axis=1, keepdims=True)
    d = depth[:, spots].sum(axis=1, keepdims=True)
    rdr, baf = hmm_nophasing.compute_emission_probability_nb_betabinom(
        x, b, res["new_log_mu"], res["new_alphas"], d, res["new_p_binom"], res["new_taus"]
    )
    emission = rdr + baf
    n_states = emission.shape[0]
    config = replayed.staged.config
    log_gamma = hmm_nophasing().get_state_posteriors(
        np.asarray(lengths), get_log_transmat(n_states, config.hmm.t), np.asarray(res["new_log_startprob"]), emission,
        np.zeros(emission.shape[1]),
    )
    return np.argmax(log_gamma, axis=0), np.asarray(res["pred_cnv"])[:, clone]


AGREEMENT = 1.0
"""The share of bins at which the recomputed decode is the stored `pred_cnv`, for every final clone: all of them on easy."""


def test_clone_to_parameter_column(sim: Any, replayed: Any) -> None:
    """Clone c's column: the integer decoder is handed clone c's pseudobulk and `pred_cnv[:, c]`, and decoding that pseudobulk
    with the shared parameters gives `pred_cnv[:, c]` at the recorded share of bins.

    `pred_cnv` is the posterior of the outer loop's last Baum-Welch, on the clones *before* its last ICM sweep, while the
    parameters are the refit after it (`hmrf.py`): the two need not agree, and on easy they agree at every bin.
    """
    res = sim.result("08_rdr/reindex_clones/out/0")
    n_clones = np.unique(res["new_assignment"]).size
    shares = []
    for c in range(n_clones):
        given = sim.stored(f"09_outputs/integer_copy_{c}/in")
        assert np.array_equal(np.asarray(given["args"][3]), np.asarray(res["pred_cnv"])[:, c])
        state, stored = posterior_states(sim, replayed, c)
        shares.append(float(np.mean(state == stored)))
    print(f"decode agreement per clone: {np.round(shares, 4)}")
    assert min(shares) >= AGREEMENT


def test_merge_groups_pick_the_parameter_columns(sim: Any) -> None:
    """`merge_by_minspots` keeps each group's first clone's states: merged column i is the fit's column `groups[i][0]`."""
    groups, merged = sim.stored("08_rdr/merge_by_minspots/out")
    fit = sim.stored("08_rdr/run_core_inference/out")
    assert np.array_equal(np.asarray(merged["pred_cnv"]), np.asarray(fit["pred_cnv"])[:, [g[0] for g in groups]])


def test_reindex_keeps_each_spots_decoded_profile(sim: Any) -> None:
    """`reindex_clones` permutes labels and columns together: every spot's decoded states are unchanged."""
    _, before = sim.stored("08_rdr/merge_by_minspots/out")
    after = sim.result("08_rdr/reindex_clones/out/0")
    pred_before = np.asarray(before["pred_cnv"])[:, np.asarray(before["new_assignment"])]
    pred_after = np.asarray(after["pred_cnv"])[:, np.asarray(after["new_assignment"])]
    assert np.array_equal(pred_before, pred_after)


def test_a_baf_state_reads_its_p_and_the_mirrored_branch_is_unreached(sim: Any) -> None:
    """The profiles handed to the normal search read state k's p at each bin: `run_cnamaste` maps a state `k + C` to
    `1 - p[k]`, but the BAF fit is `hmm_nophasing` (`run_cnamaste.py:574`), whose C states never reach `k + C`."""
    _, merged = sim.stored("05_baf/merge_by_minspots/out")
    c = np.asarray(merged["new_p_binom"]).shape[0]
    log_gamma = np.asarray(merged["log_gamma"])
    assert log_gamma.shape[0] == c
    n_clones = np.unique(merged["new_assignment"]).size
    pred = np.argmax(log_gamma, axis=0).reshape(n_clones, -1)
    p = np.asarray(merged["new_p_binom"])[:, 0]
    expected = np.where(pred < c, p[pred % c], 1 - p[pred % c])
    assert np.all(pred < c)
    assert np.array_equal(sim.stored("06_normal/determine_normal_candidates/in/args/2"), expected)


def test_state_to_pair_to_seglevel_to_genelevel(sim: Any, lineage: Any) -> None:
    """Clone c, bin b: state `pred_cnv[b, c]` -> its integer pair -> `cnv_seglevel.tsv` row b -> each gene of bin b in
    `cnv_genelevel.tsv`; the gene's row found by the lineage and by its coordinates agree."""
    res = sim.result("08_rdr/reindex_clones/out/0")
    pred = np.asarray(res["pred_cnv"])
    seglevel = table(sim, "cnv_seglevel.tsv")
    genelevel = table(sim, "cnv_genelevel.tsv", index_col=0)
    frame = sim.stored("02_blocks/assign_initial_blocks/out")
    rebinned = lineage.levels["rebinned"]
    names = frame["gene"].to_numpy()[lineage.genes.row]
    starts = frame["START"].to_numpy()[lineage.genes.row]
    called = rebinned.label != DROPPED
    by_coordinates = np.full(lineage.genes.n_genes, -1)
    for b, (chrom, start, end) in enumerate(seglevel[["CHR", "START", "END"]].itertuples(index=False)):
        inside = (lineage.genes.contig == chrom) & (starts >= start) & (starts <= end) & called
        by_coordinates[inside] = b
    assert np.array_equal(by_coordinates[called], rebinned.label[called])
    # NB one row per name: a name the reference carries twice is written 2^C times (test_defects.py)
    genelevel = genelevel[~genelevel.index.duplicated()]
    first = called & ~pd.Index(names).duplicated()
    for c in range(pred.shape[1]):
        pairs = np.asarray(sim.stored(f"09_outputs/integer_copy_{c}/out/0"))[pred[:, c]]
        assert np.array_equal(seglevel[[f"clone{c} A", f"clone{c} B"]].to_numpy(), pairs)
        gene_rows = genelevel[[f"clone{c} A", f"clone{c} B"]]
        assert list(gene_rows.index) == list(names[first])
        assert np.array_equal(gene_rows.to_numpy(), pairs[rebinned.label[first]])


def test_the_ploidy_tables_agree(sim: Any) -> None:
    """`cnv_perstate.tsv` is each clone's integer pair per state; `cnv_seglevel.tsv` its state, mu and p per bin."""
    res = sim.result("08_rdr/reindex_clones/out/0")
    pred = np.asarray(res["pred_cnv"])
    perstate, seglevel = table(sim, "cnv_perstate.tsv"), table(sim, "cnv_seglevel.tsv")
    for c in range(pred.shape[1]):
        pairs = np.asarray(sim.stored(f"09_outputs/integer_copy_{c}/out/0"))
        assert np.array_equal(perstate[[f"clone{c} A", f"clone{c} B"]].to_numpy(), pairs)
        assert np.array_equal(seglevel[f"clone{c} Z"].to_numpy(), pred[:, c])
        assert np.allclose(seglevel[f"clone{c} p"].to_numpy(), np.asarray(res["new_p_binom"])[pred[:, c], 0], rtol=1e-15, atol=1e-15)


def test_the_planted_clones_match_fitted_ones(sim: Any, truth: Any) -> None:
    """Hungarian matching of planted to fitted clones is one to one, and each pair holds most of the planted clone's spots."""
    barcodes = np.asarray(sim.stored("00_inputs/load_input_data/out/1")).astype(str)
    final = np.asarray(sim.result("08_rdr/reindex_clones/out/0")["new_assignment"])
    fitted = pd.Series(final, index=barcodes).reindex(truth.barcodes).to_numpy()
    counts = overlap(truth.labels, fitted.astype(np.int64), truth.n_clones, int(final.max()) + 1)
    pairs = matched(counts)
    assert len(set(pairs.values())) == len(pairs)
    shares = {p: counts[p, f] / counts[p].sum() for p, f in pairs.items()}
    print(f"planted -> fitted {pairs}, shares {shares}")
    assert min(shares.values()) > 0.5
