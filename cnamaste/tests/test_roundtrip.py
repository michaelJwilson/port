"""The run's input files against its output files, through the recorded intermediates and maps; no planted truth.

- **Forward:** the committed inputs, aggregated at every segment level, are
  each stage's counts; what a filter removes is accounted for; every input
  barcode reaches `clone_labels.tsv`; each clone's pseudobulk is the sum of
  its spots; the fitted parameters, through the states and integer pairs,
  are the written tables.
- **Inverse:** each clone-bin's observed pseudobulk counts lie inside the
  central 95% of the negative binomial (UMIs) and beta-binomial (B allele)
  the output states predict, at a pinned share; a removed bin's genes are
  absent from the output.
- **Files:** every written TSV reloads to the frame the run wrote, and
  writing that frame again gives the same bytes.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from audit.capture import input_files
from audit.segments import DROPPED

pytestmark = pytest.mark.roundtrip


@pytest.fixture(scope="session")
def loaded(replayed: Any) -> dict[str, Any]:
    """The input files as `load_input_data` reads them (replayed; the files' sha256 checked by `staged`): it renames each
    SNP `<chr>_<pos>_<n>` and drops spots and SNPs under its floors, so the stages index its arrays, not the files'."""
    out = [replayed.value(f"00_inputs/load_input_data/out/{k}") for k in range(7)]
    return {"barcodes": np.asarray(out[1]).astype(str), "A": np.asarray(out[4]), "B": np.asarray(out[5]), "ids": np.asarray(out[6]).astype(str)}


def snp_rows(frame: pd.DataFrame, ids: np.ndarray, key: str) -> tuple[np.ndarray, np.ndarray]:
    """Each SNP row's column in the allele matrices, and its `key`."""
    snps = frame[frame["snp_id"].notna() & frame[key].notna()]
    column = pd.Index(ids).get_indexer(snps["snp_id"].to_numpy())
    assert np.all(column >= 0)
    return column, snps[key].to_numpy(dtype=np.int64)


def test_the_loaded_inputs_are_the_files(sample: Path, loaded: dict[str, Any]) -> None:
    """Every loaded barcode is one of `barcodes.txt`'s, and the allele totals are the files' over the loaded spots and SNPs."""
    files = input_files(sample)
    import gzip

    raw = files["barcodes.txt"].read_bytes()
    barcodes = (gzip.decompress(raw) if files["barcodes.txt"].name.endswith(".gz") else raw).decode().split()
    assert set(loaded["barcodes"]) <= set(barcodes)
    a = sp.load_npz(files["cell_snp_Aallele.npz"])
    print(f"loaded {loaded['barcodes'].size} of {len(barcodes)} spots, {loaded['ids'].size} of {a.shape[1]} SNPs")
    assert loaded["A"].sum() <= a.sum()


def test_block_allele_counts_are_the_input_snps_summed(sim: Any, replayed: Any, loaded: dict[str, Any]) -> None:
    """Blocks: A and A+B per block and spot are the loaded allele matrices' SNP columns summed over each block's SNP rows."""
    column, block = snp_rows(sim.stored("02_blocks/assign_initial_blocks/out"), loaded["ids"], "block_id")
    counts = replayed.value("02_blocks/summarize_counts_for_blocks/out")
    group = sp.csr_matrix((np.ones(column.size, dtype=np.int64), (block, column)), shape=(counts.X.shape[0], loaded["A"].shape[1]))
    a = np.asarray(group @ loaded["A"].T)
    depth = a + np.asarray(group @ loaded["B"].T)
    assert np.array_equal(counts.X[:, 1, :], a) and np.array_equal(counts.total_bb_RD, depth)


def test_filtered_mass_is_accounted_for(replayed: Any, sim: Any) -> None:
    """The normal-BAF filter: kept + removed bins' UMIs and depths are the bins'. The baseline: zeroed UMIs are the low bins'."""
    bins = replayed.value("04_bins/summarize_counts_for_bins/out")
    kept = replayed.value("06_normal/normal_baf_bin_filter/out/1")
    before = sim.stored("04_bins/create_bin_ranges/out")["bin_id"].to_numpy(dtype=np.int64)
    after = sim.stored("06_normal/normal_baf_bin_filter/out/0")["bin_id"].notna().to_numpy()
    removed = np.setdiff1d(np.arange(bins.X.shape[0]), before[after])
    assert bins.X[:, 0].sum() == kept.X[:, 0].sum() + bins.X[removed, 0].sum()
    assert bins.total_bb_RD.sum() == kept.total_bb_RD.sum() + bins.total_bb_RD[removed].sum()
    print(f"normal-BAF filter: {removed.size} of {bins.X.shape[0]} bins, {bins.X[removed, 0].sum() / bins.X[:, 0].sum():.4f} of UMIs")
    rdr_in = replayed.value("07_rebin/determine_normal_baseline/in/args/0")
    rdr_normal, rdr_out = replayed.value("07_rebin/determine_normal_baseline/out/0"), replayed.value("07_rebin/determine_normal_baseline/out/1")
    assert rdr_in.sum() == rdr_out.sum() + rdr_in[np.asarray(rdr_normal) == 0].sum()


def test_every_loaded_barcode_reaches_the_labels(sim: Any, loaded: dict[str, Any]) -> None:
    """`clone_labels.tsv` holds each barcode the run loaded, once."""
    written = pd.read_csv(io.BytesIO(sim.file("clone_labels.tsv")), sep="\t", comment="#")
    assert written["barcode"].is_unique
    assert set(written["barcode"]) == set(loaded["barcodes"])


def pseudobulk(sim: Any, replayed: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Each final clone's summed UMIs, B counts, depth and baseline, from the RDR fit's spot-level input."""
    given = replayed.value("08_rdr/run_core_inference/in")
    x, base, depth = given["args"][0], given["args"][2], given["args"][3]
    labels = np.asarray(sim.result("08_rdr/reindex_clones/out/0")["new_assignment"])
    onehot = np.eye(labels.max() + 1)[labels]
    return x[:, 0, :] @ onehot, x[:, 1, :] @ onehot, depth @ onehot, base @ onehot


def test_each_clones_pseudobulk_is_its_spots_summed(sim: Any, replayed: Any) -> None:
    """The baseline the integer decoder is handed for clone c is the sum over clone c's spots."""
    _, _, _, base = pseudobulk(sim, replayed)
    for c in range(base.shape[1]):
        handed = np.asarray(sim.stored(f"09_outputs/integer_copy_{c}/in/args/1"))
        assert np.allclose(handed, base[:, c], rtol=1e-12, atol=0)


def test_parameters_to_the_written_seglevel(sim: Any) -> None:
    """`cnv_seglevel.tsv`, rebuilt from the bins, the final states, mu, p and the integer pairs, is the file."""
    res = sim.result("08_rdr/reindex_clones/out/0")
    pred = np.asarray(res["pred_cnv"])
    written = pd.read_csv(io.BytesIO(sim.file("cnv_seglevel.tsv")), sep="\t")
    for c in range(pred.shape[1]):
        pairs = np.asarray(sim.stored(f"09_outputs/integer_copy_{c}/out/0"))[pred[:, c]]
        rebuilt = np.column_stack([pred[:, c], np.asarray(res["new_log_mu"])[pred[:, c], 0], np.asarray(res["new_p_binom"])[pred[:, c], 0], pairs])
        stated = written[[f"clone{c} {k}" for k in ("Z", "logmu", "p", "A", "B")]].to_numpy()
        # NB to_csv prints 16 significant digits: the last ulp of logmu and p is the writer's
        assert np.allclose(rebuilt, stated, rtol=1e-15, atol=1e-15)


INTERVAL = 0.95
SHARES = {"umi": 0.8861, "baf": 0.9446}
"""The least share of clone-bins whose observed count lies in the central `INTERVAL` predicted: the recorded share
(UMIs 0.9061, B alleles 0.9646, of 6,728 and 6,744 clone-bins on easy) less 0.02."""


def test_observed_counts_fall_inside_the_predicted_intervals(sim: Any, replayed: Any) -> None:
    """UMIs ~ NB(mean = base * mu[state], dispersion alpha); B ~ BetaBinom(depth, p[state] tau, (1 - p) tau)."""
    from scipy import stats

    res = sim.result("08_rdr/reindex_clones/out/0")
    pred = np.asarray(res["pred_cnv"])
    umi, b, depth, base = pseudobulk(sim, replayed)
    mu, alpha = np.exp(np.asarray(res["new_log_mu"])[:, 0]), np.asarray(res["new_alphas"])[:, 0]
    p, tau = np.asarray(res["new_p_binom"])[:, 0], np.asarray(res["new_taus"])[:, 0]
    mean = base * mu[pred]
    n = 1.0 / alpha[pred]
    low, high = stats.nbinom.interval(INTERVAL, n, n / (n + mean))
    usable = mean > 0
    inside_umi = ((umi >= low) & (umi <= high))[usable]
    a_, b_ = p[pred] * tau[pred], (1 - p[pred]) * tau[pred]
    lo_b, hi_b = stats.betabinom.interval(INTERVAL, depth.astype(np.int64), a_, b_)
    covered = depth > 0
    inside_baf = ((b >= lo_b) & (b <= hi_b))[covered]
    shares = {"umi": float(inside_umi.mean()), "baf": float(inside_baf.mean())}
    print(f"inside the central {INTERVAL}: {shares} of {usable.sum()} / {covered.sum()} clone-bins")
    assert shares["umi"] >= SHARES["umi"] and shares["baf"] >= SHARES["baf"]


def test_a_removed_bins_genes_are_absent(sim: Any, lineage: Any) -> None:
    """A gene whose bin the normal-BAF filter removed has no row in `cnv_genelevel.tsv` (#105, fixed in cnamaste)."""
    genelevel = pd.read_csv(io.BytesIO(sim.file("cnv_genelevel.tsv")), sep="\t", index_col=0)
    frame = sim.stored("02_blocks/assign_initial_blocks/out")
    names = frame["gene"].to_numpy()[lineage.genes.row]
    removed = lineage.levels["kept_bins"].label == DROPPED
    assert removed.any()
    assert not set(names[removed]) & set(genelevel.index) - set(names[~removed])
    # NB the rows, not their count: a name the reference carries twice is repeated 2^C times by the
    #    per-clone joins (test_defects.py).
    assert set(genelevel.index) == set(names[~removed])


def written_tsvs(sim: Any) -> list[str]:
    return [s for s in sim.stages if "write_tsv" in s]


def test_every_written_tsv_reloads_to_its_frame(sim: Any) -> None:
    """Each output TSV, parsed, is the frame `write_tsv` was handed: same columns, same values."""
    for stage in written_tsvs(sim):
        given = sim.stored(f"{stage}/in")
        frame, kwargs = given["args"][1], given["kwargs"]
        reloaded = pd.read_csv(io.BytesIO(sim.file(sim.h5[stage].attrs["file"])), sep="\t", index_col=0 if kwargs.get("index") else None)
        assert list(reloaded.columns) == [str(c) for c in frame.columns]
        assert list(map(str, reloaded.index)) == list(map(str, frame.index)) if kwargs.get("index") else True
        for name, column in frame.items():
            found = reloaded[str(name)]
            if column.dtype == object:
                assert list(found.astype(str)) == list(column.astype(str)), (stage, name)
            else:
                assert np.allclose(found.to_numpy(dtype=float), column.to_numpy(dtype=float), rtol=1e-15, atol=1e-15, equal_nan=True), (stage, name)


def test_rewriting_a_tsv_is_byte_stable(sim: Any, tmp_path: Path) -> None:
    """`write_tsv` on the recorded frame writes the run's bytes again."""
    from cnamaste.utils import write_tsv

    for stage in written_tsvs(sim):
        given = sim.stored(f"{stage}/in")
        target = tmp_path / sim.h5[stage].attrs["file"]
        write_tsv(str(target), *given["args"][1:], **given["kwargs"])
        assert target.read_bytes() == sim.file(target.name), stage
