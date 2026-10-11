"""Per-function rows for the genome segmentation: the gene-SNP table, blocks, bins, their counts, and the counts container.

Inputs: the stage frames as stored, the counts and AnnData as replayed
(`load_input_data`'s), restricted to the first contig where a recount is per
gene; else a synthetic table with a closed form.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import anndata
import numpy as np
import pandas as pd
import pytest
from cnamaste.omics import (
    binned_gene_snp,
    create_bin_ranges,
    greedy_binning_nobreak,
    summarize_blocks,
)
from cnamaste.reference import get_reference_genes
from cnamaste.spatio_genomic_counts import SpatioGenomicCounts
from cnamaste.utils import cacher

from audit.capture import grch38
from audit.fn import Replay, Row, deep, run, table, unchanged

LOADED = "00_inputs/load_input_data/out"


def loaded(ctx: Any) -> dict[str, Any]:
    """The replayed AnnData, allele matrices and SNP ids."""
    return {"adata": ctx(f"{LOADED}/2"), "a": ctx(f"{LOADED}/4"), "b": ctx(f"{LOADED}/5"), "ids": np.asarray(ctx(f"{LOADED}/6"))}


# --- oracle --------------------------------------------------------------------


def _greedy(_: Any) -> None:
    lengths = np.full(7, 10)
    umi = np.array([5, 5, 5, 5, 5, 5, 1])
    snp = np.array([1, 3, 0, 4, 2, 2, 0])
    bins = greedy_binning_nobreak(lengths, umi, snp, np.zeros(7), 0, 4, 0, max_binlength=1_000)
    # NB [0,1] has 4 SNP UMIs, [2,3] 4, [4,5] 4, and the tail [6] (0) fails and joins the previous bin
    assert bins.tolist() == [0, 0, 1, 1, 2, 2, 2]
    every = greedy_binning_nobreak(lengths, umi, snp, np.zeros(7), 0, 0, 0, max_binlength=25)
    assert every.tolist() == list(range(7)), "with no floor each block is a bin"


def _blocks_counts(d: dict[str, Any]) -> None:
    frame, counts = d["frame"], d["counts"]
    snps = frame[frame["snp_id"].notna()]
    where = {s: i for i, s in enumerate(d["ids"])}
    first = snps["block_id"].to_numpy() < 50
    for block, group in snps[first].groupby("block_id"):
        idx = [where[s] for s in group["snp_id"]]
        assert np.array_equal(counts.X[block, 1], d["a"][:, idx].sum(axis=1)), "the block's second component is the A-allele sum"
        assert np.array_equal(counts.total_bb_RD[block], (d["a"][:, idx] + d["b"][:, idx]).sum(axis=1))
    genes = frame[frame["is_interval"] & (frame["block_id"] < 50)]
    names = d["adata"].var.index
    for block, group in genes.groupby("block_id"):
        cols = np.isin(names, group["gene"].unique())
        assert np.array_equal(counts.X[block, 0], np.asarray(d["adata"].layers["count"])[:, cols].sum(axis=1))


def _bins_counts(d: dict[str, Any]) -> None:
    frame, counts = d["frame"], d["counts"]
    names = d["adata"].var.index
    genes = frame[frame["is_interval"] & frame["bin_id"].notna() & (frame["bin_id"] < 80)]
    for b, group in genes.groupby("bin_id"):
        cols = np.isin(names, group["gene"].unique())
        assert np.array_equal(counts.X[int(b), 0], np.asarray(d["adata"].layers["count"])[:, cols].sum(axis=1)), f"bin {b}'s UMIs are its genes' counts"


def _blocks_in(ctx: Any) -> dict[str, Any]:
    return loaded(ctx) | {"frame": ctx.sim.stored("02_blocks/assign_initial_blocks/out"), "counts": ctx("02_blocks/summarize_counts_for_blocks/out")}


def _bins_in(stage: str) -> Any:
    def build(ctx: Any) -> dict[str, Any]:
        return loaded(ctx) | {"frame": ctx.sim.stored(f"{stage}/create_bin_ranges/out"), "counts": ctx(f"{stage}/summarize_counts_for_bins/out")}
    return build


def _binned(frame: pd.DataFrame) -> None:
    out = unchanged(binned_gene_snp, frame)
    kept = frame[frame["bin_id"].notna()]
    groups = kept.groupby("bin_id")
    assert out["bin_id"].tolist() == sorted(kept["bin_id"].unique().tolist())
    assert np.array_equal(out["CHR"], groups["CHR"].first()) and np.array_equal(out["START"], groups["START"].first()) and np.array_equal(out["END"], groups["END"].last())
    assert np.array_equal(out["N_SNPS"], groups["snp_id"].nunique()), "N_SNPS counts distinct SNP ids"
    for (_, g), joined in zip(groups, out["INCLUDED_GENES"], strict=True):
        assert set(joined.split(",")) == set(g["gene"].dropna()), "the joined genes are the bin's genes (as a set: Ticket#871)"


ORACLE: list[Row] = table(
    "oracle",
    ("omics:greedy_binning_nobreak", "synthetic: 7 blocks", lambda c: None, _greedy, "the shortest runs meeting the SNP-UMI floor; a failing tail joins the previous bin; no floor, one block per bin"),
    ("omics:summarize_counts_for_blocks", Replay("02_blocks out, load_input_data out (replayed), blocks 0-49"), _blocks_in, _blocks_counts, "A-allele and A+B sums over each block's SNPs; UMIs over its genes"),
    ("omics:summarize_counts_for_bins", Replay("04_bins out, load_input_data out (replayed), bins 0-79"), _bins_in("04_bins"), _bins_counts, "a bin's UMIs are its genes' counts in the AnnData"),
    ("omics:summarize_counts_for_bins", Replay("07_rebin out, load_input_data out (replayed), bins 0-79"), _bins_in("07_rebin"), _bins_counts, "after re-binning too"),
    ("omics:binned_gene_snp", "07_rebin/create_bin_ranges out", lambda c: c.sim.stored("07_rebin/create_bin_ranges/out"), _binned,
     "per bin: first CHR and START, last END, distinct SNPs, its genes; no input mutation"),
)


# --- invariants ---------------------------------------------------------------------


def _table(frame: pd.DataFrame) -> None:
    genes = frame[frame["is_interval"]]
    snps = frame[~frame["is_interval"]]
    assert genes["snp_id"].isna().all() and snps["snp_id"].notna().all()
    spans = genes.groupby(["CHR", "gene"]).agg(start=("START", "min"), end=("END", "max"))
    keyed = spans.reindex(pd.MultiIndex.from_arrays([snps["CHR"], snps["gene"]])).to_numpy()
    assert np.all((keyed[:, 0] <= snps["START"].to_numpy()) & (snps["START"].to_numpy() < keyed[:, 1])), "every SNP lies inside a gene it is assigned"


def _reference(ctx: Any) -> pd.DataFrame:
    resources = grch38()
    if resources is None:
        pytest.skip("CalicoST's GRCh38_resources not found")
    return ctx.once("omics/reference_genes", lambda: get_reference_genes(str(resources / "hgTables_hg38_gencode.txt")))


def _reference_rows(genes: pd.DataFrame) -> None:
    assert set(genes["CHR"].unique()) == set(range(1, 23)) and genes["is_interval"].all() and genes["snp_id"].isna().all()
    assert np.all(genes["END"] >= genes["START"])


def _blocks(d: dict[str, Any]) -> None:
    frame, total, floor = d["frame"], d["total"], d["floor"]
    ids = frame["block_id"].to_numpy()
    snp_umis = total.sum(axis=1)
    contig = frame.groupby("block_id")["CHR"].first().to_numpy()
    last = np.r_[contig[1:] != contig[:-1], True]
    first = np.r_[True, contig[1:] != contig[:-1]]
    under = snp_umis < floor
    assert np.all(~under | last | first), "only a contig's first or last block falls below the SNP-UMI floor"
    assert np.all(np.diff(ids) >= 0)


def _blocks_in_total(ctx: Any) -> dict[str, Any]:
    return {"frame": ctx.sim.stored("02_blocks/assign_initial_blocks/out"), "total": ctx("02_blocks/summarize_counts_for_blocks/out/total_bb_RD"),
            "floor": ctx.config.quality.phasing_min_snp_umis}


def _bins_respect(ctx: Any) -> dict[str, Any]:
    given = ctx("04_bins/create_bin_ranges/in")
    return {"in": given, "out": ctx.sim.stored("04_bins/create_bin_ranges/out")}


def _bins(d: dict[str, Any]) -> None:
    args, kwargs = d["in"]["args"], d["in"]["kwargs"]
    frame, total, refined = args[0], args[6], np.asarray(args[7])
    out = create_bin_ranges(deep(frame), *args[1:], **kwargs)
    pd.testing.assert_frame_equal(out, d["out"])
    of = out.groupby("block_id")["bin_id"].first().to_numpy()
    segment = np.searchsorted(np.cumsum(refined), np.arange(len(of)), side="right")
    assert (pd.Series(segment).groupby(of).nunique() == 1).all(), "no bin crosses a phase-refined segment"
    spans = frame.groupby("block_id").agg(start=("START", "first"), end=("END", "last"))
    long = np.flatnonzero((spans["end"] - spans["start"]).to_numpy() > kwargs["max_binlength"])
    cuts = np.unique(np.concatenate([np.cumsum(refined), long, long + 1]))
    piece = np.searchsorted(cuts, np.arange(len(of)), side="right")  # NB the runs the greedy binner sees
    snp = pd.Series(total.sum(axis=1)).groupby(of).sum().to_numpy()
    pieces = pd.Series(piece).groupby(of).first()
    lonely = pieces.map(pieces.value_counts()).to_numpy() == 1
    blen = (spans["end"] - spans["start"]).to_numpy()
    last = pd.Series(np.arange(len(of))).groupby(of).max().to_numpy()
    following = np.append(blen, 0)[np.minimum(last + 1, len(blen))]
    capped = pd.Series(blen).groupby(of).sum().to_numpy() + following >= kwargs["max_binlength"]
    assert np.all((snp >= args[9]) | lonely | capped), "a bin below the SNP-UMI floor is its run's only bin, or stopped at max_binlength"


def _bins_pure(d: dict[str, Any]) -> None:
    # NB copies: the input is the replay's cached value, and the call this row expects to fail
    #    (Ticket#189) writes bin_id into it, which every later reader of the stage would inherit.
    args, kwargs = deep(d["in"]["args"]), deep(d["in"]["kwargs"])
    unchanged(create_bin_ranges, *args, **kwargs)


def _summary(_: Any) -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger("cnamaste.omics")
    logger.addHandler(handler)
    frame = pd.DataFrame({"CHR": [1, 1, 1], "START": [0, 5, 20], "END": [10, 6, 30], "snp_id": [None, "1_5", None], "gene": ["g0", "g0", "g1"],
                          "is_interval": [True, False, True], "block_id": [0, 0, 1]})
    adata = anndata.AnnData(np.ones((2, 2)), var=pd.DataFrame(index=["g0", "g1"]))
    adata.layers["count"] = np.array([[3, 1], [4, 2]])
    try:
        assert summarize_blocks(frame, adata, np.array([[2], [5]]), np.array([[1], [1]]), np.array(["1_5"]), block_key="block_id") is None
    finally:
        logger.removeHandler(handler)
    text = stream.getvalue()
    assert "total blocks: 2" in text and "total umis: 10" in text and "total snp-umis: 9" in text


def _counts_container(_: Any) -> None:
    x = np.arange(12).reshape(2, 2, 3)
    c = SpatioGenomicCounts(np.array([1, 1]), x, np.zeros((2, 3)), np.ones((2, 3), dtype=int) * 20)
    assert (c.n_segments, c.n_spots) == (2, 3) and c.keys() == ["lengths", "X", "base_nb_mean", "total_bb_RD"]
    lengths, x2, base, total = c
    assert x2 is x and c.values()[1] is x and "n_spots" not in str(c) and "X:" in str(c)
    with pytest.raises(ValueError):
        SpatioGenomicCounts(np.array([3]), x, np.zeros((2, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        SpatioGenomicCounts(np.array([2]), x, np.zeros((2, 4)), np.zeros((2, 3)))


def _cached(tmp: Any) -> None:
    calls = []

    @cacher("probe.tsv")
    def make() -> pd.DataFrame:
        calls.append(1)
        return pd.DataFrame({"snp_id": [None, "1_5"], "x": [1, 2]})

    first, second = make(), make()
    assert len(calls) == 1 and second.equals(first), "the second call reads the cache"
    assert first["snp_id"].isna().tolist() == [True, False], "a None survives the cache"


def _cache_config(ctx: Any) -> Any:
    config = ctx.config
    config.run.cache = True
    config.paths.output_dir = str(ctx.tmp_path)
    return ctx.tmp_path


INVARIANT: list[Row] = table(
    "invariant",
    ("omics:form_gene_snp_table", "01_genes/form_gene_snp_table out", lambda c: c.sim.stored("01_genes/form_gene_snp_table/out"), _table, "gene rows carry no SNP id; every SNP lies inside its gene's span"),
    ("reference:get_reference_genes", "CalicoST hgTables", _reference, _reference_rows, "chr1-22 as integers; gene rows only"),
    ("omics:assign_initial_blocks", Replay("02_blocks out and summarize_counts_for_blocks (replayed)"), _blocks_in_total, _blocks, "blocks ascend; only a contig's end block falls under phasing_min_snp_umis"),
    ("omics:create_bin_ranges", Replay("04_bins/create_bin_ranges in (replayed)"), _bins_respect, _bins, "recomputed, the run's frame; no bin crosses a refined segment; under-floor bins are a run's only bin or length-capped"),
    ("omics:create_bin_ranges", Replay("04_bins/create_bin_ranges in (replayed)"), _bins_respect, _bins_pure, "the caller's gene table is not rewritten",
     "Ticket#189: Inherited defect: assign_initial_blocks writes by positional column and is not re-entrant (create_bin_ranges adds bin_id to its caller's frame the same way)"),
    ("omics:summarize_blocks", "synthetic: 2 blocks", lambda c: None, _summary, "returns None; logs block, UMI and SNP-UMI totals (Ticket#191: the result is discarded)"),
    ("spatio_genomic_counts:SpatioGenomicCounts.__post_init__", "synthetic", lambda c: None, _counts_container, "validate refuses lengths or shapes that disagree with X"),
    ("spatio_genomic_counts:SpatioGenomicCounts.validate", "synthetic", lambda c: None, _counts_container, "sum(lengths) == segments; base and depth (segments, spots)"),
    ("spatio_genomic_counts:SpatioGenomicCounts.n_segments", "synthetic", lambda c: None, _counts_container, "X's first axis"),
    ("spatio_genomic_counts:SpatioGenomicCounts.n_spots", "synthetic", lambda c: None, _counts_container, "X's last axis"),
    ("spatio_genomic_counts:SpatioGenomicCounts.__iter__", "synthetic", lambda c: None, _counts_container, "unpacks as (lengths, X, base, depth), by identity"),
    ("spatio_genomic_counts:SpatioGenomicCounts.keys", "synthetic", lambda c: None, _counts_container, "the four field names in order"),
    ("spatio_genomic_counts:SpatioGenomicCounts.values", "synthetic", lambda c: None, _counts_container, "the four fields in order"),
    ("spatio_genomic_counts:SpatioGenomicCounts.__str__", "synthetic", lambda c: None, _counts_container, "names each field"),
    ("spatio_genomic_counts:LockableMixin.__setattr__", "synthetic", lambda c: None, _counts_container, "an unlocked container takes writes"),
    ("utils:cacher", "synthetic: a cached two-row table, run.cache on", _cache_config, _cached, "the second call reads the cache, and the cached table equals the computed one",
     "new: cacher's .tsv round trip turns None into '' (keep_default_na=False), and the first call returns the reloaded table: with run.cache on, form_gene_snp_table's gene rows carry snp_id ''"),
)


@pytest.mark.parametrize("row", ORACLE, ids=[r.id for r in ORACLE])
def test_oracle(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """Counts recounted from the AnnData and allele matrices; greedy binning by hand."""
    run(row, ctx, request)


@pytest.mark.parametrize("row", INVARIANT, ids=[r.id for r in INVARIANT])
def test_invariant(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """Spans, floors, segment boundaries, container checks, the cache."""
    run(row, ctx, request)
