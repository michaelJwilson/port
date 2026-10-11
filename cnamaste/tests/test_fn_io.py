"""Per-function rows for loading and the run's scaffolding: readers, the input loader, configuration, output directories, logging.

Inputs: the staged sample (read by the `staged` fixture's paths) and the
replayed `load_input_data` output; else a few-line synthetic file with a
closed-form reading.
"""

from __future__ import annotations

import gzip
import io
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anndata
import numba
import numpy as np
import pandas as pd
import polars as pl
import pytest
import scipy.sparse as sp
import yaml
from cnamaste.annotation import assign_clone_ranges, load_clone_labels, load_clone_ranges
from cnamaste.config import YAMLConfig, get_global_config, set_global_config
from cnamaste.filter import get_filter_genes, get_filter_ranges
from cnamaste.he import get_he_image, join_tables_xy
from cnamaste.io import (
    get_aggregated_barcodes,
    get_alignments,
    get_barcodes,
    get_sample_list,
    get_sample_sheet,
    get_spaceranger_counts,
    get_spatial_positions,
    map_unique_snps_enum,
    read_tumor_prop,
)
from cnamaste.logger import RuntimeFormatter, RuntimePhaseFilter, SharedStateLogger, get_logger
from cnamaste.reference import exp_cancer_gene
from cnamaste.scripts.run_cnamaste import set_numba_seed
from cnamaste.utils import configure_output_dir, get_output_dir, pause

from cnamaste.audit.capture import RESOURCES, grch38
from fn import Replay, Row, run, table, unchanged

LOADED = Replay("00_inputs/load_input_data, replayed")


def captured_log(name: str, call: Any) -> str:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger(name)
    logger.addHandler(handler)
    try:
        call()
    finally:
        logger.removeHandler(handler)
    return stream.getvalue()


# --- oracle --------------------------------------------------------------------


def _snp_ids_number_repeats(_: Any) -> None:
    ids = np.array(["chr1_100_A_G", "1_200_C_T", "chr1_100_A_G", "2_5_N_N", "chr1_100_A_G"], dtype=object)
    assert map_unique_snps_enum(ids).tolist() == ["1_100_0", "1_200_0", "1_100_1", "2_5_0", "1_100_2"], "repeats numbered in order; chr dropped"


def _barcodes_first_extension(tmp: Path) -> None:
    (tmp / "barcodes.tsv.gz").write_bytes(gzip.compress(b"AAAC-1\nAAAG-1\n"))
    assert get_barcodes(str(tmp / "barcodes.txt"))["combined_barcode"].tolist() == ["AAAC-1", "AAAG-1"], "the .tsv.gz alternative is found first"
    with pytest.raises(RuntimeError):
        get_barcodes(str(tmp / "none" / "barcodes.txt"))


def _aggregated_carry_sample_id(tmp: Path) -> None:
    (tmp / "barcodes.txt").write_text("AAAC-1\nAAAG-1\n")
    found = get_aggregated_barcodes(str(tmp / "barcodes.txt"), known_sample_id="s1")
    assert found["barcode"].tolist() == ["AAAC-1", "AAAG-1"] and found["sample_id"].tolist() == ["s1", "s1"]


def _positions_in_tissue(tmp: Path) -> None:
    (tmp / "spatial").mkdir()
    (tmp / "spatial" / "tissue_positions_list.csv").write_text("a,1,0,0,10,20\nb,0,0,1,11,21\nc,1,1,0,12,22\n")
    kept = get_spatial_positions(str(tmp))
    assert kept["barcode"].tolist() == ["a", "c"] and kept["x"].tolist() == [0, 1]
    assert len(get_spatial_positions(str(tmp), filter_in_tissue=False)) == 3
    with pytest.raises(RuntimeError):
        get_spatial_positions(str(tmp / "missing"))


def _counts_integer_unique_genes(tmp: Path) -> None:
    x = sp.csr_matrix(np.array([[1.0, np.nan], [2.0, 3.0]]))
    adata = anndata.AnnData(x, obs=pd.DataFrame(index=["a", "b"]), var=pd.DataFrame(index=["g", "g"]))
    adata.write_h5ad(tmp / "filtered_feature_bc_matrix.h5ad")
    found = get_spaceranger_counts(str(tmp))
    assert found.layers["count"].tolist() == [[1, 0], [2, 3]] and found.layers["count"].dtype.kind == "i", "NaN counts become 0, integers"
    assert found.var_names.is_unique


def _sheet_requires_columns(tmp: Path) -> None:
    (tmp / "sheet.tsv").write_text("# comment\nbam sample_id spaceranger_dir snp_dir\nx.bam s1 /a /b\n")
    assert get_sample_sheet(str(tmp / "sheet.tsv"))["sample_id"].tolist() == ["s1"]
    (tmp / "bad.tsv").write_text("bam sample_id\nx.bam s1\n")
    with pytest.raises(AssertionError):
        get_sample_sheet(str(tmp / "bad.tsv"))


def _filters_parse_lines(tmp: Path) -> None:
    (tmp / "genes.txt").write_text("IGHA1\nIGHG1\n")
    assert get_filter_genes(tmp / "genes.txt").iloc[:, 0].tolist() == ["IGHA1", "IGHG1"]
    (tmp / "ranges.bed").write_text("chr6\t300\t400\nchr2\t10\t20\nchr6\t100\t200\n")
    ranges = get_filter_ranges(tmp / "ranges.bed")
    assert ranges[["Chr", "Start"]].values.tolist() == [[2, 10], [6, 100], [6, 300]], "chr dropped, sorted by (Chr, Start)"


def _cancer_genes_by_pattern(_: Any) -> None:
    assert [exp_cancer_gene(g) for g in ("MT-CO1", "RPL3", "B2M", "B2MX", "TP53", "FTH1")] == [True, True, True, False, False, True]


def _samples_in_order(_: Any) -> None:
    adata = anndata.AnnData(np.zeros((5, 1)), obs=pd.DataFrame({"sample": ["a", "a", "b", "b", "b"]}, index=list("vwxyz")))
    names, ids = get_sample_list(adata)
    assert names == ["a", "b"] and ids.tolist() == [0, 0, 1, 1, 1]


def _tumor_prop_by_barcode(tmp: Path) -> None:
    adata = anndata.AnnData(np.zeros((2, 1)), obs=pd.DataFrame(index=["a", "b"]))
    (tmp / "tp.tsv").write_text("barcode\tTumor\na\t0.25\nb\t0.75\n")
    config = YAMLConfig({"preprocessing": {"tumorprop_file": str(tmp / "tp.tsv")}})
    assert read_tumor_prop(adata, config=config).tolist() == [0.25, 0.75]
    assert read_tumor_prop(adata, config=YAMLConfig({"preprocessing": {"tumorprop_file": "None"}})) is None


def _clone_labels_scale_normal(tmp: Path) -> None:
    path = tmp / "labels.tsv"
    pd.DataFrame({"labels": ["clone_1", "normal", "clone_0", "normal", "clone_1"]}, index=list("abcde")).to_csv(path, sep="\t")
    config = SimpleNamespace(annotation=SimpleNamespace(clone_label=str(path)))
    index, base = load_clone_labels(None, config)
    assert [i.tolist() for i in index] == [[1, 3], [2], [0, 4]] and base is None, "normal is clone 0, clone_k is k + 1"
    x = np.arange(30, dtype=float).reshape(3, 2, 5)  # NB genes x[b, 0, s] = 10 b + s
    index, base = load_clone_labels(x, config)
    normal = x[:, 0, [1, 3]].sum(axis=1) / x[:, 0, [1, 3]].sum()  # NB [4, 24, 44] / 72
    coverage = x[:, 0, :].sum(axis=0)  # NB 30 + 3 s
    want = np.column_stack([normal * coverage[i].sum() for i in index])
    assert np.allclose(base, want, rtol=1e-12, atol=0) and np.allclose(base[:, 1], [2, 12, 22]), "base: the normal profile times each clone's coverage"


RANGES = pd.DataFrame({"chr": ["1", "1", "1", "2"], "start": [0, 100, 250, 0], "end": [100, 200, 400, 2500], "state": ["A", "A", "B", "A"]})
ITEMS = pd.DataFrame({"CHR": [1, 1, 1, 1, 2, 3, 2], "START": [10, 180, 195, 210, 900, 0, 2400], "END": [20, 260, 300, 240, 1300, 10, 2600]})


def overlap_brute(items: pd.DataFrame, ranges: list[tuple[str, int, int]]) -> list[Any]:
    """Each item's range of greatest overlap on its contig, the first on a tie; None where none overlaps."""
    out: list[Any] = []
    for chrom, start, end in zip(items["CHR"], items["START"], items["END"], strict=True):
        best, at = 0, None
        for k, (c, s, e) in enumerate(ranges):
            o = max(0, min(end, e) - max(start, s)) if c == str(chrom) else 0
            if o > best:
                best, at = o, k
        out.append(at)
    return out


def _ranges_greatest_overlap(_: Any) -> None:
    found = unchanged(assign_clone_ranges, ITEMS, RANGES, max_length=1000)
    merged = [("1", 0, 200), ("1", 250, 400), ("2", 0, 1000), ("2", 1000, 2000), ("2", 2000, 2500)]  # NB A A merge; chr 2 split at 1000
    want = overlap_brute(ITEMS, merged)
    assert want == [0, 0, 1, None, 3, None, 4], "by hand"
    assert [None if pd.isna(v) else int(v) for v in found["known_id"]] == want and str(found["known_id"].dtype) == "Int64"
    assert found[["CHR", "START", "END"]].equals(ITEMS), "the items pass through"
    raw = assign_clone_ranges(ITEMS, RANGES, collapse=False, max_length=None)
    rows = list(zip(RANGES["chr"], RANGES["start"], RANGES["end"], strict=True))
    assert [None if pd.isna(v) else int(v) for v in raw["known_id"]] == overlap_brute(ITEMS, rows) == [0, 1, 2, None, 3, None, 3]


def _join_xy_nearest(_: Any) -> None:
    first = pl.DataFrame({"x": [0.0, 5.0, 9.0], "y": [0.0, 5.0, 1.0], "id": [1, 2, 3]})
    second = pl.DataFrame({"x": [1.0, 4.0, 10.0, 6.0], "y": [0.0, 4.0, 10.0, 0.0], "lab": ["p", "q", "r", "s"], "v": [1.0, 2.0, 3.0, 4.0]})
    out = join_tables_xy(first, second, ["lab", "v"])
    assert out["lab"].to_list() == ["p", "q", "s"] and np.allclose(out["dist"].to_numpy(), [1.0, np.sqrt(2), np.sqrt(10)], rtol=1e-12), "by hand"
    rng = np.random.default_rng(3)
    a, b = rng.uniform(0, 10, (40, 2)), rng.uniform(0, 10, (15, 2))
    first = pl.DataFrame({"x": a[:, 0], "y": a[:, 1], "spot": np.arange(40)})
    second = pl.DataFrame({"x": b[:, 0], "y": b[:, 1], "cell": np.arange(15) * 7})
    out = join_tables_xy(first, second, ["cell"])
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
    assert np.array_equal(out["cell"].to_numpy(), d.argmin(axis=1) * 7), "the nearest row of second, by brute force"
    assert np.allclose(out["dist"].to_numpy(), d.min(axis=1), rtol=1e-12, atol=0) and out.select(first.columns).equals(first), "its distance; first passes through"


ORACLE: list[Row] = table(
    "oracle",
    ("io:map_unique_snps_enum", "synthetic: 5 SNP ids, one repeated 3 times", lambda c: None, _snp_ids_number_repeats, "{contig}_{pos}_{k}, k numbering repeats in order"),
    ("io:get_barcodes", "synthetic: a gzipped barcode file", lambda c: c.tmp_path, _barcodes_first_extension, "reads the first of the known extensions; none found raises"),
    ("io:get_aggregated_barcodes", "synthetic: 2 barcodes, one sample", lambda c: c.tmp_path, _aggregated_carry_sample_id, "one sample: every barcode carries the known sample id (Ticket#446 for 2+)"),
    ("io:get_spatial_positions", "synthetic: 3 positions, 2 in tissue", lambda c: c.tmp_path, _positions_in_tissue, "the in-tissue rows, or all; no file raises"),
    ("io:get_spaceranger_counts", "synthetic: a sparse .h5ad with a NaN and a repeated gene", lambda c: c.tmp_path, _counts_integer_unique_genes, "integer counts, NaN to 0, unique gene names"),
    ("io:get_sample_sheet", "synthetic: a sheet with a comment line", lambda c: c.tmp_path, _sheet_requires_columns, "reads the sheet; a missing required column raises"),
    ("filter:get_filter_genes", "synthetic: 2 genes", lambda c: c.tmp_path, _filters_parse_lines, "one gene per line"),
    ("filter:get_filter_ranges", "synthetic: 3 BED rows", lambda c: c.tmp_path, _filters_parse_lines, "chr prefix dropped, sorted by (Chr, Start)"),
    ("reference:exp_cancer_gene", "synthetic: 6 names", lambda c: None, _cancer_genes_by_pattern, "mitochondrial, ribosomal and five exact names"),
    ("io:get_sample_list", "synthetic: 5 spots, 2 samples", lambda c: None, _samples_in_order, "samples in order of appearance; ids index them"),
    ("io:read_tumor_prop", "synthetic: a 2-row tumor proportion file", lambda c: c.tmp_path, _tumor_prop_by_barcode, "the Tumor column by barcode; None without a file"),
    ("annotation:load_clone_labels", "synthetic: 5 labelled spots, 3 bins", lambda c: c.tmp_path, _clone_labels_scale_normal,
     "clone indices by label (normal 0, clone_k k+1); base = the normal spots' gene profile x each clone's coverage, recomputed"),
    ("annotation:assign_clone_ranges", "synthetic: 4 ranges, 7 items", lambda c: None, _ranges_greatest_overlap,
     "each item's range of greatest overlap, by brute force, after collapsing equal neighbours and splitting at max_length; none: NA"),
    ("he:join_tables_xy", "synthetic: 3 to 4 spots by hand, 40 to 15 at random", lambda c: None, _join_xy_nearest,
     "each row of first takes its nearest row of second and the distance, by brute force (Ticket#113, Ticket#311 for the image path)"),
)


# --- invariants ---------------------------------------------------------------------


def _loaded(ctx: Any) -> dict[str, Any]:
    return {"out": ctx("00_inputs/load_input_data/out"), "config": ctx.config}


def _loader_filters_hold(d: dict[str, Any]) -> None:
    coords, barcodes, adata, exp, a, b, ids, across = d["out"]
    q = d["config"].quality
    n = len(barcodes)
    assert coords.shape == (n, 2) and adata.shape[0] == exp.shape[0] == a.shape[0] == b.shape[0] == n and a.shape[1] == ids.size
    counts = np.asarray(adata.layers["count"])
    assert np.all(a.sum(axis=1) + b.sum(axis=1) >= q.spot_min_snp_umis), "every spot keeps the SNP-UMI floor"
    expressed = np.asarray((adata.X > 0).sum(axis=0)).ravel()
    assert np.all(expressed >= q.min_percent_expressed_spots * n), "every gene is expressed in the kept fraction of spots"
    # NB the staged YAML names the capture host's resources; read the same file from this host's (`cnamaste.audit.capture.grch38`)
    resources = grch38()
    assert resources is not None, "load_input_data replayed without GRCh38_resources"
    genes = set(get_filter_genes(resources / RESOURCES["filtergenelist_file"]).iloc[:, 0])
    assert not genes & set(adata.var.index), "filtered genes are gone"
    assert np.array_equal(exp.sparse.to_dense().to_numpy(), counts) and across is None
    assert np.array_equal(np.asarray(adata.obsm["X_pos"]), coords) and list(adata.obs.index) == list(barcodes)


def _config_nested_round_trip(tmp: Path) -> None:
    (tmp / "c.yaml").write_text(yaml.safe_dump({"a": {"b": 1, "c": "None", "d": "x"}, "e": [1, 2]}))
    c = YAMLConfig.from_file(tmp / "c.yaml")
    assert c.a.b == 1 and c.a.c is None and c.a.d == "x" and c.e == [1, 2]
    text = repr(c)
    assert "b: 1" in text and "d: 'x'" in text and "e: [1, 2]" in text
    c.over_ride(["a.b=7"])
    assert c.a.b == "7", "an over-ride is stored as text"
    with pytest.raises(ValueError):
        c.over_ride(["a.b"])


def _over_ride_keeps_type(tmp: Path) -> None:
    c = YAMLConfig({"hmrf": {"spatial_weight": 1.0, "n_clones": 3, "random_state": 0}, "paths": {"output_dir": str(tmp)}})
    c.over_ride(["hmrf.spatial_weight=0.5"])
    get_output_dir(c)


def _global_config_round_trip(_: Any) -> None:
    c = YAMLConfig({"x": 1})
    set_global_config(c)
    assert get_global_config() is c
    with pytest.raises(AssertionError):
        set_global_config({"x": 1})
    set_global_config(None)
    text = captured_log("cnamaste.config", get_global_config)
    assert "has not been defined" in text


def _config_warnings_name_departures(ctx: Any) -> None:
    config = ctx.config
    assert captured_log("cnamaste.config", config.issue_warnings) == "", "easy's configuration warns of nothing"
    config.hmrf.n_clones_rdr, config.phasing.run = 1, False
    text = captured_log("cnamaste.config", config.issue_warnings)
    assert "no rdr-based clone identification" in text and "no baf-based phasing" in text


def _output_dirs_named(ctx: Any) -> None:
    config = ctx.config
    config.paths.output_dir = str(ctx.tmp_path / "out")
    (ctx.tmp_path / "out").mkdir()
    out, plots = configure_output_dir(config)
    assert out == get_output_dir(config) == f"{ctx.tmp_path}/out/clone3_rectangle0_w1.0/"
    assert Path(out).is_dir() and Path(plots).is_dir()


def _pause_returns_unless_set(ctx: Any) -> None:
    config = ctx.config
    assert pause(config) is None
    config.run.pause = True
    with pytest.raises(OSError):
        pause(config)  # NB pytest's stdin raises on read: pause() did ask for input


def _logs_once(_: Any) -> None:
    name = "cnamaste.probe_once"
    logger = get_logger(name, start_time=0.0)
    for seen in ("_seen_warnings", "_seen_infos"):
        vars(logger).pop(seen, None)  # NB one name for every call, so the logger (and its pin) is the same whichever row ran first
    text = captured_log(name, lambda: [logger.warning_once("w"), logger.warning_once("w"), logger.info_once("i"), logger.info_once("i")])
    assert text.count("w\n") == 1 and text.count("i\n") == 1


def _formatter_stamps_phase(_: Any) -> None:
    record = logging.LogRecord("n", logging.INFO, "f", 1, "m", None, None)
    SharedStateLogger._shared_runtime_phase = "phase"
    assert RuntimePhaseFilter().filter(record) and record.runtime_phase_str == " (phase)"
    logger = logging.getLogger("cnamaste.probe_phase")
    assert isinstance(logger, SharedStateLogger)
    logger.runtime_phase = "other"
    assert SharedStateLogger._shared_runtime_phase == "other" and logging.getLogger("cnamaste.another").runtime_phase == "other", "shared across loggers"
    SharedStateLogger._shared_runtime_phase = None
    text = RuntimeFormatter("%(runtime)s|%(runtime_phase_str)s|%(message)s", start_time=0.0).format(logging.LogRecord("n", 20, "f", 1, "m", None, None))
    assert text.endswith("||m") and text.split("|")[0].endswith("m")


def _numba_seed_repeats(_: Any) -> None:
    @numba.njit
    def draw() -> float:
        return np.random.random()

    set_numba_seed(3)
    first = draw()
    set_numba_seed(3)
    assert draw() == first, "numba's generator restarts from the seed"


def _he_absent_passes_through(tmp: Path) -> None:
    pos = pd.DataFrame({"barcode": ["a"], "x": [0.0], "y": [0.0]})
    assert get_he_image(str(tmp), pos=pos) is pos, "no image: the positions pass through unchanged"


def _ranges_read_typed(tmp: Path) -> None:
    pd.DataFrame({"chr": [1, "X"], "start": [0.0, 5], "end": [10, 20], "state": [1, 2]}).to_csv(tmp / "ranges.tsv", sep="\t", index=False)
    found = load_clone_ranges(tmp / "ranges.tsv")
    assert found["chr"].tolist() == ["1", "X"] and found["start"].tolist() == [0, 5] and found["end"].tolist() == [10, 20]
    assert found["start"].dtype == np.int64 and found["end"].dtype == np.int64 and found["state"].tolist() == [1, 2], "contigs as text, coordinates as integers"
    with pytest.raises(FileNotFoundError):
        load_clone_ranges(tmp / "absent.tsv")


def _ranges_without_chr_load(tmp: Path) -> None:
    pd.DataFrame({"start": [0], "end": [10]}).to_csv(tmp / "ranges.tsv", sep="\t", index=False)
    assert load_clone_ranges(tmp / "ranges.tsv")["end"].tolist() == [10], "a missing column warns, and the file still loads"


def _no_alignments_none(_: Any) -> None:
    assert get_alignments(None, None, None) is None


def _alignment_builds_adjacency(tmp: Path) -> None:
    np.save(tmp / "pi.npy", np.eye(2))
    meta = pd.DataFrame({"sample_id": ["s1", "s2"]})
    barcodes = pd.DataFrame({"sample_id": ["s1", "s1", "s2", "s2"]})
    found = get_alignments([str(tmp / "pi.npy")], meta, barcodes)
    assert found.shape == (4, 4)


INVARIANT: list[Row] = table(
    "invariant",
    ("io:load_input_data", LOADED, _loaded, _loader_filters_hold, "aligned shapes; every spot over the SNP-UMI floor, every gene over the expressed fraction, filtered genes gone"),
    ("config:YAMLConfig.from_file", "synthetic: a 2-level YAML", lambda c: c.tmp_path, _config_nested_round_trip, "nested sections, 'None' read as None"),
    ("config:YAMLConfig.__init__", "synthetic: a 2-level YAML", lambda c: c.tmp_path, _config_nested_round_trip, "a dict becomes nested attributes"),
    ("config:YAMLConfig.__repr__", "synthetic: a 2-level YAML", lambda c: c.tmp_path, _config_nested_round_trip, "every key and value, quoted strings"),
    ("config:YAMLConfig._format_dict", "synthetic: a 2-level YAML", lambda c: c.tmp_path, _config_nested_round_trip, "nested sections indented"),
    ("config:YAMLConfig.over_ride", "synthetic: a 2-level YAML", lambda c: c.tmp_path, _config_nested_round_trip, "dot paths set a value; a term without '=' raises"),
    ("config:YAMLConfig.over_ride", "synthetic: one numeric key over-ridden", lambda c: c.tmp_path, _over_ride_keeps_type, "a numeric key over-ridden keeps its type",
     "new: YAMLConfig.over_ride stores every value as text, so `-o hmrf.spatial_weight=0.5` breaks get_output_dir's {:.1f} (and any arithmetic on the key)", ValueError),
    ("config:set_global_config", "synthetic", lambda c: None, _global_config_round_trip, "get returns what set installed; a non-config is refused"),
    ("config:get_global_config", "synthetic", lambda c: None, _global_config_round_trip, "None installed: returns None and warns"),
    ("config:YAMLConfig.issue_warnings", "the staged configuration", lambda c: c, _config_warnings_name_departures, "silent on easy; names each departure"),
    ("utils:configure_output_dir", "the staged configuration, a scratch output_dir", lambda c: c, _output_dirs_named, "creates clone{n}_rectangle{seed}_w{weight} and its plots/"),
    ("utils:get_output_dir", "the staged configuration, a scratch output_dir", lambda c: c, _output_dirs_named, "clone{n}_rectangle{seed}_w{weight:.1f}/ under paths.output_dir"),
    ("utils:pause", "the staged configuration", lambda c: c, _pause_returns_unless_set, "returns at once unless run.pause"),
    ("logger:warning_once", "synthetic", lambda c: None, _logs_once, "a message logs once"),
    ("logger:info_once", "synthetic", lambda c: None, _logs_once, "a message logs once"),
    ("logger:RuntimePhaseFilter.filter", "synthetic", lambda c: None, _formatter_stamps_phase, "stamps the shared phase"),
    ("logger:RuntimeFormatter.format", "synthetic", lambda c: None, _formatter_stamps_phase, "stamps minutes since start"),
    ("logger:SharedStateLogger.runtime_phase", "synthetic", lambda c: None, _formatter_stamps_phase, "one phase shared by every cnamaste logger"),
    ("scripts.run_cnamaste:set_numba_seed", "synthetic", lambda c: None, _numba_seed_repeats, "seeds numba's generator"),
    ("he:get_he_image", "synthetic: a directory with no image", lambda c: c.tmp_path, _he_absent_passes_through, "without an image the positions pass through (Ticket#113, Ticket#311 for the image path)"),
    ("annotation:load_clone_ranges", "synthetic: a 2-range file", lambda c: c.tmp_path, _ranges_read_typed, "contigs as text, coordinates as integers; a missing file raises"),
    ("annotation:load_clone_ranges", "synthetic: a file without chr", lambda c: c.tmp_path, _ranges_without_chr_load, "a missing column warns, and the file still loads",
     "new: load_clone_ranges warns on a missing chr column, then counts df['chr'] in its next log line (KeyError)", KeyError),
    ("io:get_alignments", "synthetic", lambda c: None, _no_alignments_none, "no alignment files: None"),
    ("io:get_alignments", "synthetic: one 2x2 alignment", lambda c: c.tmp_path, _alignment_builds_adjacency, "an alignment builds the across-slice adjacency",
     "new: get_alignments reads `adata`, a name it never binds (NameError on any alignment file); load_input_data reads df_meta before assigning it", NameError),
)


@pytest.mark.parametrize("row", ORACLE, ids=[r.id for r in ORACLE])
def test_oracle_agrees(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """Closed-form readings of synthetic files."""
    run(row, ctx, request)


@pytest.mark.parametrize("row", INVARIANT, ids=[r.id for r in INVARIANT])
def test_invariants_hold(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """The loader's filters, configuration parsing, directories, logging."""
    run(row, ctx, request)

