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
from typing import Any

import anndata
import numba
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import yaml
from cnamaste.config import YAMLConfig, get_global_config, set_global_config
from cnamaste.filter import get_filter_genes, get_filter_ranges
from cnamaste.he import get_he_image
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

from audit.fn import Replay, Row, run, table

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


def _snp_enum(_: Any) -> None:
    ids = np.array(["chr1_100_A_G", "1_200_C_T", "chr1_100_A_G", "2_5_N_N", "chr1_100_A_G"], dtype=object)
    assert map_unique_snps_enum(ids).tolist() == ["1_100_0", "1_200_0", "1_100_1", "2_5_0", "1_100_2"], "repeats numbered in order; chr dropped"


def _barcodes(tmp: Path) -> None:
    (tmp / "barcodes.tsv.gz").write_bytes(gzip.compress(b"AAAC-1\nAAAG-1\n"))
    assert get_barcodes(str(tmp / "barcodes.txt"))["combined_barcode"].tolist() == ["AAAC-1", "AAAG-1"], "the .tsv.gz alternative is found first"
    with pytest.raises(RuntimeError):
        get_barcodes(str(tmp / "none" / "barcodes.txt"))


def _aggregated(tmp: Path) -> None:
    (tmp / "barcodes.txt").write_text("AAAC-1\nAAAG-1\n")
    found = get_aggregated_barcodes(str(tmp / "barcodes.txt"), known_sample_id="s1")
    assert found["barcode"].tolist() == ["AAAC-1", "AAAG-1"] and found["sample_id"].tolist() == ["s1", "s1"]


def _positions(tmp: Path) -> None:
    (tmp / "spatial").mkdir()
    (tmp / "spatial" / "tissue_positions_list.csv").write_text("a,1,0,0,10,20\nb,0,0,1,11,21\nc,1,1,0,12,22\n")
    kept = get_spatial_positions(str(tmp))
    assert kept["barcode"].tolist() == ["a", "c"] and kept["x"].tolist() == [0, 1]
    assert len(get_spatial_positions(str(tmp), filter_in_tissue=False)) == 3
    with pytest.raises(RuntimeError):
        get_spatial_positions(str(tmp / "missing"))


def _counts(tmp: Path) -> None:
    x = sp.csr_matrix(np.array([[1.0, np.nan], [2.0, 3.0]]))
    adata = anndata.AnnData(x, obs=pd.DataFrame(index=["a", "b"]), var=pd.DataFrame(index=["g", "g"]))
    adata.write_h5ad(tmp / "filtered_feature_bc_matrix.h5ad")
    found = get_spaceranger_counts(str(tmp))
    assert found.layers["count"].tolist() == [[1, 0], [2, 3]] and found.layers["count"].dtype.kind == "i", "NaN counts become 0, integers"
    assert found.var_names.is_unique


def _sheet(tmp: Path) -> None:
    (tmp / "sheet.tsv").write_text("# comment\nbam sample_id spaceranger_dir snp_dir\nx.bam s1 /a /b\n")
    assert get_sample_sheet(str(tmp / "sheet.tsv"))["sample_id"].tolist() == ["s1"]
    (tmp / "bad.tsv").write_text("bam sample_id\nx.bam s1\n")
    with pytest.raises(AssertionError):
        get_sample_sheet(str(tmp / "bad.tsv"))


def _filters(tmp: Path) -> None:
    (tmp / "genes.txt").write_text("IGHA1\nIGHG1\n")
    assert get_filter_genes(tmp / "genes.txt").iloc[:, 0].tolist() == ["IGHA1", "IGHG1"]
    (tmp / "ranges.bed").write_text("chr6\t300\t400\nchr2\t10\t20\nchr6\t100\t200\n")
    ranges = get_filter_ranges(tmp / "ranges.bed")
    assert ranges[["Chr", "Start"]].values.tolist() == [[2, 10], [6, 100], [6, 300]], "chr dropped, sorted by (Chr, Start)"


def _cancer(_: Any) -> None:
    assert [exp_cancer_gene(g) for g in ("MT-CO1", "RPL3", "B2M", "B2MX", "TP53", "FTH1")] == [True, True, True, False, False, True]


def _sample_list(_: Any) -> None:
    adata = anndata.AnnData(np.zeros((5, 1)), obs=pd.DataFrame({"sample": ["a", "a", "b", "b", "b"]}, index=list("vwxyz")))
    names, ids = get_sample_list(adata)
    assert names == ["a", "b"] and ids.tolist() == [0, 0, 1, 1, 1]


def _tumor_prop(tmp: Path) -> None:
    adata = anndata.AnnData(np.zeros((2, 1)), obs=pd.DataFrame(index=["a", "b"]))
    (tmp / "tp.tsv").write_text("barcode\tTumor\na\t0.25\nb\t0.75\n")
    config = YAMLConfig({"preprocessing": {"tumorprop_file": str(tmp / "tp.tsv")}})
    assert read_tumor_prop(adata, config=config).tolist() == [0.25, 0.75]
    assert read_tumor_prop(adata, config=YAMLConfig({"preprocessing": {"tumorprop_file": "None"}})) is None


ORACLE: list[Row] = table(
    "oracle",
    ("io:map_unique_snps_enum", "synthetic: 5 SNP ids, one repeated 3 times", lambda c: None, _snp_enum, "{contig}_{pos}_{k}, k numbering repeats in order"),
    ("io:get_barcodes", "synthetic: a gzipped barcode file", lambda c: c.tmp_path, _barcodes, "reads the first of the known extensions; none found raises"),
    ("io:get_aggregated_barcodes", "synthetic: 2 barcodes, one sample", lambda c: c.tmp_path, _aggregated, "one sample: every barcode carries the known sample id (Ticket#446 for 2+)"),
    ("io:get_spatial_positions", "synthetic: 3 positions, 2 in tissue", lambda c: c.tmp_path, _positions, "the in-tissue rows, or all; no file raises"),
    ("io:get_spaceranger_counts", "synthetic: a sparse .h5ad with a NaN and a repeated gene", lambda c: c.tmp_path, _counts, "integer counts, NaN to 0, unique gene names"),
    ("io:get_sample_sheet", "synthetic: a sheet with a comment line", lambda c: c.tmp_path, _sheet, "reads the sheet; a missing required column raises"),
    ("filter:get_filter_genes", "synthetic: 2 genes", lambda c: c.tmp_path, _filters, "one gene per line"),
    ("filter:get_filter_ranges", "synthetic: 3 BED rows", lambda c: c.tmp_path, _filters, "chr prefix dropped, sorted by (Chr, Start)"),
    ("reference:exp_cancer_gene", "synthetic: 6 names", lambda c: None, _cancer, "mitochondrial, ribosomal and five exact names"),
    ("io:get_sample_list", "synthetic: 5 spots, 2 samples", lambda c: None, _sample_list, "samples in order of appearance; ids index them"),
    ("io:read_tumor_prop", "synthetic: a 2-row tumor proportion file", lambda c: c.tmp_path, _tumor_prop, "the Tumor column by barcode; None without a file"),
)


# --- invariants ---------------------------------------------------------------------


def _loaded(ctx: Any) -> dict[str, Any]:
    return {"out": ctx("00_inputs/load_input_data/out"), "config": ctx.config}


def _load_input(d: dict[str, Any]) -> None:
    coords, barcodes, adata, exp, a, b, ids, across = d["out"]
    q = d["config"].quality
    n = len(barcodes)
    assert coords.shape == (n, 2) and adata.shape[0] == exp.shape[0] == a.shape[0] == b.shape[0] == n and a.shape[1] == ids.size
    counts = np.asarray(adata.layers["count"])
    assert np.all(a.sum(axis=1) + b.sum(axis=1) >= q.spot_min_snp_umis), "every spot keeps the SNP-UMI floor"
    expressed = np.asarray((adata.X > 0).sum(axis=0)).ravel()
    assert np.all(expressed >= q.min_percent_expressed_spots * n), "every gene is expressed in the kept fraction of spots"
    genes = set(get_filter_genes(d["config"].references.filtergenelist_file).iloc[:, 0])
    assert not genes & set(adata.var.index), "filtered genes are gone"
    assert np.array_equal(exp.sparse.to_dense().to_numpy(), counts) and across is None
    assert np.array_equal(np.asarray(adata.obsm["X_pos"]), coords) and list(adata.obs.index) == list(barcodes)


def _config_io(tmp: Path) -> None:
    (tmp / "c.yaml").write_text(yaml.safe_dump({"a": {"b": 1, "c": "None", "d": "x"}, "e": [1, 2]}))
    c = YAMLConfig.from_file(tmp / "c.yaml")
    assert c.a.b == 1 and c.a.c is None and c.a.d == "x" and c.e == [1, 2]
    text = repr(c)
    assert "b: 1" in text and "d: 'x'" in text and "e: [1, 2]" in text
    c.over_ride(["a.b=7"])
    assert c.a.b == "7", "an over-ride is stored as text"
    with pytest.raises(ValueError):
        c.over_ride(["a.b"])


def _over_ride_type(tmp: Path) -> None:
    c = YAMLConfig({"hmrf": {"spatial_weight": 1.0, "n_clones": 3, "random_state": 0}, "paths": {"output_dir": str(tmp)}})
    c.over_ride(["hmrf.spatial_weight=0.5"])
    get_output_dir(c)


def _global(_: Any) -> None:
    c = YAMLConfig({"x": 1})
    set_global_config(c)
    assert get_global_config() is c
    with pytest.raises(AssertionError):
        set_global_config({"x": 1})
    set_global_config(None)
    text = captured_log("cnamaste.config", get_global_config)
    assert "has not been defined" in text


def _warnings(ctx: Any) -> None:
    config = ctx.config
    assert captured_log("cnamaste.config", config.issue_warnings) == "", "easy's configuration warns of nothing"
    config.hmrf.n_clones_rdr, config.phasing.run = 1, False
    text = captured_log("cnamaste.config", config.issue_warnings)
    assert "no rdr-based clone identification" in text and "no baf-based phasing" in text


def _output_dirs(ctx: Any) -> None:
    config = ctx.config
    config.paths.output_dir = str(ctx.tmp_path / "out")
    (ctx.tmp_path / "out").mkdir()
    out, plots = configure_output_dir(config)
    assert out == get_output_dir(config) == f"{ctx.tmp_path}/out/clone3_rectangle0_w1.0/"
    assert Path(out).is_dir() and Path(plots).is_dir()


def _pause(ctx: Any) -> None:
    config = ctx.config
    assert pause(config) is None
    config.run.pause = True
    with pytest.raises(OSError):
        pause(config)  # NB pytest's stdin raises on read: pause() did ask for input


def _once(_: Any) -> None:
    name = "cnamaste.probe_once"
    logger = get_logger(name, start_time=0.0)
    for seen in ("_seen_warnings", "_seen_infos"):
        vars(logger).pop(seen, None)  # NB one name for every call, so the logger (and its pin) is the same whichever row ran first
    text = captured_log(name, lambda: [logger.warning_once("w"), logger.warning_once("w"), logger.info_once("i"), logger.info_once("i")])
    assert text.count("w\n") == 1 and text.count("i\n") == 1


def _formatter(_: Any) -> None:
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


def _numba_seed(_: Any) -> None:
    @numba.njit
    def draw() -> float:
        return np.random.random()

    set_numba_seed(3)
    first = draw()
    set_numba_seed(3)
    assert draw() == first, "numba's generator restarts from the seed"


def _he_absent(tmp: Path) -> None:
    pos = pd.DataFrame({"barcode": ["a"], "x": [0.0], "y": [0.0]})
    assert get_he_image(str(tmp), pos=pos) is pos, "no image: the positions pass through unchanged"


def _alignments(_: Any) -> None:
    assert get_alignments(None, None, None) is None


def _alignments_given(tmp: Path) -> None:
    np.save(tmp / "pi.npy", np.eye(2))
    meta = pd.DataFrame({"sample_id": ["s1", "s2"]})
    barcodes = pd.DataFrame({"sample_id": ["s1", "s1", "s2", "s2"]})
    found = get_alignments([str(tmp / "pi.npy")], meta, barcodes)
    assert found.shape == (4, 4)


INVARIANT: list[Row] = table(
    "invariant",
    ("io:load_input_data", LOADED, _loaded, _load_input, "aligned shapes; every spot over the SNP-UMI floor, every gene over the expressed fraction, filtered genes gone"),
    ("config:YAMLConfig.from_file", "synthetic: a 2-level YAML", lambda c: c.tmp_path, _config_io, "nested sections, 'None' read as None"),
    ("config:YAMLConfig.__init__", "synthetic: a 2-level YAML", lambda c: c.tmp_path, _config_io, "a dict becomes nested attributes"),
    ("config:YAMLConfig.__repr__", "synthetic: a 2-level YAML", lambda c: c.tmp_path, _config_io, "every key and value, quoted strings"),
    ("config:YAMLConfig._format_dict", "synthetic: a 2-level YAML", lambda c: c.tmp_path, _config_io, "nested sections indented"),
    ("config:YAMLConfig.over_ride", "synthetic: a 2-level YAML", lambda c: c.tmp_path, _config_io, "dot paths set a value; a term without '=' raises"),
    ("config:YAMLConfig.over_ride", "synthetic: one numeric key over-ridden", lambda c: c.tmp_path, _over_ride_type, "a numeric key over-ridden keeps its type",
     "new: YAMLConfig.over_ride stores every value as text, so `-o hmrf.spatial_weight=0.5` breaks get_output_dir's {:.1f} (and any arithmetic on the key)", ValueError),
    ("config:set_global_config", "synthetic", lambda c: None, _global, "get returns what set installed; a non-config is refused"),
    ("config:get_global_config", "synthetic", lambda c: None, _global, "None installed: returns None and warns"),
    ("config:YAMLConfig.issue_warnings", "the staged configuration", lambda c: c, _warnings, "silent on easy; names each departure"),
    ("utils:configure_output_dir", "the staged configuration, a scratch output_dir", lambda c: c, _output_dirs, "creates clone{n}_rectangle{seed}_w{weight} and its plots/"),
    ("utils:get_output_dir", "the staged configuration, a scratch output_dir", lambda c: c, _output_dirs, "clone{n}_rectangle{seed}_w{weight:.1f}/ under paths.output_dir"),
    ("utils:pause", "the staged configuration", lambda c: c, _pause, "returns at once unless run.pause"),
    ("logger:warning_once", "synthetic", lambda c: None, _once, "a message logs once"),
    ("logger:info_once", "synthetic", lambda c: None, _once, "a message logs once"),
    ("logger:RuntimePhaseFilter.filter", "synthetic", lambda c: None, _formatter, "stamps the shared phase"),
    ("logger:RuntimeFormatter.format", "synthetic", lambda c: None, _formatter, "stamps minutes since start"),
    ("logger:SharedStateLogger.runtime_phase", "synthetic", lambda c: None, _formatter, "one phase shared by every cnamaste logger"),
    ("scripts.run_cnamaste:set_numba_seed", "synthetic", lambda c: None, _numba_seed, "seeds numba's generator"),
    ("he:get_he_image", "synthetic: a directory with no image", lambda c: c.tmp_path, _he_absent, "without an image the positions pass through (Ticket#113, Ticket#311 for the image path)"),
    ("io:get_alignments", "synthetic", lambda c: None, _alignments, "no alignment files: None"),
    ("io:get_alignments", "synthetic: one 2x2 alignment", lambda c: c.tmp_path, _alignments_given, "an alignment builds the across-slice adjacency",
     "new: get_alignments reads `adata`, a name it never binds (NameError on any alignment file); load_input_data reads df_meta before assigning it", NameError),
)


@pytest.mark.parametrize("row", ORACLE, ids=[r.id for r in ORACLE])
def test_oracle(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """Closed-form readings of synthetic files."""
    run(row, ctx, request)


@pytest.mark.parametrize("row", INVARIANT, ids=[r.id for r in INVARIANT])
def test_invariant(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """The loader's filters, configuration parsing, directories, logging."""
    run(row, ctx, request)

