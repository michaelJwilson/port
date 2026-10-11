"""Smoke rows for what the staged run does not exercise: the figures, the console script, the output tree, logging and perf.

The capture stubs plots, so each `plot_*` that `run_cnamaste` reaches is
called here on arguments built from the staged stage outputs, the way its call
site builds them, and written with `write_fig`. A row asserts what it can of
the result (a PDF, non-empty) and says so: these are smoke, not a judgement
of the figure. `plot_he` is not reached on easy (no H&E image).
"""

from __future__ import annotations

import ast
import io
import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from cnamaste.config import YAMLConfig
from cnamaste.hmrf_utils import get_clone_indices
from cnamaste.logger import get_logger
from cnamaste.plot_copy_number_profile import plot_copy_number_profile
from cnamaste.plot_genomic import plot_clones_genomic
from cnamaste.plotting import plot_clones_spatial
from cnamaste.scripts import run_cnamaste as script
from cnamaste.utils import write_fig

from audit.fn import Replay, Row, fitted, run, table

RDR = "08_rdr"
SCRIPT = Path(script.__file__)
COUNTS = Replay("08_rdr/run_core_inference in (counts, lengths), 08_rdr/reindex_clones out, get_sample_list out")
STUBBED = "the capture stubs plots, so the staged listing holds no figure; the plotting rows render each from staged inputs"


def figure_inputs(ctx: Any) -> dict[str, Any]:
    """The final fit's counts and result, the coordinates, the sample list and the segment table, as run_cnamaste's plot calls take them."""
    def make() -> dict[str, Any]:
        f = fitted(ctx, RDR)
        res = ctx(f"{RDR}/reindex_clones/out/0")
        sample_list, sample_ids = ctx.sim.stored("00_inputs/get_sample_list/out")
        return {"lengths": f["lengths"], "X": f["single_X"], "base": f["single_base_nb_mean"], "total": f["single_total_bb_RD"], "res": res,
                "coords": np.asarray(ctx.sim.stored("09_outputs/construct_df_clone_label/in/args/1")), "sample_list": sample_list,
                "sample_ids": np.asarray(sample_ids), "seg": pd.read_csv(io.BytesIO(ctx.sim.file("cnv_seglevel.tsv")), sep="\t")}
    return ctx.once("smoke/figures", make) | {"tmp": ctx.tmp_path}


def rendered(name: str, draw: Any) -> Any:
    """`draw(inputs)` written by write_fig to `<tmp>/<name>.pdf`: a non-empty PDF."""
    def check(d: dict[str, Any]) -> None:
        path = d["tmp"] / f"{name}.pdf"
        write_fig(str(path), draw(d), transparent=True, bbox_inches="tight")
        data = path.read_bytes()
        assert data[:5] == b"%PDF-" and len(data) > 1000, f"{name}: {len(data)} bytes"
    return check


def _pseudobulk(d: dict[str, Any]) -> Any:
    return plot_clones_genomic(lengths=d["lengths"], single_X=d["X"], single_base_nb_mean=d["base"], single_total_bb_RD=d["total"], df_cnv=None,
                               clone_index=[list(range(d["coords"].shape[0]))], single_tumor_prop=None, sample_list=d["sample_list"])


def _clones(d: dict[str, Any]) -> Any:
    labels = np.asarray(d["res"]["new_assignment"])
    return plot_clones_genomic(lengths=d["lengths"], single_X=d["X"], single_base_nb_mean=d["base"], single_total_bb_RD=d["total"], df_cnv=None,
                               clone_index=get_clone_indices(labels, np.unique(labels)), res_combine=d["res"], single_tumor_prop=None,
                               sample_list=d["sample_list"], palette_name="chisel_single")


def _segments(d: dict[str, Any]) -> Any:
    return plot_clones_genomic(d["lengths"], d["X"], d["base"], d["total"], df_cnv=d["seg"], res_combine=d["res"], single_tumor_prop=None,
                               sample_list=d["sample_list"], chrtext_shift=-0.3)


def _spatial(d: dict[str, Any]) -> Any:
    assignment = pd.Series([f"clone {x}" for x in np.asarray(d["res"]["new_assignment"])])
    return plot_clones_spatial(d["coords"], assignment, single_tumor_prop=None, sample_list=d["sample_list"], sample_ids=d["sample_ids"])


def _profile(d: dict[str, Any]) -> Any:
    return plot_copy_number_profile(d["seg"])


def _empty_figure(tmp: Path) -> None:
    write_fig(str(tmp / "empty.pdf"))
    assert (tmp / "empty.pdf").read_bytes()[:5] == b"%PDF-", "write_fig with no figure writes an empty page"


# --- console script ------------------------------------------------------------


def _help(_: Any) -> None:
    exe = Path(sys.executable).with_name("run_cnamaste")
    assert exe.exists(), f"no console script beside {sys.executable}"
    out = subprocess.run([str(exe), "--help"], capture_output=True, text=True, timeout=120, check=False)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "config_path" in out.stdout and "--over_rides" in out.stdout and "-o" in out.stdout


def _override(ctx: Any) -> Any:
    """main()'s `-o` list, applied the way run_cnamaste applies it, on the staged configuration."""
    path = ctx.tmp_path / "config.yaml"
    path.write_text(ctx.sim.config["yaml"])
    seen: dict[str, Any] = {}

    def capture(config_path: str, over_rides: list[str] | None = None) -> None:
        config = YAMLConfig.from_file(config_path)
        config.over_ride(over_rides)
        seen["config"] = config

    monkeypatch = ctx.request.getfixturevalue("monkeypatch")
    monkeypatch.setattr(script, "run_cnamaste", capture)
    monkeypatch.setattr(sys, "argv", ["run_cnamaste", str(path), "-o", "hmrf.spatial_weight=0.5", "-o", "hmrf.n_clones=4", "-o", "run.cache=true"])
    script.main()
    return seen["config"]


def _typed(config: Any) -> None:
    found = (config.hmrf.spatial_weight, config.hmrf.n_clones, config.run.cache)
    assert found == (0.5, 4, True) and [type(v) for v in found] == [float, int, bool], f"parsed as {found!r}"


# --- output tree ---------------------------------------------------------------


def declared(writer: str) -> list[str]:
    """The file names run_cnamaste's `writer` calls name, `{medfix[o]}` as the first (unsuffixed) pass."""
    names = []
    for node in ast.walk(ast.parse(SCRIPT.read_text())):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == writer and node.args and isinstance(node.args[0], ast.JoinedStr):
            parts = [v.value if isinstance(v, ast.Constant) else "" if "medfix" in ast.unparse(v) else "/" for v in node.args[0].values]
            names.append("".join(parts).rsplit("/", 1)[-1])
    return names


def _tables(ctx: Any) -> tuple[list[str], set[str]]:
    return declared("write_tsv"), set(json.loads(ctx.sim.config["files"]))


def _tables_written(d: tuple[list[str], set[str]]) -> None:
    names, listing = d
    assert len(names) == 5 and set(names) <= listing, f"declared {sorted(names)}, written {sorted(listing)}"


def _figures(ctx: Any) -> tuple[list[str], set[str]]:
    if ctx.sim.config["plots"] == "stubbed":
        pytest.skip(STUBBED)
    return declared("write_fig"), set(json.loads(ctx.sim.config["plots"]))


def _figures_written(d: tuple[list[str], set[str]]) -> None:
    names, listing = d
    assert set(names) <= listing, f"declared but not written: {sorted(set(names) - listing)}"


# --- logging and perf -------------------------------------------------------------


def _log_sink(ctx: Any) -> None:
    """No configuration key names a log file; every cnamaste logger writes to stdout alone, once (no propagation)."""
    keys = {f"{s}.{k}" for s, sub in ctx.config.__dict__.items() if isinstance(sub, YAMLConfig) for k in sub.__dict__}
    assert not [k for k in keys if re.search(r"(^|_)log(ging|_file|_path|_dir|_level)?$", k.split(".")[-1])], "a log key the logger would have to read"
    logger = get_logger("cnamaste.smoke_probe", start_time=0.0)
    handlers = logger.handlers
    assert not logger.propagate and len(handlers) == 1 and isinstance(handlers[0], logging.StreamHandler) and handlers[0].stream is sys.stdout


ROWS: list[Row] = [
    *table(
        "plot",
        ("plot_genomic:plot_clones_genomic", COUNTS, figure_inputs, rendered("pseudobulk_clones_genomic", _pseudobulk),
         "the all-spot pseudobulk, no result (run_cnamaste.py:214): a non-empty PDF"),
        ("plot_genomic:plot_clones_genomic", COUNTS, figure_inputs, rendered("rdr_baf_clones_genomic", _clones),
         "per clone with the fit, chisel_single (run_cnamaste.py:1143): a non-empty PDF"),
        ("plot_genomic:plot_clones_genomic", COUNTS, figure_inputs, rendered("clones_genomic", _segments),
         "per clone with cnv_seglevel.tsv's integer states (run_cnamaste.py:1640): a non-empty PDF"),
        ("plotting:plot_clones_spatial", COUNTS, figure_inputs, rendered("clones_spatial", _spatial),
         "the final labels on the coordinates (run_cnamaste.py:1699): a non-empty PDF"),
        ("plot_copy_number_profile:plot_copy_number_profile", COUNTS, figure_inputs, rendered("copy_number_profile", _profile),
         "cnv_seglevel.tsv's allele-specific profile (run_cnamaste.py:1714): a non-empty PDF"),
        ("utils:write_fig", "synthetic", lambda c: c.tmp_path, _empty_figure, "no figure writes a blank PDF page"),
    ),
    *table(
        "script",
        ("scripts.run_cnamaste:main", "the installed console script", lambda c: None, _help, "`run_cnamaste --help` exits 0 and names config_path and -o/--over_rides"),
        ("scripts.run_cnamaste:main", "the staged config.yaml", _override, _typed, "`-o key=value` reaches the config as the key's type (float, int, bool)",
         "new: YAMLConfig.over_ride stores every value as text (test_fn_io's over_ride row), so `-o hmrf.spatial_weight=0.5` is '0.5'"),
    ),
    *table(
        "outputs",
        ("scripts.run_cnamaste:run_cnamaste", "/config files, run_cnamaste.py's write_tsv calls", _tables, _tables_written,
         "every declared table of the first pass is in the staged listing (the ploidy pass: test_stages' int_copy_num.ploidy row)"),
        ("scripts.run_cnamaste:run_cnamaste", "/config plots, run_cnamaste.py's write_fig calls", _figures, _figures_written, "every declared figure is in the staged listing"),
    ),
    *table(
        "logging",
        ("logger:get_logger", "the staged config.yaml", lambda c: c, _log_sink,
         "logs go to stdout, which is where the config says (it names no log file); perf: test_fn_normal's flush_perf row (paths.perf_path ignored)"),
    ),
]


@pytest.mark.parametrize("row", ROWS, ids=[f"{r.kind}: {r.id}" for r in ROWS])
def test_smoke(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """Runs and writes what it should; asserts no more than the row says."""
    run(row, ctx, request)
