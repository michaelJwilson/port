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
from cnamaste.hmrf_utils import get_clone_assignment, get_clone_indices
from cnamaste.logger import get_logger
from cnamaste.plot_copy_number_profile import plot_copy_number_profile
from cnamaste.plot_genomic import plot_clones_genomic
from cnamaste.plotting import plot_clones_spatial
from cnamaste.scripts import run_cnamaste as script
from cnamaste.utils import write_fig

from fn import Replay, Row, fitted, run, table

RDR = "08_rdr"
SCRIPT = Path(script.__file__)
COUNTS = Replay("08_rdr/run_core_inference in (counts, lengths), 08_rdr/reindex_clones out, get_sample_list out, 03_phasing out")
STUBBED = "the capture stubs plots, so the staged listing holds no figure; the plotting rows render each from staged inputs"


def figure_inputs(ctx: Any) -> dict[str, Any]:
    """The final fit's counts and result, the coordinates, the sample list and the segment table, as run_cnamaste's plot calls take them."""
    def make() -> dict[str, Any]:
        f = fitted(ctx, RDR)
        res = ctx(f"{RDR}/reindex_clones/out/0")
        sample_list, sample_ids = ctx.sim.stored("00_inputs/get_sample_list/out")
        return {"lengths": f["lengths"], "X": f["single_X"], "base": f["single_base_nb_mean"], "total": f["single_total_bb_RD"], "res": res,
                "coords": np.asarray(ctx.sim.stored("09_outputs/construct_df_clone_label/in/args/1")), "sample_list": sample_list,
                "sample_ids": np.asarray(sample_ids), "seg": pd.read_csv(io.BytesIO(ctx.sim.file("cnv_seglevel.tsv")), sep="\t"),
                "initial": ctx.sim.stored("03_phasing/initialize_clones/out"),
                "phasing": ctx("03_phasing/initial_phase_given_partition/out/0")}
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


def _postphasing(d: dict[str, Any]) -> Any:
    """run_cnamaste.py:406's figure from phasing's own result: its parameters (n_states,), as every fit returns them."""
    res = dict(d["phasing"])
    assert all(np.ndim(res[k]) == 1 for k in ("new_log_mu", "new_alphas", "new_p_binom", "new_taus")), "phasing's parameters (n_states,)"
    res["new_assignment"] = get_clone_assignment(d["coords"], d["initial"])
    return plot_clones_genomic(df_cnv=None, lengths=d["lengths"], single_X=d["X"], single_base_nb_mean=d["base"], single_total_bb_RD=d["total"],
                               clone_index=d["initial"], res_combine=res, single_tumor_prop=None, sample_list=d["sample_list"])


def _spatial(d: dict[str, Any]) -> Any:
    assignment = pd.Series([f"clone {x}" for x in np.asarray(d["res"]["new_assignment"])])
    return plot_clones_spatial(d["coords"], assignment, single_tumor_prop=None, sample_list=d["sample_list"], sample_ids=d["sample_ids"])


def _profile(d: dict[str, Any]) -> Any:
    return plot_copy_number_profile(d["seg"])


def _slide(tmp: Path, coords: np.ndarray, scalef: float = 4.0) -> Path:
    """A random H&E slide over `coords`, in get_he_image's files: pixel (i, j) sits at (x, y) = (j, i) / scalef."""
    import matplotlib.pyplot as plt

    (tmp / "spatial").mkdir(exist_ok=True)
    rows, cols = int(scalef * (coords[:, 1].max() + 2)), int(scalef * (coords[:, 0].max() + 2))
    plt.imsave(tmp / "spatial" / "tissue_hires_image.png", np.random.default_rng(836).random((rows, cols, 3)))
    (tmp / "spatial" / "scalefactors_json.json").write_text(json.dumps({"tissue_hires_scalef": scalef}))
    return tmp


def _on_page(fig: Any, page: str) -> None:
    """At the page's size in inches and pixels, nothing drawn off it, no two axes overlapping, one font size."""
    import matplotlib.pyplot as plt
    from matplotlib.text import Text

    from cnamaste.layout import page_size, style

    assert tuple(fig.get_size_inches()) == pytest.approx(page_size(page), abs=1e-12), "the page's size"
    fig.canvas.draw()
    w, h = fig.get_size_inches()
    box = fig.get_tightbbox(fig.canvas.get_renderer())
    assert box.x0 >= -1e-6 and box.y0 >= -1e-6 and box.x1 <= w + 1e-6 and box.y1 <= h + 1e-6, f"drawn off the page: {box}"
    boxes = [ax.get_position() for ax in fig.get_axes()]
    for i, p in enumerate(boxes):
        for q in boxes[i + 1:]:
            assert min(p.x1, q.x1) <= max(p.x0, q.x0) + 1e-9 or min(p.y1, q.y1) <= max(p.y0, q.y0) + 1e-9, "two axes overlap"
    sizes = {t.get_fontsize() for t in fig.findobj(Text) if t.get_text()}
    assert sizes == {style()["font_size"]}, f"font sizes {sorted(sizes)}"
    plt.close("all")


def _combined(he: bool) -> Any:
    def check(d: dict[str, Any]) -> None:
        import matplotlib.pyplot as plt

        from cnamaste.he import get_he_image
        from cnamaste.layout import DPI, GenomicAxis, page_size, plot_combined_page
        from cnamaste.utils import cast_clone_label

        labels = np.asarray(d["res"]["new_assignment"])
        assignment = pd.Series([f"clone {x}" for x in labels])
        frame = get_he_image(str(_slide(d["tmp"], d["coords"])), pos=None) if he else None
        fig = plot_combined_page(d["lengths"], d["X"], d["base"], d["total"], d["seg"], d["res"], d["coords"], assignment,
                                 sample_list=d["sample_list"], sample_ids=d["sample_ids"], he_frame=frame)  # fmt: skip
        out = d["tmp"] / "combined.png"
        fig.savefig(out, format="png", dpi=DPI, facecolor="white")
        w, h = page_size("full")
        assert plt.imread(out).shape[:2] == (round(h * DPI), round(w * DPI)), "the PNG at the page's size"
        genome = GenomicAxis.from_bins(d["seg"]["CHR"])
        genomic = [ax for ax in fig.get_axes() if ax.get_xlim() == (0.0, float(genome.n_bins))]
        assert len(genomic) == 1 + 2 * np.unique(labels).size, "the profile and two tracks per clone on the genomic axis"
        edges = {(round(ax.get_position().x0, 12), round(ax.get_position().x1, 12)) for ax in genomic}
        assert len(edges) == 1, f"the genomic panels' left and right edges differ: {edges}"
        spatial = [ax for ax in fig.get_axes() if ax.get_aspect() == 1.0]
        assert len(spatial) == 2 and len({(ax.get_xlim(), ax.get_ylim()) for ax in spatial}) == 1, "slide and clones on one extent"
        assert bool(spatial[0].get_images()) == he, "the slide where there is one, a blank frame where there is none"
        assert {t.get_text() for t in fig.texts} >= {"(a)", "(b)", "(c)"}, "the panel letters"
        names = [t.get_text() for ax in genomic for t in ax.texts if t.get_position() == (0.0, 1.02)]
        assert names == [cast_clone_label(f"clone {c}") for c in np.unique(labels)], f"cnaster's clone names over the tracks: {names}"
        profile = [ax for ax in genomic if ax.get_yticklabels() and ax.get_ylim() == (0.0, 1.0)]
        rows = [t.get_text() for t in profile[0].get_yticklabels()] if profile else []
        assert rows == names, f"the profile's rows in the tracks' order: {rows}, {names}"
        _on_page(fig, "full")
    return check


def _he_page(d: dict[str, Any]) -> None:
    """he_clones.png on easy's spots: a synthetic slide over them, each spot's H&E class as get_he_image gives it, and
    the run's initial clones (03_phasing/initialize_clones)."""
    from cnamaste.he import get_he_image
    from cnamaste.layout import plot_he_page

    coords = d["coords"]
    spots = pd.DataFrame({"barcode": np.arange(len(coords)), "x": coords[:, 0], "y": coords[:, 1]})
    frame = get_he_image(str(_slide(d["tmp"], coords)), pos=spots)
    assignment = pd.Series([f"clone {x}" for x in get_clone_assignment(coords, d["initial"])])
    fig = plot_he_page(coords, frame["label"].to_numpy(), assignment)
    fig.savefig(d["tmp"] / "he_clones.png", format="png", dpi=300, facecolor="white")
    spatial = [ax for ax in fig.get_axes() if ax.get_aspect() == 1.0]
    assert len(spatial) == 2 and len({(ax.get_xlim(), ax.get_ylim()) for ax in spatial}) == 1, "one extent"
    keys = [t.get_text() for legend in fig.findobj(lambda a: type(a).__name__ == "Legend") for t in legend.get_texts()]
    assert [k for k in keys if k.startswith("H&E")] == [f"H&E {k}" for k in range(4)], f"the H&E classes' key: {keys}"
    assert len([k for k in keys if k.startswith("Clone")]) == len(d["initial"]), f"the initial clones' key: {keys}"
    assert {t.get_text() for t in fig.texts} >= {"(a)", "(b)"}, "the panel letters"
    _on_page(fig, "third")


def _genomic_axis(_: Any) -> None:
    from cnamaste.layout import GenomicAxis

    genome = GenomicAxis.from_bins([1, 1, 2, 2, 2, 10])
    assert genome.lengths.tolist() == [2, 3, 1] and genome.names == ("1", "2", "10") and genome.starts.tolist() == [0, 2, 5]
    assert genome.n_bins == 6
    with pytest.raises(AssertionError):
        GenomicAxis.from_bins([1, 2, 1])


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
         "the all-spot pseudobulk, no result (run_cnamaste.py:219): a non-empty PDF"),
        ("plot_genomic:plot_clones_genomic", COUNTS, figure_inputs, rendered("rdr_baf_clones_genomic", _clones),
         "per clone with the fit, chisel_single (run_cnamaste.py:1145): a non-empty PDF"),
        ("plot_genomic:plot_clones_genomic", COUNTS, figure_inputs, rendered("clones_genomic", _segments),
         "per clone with cnv_seglevel.tsv's integer states (run_cnamaste.py:1640): a non-empty PDF"),
        ("plot_genomic:plot_clones_genomic", COUNTS + "; 03_phasing/initial_phase_given_partition out", figure_inputs,
         rendered("postphasing_clones_genomic", _postphasing),
         "phasing's result, its parameters (n_states,), on the initial clones (run_cnamaste.py:406): a non-empty PDF"),
        ("plotting:plot_clones_spatial", COUNTS, figure_inputs, rendered("clones_spatial", _spatial),
         "the final labels on the coordinates (run_cnamaste.py:1699): a non-empty PDF"),
        ("plot_copy_number_profile:plot_copy_number_profile", COUNTS, figure_inputs, rendered("copy_number_profile", _profile),
         "cnv_seglevel.tsv's allele-specific profile (run_cnamaste.py:1714): a non-empty PDF"),
        ("layout:plot_combined_page", COUNTS, figure_inputs, _combined(False),
         "combined.png, no H&E: full page, a blank slide panel; profile and tracks on one genomic axis and one clone order"),
        ("layout:plot_combined_page", COUNTS + "; a synthetic slide over easy's spots", figure_inputs, _combined(True),
         "combined.png with an H&E slide: slide and clones on one extent"),
        ("layout:plot_he_page", COUNTS + "; 03_phasing/initialize_clones out; a synthetic slide over easy's spots", figure_inputs, _he_page,
         "he_clones.png: a third of a page; H&E classes 0-3 and the initial clones on one extent"),
        ("layout:GenomicAxis.from_bins", "synthetic: 6 bins, 3 chromosomes", lambda c: None, _genomic_axis,
         "one run per chromosome, in bin order; a chromosome's bins split is refused"),
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
         "logs go to stdout, which is where the config says (it names no log file); perf: test_fn_normal's flush_perf row (<paths.output_dir>/cnamaste.perf)"),
    ),
]


@pytest.mark.parametrize("row", ROWS, ids=[f"{r.kind}: {r.id}" for r in ROWS])
def test_smoke(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """Runs and writes what it should; asserts no more than the row says."""
    run(row, ctx, request)
