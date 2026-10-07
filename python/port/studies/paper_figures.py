"""The paper's figure set for one `dev_tree` fixture's r0: truth, run, and the two compared (#624).

    run_study --paper-figures [--fixture dev_tree_1s_easy] [--draw DIR]
        [--out docs/plots/paper] [--truth-only]
    run_study --paper-figures --solvers POTTS.record COPY.record [--out docs/plots/paper]

draws `sim/manifests/<fixture>.toml`'s r0 (or reads it from `--draw`, the
directory holding `<fixture>/r0`), refuses it unless it hashes to the
manifest's `r0_hash`, and writes into `OUT`:

- `truth/`, figures 1-8: `port.sim.analysis`'s figures and
  `port.sim.truth_figure.truth_combined_figure`, no run needed;
- `run/`, figures 9-13: one `run_cnaster_port --sal --png-copies` run through
  `port.qa.audit.audit_sample`, its named PNGs copied out, and the combined,
  genomic and spatial pages drawn from it as `run_figures` draws
  them, beside a slide mocked from the planted labels (`port.sim.he_slide`);
- `compare/`, figures 14-17: the run against the truth, through
  `port.qa.audit.score_sample`'s matching, `copy_confusion` and planted classes;
- `solver_combined.png`, figure 18 (`--solvers POTTS.record COPY.record`, no
  fixture run), at the top level beside the run's pages: the spatial solvers
  (`port.studies.potts_plot`) left of the copy-state starts
  (`port.studies.copy_state_plot`), each keyed below, no table (T- #660), on
  the text column `combined.png` is drawn on.

Every figure carries `<fixture> <hash> · code <sha>`, the commit read before
anything is written, `+` where the tree differs from it. A run is appended to
the metrics ledger (`port.qa.ledger.write`) as `<fixture>_r0`, its hash in
the ledger's `fixture_hash` column (`<fixture>_ln_r0` before T- #660), and
`OUT/README.md` is written with its `run_id`.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import shutil
import tempfile
import tomllib
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from port.extensions.figure_style import GRID, INK, MUTED, axes_style
from port.qa import provenance
from port.qa.provenance import ROOT
from port.qa.statistics import measured

MANIFESTS = ROOT / "sim" / "manifests"
OUT = ROOT / "docs" / "plots" / "paper"
FLAGS = ("--sal", "--png-copies")
"""The run: `--sal`, with a PNG written beside each figure's PDF."""

TRUTH = (
    "truth_combined.png",
    "simulated_tree.png",
    "spatial.png",
    "clones_genomic.png",
    "clone_profiles.png",
    "coverage.png",
    "phase.png",
    "baseline.png",
)
"""Figures 1-8, written under `truth/`."""

RUN_COPIED = (
    "copy_number_profile.png",
    "clones_spatial.png",
    "clones_genomic.png",
    "rdr_baf_clones_genomic.png",
)
RUN_PAGES = ("combined.png", "genomic.png", "spatial.png")
"""Figures 9-13, under `run/`: the pages drawn here, the rest copied."""

COMPARE = (
    "clones_truth_vs_fit.png",
    "copy_confusion.png",
    "copy_genomic_truth_vs_fit.png",
    "exact_by_class.png",
)
"""Figures 14-17, under `compare/`."""

CLASSES = {
    "loh": "LOH",
    "balanced_gain": "balanced gain",
    "unbalanced_gain": "unbalanced gain",
    "neutral": "neutral",
}
"""`port.qa.scoring.CLASSES`, in figure 17's order, with their labels."""

DPI = 150
"""`port.sim.analysis`'s and `port.pipeline.FIGURE_DPI`'s."""

WRONG = "#e34948"
SWAPPED = "#4a3aa7"
"""Figure 16's marks: a bin decoded to another pair, and one decoded to the planted pair's swap."""


def stamp(figure: Any, text: str, *, top: bool = False) -> None:
    """`text` in the figure's bottom-right corner, or its top-right where a
    page fills the bottom, muted, at `MIN_FONT_SIZE`, in the stated face."""
    from port.extensions.figure_style import MIN_FONT_SIZE, figure_font

    y, va = (0.998, "top") if top else (0.002, "bottom")
    # NB a warped genomic axis says so (`genomic_axis.disclose`, T- #683).
    label = figure.get_label()
    if label.startswith("axis:"):
        text = f"{text} · {label}"
    with figure_font():
        figure.text(
            0.998, y, text, ha="right", va=va, fontsize=MIN_FONT_SIZE, color=MUTED
        )


@contextlib.contextmanager
def stamping(text: str) -> Iterator[None]:
    """Stamp every figure `port.sim.analysis` saves and the run writes, for the block."""
    from port.patch import utils
    from port.sim import analysis

    save, write_fig = analysis._save, utils.write_fig

    def stamped_save(figure: Any, path: Path, **kwargs: Any) -> Path:
        stamp(figure, text)
        return save(figure, path, **kwargs)

    def stamped_write(opath: str, fig: Any = None, *args: Any, **kwargs: Any) -> None:
        if fig is not None:
            stamp(fig, text)
        write_fig(opath, fig, *args, **kwargs)

    analysis._save = stamped_save  # type: ignore[assignment]
    utils.write_fig = stamped_write
    try:
        yield
    finally:
        analysis._save, utils.write_fig = save, write_fig


def switches(path: Path) -> int:
    """Phase switches `truth_phase.npy` plants: SNPs whose phase differs from the previous one's."""
    phase = np.load(path / "truth_phase.npy").astype(bool)
    return int(np.count_nonzero(phase[1:] != phase[:-1]))


def stated_hash(fixture: str) -> str:
    """`[sample] r0_hash` as `sim/manifests/<fixture>.toml` states it."""
    document = tomllib.loads((MANIFESTS / f"{fixture}.toml").read_text())
    return str(document["sample"]["r0_hash"])


def realization(fixture: str, draw: Path | None) -> Path:
    """`<draw>/<fixture>/r0`, drawn into `.cache/paper_figures/` where no `draw`
    is given; refused unless it hashes to the manifest's `r0_hash`."""
    from port.sim.fixtures import realization_hash

    if draw is None:
        draw = ROOT / ".cache" / "paper_figures"
        path = draw / fixture / "r0"
        if not (path / "truth_clone_labels.tsv").is_file():
            from dataclasses import replace

            from port.sim.draw import _merge, read_manifest
            from port.sim.draw import draw as drawn
            from port.sim.fixtures import references

            manifest = read_manifest(MANIFESTS / f"{fixture}.toml")
            one = {"sample": {"realizations": 1}}
            manifest = replace(manifest, tables=_merge(manifest.tables, one))
            drawn(manifest, draw, resources=references())

    path = draw / fixture / "r0"
    found, stated = realization_hash(path), stated_hash(fixture)
    if found != stated:
        msg = f"{path} hashes to {found}, not the manifest's {stated}"
        raise RuntimeError(msg)
    return path


def truth_figures(path: Path, out: Path, text: str) -> list[Path]:
    """Figures 1-8 into `out`, stamped."""
    import matplotlib.pyplot as plt

    from port.extensions.combined_figure import page_style
    from port.sim import analysis
    from port.sim.truth_figure import simulated_tree_figure, truth_combined_figure

    r = analysis.read(path)
    with stamping(text):
        written = r.plot(out, metric=True)

    # NB `mutation_tree.png` is not a paper figure: `simulated_tree.png` is
    #    `truth_combined`'s panel (a) alone (T- #660).
    (out / "mutation_tree.png").unlink(missing_ok=True)
    written = [w for w in written if w.name != "mutation_tree.png"]
    pages = (
        ("truth_combined.png", truth_combined_figure(r, metric=True)),
        ("simulated_tree.png", simulated_tree_figure(r)),
    )
    for name, figure in pages:
        stamp(figure, text, top=True)
        with page_style():
            figure.savefig(
                out / name, dpi=300, facecolor="white",
                metadata={"Software": None},
            )  # fmt: skip
        plt.close(figure)
    return [*(out / name for name, _ in pages), *written]


def mock_slide(coords: np.ndarray, labels: np.ndarray, root: Path) -> Any:
    """A slide stained by the planted clones (`port.sim.he_slide`), read back as
    `run_cnaster` reads one; each lattice cell takes its nearest spot's clone.

    Written beside the run's inputs, not into them, so the run never reads it.
    """
    from cnaster.he import get_he_image
    from scipy.spatial import cKDTree

    from port.sim.he_slide import mock_he, write_he_slide

    lattice = (int(coords[:, 0].max()) + 1, int(coords[:, 1].max()) + 1)
    cells = np.indices(lattice).reshape(2, -1).T
    _, nearest = cKDTree(coords).query(cells)
    write_he_slide(mock_he(labels[nearest], lattice), root / "slide")
    return get_he_image(str(root / "slide"), res="hires", pos=None)


@dataclass
class Run:
    """One `--sal` run on the realization, scored."""

    recovery: dict[str, Any]
    output: Path
    wall: float
    peak_gb: float


def run_figures(sample: Any, root: Path, out: Path, text: str) -> Run:
    """Figures 9-13 into `out` from one run under `root`, stamped."""
    import matplotlib.pyplot as plt

    from port.extensions.combined_figure import (
        combined_figure,
        genomic_figure,
        page_style,
        recording,
        spatial_figure,
    )
    from port.qa.audit import audit_sample

    with stamping(text), recording() as recorded, measured() as cost:
        recovery, output = audit_sample(sample, list(FLAGS), None, root / "run")

    frame = mock_slide(sample.coords, sample.labels, root)

    # NB at their declared size, not a tight box, as `run_figures`
    #    writes them: the page is included at 1:1.
    with page_style():
        for name, figure in (
            ("genomic.png", genomic_figure(recorded, metric=True)),
            ("spatial.png", spatial_figure(recorded, frame)),
            ("combined.png", combined_figure(recorded, frame, metric=True)),
        ):
            stamp(figure, text, top=True)
            figure.savefig(
                out / name, dpi=300, facecolor="white", metadata={"Software": None}
            )
            plt.close(figure)

    for name in RUN_COPIED:
        shutil.copy(next(output.rglob(f"plots/{name}")), out / name)

    return Run(asdict(recovery), output, round(cost.wall_s, 1), round(cost.peak_gb, 2))


@dataclass
class Compared:
    """The truth and the fit, as figures 14-17 read them.

    Spots: `coords`, `planted` and `fitted` labels (`-1` unscored). Bins: the
    covered clone-bins of `port.qa.audit.score_sample`, per matched clone, in
    `clone_of`'s order, with genome coordinates `start`, `end` and the
    chromosome `edges`.
    """

    coords: np.ndarray
    planted: np.ndarray
    fitted: np.ndarray
    names: tuple[str, ...]
    clone_of: dict[int, int]
    ari: float
    start: np.ndarray
    end: np.ndarray
    edges: np.ndarray
    truth: np.ndarray
    """`(n_bins, n_matched, 2)` planted `(A, B)`."""
    decoded: np.ndarray
    """`(n_bins, n_matched, 2)` decoded `(A, B)` of the matched fitted clone."""


def compared(
    sample: Any, path: Path, output: Path, recovery: dict[str, Any]
) -> Compared:
    """`score`'s bins, rebuilt, and refused unless they give its `confusion` and `exact`."""
    from port.extensions.combined_figure import clone_symbol
    from port.extensions.integer_copy import DEFAULT_MAX_TOTAL_COPY
    from port.qa.audit import read_run
    from port.qa.scoring import copy_confusion
    from port.sim.analysis import display, read

    r = read(path)
    run = read_run(sample, output)
    seglevel = run["seglevel"]
    middle = ((seglevel["START"] + seglevel["END"]) // 2).to_numpy()
    chromosome = seglevel["CHR"].astype(str).str.removeprefix("chr").to_numpy()
    planted = sample.copies_at(chromosome, middle)
    covered = planted[:, 0, 0] >= 0
    clone_of = {int(k): int(v) for k, v in recovery["clone_of"].items()}

    truth = np.stack([planted[covered, c] for c in clone_of], 1)
    decoded = np.stack(
        [np.stack([run["a"][covered, f], run["b"][covered, f]], 1)
         for f in clone_of.values()], 1,
    )  # fmt: skip
    t = (truth[..., 0] * 1_000 + truth[..., 1]).T.ravel()
    ab = (decoded[..., 0] * 1_000 + decoded[..., 1]).T.ravel()

    if copy_confusion(t, ab, DEFAULT_MAX_TOTAL_COPY) != recovery["confusion"]:
        msg = "the rebuilt bins do not give score's confusion"
        raise RuntimeError(msg)
    if round(float(np.mean(t == ab)), 4) != recovery["exact"]:
        msg = "the rebuilt bins do not give score's exact"
        raise RuntimeError(msg)

    chromosome, starts, ends = (
        chromosome[covered],
        seglevel["START"].to_numpy()[covered],
        seglevel["END"].to_numpy()[covered],
    )
    return Compared(
        coords=sample.coords,
        planted=sample.labels,
        fitted=run["labels"],
        names=tuple(clone_symbol(display(c, r.clones)) for c in sample.clones),
        clone_of=clone_of,
        ari=float(recovery["ari"]),
        start=r.genome(chromosome, starts),
        end=r.genome(chromosome, ends),
        edges=np.concatenate([r.offsets, [r.lengths.sum()]]).astype(np.float64),
        truth=truth,
        decoded=decoded,
    )


def _colours(n: int) -> list[str]:
    """`port.sim.analysis`'s clone colours: the normal grey, then the series."""
    from port.sim.analysis import NEUTRAL, SERIES

    return [NEUTRAL, *(SERIES[k % len(SERIES)] for k in range(n - 1))]


def labels_figure(c: Compared) -> Any:
    """14: planted and fitted clone per spot; a fitted clone takes its matched planted clone's colour."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colours = _colours(len(c.names))
    planted_of = {f: p for p, f in c.clone_of.items()}
    n_fitted = int(c.fitted.max()) + 1
    extra = ["#d4d2cc", "#8a8882", "#2b2a28"]
    fitted_colour = [
        colours[planted_of[f]] if f in planted_of else extra[f % len(extra)]
        for f in range(n_fitted)
    ]
    # NB `port.sim.analysis.draw_spatial`'s frame: hex rows at sqrt(3)/2.
    x, y = c.coords[:, 1] / 2.0, -c.coords[:, 0] * np.sqrt(3.0) / 2.0
    aspect = np.ptp(y) / max(np.ptp(x), 1.0)
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.3 * aspect + 1.2))
    size = 7.0 if c.coords.shape[0] > 2_000 else 30.0

    for ax, title, labels, palette in (
        (axes[0], "planted", c.planted, colours),
        (axes[1], "fitted", c.fitted, fitted_colour),
    ):
        scored = labels >= 0
        ax.scatter(
            x[scored], y[scored], s=size, edgecolors="white", linewidths=0.2,
            c=[palette[k] for k in labels[scored]],
        )  # fmt: skip
        ax.scatter(x[~scored], y[~scored], s=size, marker="x", linewidths=0.4,
                   color=MUTED)  # fmt: skip
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)

    key = [
        Line2D([], [], marker="o", linestyle="", color=colours[p], markersize=5,
               label=f"{c.names[p]} $\\leftrightarrow$ fit {f}")
        for p, f in c.clone_of.items()
    ]  # fmt: skip
    key += [
        Line2D([], [], marker="o", linestyle="", color=fitted_colour[f],
               markersize=5, label=f"fit {f}, unmatched")
        for f in range(n_fitted) if f not in planted_of
    ]  # fmt: skip
    figure.legend(
        handles=key, loc="lower center", ncol=min(len(key), 5), fontsize=7,
        frameon=False, bbox_to_anchor=(0.5, 0.04),
    )  # fmt: skip
    # NB no title: the clone ARI is the README's TL;DR (T- #660).
    figure.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.14, wspace=0.05)
    return figure


def _codes(pairs: np.ndarray) -> np.ndarray:
    return np.asarray(pairs[..., 0] * 1_000 + pairs[..., 1])


def confusion_figure(c: Compared) -> Any:
    """15: `port.qa.scoring.copy_confusion` as a heatmap, planted rows and decoded columns that occur."""
    import matplotlib.pyplot as plt

    from port.extensions.integer_copy import DEFAULT_MAX_TOTAL_COPY
    from port.qa.scoring import OTHER, copy_confusion, copy_states

    t, ab = _codes(c.truth).T.ravel(), _codes(c.decoded).T.ravel()
    confusion = copy_confusion(t, ab, DEFAULT_MAX_TOTAL_COPY)
    pairs = [f"{a},{b}" for a, b in copy_states(DEFAULT_MAX_TOTAL_COPY)]
    decoded = {d for row in confusion.values() for d in row}
    rows = [p for p in pairs if p in confusion]
    columns = [p for p in pairs if p in decoded] + ([OTHER] if OTHER in decoded else [])
    matrix = np.array([[confusion[r].get(k, 0.0) for k in columns] for r in rows])
    counts = {r: int(np.sum(t == int(r.split(",")[0]) * 1_000 + int(r.split(",")[1])))
              for r in rows}  # fmt: skip

    figure, ax = plt.subplots(
        figsize=(1.6 + 0.62 * len(columns), 1.2 + 0.42 * len(rows))
    )
    image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    for i, j in np.ndindex(matrix.shape):
        if matrix[i, j] > 0:
            ax.text(
                j, i, f"{matrix[i, j]:.3f}" if matrix[i, j] >= 5e-4 else "<0.001", ha="center", va="center", fontsize=7,
                color="white" if matrix[i, j] > 0.55 else INK,
            )  # fmt: skip
    ax.set_xticks(range(len(columns)), [f"({k})" if k != OTHER else k for k in columns])
    ax.set_yticks(range(len(rows)), [f"({r})  n={counts[r]}" for r in rows])
    ax.set_xlabel("decoded (A, B)", fontsize=8, color=INK)
    ax.set_ylabel("planted (A, B), clone-bins", fontsize=8, color=INK)
    ax.tick_params(labelsize=7, colors=MUTED, labelcolor=INK)
    bar = figure.colorbar(image, ax=ax, fraction=0.05, pad=0.03)
    bar.set_label("share of the planted row", fontsize=7)
    bar.ax.tick_params(labelsize=6)
    ax.set_title("planted against decoded, matched clones", fontsize=9, color=INK)
    figure.tight_layout()
    return figure


MB_PAD = 12.0
"""Points from the axis to the chromosome numbers, under the Mb labels."""


def genomic_compare_figure(c: Compared, *, metric: bool = False) -> Any:
    """16: planted and decoded `(A, B)` along the genome per matched clone,
    mismatched bins marked; on `metric`, every bin planted or decoded altered
    in any matched clone drawn wider (T- #683)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from port.extensions.genomic_axis import GenomicAxis, disclose
    from port.sim.analysis import SERIES

    lengths = np.diff(c.edges)
    altered = np.any(c.truth != 1, axis=(1, 2)) | np.any(c.decoded != 1, axis=(1, 2))
    genome = GenomicAxis(
        lengths.astype(np.int64),
        np.column_stack([c.start, c.end])[altered] if metric else None,
    )
    start, end, edges = genome.warp(c.start), genome.warp(c.end), genome.edges
    n = len(c.clone_of)
    top = int(max(c.truth.max(), c.decoded.max(), 2))
    figure, axes = plt.subplots(n, 1, figsize=(7.0, 0.4 + 1.25 * n), sharex=True)
    axes = np.atleast_1d(axes)
    allele = ((0, SERIES[0], 0.07), (1, SERIES[1], -0.07))

    for k, (ax, (p, f)) in enumerate(zip(axes, c.clone_of.items(), strict=True)):
        truth, decoded = c.truth[:, k], c.decoded[:, k]
        for column, colour, shift in allele:
            ax.hlines(truth[:, column] + shift, start, end, color=colour,
                      linewidth=3.0, alpha=0.35)  # fmt: skip
            ax.hlines(decoded[:, column] + shift, start, end, color=colour,
                      linewidth=0.9)  # fmt: skip
        same = np.all(truth == decoded, axis=1)
        swapped = ~same & np.all(truth == decoded[:, ::-1], axis=1)
        wrong = ~same & ~swapped
        middle = (start + end) / 2
        ax.plot(middle[swapped], np.full(swapped.sum(), -0.75), "|", color=SWAPPED,
                markersize=5, markeredgewidth=0.6)  # fmt: skip
        ax.plot(middle[wrong], np.full(wrong.sum(), -0.75), "|", color=WRONG,
                markersize=5, markeredgewidth=0.6)  # fmt: skip
        for edge in edges:
            ax.axvline(edge, color=GRID, linewidth=0.6, zorder=0)
        ax.set_ylim(-1.0, top + 0.5)
        ax.set_yticks(range(top + 1))
        axes_style(ax, labelsize=7, grid=False)
        ax.set_ylabel(f"{c.names[p]} / fit {f}", fontsize=7, color=INK)
        ax.text(
            1.0, 1.0, f"{int(swapped.sum())} swapped, {int(wrong.sum())} wrong of {same.size}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color=MUTED,
        )  # fmt: skip

    axes[-1].set_xticks((edges[:-1] + edges[1:]) / 2,
                        [str(i) for i in range(1, lengths.size + 1)])  # fmt: skip
    axes[-1].tick_params(axis="x", labelsize=6)
    # NB a tick every 10 Mb, labelled on the last row; the chromosome numbers
    #    a row below their Mb (T- #683).
    for ax in axes:
        genome.draw(ax, labels=ax is axes[-1])
    axes[-1].tick_params(axis="x", which="major", pad=MB_PAD)
    axes[-1].set_xlim(edges[0], edges[-1])
    axes[-1].set_xlabel("chromosome", fontsize=8, color=INK)
    key = [
        Line2D([], [], color=SERIES[0], linewidth=3.0, alpha=0.35, label="planted A"),
        Line2D([], [], color=SERIES[0], linewidth=0.9, label="decoded A"),
        Line2D([], [], color=SERIES[1], linewidth=3.0, alpha=0.35, label="planted B"),
        Line2D([], [], color=SERIES[1], linewidth=0.9, label="decoded B"),
        Line2D([], [], color=SWAPPED, marker="|", linestyle="", label="decoded (B, A)"),
        Line2D([], [], color=WRONG, marker="|", linestyle="", label="decoded other"),
    ]
    figure.legend(handles=key, loc="upper center", ncol=6, fontsize=6.5,
                  frameon=False)  # fmt: skip
    figure.tight_layout(rect=(0, 0.01, 1, 0.97))
    disclose(figure, genome)
    return figure


def exact_by_class(c: Compared) -> dict[str, tuple[float, float, int]]:
    """Per `CLASSES`: exact, exact phase-free, and clone-bins; NaN where none is planted."""
    from port.qa import scoring

    t, ab = _codes(c.truth).T.ravel(), _codes(c.decoded).T.ravel()
    return scoring.exact_by_class(t, ab)


def exact_figure(c: Compared) -> Any:
    """17: exact recovery by planted class, phased and phase-free."""
    import matplotlib.pyplot as plt

    from port.sim.analysis import SERIES

    shares = exact_by_class(c)
    figure, ax = plt.subplots(figsize=(5.0, 2.8))
    x = np.arange(len(CLASSES))
    for offset, column, colour, label in (
        (-0.19, 0, SERIES[0], "phased: (A, B)"),
        (0.19, 1, SERIES[2], "phase-free: (A, B) or (B, A)"),
    ):
        values = np.array([shares[k][column] for k in CLASSES])
        ax.bar(x + offset, np.nan_to_num(values), width=0.36, color=colour, label=label)
        for xi, v in zip(x + offset, values, strict=True):
            if not np.isnan(v):
                ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=6.5, color=INK)
    names = [
        f"{label}\nn={shares[k][2]}" if shares[k][2] else f"{label}\nnot planted"
        for k, label in CLASSES.items()
    ]
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel("exact share of clone-bins", fontsize=8, color=INK)
    ax.legend(fontsize=7, frameon=False, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, 1.13))  # fmt: skip
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    axes_style(ax, labelsize=7, grid=False)
    figure.tight_layout()
    return figure


FIGURES: dict[str, Callable[[Compared], Any]] = {
    "clones_truth_vs_fit.png": labels_figure,
    "copy_confusion.png": confusion_figure,
    "copy_genomic_truth_vs_fit.png": functools.partial(
        genomic_compare_figure, metric=True
    ),
    "exact_by_class.png": exact_figure,
}


def compare_figures(c: Compared, out: Path, text: str) -> list[Path]:
    """Figures 14-17 into `out`, stamped, in the stated face."""
    import matplotlib.pyplot as plt

    from port.extensions.figure_style import figure_font

    out.mkdir(parents=True, exist_ok=True)
    written = []
    with figure_font():
        for name, draw in FIGURES.items():
            figure = draw(c)
            stamp(figure, text)
            figure.savefig(out / name, dpi=DPI, facecolor="white",
                           metadata={"Software": None})  # fmt: skip
            plt.close(figure)
            written.append(out / name)
    return written


SOLVERS = "solver_combined.png"
"""Figure 18, beside the run's pages in `docs/plots/paper/`."""

SOLVER_PANEL = 2.2
"""Inches: each panel's axes height. (a) left of (b), each keyed below in one column, on one text-column page."""


def solver_figure(potts: dict[str, Any], copies: dict[str, Any]) -> Any:
    """18: the spatial solvers' gap panel left of the copy-state starts', each keyed below its axes (`figures.key_below`).

    Sized as `combined.png` is: the paper's text column, every text at
    `combined_figure.FONT_SIZE`, saved at 300 dpi, so it is included at
    `width=\\linewidth` with nothing scaled.
    """
    import matplotlib.pyplot as plt
    from matplotlib.text import Text

    from port.extensions.combined_figure import FONT_SIZE
    from port.extensions.figure_style import PAPER_WIDTH
    from port.studies import copy_state_plot, potts_plot

    width = PAPER_WIDTH
    line = FONT_SIZE * 1.5 / 72.0
    # NB the key's rows under the x label: the longer of the two panels' methods sets the height
    rows = max(len(potts_plot.KEY_NAMES), len(copy_state_plot.KEY_NAMES))
    above, label, foot = 0.2, 0.4, 0.12
    height = above + SOLVER_PANEL + label + line * (rows + 1) + foot
    figure = plt.figure(figsize=(width, height))
    # NB (b) shares (a)'s y label, so the gap between them holds only (b)'s tick labels
    left, gap, right = 0.42, 0.3, 0.06
    panel = (width - left - gap - right) / 2
    bottom = height - above - SOLVER_PANEL
    axes = {k: figure.add_axes(((left + i * (panel + gap)) / width, bottom / height, panel / width, SOLVER_PANEL / height))
            for i, k in enumerate("ab")}  # fmt: skip
    style = {
        "fontsize": FONT_SIZE,
        "row": line / SOLVER_PANEL,
        "top": -label / SOLVER_PANEL,
        "columns": 1,
    }
    potts_plot.draw(axes["a"], potts, key=True, centre=True, key_style=style)
    # NB Initial, Polish and Truth mean the same in both panels: keyed once, under (a)
    copy_state_plot.draw(
        axes["b"], copies, key=True, key_style={**style, "marks": False}
    )
    for text in figure.findobj(Text):
        text.set_fontsize(FONT_SIZE)
    # NB each key's title names the fixture; at half the page two titles and two letters
    #    collide, so the fixture moves to the stamp, each title keeps its count and starts at
    #    its panel's left edge, the letter before it
    fixture = Path(potts["manifest"]).stem
    for text in figure.findobj(Text):
        if text.get_text().startswith(f"{fixture}: "):
            text.set_text(text.get_text().removeprefix(f"{fixture}: ").capitalize())
            text.set_position((0.0, 1.01))
            text.set_horizontalalignment("left")
    axes["b"].set_ylabel("")
    for k, ax in axes.items():
        ax.tick_params(labelsize=FONT_SIZE, length=2.5, pad=1.5)
        ax.text(-0.27 / panel, 1.01, f"({k})", transform=ax.transAxes, fontsize=FONT_SIZE,
                ha="left", va="bottom", color=INK)  # fmt: skip
    return figure


def solver_figures(potts: Path, copies: Path, out: Path, commit: str) -> Path:
    """Figure 18 into `out`, stamped with each record's data hash and the code."""
    import matplotlib.pyplot as plt

    from port.extensions.figure_style import figure_font
    from port.studies import records as stored

    out.mkdir(parents=True, exist_ok=True)
    records = [stored.read(p) for p in (potts, copies)]
    text = (
        f"{Path(records[0]['manifest']).stem} · potts {stored.digest(records[0])}"
        f" · copy states {stored.digest(records[1])}"
        f" · code {commit}"
    )
    with figure_font():
        figure = solver_figure(*records)
        stamp(figure, text)
        figure.savefig(out / SOLVERS, dpi=300, facecolor="white",
                       metadata={"Software": None})  # fmt: skip
        plt.close(figure)
    return out / SOLVERS


QUESTIONS: dict[str, tuple[str, str]] = {
    "truth/truth_combined.png": (
        "What was planted, on one page: tree, (A, B) profile, RDR and BAF per clone?",
        "`port.sim.truth_figure.truth_combined_figure`",
    ),
    "truth/simulated_tree.png": (
        "Which events sit on which edge of the simulated clone tree?",
        "`port.sim.truth_figure.simulated_tree_figure`, `truth_combined`'s panel (a) alone",
    ),
    "truth/spatial.png": (
        "Which clone was each spot drawn from?",
        "`port.sim.analysis.plot_spatial`",
    ),
    "truth/clones_genomic.png": (
        "What RDR and BAF does each planted clone give along the genome?",
        "`port.sim.analysis.plot_clones_genomic_truth`",
    ),
    "truth/clone_profiles.png": (
        "What (A, B) does each clone carry along the genome?",
        "`port.sim.analysis.plot_clone_profiles`",
    ),
    "truth/coverage.png": (
        "Do spot UMI and SNP reads follow their laws?",
        "`port.sim.analysis.plot_coverage`",
    ),
    "truth/phase.png": (
        "Where does the planted phase switch?",
        "`port.sim.analysis.plot_phase`",
    ),
    "truth/baseline.png": (
        "What normal expression baseline were counts drawn from?",
        "`port.sim.analysis.plot_baseline`",
    ),
    "run/combined.png": (
        "What did the run fit, genome and array on one page?",
        "`port.extensions.combined_figure.combined_figure`",
    ),
    "run/genomic.png": (
        "What RDR, BAF and integer (A, B) did the run fit per clone?",
        "`port.extensions.combined_figure.genomic_figure`",
    ),
    "run/spatial.png": (
        "Where are the fitted clones on the array?",
        "`port.extensions.combined_figure.spatial_figure`",
    ),
    "run/copy_number_profile.png": (
        "What integer (A, B) did the run decode per clone?",
        "the run's `plots/copy_number_profile.png`",
    ),
    "run/clones_spatial.png": (
        "Which fitted clone is each spot?",
        "the run's `plots/clones_spatial.png`",
    ),
    "run/clones_genomic.png": (
        "What RDR and BAF did the run fit per clone?",
        "the run's `plots/clones_genomic.png`",
    ),
    "run/rdr_baf_clones_genomic.png": (
        "What RDR and BAF were the clones fitted to?",
        "the run's `plots/rdr_baf_clones_genomic.png`",
    ),
    "compare/clones_truth_vs_fit.png": (
        "Do the fitted clones recover the planted ones, and which matches which?",
        "`labels_figure`: `port.qa.audit.score_sample`'s ARI and matching",
    ),
    "compare/copy_confusion.png": (
        "Which (A, B) is each planted pair decoded as?",
        "`confusion_figure`: `port.qa.scoring.copy_confusion`",
    ),
    "compare/copy_genomic_truth_vs_fit.png": (
        "Where along the genome is a matched clone's (A, B) decoded wrong, or swapped?",
        "`genomic_compare_figure`: `score`'s clone-bins",
    ),
    "compare/exact_by_class.png": (
        "Which planted classes are recovered exactly, with and without phase?",
        "`exact_figure`: `port.qa.scoring.exact_by_class`",
    ),
    "solver_combined.png": (
        "How far above the best does each spatial solver and each copy-state start end, and how fast?",
        "`solver_figure`: `port.studies.potts_plot.draw`, `port.studies.copy_state_plot.draw`",
    ),
}
"""Each committed file under `OUT`: the question it answers, and its source."""

KEY_STUDIES: dict[str, tuple[str, str, str]] = {
    "key_studies/557_copy-states.png": (
        "Which copy-state start, polished by `--sal` Baum-Welch, recovers the planted states at known clones?",
        "`port.studies.copy_state_plot` (#540, PR #557)",
        "`run_study --copy-state-stream sim/manifests/baseline/dev_tree_1s_hard.toml OUT "
        "--problems 10 --seeds 10 --held-out 3 --settings python/port/studies/copy_sampler_settings.json`",
    ),
    "key_studies/554_clone-starts.png": (
        "Does the clone-label start or the Potts solver decide the clones, and what does each start reach?",
        "`docs/nb/clone_label_study.ipynb` via `port.studies.clone_label_notebook` (#541, PR #554)",
        "`run_study --clone-labels capture SAMPLE CAPTURE.npz`, `... run CAPTURE.npz OUT.pkl`, "
        "`run_study --clone-label-notebook OUT.pkl`",
    ),
    "key_studies/546_population.png": (
        "At J = 1, how many UMIs does a clone need to be detected, how long must a CNA be to be recovered, "
        "and how often is a true-(1,1) segment called altered?",
        "`port.studies.population_report.figures` (#544, PR #546)",
        "`run_study --population run --seeds 0:200 --J 1 --out DIR`, then `... run --seeds 1000:1260 "
        "--J 1 --manifest sim/manifests/population_long.toml --out DIR`, then `... report --out DIR --study2-J 1`",
    ),
}
"""Each key study's figure (label `key_study`) under `OUT/key_studies/`: question, source, regenerate command."""


def ledger_name(fixture: str) -> str:
    """The run's fixture name in the metrics ledger; its hash, the generation, is its own column."""
    return f"{fixture}_r0"


def readme(
    fixture: str,
    digest: str,
    commit: str,
    run_id: str,
    recovery: dict[str, Any],
    switches: int,
) -> str:
    """`OUT/README.md`: the run's headline metrics, then each file's question and source."""
    rows = "\n".join(f"| `{k}` | {q} | {s} |" for k, (q, s) in QUESTIONS.items())
    studies = "\n".join(
        f"| `{k}` | {q} | {s} | {r} |" for k, (q, s, r) in KEY_STUDIES.items()
    )
    return f"""# Paper figures: {fixture} r0 ({digest})

**TL;DR:** clone ARI {recovery["ari"]:.4f}, copy ARI (phase-free)
{recovery["copy_ari_pf"]:.4f}, exact altered (phase-free)
{recovery["exact_altered_minor"]:.4f} (phased {recovery["exact_altered"]:.4f}), from one
`run_cnaster_port {" ".join(FLAGS)}` run on `sim/manifests/{fixture}.toml` r0 at
code `{commit}`: {recovery["wall"]:.0f} s wall, {recovery["peak_gb"]:.2f} GB peak.
Ledger `run_id` `{run_id}` (`docs/metrics/`, fixture `{ledger_name(fixture)}` `{digest}`).

Regenerate from a clean tree, so the stamp carries no `+`; it draws r0 into
`.cache/paper_figures/` where `--draw` is not given, and refuses any r0 not
hashing to `{digest}`:

    run_study --paper-figures --fixture {fixture} --out docs/plots/paper

`--truth-only` writes `truth/` alone, with no run. Every figure is stamped
`{fixture} {digest} · code <sha>`. `run/spatial.png` and `run/combined.png`
draw panel (a) on a slide mocked from the planted labels (`port.sim.he_slide`):
the fixture has no H&E image, and the run never reads the mock.
`truth/phase.png` is flat: this r0 plants {switches} phase switches.
`solver_combined.png` is not drawn from this fixture: `--solvers POTTS.record COPY.record`
draws it from a `port.studies.potts_stream` and a `port.studies.copy_state_stream` record,
and its stamp names both records' data hashes.
The genomic panels of `truth/truth_combined.png`, `truth/clones_genomic.png`,
`truth/clone_profiles.png`, `run/genomic.png`, `run/combined.png` and
`compare/copy_genomic_truth_vs_fit.png` draw every altered bin at 2x its extent
(`port.extensions.genomic_axis`, T- #683), and their stamps end `· axis: altered` and the scale used.

| File | Question | Source |
| --- | --- | --- |
{rows}

## Key studies

Each figure is redrawn when its study is rerun, and stamped `data <hash> · code <sha>`.

| File | Question | Source | Regenerate |
| --- | --- | --- | --- |
{studies}
"""


def main(argv: list[str] | None = None) -> int:
    import matplotlib as mpl

    mpl.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixture", default="dev_tree_1s_easy")
    parser.add_argument("--draw", type=Path, default=None,
                        help="the directory holding <fixture>/r0; drawn if not given")  # fmt: skip
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--truth-only", action="store_true", help="figures 1-8 alone")
    parser.add_argument("--solvers", nargs=2, type=Path, default=None,
                        metavar=("POTTS.record", "COPY.record"),
                        help="figure 18 alone, from a potts_stream and a copy_state_stream record")  # fmt: skip
    arguments = parser.parse_args(argv)
    if arguments.solvers is not None:
        potts, copies = arguments.solvers
        print(solver_figures(potts, copies, arguments.out, provenance.commit()))
        return 0

    # NB read before anything is written, so the set's own files never mark it `+`
    commit = provenance.commit()
    path = realization(arguments.fixture, arguments.draw)
    digest = stated_hash(arguments.fixture)
    text = f"{arguments.fixture} {digest} · code {commit}"
    out: Path = arguments.out
    for part in ("truth", "run", "compare"):
        (out / part).mkdir(parents=True, exist_ok=True)

    for written in truth_figures(path, out / "truth", text):
        print(written)
    if arguments.truth_only:
        return 0

    from port.qa.ledger import SIM_TEST, write
    from port.sim.fixtures import load_simulated

    sample = load_simulated(path.name, path.parent)
    with tempfile.TemporaryDirectory() as scratch:
        run = run_figures(sample, Path(scratch), out / "run", text)
        c = compared(sample, path, run.output, run.recovery)
        for written in compare_figures(c, out / "compare", text):
            print(written)

    recovery = {**run.recovery, "fixture_hash": digest, "peak_gb": run.peak_gb}
    run_id = write(
        recovery,
        fixture=ledger_name(arguments.fixture),
        args=" ".join(FLAGS),
        note=f"#624 paper figures: {arguments.fixture} r0 {digest}, --sal",
        dirty=commit.endswith("+"),
        test=SIM_TEST,
    )
    (out / "README.md").write_text(
        readme(arguments.fixture, digest, commit, run_id, recovery, switches(path))
    )
    return 0
