"""Pages for run_cnamaste's figures: a fixed page, panels placed in inches, one genomic axis (T- #836 F1-F3).

A standalone version of port's `combined_figure`, on cnamaste's own plotters. A page is the text column wide and a
supported share of the text block tall, less the caption's room: a third, a half or the full height, as port's
`figure_style` sets them. Every text on a page is `[tool.cnamaste.figures]`'s `font_size`, in its face.

- `plot_combined_page`, the full page: (a) the H&E slide, a blank frame where there is none, and the inferred clones, framed on the
  spots' extent and keyed to the right; (b) the hatched copy-number profile under its key; (c) each clone's RDR and BAF
  tracks, in the profile's clone order.
- `plot_he_page`, a third of a page: (a) each spot's H&E class and (b) the initial clones, on one extent.

The genomic panels share one `GenomicAxis`: the profile and every track are drawn on bin index with one set of
chromosome boundaries, so a bin sits at one x on each. The spatial panels share one extent, so a spot sits at one
place on each. Clones are named as `cnaster` names them (`cast_clone_label`).
"""

from __future__ import annotations

import contextlib
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.transforms import offset_copy

from cnamaste.plot_copy_number_profile import plot_hatched_key, plot_hatched_profile
from cnamaste.plot_genomic import plot_clones_genomic
from cnamaste.plotting import plot_clones_spatial

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"

DEFAULT = {"family": "serif", "font": "STIXGeneral", "mathtext": "stix", "font_size": 8.0}
"""`[tool.cnamaste.figures]` as shipped; used where no `pyproject.toml` is found."""

PAPER_WIDTH = 8.5 - 2 * 1.01
"""Inches: a text column, US letter less 1.01 in margins, as port's `figure_style.PAPER_WIDTH`."""

TEXT_HEIGHT = 11.0 - 2 * 1.01 - 58.0 / 72.27
"""Inches: the text block less its running head and folio, as port's `figure_style.TEXT_HEIGHT`."""

CAPTION_ROOM = 1.0
"""Inches of the text block a page leaves for its caption."""

PAGES = {"third": 1 / 3, "half": 1 / 2, "full": 1.0}
"""The supported page heights, each a share of `TEXT_HEIGHT - CAPTION_ROOM`."""

DPI = 300

MARGIN = {"left": 0.62, "right": 0.05, "top": 0.05, "bottom": 0.35}
"""Inches around the panels: tick labels left, staggered chromosome names below."""

LETTER = 0.16
"""Inches of a panel letter's line."""

TICK_LINE = 0.25
"""Inches under the spatial panels for their tick labels."""

GAP = 0.1
"""Inches between panels."""

ROW = 1.9
"""Inches, a spatial panel's height at most: port's share of the page, so the tracks keep theirs."""

KEY_COLUMN = 0.85
"""Inches right of the spatial panels for their key."""

PROFILE_KEY = 0.3
"""Inches of the profile's key, above it."""

PROFILE_ROW = 0.22
"""Inches of profile per clone."""

NAME_LINE = 0.14
"""Inches above a clone's tracks for its name and state key."""

TRACK_GAP = 0.05
"""Inches between a clone's RDR and BAF tracks, so their end ticks do not meet."""

HE_PALETTE = "mako"
"""The H&E classes' palette, distinct from the clones' `rocket`."""


def style() -> dict[str, Any]:
    """`[tool.cnamaste.figures]` from cnamaste's `pyproject.toml`, else `DEFAULT`."""
    if not PYPROJECT.exists():
        return dict(DEFAULT)
    stated = tomllib.loads(PYPROJECT.read_text()).get("tool", {}).get("cnamaste", {}).get("figures")
    return dict(DEFAULT) if stated is None else dict(DEFAULT) | stated


@contextlib.contextmanager
def page_style() -> Iterator[float]:
    """The stated face for the block, over cnaster's DejaVu Serif; yields the font size."""
    stated = style()
    family = stated["family"]
    with mpl.rc_context({"font.family": family, f"font.{family}": [stated["font"]], "mathtext.fontset": stated["mathtext"]}):
        yield float(stated["font_size"])


def page_size(page: str) -> tuple[float, float]:
    """Inches, width by height, of a `page` ("third", "half" or "full")."""
    return PAPER_WIDTH, PAGES[page] * (TEXT_HEIGHT - CAPTION_ROOM)


@dataclass(frozen=True)
class GenomicAxis:
    """Bins in genome order, one run per chromosome: the x axis every genomic panel draws on."""

    lengths: np.ndarray
    names: tuple[str, ...]

    @classmethod
    def from_bins(cls, chromosomes: Any) -> GenomicAxis:
        """From each bin's chromosome, in bin order (`cnv_seglevel.tsv`'s `CHR`)."""
        chromosomes = np.asarray(chromosomes)
        starts = np.flatnonzero(np.r_[True, chromosomes[1:] != chromosomes[:-1]])
        lengths = np.diff(np.r_[starts, chromosomes.size])
        names = tuple(str(c) for c in chromosomes[starts])
        assert len(set(names)) == len(names), "a chromosome's bins are not contiguous"
        return cls(lengths, names)

    @property
    def n_bins(self) -> int:
        return int(np.sum(self.lengths))

    @property
    def starts(self) -> np.ndarray:
        return np.r_[0, np.cumsum(self.lengths)[:-1]]

    def frame(self, ax: Any, labels: bool = False) -> None:
        """`ax` on bins `[0, n_bins)`, a line at each chromosome's start; with `labels`, each chromosome named level
        under its middle, a name that would meet its neighbour's on a second line, and `chr` at the left."""
        ax.set_xlim(0, self.n_bins)
        ax.set_xticks([])
        for x in self.starts[1:]:
            ax.axvline(x, color="black", linewidth=0.5, zorder=3)
        if not labels:
            return
        size = style()["font_size"]
        points = ax.get_position().width * ax.figure.get_size_inches()[0] * 72.0 / self.n_bins
        right = [-np.inf, -np.inf]
        for start, length, name in zip(self.starts, self.lengths, self.names):
            middle = (start + length / 2.0) * points
            half = 0.3 * size * len(name)
            line = 0 if middle - half >= right[0] + 1.0 else 1
            right[line] = middle + half
            at = offset_copy(ax.get_xaxis_transform(), ax.figure, y=-2.0 - line * 1.1 * size, units="points")
            ax.text(start + length / 2.0, 0.0, name, transform=at, ha="center", va="top")
        at = offset_copy(ax.transAxes, ax.figure, x=-4.0, y=-2.0, units="points")
        ax.text(0.0, 0.0, "chr", transform=at, ha="right", va="top")


def add_axes(figure: Any, x0: float, top: float, width: float, height: float) -> Any:
    """An axes `width` by `height` inches, its left edge `x0` and its top `top` inches from the page's left and top."""
    w, h = figure.get_size_inches()
    return figure.add_axes((x0 / w, 1.0 - (top + height) / h, width / w, height / h))


def place(ax: Any, x0: float, top: float, width: float, height: float) -> None:
    """`ax` moved to `width` by `height` inches at `x0`, `top` inches down."""
    w, h = ax.figure.get_size_inches()
    ax.set_position((x0 / w, 1.0 - (top + height) / h, width / w, height / h))


def letter(figure: Any, top: float, label: str, x0: float = 0.02) -> None:
    """A panel letter `(label)` `x0` inches from the page's left edge, `top` inches from its top."""
    w, h = figure.get_size_inches()
    figure.text(x0 / w, 1.0 - top / h, f"({label})", va="top", ha="left")


def uniform(figure: Any, size: float) -> None:
    """Every text on the page at `size` points."""
    for text in figure.findobj(Text):
        text.set_fontsize(size)


def draw_slide(ax: Any, he_frame: pd.DataFrame) -> None:
    """The H&E image `get_he_image` returns, one row per pixel, on the spots' coordinates `(x, -y)`."""
    rows, cols = he_frame["array_row"].to_numpy(), he_frame["array_col"].to_numpy()
    image = np.zeros((rows.max() + 1, cols.max() + 1, 3))
    image[rows, cols] = he_frame[["red", "green", "blue"]].to_numpy()
    image = np.clip(image / (255.0 if image.max() > 1.0 else 1.0), 0.0, 1.0)
    # NB a pixel at column j, row i sits at x = j / scalef, y = i / scalef (get_he_image).
    x_scale = float(he_frame["x"].to_numpy()[cols > 0][0] / cols[cols > 0][0])
    y_scale = float(he_frame["y"].to_numpy()[rows > 0][0] / rows[rows > 0][0])
    height, width = image.shape[:2]
    extent = (-0.5 * x_scale, (width - 0.5) * x_scale, -(height - 0.5) * y_scale, 0.5 * y_scale)
    ax.imshow(image, extent=extent, interpolation="none")


def square_handle(colour: Any) -> Line2D:
    return Line2D([0], [0], marker="s", color="w", markerfacecolor=colour, markeredgewidth=0, markersize=6)


def draw_classes(ax: Any, coords: np.ndarray, he_labels: np.ndarray) -> tuple[list[Any], list[str]]:
    """Each spot's H&E class (`get_he_image`'s `label`, 0 darkest) in `HE_PALETTE`; returns its key."""
    he_labels = np.asarray(he_labels).astype(int)
    classes = np.arange(he_labels.max() + 1)
    colours = sns.color_palette(HE_PALETTE, classes.size).as_hex()
    size = np.clip(12_000.0 / len(coords), 0.1, 25.0)
    ax.scatter(coords[:, 0], -coords[:, 1], s=size, c=np.array(colours)[he_labels], linewidth=0)
    return [square_handle(colours[k]) for k in classes], [f"H&E {k}" for k in classes]


def clone_key(ax: Any) -> tuple[list[Any], list[str]]:
    """`plot_clones_spatial`'s key as square handles, its own removed from the panel."""
    legend = ax.get_legend()
    handles = [square_handle(h.get_markerfacecolor()) for h in legend.legend_handles]
    labels = [t.get_text() for t in legend.get_texts()]
    legend.remove()
    return handles, labels


def extent(ax: Any) -> tuple[float, float, float, float]:
    """The spots' extent on `ax` as drawn, `(x0, x1, y0, y1)` on `(x, -y)`."""
    (x0, x1), (y0, y1) = ax.dataLim.intervalx, ax.dataLim.intervaly
    return x0, x1, y0, y1


def frame_extent(ax: Any, box: tuple[float, float, float, float], labels: bool, pad: float = 0.02) -> None:
    """`ax` on `box` padded by `pad` of each side, aspect equal, framed left and below on it: ticks at its first,
    middle and last coordinate, `y` read as the spots' `y` (drawn as `-y`), labelled where `labels`."""
    x0, x1, y0, y1 = box
    dx, dy = pad * (x1 - x0), pad * (y1 - y0)
    ax.set_xlim(x0 - dx, x1 + dx)
    ax.set_ylim(y0 - dy, y1 + dy)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("on")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_bounds(x0, x1)
    ax.spines["left"].set_bounds(y0, y1)
    xs, ys = np.linspace(x0, x1, 3), np.linspace(y1, y0, 3)
    ax.set_xticks(xs)
    ax.set_yticks(ys)
    ax.set_xticklabels([f"{v:.1f}" for v in xs] if labels else [])
    ax.set_yticklabels([f"{0.0 - v:.1f}" for v in ys] if labels else [])


def spatial_row(figure: Any, top: float, panels: list[Any], keys: list[tuple[list[Any], list[str]]], most: float) -> float:
    """`panels`, drawn, placed side by side `top` inches down on the spots' extent of the last, each as wide as the
    row allows and at most `most` inches tall; `keys` in one column to their right. Returns the row's height."""
    box = extent(panels[-1])
    ratio = (box[3] - box[2]) / (box[1] - box[0])  # NB the padding is equal on both axes, so it keeps the ratio
    room = figure.get_size_inches()[0] - MARGIN["left"] - MARGIN["right"] - KEY_COLUMN - (len(panels) - 1) * GAP
    width = min(room / len(panels), most / ratio)
    for k, ax in enumerate(panels):
        frame_extent(ax, box, labels=k == 0)
        place(ax, MARGIN["left"] + k * (width + GAP), top, width, width * ratio)
    # NB one key, each panel's entries in turn, a blank entry between them.
    handles, labels = [], []
    for k, (h, l) in enumerate(keys):
        if k:
            handles.append(Line2D([], [], linestyle="none"))
            labels.append("")
        handles, labels = handles + list(h), labels + list(l)
    panels[-1].legend(handles, labels, loc="lower left", bbox_to_anchor=(1.0 + GAP / width, 0.0), frameon=False,
                      handlelength=0.8, handletextpad=0.4, labelspacing=0.3, borderaxespad=0.0)  # fmt: skip
    return width * ratio


def name_line(ax: Any) -> None:
    """The clone's name alone above its tracks, its state key on the same line at the right."""
    names = [t for t in ax.texts if t.get_rotation() == 90.0]
    stats = [t for t in ax.texts if t.get_rotation() == 0.0 and t.get_position() == (0.0, 1.02)]
    if names and stats:
        stats[0].set_text(names[0].get_text())
        for t in names:
            t.remove()
    legend = ax.get_legend()
    if legend is not None:
        handles, labels = legend.legend_handles, [t.get_text() for t in legend.get_texts()]
        for handle in handles:
            handle.set_markersize(3)
        ax.legend(handles, labels, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(labels), frameon=False,
                  handlelength=0.6, handletextpad=0.3, columnspacing=1.0, borderaxespad=0.0, borderpad=0.0)  # fmt: skip


def track_furniture(ax: Any, name: str) -> None:
    """End y ticks alone, and the track named `name` beside them."""
    low, high = ax.get_yticks()[[0, -1]]
    ax.set_yticks([low, high])
    ax.set_yticklabels([f"{low:.1f}", f"{high:.1f}"])
    ax.set_ylabel(name)


def plot_combined_page(
    lengths: np.ndarray,
    single_X: np.ndarray,
    single_base_nb_mean: np.ndarray,
    single_total_bb_RD: np.ndarray,
    df_cnv: pd.DataFrame,
    res_combine: dict[str, Any],
    coords: np.ndarray,
    assignment: pd.Series,
    single_tumor_prop: np.ndarray | None = None,
    sample_list: list | None = None,
    sample_ids: np.ndarray | None = None,
    he_frame: pd.DataFrame | None = None,
) -> Any:
    """The full page: (a) the H&E slide, a blank frame where there is none, and the inferred clones; (b) the hatched copy-number
    profile; (c) each clone's RDR and BAF tracks, on the profile's genomic axis. Arguments as `plot_clones_genomic`
    and `plot_clones_spatial` take them in run_cnamaste."""
    genome = GenomicAxis.from_bins(df_cnv["CHR"])
    n_clones = len(np.unique(res_combine["new_assignment"]))
    has_rdr = single_base_nb_mean is not None and np.max(single_base_nb_mean) > 0
    per = 2 if has_rdr else 1

    with page_style() as size:
        figure = plt.figure(figsize=page_size("full"), dpi=DPI, facecolor="white")
        page_width, page_height = figure.get_size_inches()
        width = page_width - MARGIN["left"] - MARGIN["right"]

        top = MARGIN["top"]
        letter(figure, top, "a")
        top += LETTER
        # NB two panels whether or not there is a slide, so the page is laid out alike: a blank frame without one.
        panels = [add_axes(figure, MARGIN["left"], top, 1.0, 1.0) for _ in range(2)]
        plot_clones_spatial(coords, assignment, single_tumor_prop=single_tumor_prop, sample_ids=sample_ids,
                            sample_list=sample_list, ax=panels[-1])  # fmt: skip
        for text in list(panels[-1].texts):
            text.remove()
        if he_frame is not None:
            draw_slide(panels[0], he_frame)
        top += spatial_row(figure, top, panels, [clone_key(panels[-1])], ROW) + TICK_LINE

        letter(figure, top, "b")
        plot_hatched_key(add_axes(figure, MARGIN["left"], top, width, PROFILE_KEY))
        top += PROFILE_KEY + GAP
        profile_ax = add_axes(figure, MARGIN["left"], top, width, PROFILE_ROW * n_clones)
        # NB the tracks' order, clone index from the normal clone down, on the profile's rows too.
        plot_hatched_profile(df_cnv, profile_ax, genome, order=np.unique(res_combine["new_assignment"]))
        # NB the clones' names from the page's left edge, as the panel letters.
        for label in profile_ax.get_yticklabels():
            label.set_horizontalalignment("left")
        profile_ax.tick_params(axis="y", pad=72.0 * (MARGIN["left"] - 0.02))
        top += PROFILE_ROW * n_clones + GAP

        letter(figure, top, "c")
        block = (page_height - top - MARGIN["bottom"]) / n_clones
        track = (block - NAME_LINE - GAP / 2.0 - (per - 1) * TRACK_GAP) / per
        axes = []
        for c in range(n_clones):
            for t in range(per):
                axes.append(add_axes(figure, MARGIN["left"], top + c * block + NAME_LINE + t * (track + TRACK_GAP), width, track))
        plot_clones_genomic(lengths, single_X, single_base_nb_mean, single_total_bb_RD, df_cnv=df_cnv,
                            res_combine=res_combine, single_tumor_prop=single_tumor_prop, sample_list=sample_list,
                            axes=axes, genome=genome)  # fmt: skip
        for k, ax in enumerate(axes):
            track_furniture(ax, ("RDR", "BAF")[k % per] if has_rdr else "BAF")
            if k % per == 0:
                name_line(ax)

        uniform(figure, size)
    return figure


def plot_he_page(coords: np.ndarray, he_labels: np.ndarray, assignment: pd.Series, single_tumor_prop: np.ndarray | None = None) -> Any:
    """A third of a page: (a) each spot's H&E class, (b) the initial clones, on one extent, keyed to the right."""
    with page_style() as size:
        figure = plt.figure(figsize=page_size("third"), dpi=DPI, facecolor="white")
        top = MARGIN["top"] + LETTER
        classes_ax, clones_ax = (add_axes(figure, MARGIN["left"], top, 1.0, 1.0) for _ in range(2))
        he_key = draw_classes(classes_ax, coords, he_labels)
        plot_clones_spatial(coords, assignment, single_tumor_prop=single_tumor_prop, ax=clones_ax)
        most = figure.get_size_inches()[1] - top - 0.3
        spatial_row(figure, top, [classes_ax, clones_ax], [he_key, clone_key(clones_ax)], most)
        letter(figure, MARGIN["top"], "a")
        letter(figure, MARGIN["top"], "b", x0=clones_ax.get_position().x0 * figure.get_size_inches()[0])
        uniform(figure, size)
    return figure


def write_page(path: str, figure: Any, dpi: int = DPI) -> None:
    """`figure` as a PNG at its page size, on white; nothing trimmed."""
    figure.savefig(path, format="png", dpi=dpi, facecolor="white")
    plt.close(figure)
