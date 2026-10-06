"""`cnaster.plot_copy_number_profile`, one row per clone and aberrations hatched.

`cnaster` draws each clone as two half-rows, A above B, and marks a segment
whose alleles are swapped in another clone with chevrons. At a text column's
width the half-rows are 0.1 in tall and the chevrons fill them.

**One row per clone.** A normal segment, `(1, 1)`, is the faint normal
colour as before. Any other segment is filled with A's colour and hatched
with B's -- A first, each as its box on `plot_ascn_legend`'s colour bar shows
it (`swatch`) -- so both alleles read at the row's full height.

**The hatch orientation is the mirror.** Hatching rises to the right where
A >= B and to the left where A < B, so a pair of segments whose alleles are
swapped between clones -- what the chevrons marked -- hatch in opposite
directions, and every segment carries its orientation rather than only
mirrored ones. The legend shows the two orientations, unnumbered, stacked,
with `MIRROR` to their right: what it keys is that a pair hatches opposite
ways.

**The hatch is drawn, not a matplotlib hatch.** A hatch pattern is fixed at
45 degrees and a spacing matplotlib chooses. Here each aberrant segment
carries a `LineCollection` of B's colour clipped to it, at `HATCH_ANGLE`
from the horizontal and `HATCH_SPACING` apart, both in inches on the page,
so the lines keep their angle and density whatever the axis's data scale.

A `FIGURE_SWAPS` row: the figure changes by design.
"""

from __future__ import annotations

from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cnaster.palette import get_full_palette
from cnaster.plot_copy_number_profile import get_intervals
from cnaster.utils import cast_clone_label
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib.transforms import IdentityTransform

from port.extensions.genomic_axis import GenomicAxis, Ticks, resolve

__all__ = [
    "HATCH",
    "HATCH_ANGLE",
    "HATCH_SPACING",
    "KEY_GROWTH",
    "MIRROR",
    "MIRROR_GAP",
    "hatch_of",
    "plot_ascn_legend",
    "plot_copy_number_profile",
    "swatch",
]

COPY_COLOURS = {
    2: "#feb24c",
    3: "#fd8d3c",
    4: "#fc4e2a",
    5: "#e31a1c",
    6: "#bd0026",
    "7+": "#660013",
}
"""Copies 2 to 7+ on `chisel_single`'s scale, its `khaki` 2 dropped: each
moves one step redder along ColorBrewer's YlOrRd, which adds a red (#339)."""

HATCH = {1: 1, -1: -1}
"""Rising to the right where A >= B (`h=0`), to the left where A < B (`h=1`)."""

HATCH_ANGLE = 35.0
"""Degrees from the horizontal: flatter than a 45-degree hatch, so a row a
tenth of an inch tall carries several lines, and steep enough that the two
phases cross at 70 degrees and read apart."""

HATCH_SPACING = 0.10
"""Inches between lines, along the row: wide enough that each stripe, A's
and B's, reads at print size."""

HATCH_LINEWIDTH = 72.0 * HATCH_SPACING * float(np.sin(np.radians(HATCH_ANGLE))) / 2.0
"""Points: half the lines' period measured across them, `HATCH_SPACING`
along the row times the sine of `HATCH_ANGLE`, so A's stripes and B's are the
same width (2.065 pt at 0.10 in and 35 degrees)."""

NORMAL_OPACITY = 0.35
"""Copy 1's opacity over white, for a normal `(1, 1)` segment and a hatch's
1: over `cnaster`'s 0.25, so it reads as blue rather than as the page (#339
had halved it to 0.125, which read as white), and still paler than copy 2's
orange, so the aberrations are what the eye finds."""

LINEWIDTH = 0.5
"""Points, for each row's outline and the chromosome boundaries: what
`plot_clones_genomic` draws its boundaries at."""


def _palette(palette_name: str) -> tuple[dict[Any, Any], Any]:
    """`cnaster`'s palette, `COPY_COLOURS` over it for `chisel_single`."""
    state_style, ordered_acn = get_full_palette(palette_name)

    if palette_name == "chisel_single":
        state_style = {**state_style, **COPY_COLOURS}

    return state_style, ordered_acn


def swatch(style: Any, copies: Any) -> tuple[float, float, float]:
    """Copy number `copies`'s colour as its legend box shows it, opaque.

    Copy 1 is `cnaster`'s colour at `NORMAL_OPACITY` over white, so a fill or
    hatch line of it is the colour of its box rather than the full colour the
    box fades: lines of a translucent colour would vanish into the fill.
    """
    rgb = np.asarray(
        mcolors.to_rgb(style.get(copies, style.get("default", "lightgray")))
    )
    alpha = NORMAL_OPACITY if copies == 1 else 1.0
    r, g, b = alpha * rgb + (1.0 - alpha)
    return float(r), float(g), float(b)


def _order(df_cnv: pd.DataFrame, clone_ids: list[str]) -> list[str]:
    """`cnaster`'s order: least aberrant clone first, by segment deviation."""
    deviations = []

    for cid in clone_ids:
        a_col = df_cnv[f"clone{cid} A"].fillna(1).to_numpy()
        b_col = df_cnv[f"clone{cid} B"].fillna(1).to_numpy()
        starts = [s for s, _ in get_intervals(a_col * 1_000 + b_col)[0]]
        deviations.append(np.sum(np.abs(a_col[starts] - 1) + np.abs(b_col[starts] - 1)))

    return [clone_ids[i] for i in np.argsort(deviations)]


def _segment(
    ax: Any, x0: float, y0: float, w: float, h: float, a: Any, b: Any, style: Any
) -> None:
    """One segment: faint if normal, else A's fill under B's hatch, its two
    ends drawn as boundaries."""
    default = style.get("default", "lightgray")

    if a == 1 and b == 1:
        ax.add_patch(
            Rectangle(
                (x0, y0),
                w,
                h,
                facecolor=mcolors.to_rgba(style.get(1, default), NORMAL_OPACITY),
                edgecolor="none",
                linewidth=0,
            )
        )
        return

    fill = Rectangle(
        (x0, y0), w, h, facecolor=swatch(style, a), edgecolor="none", linewidth=0
    )
    ax.add_patch(fill)
    _hatch(ax, fill, swatch(style, b), HATCH[1 if a >= b else -1])
    # NB the aberration's ends, at the outline's weight: where it starts and
    #    stops reads without following the fill's edge into the next colour.
    ax.vlines(
        [x0, x0 + w],
        ymin=y0,
        ymax=y0 + h,
        linewidth=LINEWIDTH,
        colors="black",
        zorder=3,
        clip_on=False,
    )


class _Hatch(LineCollection):
    """B's lines over one fill, laid out when drawn, in the page's inches.

    At draw time the fill's extent on the page is known, so exactly the
    lines that cross it are made: `HATCH_SPACING` apart along its bottom
    edge, at `HATCH_ANGLE`, clipped to it.
    """

    def __init__(self, fill: Rectangle, colour: Any, orientation: int) -> None:
        # NB under the outlines (zorder 3), over the fill (1).
        super().__init__([], colors=[colour], linewidths=HATCH_LINEWIDTH, zorder=1.5)
        self.fill = fill
        self.orientation = orientation
        self.set_transform(IdentityTransform())

    def draw(self, renderer: Any) -> None:
        box = self.fill.get_window_extent(renderer)
        dpi = self.figure.dpi if self.figure is not None else 72.0
        spacing = HATCH_SPACING * dpi
        run = box.height / np.tan(np.radians(HATCH_ANGLE))
        starts = np.arange(box.x0 - run, box.x1 + run + spacing, spacing)
        ends = starts + self.orientation * run
        self.set_segments(
            [[(x0, box.y0), (x1, box.y1)] for x0, x1 in zip(starts, ends, strict=True)]
        )
        super().draw(renderer)


def _hatch(ax: Any, fill: Rectangle, colour: Any, orientation: int) -> None:
    """Attach B's lines to `fill`, drawn over it."""
    hatch = _Hatch(fill, colour, orientation)
    ax.add_collection(hatch, autolim=False)
    # NB as a path and its transform, after `add_collection`: a `Rectangle`
    #    is turned into a clip box, and `add_collection` then replaces it
    #    with the axis's own, which is what let the lines cross rows.
    hatch.set_clip_path(fill.get_path(), fill.get_transform())
    fill.set_gid(f"hatch{orientation:+d}")


def hatch_of(ax: Any, fill: Rectangle) -> tuple[int, Any] | None:
    """A fill's hatch orientation and B's colour, or `None` where it is plain."""
    for collection in ax.collections:
        if isinstance(collection, _Hatch) and collection.fill is fill:
            return collection.orientation, collection.get_edgecolor()[0]

    return None


def plot_copy_number_profile(
    df_cnv: pd.DataFrame,
    ax: Any = None,
    height: float = 1.0,
    title: Any = None,
    show_clone_name: bool = True,
    plot_chrname: bool = True,
    figsize: Any = None,
    palette_name: str = "chisel_single",
    *,
    axis: GenomicAxis | Ticks | None = None,
    rows: list[str] | None = None,
) -> Any:
    """`cnaster`'s profile, one row per clone, aberrations hatched A then B.

    `rows`, the clone ids of `df_cnv`'s `clone<id> A` columns top to bottom,
    sets the rows' order; `None` is `cnaster`'s, least aberrant first
    (PR- #701).

    `axis`, a `port.extensions.genomic_axis.GenomicAxis` on `df_cnv`'s bins
    or a `Ticks` made on them, draws the segments and boundaries on its
    coordinate and ticks every 10 Mb, unlabelled: the chromosome names hold
    the line below the rows (T- #683). `None` is `cnaster`'s axis.
    """
    state_style, _ = _palette(palette_name)
    clone_ids = [c.split(" ")[0][5:] for c in df_cnv.columns if c.endswith(" A")]
    clone_ids = _order(df_cnv, clone_ids) if rows is None else list(rows)
    num_clones = len(clone_ids)

    if ax is None:
        figsize = figsize or (15, max(2.0, 0.6 * num_clones))
        fig, ax = plt.subplots(figsize=figsize, dpi=300, facecolor="white")
        fig.subplots_adjust(bottom=0.25)
    else:
        fig = ax.figure

    genome = resolve(axis, df_cnv, len(df_cnv))

    def at(u: Any) -> Any:
        return u if genome is None else genome.warp(u)

    h = height / num_clones
    gap = 0.2 * h
    row = h - gap

    ch_offset = 0
    ch_coords: list[int] = []
    chs: list[Any] = []

    for ch, df_ch in df_cnv.groupby("CHR", sort=False):
        chs.append(ch)
        ch_coords.append(ch_offset)

        for k, cid in enumerate(clone_ids):
            a_states = df_ch[f"clone{cid} A"].to_numpy()
            b_states = df_ch[f"clone{cid} B"].to_numpy()
            y0 = gap / 2 + h * (num_clones - k - 1)

            for s, e in get_intervals(a_states * 1_000 + b_states)[0]:
                x0, x1 = at(ch_offset + s), at(ch_offset + e)
                _segment(
                    ax,
                    x0,
                    y0,
                    x1 - x0,
                    row,
                    a_states[s],
                    b_states[s],
                    state_style,
                )

        ch_offset += len(df_ch)

    ch_coords.append(ch_offset)

    if genome is not None:
        ch_coords = list(genome.edges)

    # NB unclipped: the first and last edges sit on the x limits, where the
    #    axes clip would cut them to half the width of every other line.
    for k in range(num_clones):
        y0 = gap / 2 + k * h
        ax.vlines(
            ch_coords,
            ymin=y0,
            ymax=y0 + row,
            linewidth=LINEWIDTH,
            colors="black",
            zorder=3,
            clip_on=False,
        )
        ax.add_patch(
            Rectangle(
                (0, y0),
                ch_offset,
                row,
                facecolor="none",
                edgecolor="black",
                linewidth=LINEWIDTH,
                zorder=3,
                clip_on=False,
            )
        )

    ax.grid(False)
    ax.set_xlim(0, ch_offset)
    ax.set_ylim(0, num_clones * h)
    ax.set_xlabel("")

    for spine in ax.spines.values():
        spine.set_visible(False)

    if plot_chrname and chs:
        ax.set_xticks(ch_coords[:-1])
        ax.set_xticklabels(
            [f"chr{ch}" if str(ch).isdigit() else str(ch) for ch in chs],
            rotation=45,
            fontsize=8,
            ha="left",
        )
        ax.tick_params(axis="x", labelbottom=True, bottom=False, pad=-5)
    else:
        ax.set_xticks([])

    if genome is not None:
        genome.draw(ax, labels=False)

    ax.set_yticks([h * (i + 0.5) for i in range(num_clones)])
    ax.set_yticklabels(
        [
            cast_clone_label(f"Clone {cid}" if show_clone_name else str(cid))
            for cid in reversed(clone_ids)
        ],
        fontsize=8,
        va="center",
    )
    ax.tick_params(axis="y", which="major", left=True, right=False, length=4)

    legend_ax = fig.add_axes((0.15, 0.025, 0.7, 0.05 * KEY_GROWTH))
    plot_ascn_legend(legend_ax, palette_name=palette_name)

    if title:
        ax.set_title(title)

    return fig


MIRROR = "Co-located Mirror"
"""The mirror swatches' label, right of them in every figure that draws the key (PR- #701)."""

MIRROR_GAP = 0.15
"""The white between the stacked mirror swatches, against a box's height."""

KEY_GROWTH = (0.6 + 1.575 * 0.8 + 0.05) / (0.6 + 0.8 + 0.05)
"""A key axis's height against its height before the swatches stacked
(PR- #701), at the default `box_h` 0.8: its y range runs from the numerals'
-0.6 to 0.05 over the upper swatch, at `(1.5 + MIRROR_GAP / 2) * box_h`,
where it ran to 0.05 over the bar. A caller grows its key row by this so a
box keeps its size on the page."""


def plot_ascn_legend(
    ax: Any,
    box_w: float = 0.8,
    box_h: float = 0.8,
    tick_len: float = 0.08,
    label_fontsize: float = 10,
    palette_name: str = "chisel_single",
    *,
    span: float | None = None,
) -> Any:
    """The mirror swatches, `MIRROR` to their right, then `cnaster`'s colour bar titled on its left.

    The swatches, black lines on white, one per hatch orientation, are
    stacked, `MIRROR_GAP` of a box apart, about the bar's centre, and start
    on the axis's left edge, so a caller that sets the axis over its plot
    gets them on the plot's left edge. `MIRROR` is centred on the white
    between them. With `span`, the axis runs `0` to `span` and the bar ends
    there, on the plot's right edge; without it, the axis is sized to the
    swatches, their label, the bar's title and the bar. The axis's y range
    is `KEY_GROWTH` of what it was with the swatches side by side.
    """
    state_style, ordered_acn = _palette(palette_name)
    ax.axis("off")

    gap = 0.15 * box_w
    label_y = -tick_len - 0.04
    text = {"fontsize": label_fontsize, "clip_on": False}
    start = 0.0

    between = MIRROR_GAP * box_h
    for k, orientation in enumerate((HATCH[1], HATCH[-1])):
        y = box_h / 2 + between / 2 if k == 0 else box_h / 2 - between / 2 - box_h
        box = Rectangle(
            (start, y), box_w, box_h, facecolor="white", edgecolor="none", linewidth=0
        )
        ax.add_patch(box)
        _hatch(ax, box, "black", orientation)
        ax.add_patch(
            Rectangle(
                (start, y),
                box_w,
                box_h,
                facecolor="none",
                edgecolor="black",
                linewidth=LINEWIDTH,
                zorder=3,
            )
        )

    phase_end = start + box_w
    mirror = ax.text(phase_end + gap, box_h / 2, MIRROR, ha="left", va="center", **text)
    title = ax.text(
        0.0, box_h / 2, r"$\mathbb{N}$-CNA", ha="right", va="center_baseline", **text
    )

    bar = len(ordered_acn) * box_w
    if span is None:
        # NB data units per pixel follow from the axis's width once its
        #    limits are set: the two labels' pixels and the rest's units
        #    solve for the end that fits both.
        renderer = ax.figure.canvas.get_renderer()
        pixels = ax.get_window_extent(renderer).width
        words = sum(t.get_window_extent(renderer).width for t in (mirror, title))
        rest = phase_end + gap + 3 * gap + 3 * gap + bar
        end = rest / max(1.0 - words / pixels, 0.5)
    else:
        end = span
    x0 = end - bar
    title.set_x(x0 - 3 * gap)

    for i, label in enumerate(ordered_acn):
        ax.add_patch(
            Rectangle(
                (x0 + i * box_w, 0.0),
                box_w,
                box_h,
                facecolor=swatch(state_style, label),
                edgecolor="black",
                linewidth=LINEWIDTH,
            )
        )
        xc = x0 + i * box_w + box_w / 2.0
        ax.plot([xc, xc], [-tick_len, 0.0], color="black", linewidth=LINEWIDTH)
        ax.text(xc, label_y, str(label), ha="center", va="top", **text)

    ax.set_xlim(0.0, end)
    ax.set_ylim(-0.6, (1.5 + MIRROR_GAP / 2) * box_h + 0.05)
    ax.set_aspect("auto")

    return ax
