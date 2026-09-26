"""`cnaster.plot_copy_number_profile`, one row per clone and aberrations hatched.

`cnaster` draws each clone as two half-rows, A above B, and marks a segment
whose alleles are swapped in another clone with chevrons. At a text column's
width the half-rows are 0.1 in tall and the chevrons fill them.

**One row per clone.** A normal segment, `(1, 1)`, is the faint normal
colour as before. Any other segment is filled with A's colour and hatched
with B's -- A first, on the same colour bar `plot_ascn_legend` draws -- so
both alleles read at the row's full height.

**The hatch orientation is the mirror.** Hatching runs at +45 degrees where
A >= B and at -45 where A < B, so a pair of segments whose alleles are swapped
between clones -- what the chevrons marked -- hatch in opposite directions,
and every segment carries its orientation rather than only mirrored ones.

A `FIGURE_SWAPS` row: the figure changes by design.
"""

from __future__ import annotations

from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cnaster.palette import get_full_palette
from cnaster.plot_copy_number_profile import NORMAL_OPACITY, get_intervals
from cnaster.utils import cast_clone_label
from matplotlib.patches import Rectangle

__all__ = ["HATCH", "plot_ascn_legend", "plot_copy_number_profile"]

HATCH = {1: "////", -1: "\\\\\\\\"}
"""+45 degrees where A >= B, -45 where A < B."""

HATCH_LINEWIDTH = 1.6
"""Wide enough that B's colour reads as a band beside A's, not a thin line."""


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
    """One segment: faint if normal, else A's fill under B's hatch."""
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

    ax.add_patch(
        Rectangle(
            (x0, y0),
            w,
            h,
            facecolor=style.get(a, default),
            hatch=HATCH[1 if a >= b else -1],
            hatchcolor=style.get(b, default),
            hatch_linewidth=HATCH_LINEWIDTH,
            edgecolor="none",
            linewidth=0,
        )
    )


def plot_copy_number_profile(
    df_cnv: pd.DataFrame,
    ax: Any = None,
    height: float = 1.0,
    title: Any = None,
    show_clone_name: bool = True,
    plot_chrname: bool = True,
    figsize: Any = None,
    palette_name: str = "chisel_single",
) -> Any:
    """`cnaster`'s profile, one row per clone, aberrations hatched A then B."""
    state_style, _ = get_full_palette(palette_name)
    clone_ids = [c.split(" ")[0][5:] for c in df_cnv.columns if c.endswith(" A")]
    clone_ids = _order(df_cnv, clone_ids)
    num_clones = len(clone_ids)

    if ax is None:
        figsize = figsize or (15, max(2.0, 0.6 * num_clones))
        fig, ax = plt.subplots(figsize=figsize, dpi=300, facecolor="white")
        fig.subplots_adjust(bottom=0.25)
    else:
        fig = ax.figure

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
                _segment(
                    ax,
                    ch_offset + s,
                    y0,
                    e - s,
                    row,
                    a_states[s],
                    b_states[s],
                    state_style,
                )

        ch_offset += len(df_ch)

    ch_coords.append(ch_offset)

    for k in range(num_clones):
        y0 = gap / 2 + k * h
        ax.vlines(ch_coords, ymin=y0, ymax=y0 + row, linewidth=1, colors="black")
        ax.add_patch(
            Rectangle(
                (0, y0),
                ch_offset,
                row,
                facecolor="none",
                edgecolor="black",
                linewidth=1,
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

    legend_ax = fig.add_axes((0.15, 0.025, 0.7, 0.05))
    plot_ascn_legend(legend_ax, palette_name=palette_name)

    if title:
        ax.set_title(title)

    return fig


def plot_ascn_legend(
    ax: Any,
    box_w: float = 0.8,
    box_h: float = 0.8,
    tick_len: float = 0.08,
    label_fontsize: float = 10,
    palette_name: str = "chisel_single",
) -> Any:
    """`cnaster`'s colour bar, with swatches for the fill, the hatch and its turn."""
    state_style, ordered_acn = get_full_palette(palette_name)
    ax.axis("off")

    example_a, example_b = 3, 2
    key = [
        (f"A{example_a}B{example_b}", example_a, example_b),
        (f"A{example_b}B{example_a}", example_b, example_a),
    ]
    x = 0.0

    for label, a, b in key:
        _segment(ax, x, 0.0, box_w, box_h, a, b, state_style)
        ax.add_patch(
            Rectangle((x, 0.0), box_w, box_h, facecolor="none", edgecolor="black")
        )
        ax.text(
            x + box_w / 2,
            -tick_len - 0.04,
            label,
            ha="center",
            va="top",
            fontsize=label_fontsize,
        )
        x += box_w + 0.15

    x0 = x + 0.35

    for i, label in enumerate(ordered_acn):
        color = state_style.get(label)
        ax.add_patch(
            Rectangle(
                (x0 + i * box_w, 0.0),
                box_w,
                box_h,
                facecolor=mcolors.to_rgba(color, NORMAL_OPACITY if label == 1 else 1.0),
                edgecolor="black",
            )
        )
        xc = x0 + i * box_w + box_w / 2.0
        ax.plot([xc, xc], [-tick_len, 0.0], color="black", linewidth=0.8)
        ax.text(
            xc,
            -tick_len - 0.04,
            str(label),
            ha="center",
            va="top",
            fontsize=label_fontsize,
        )

    end = x0 + len(ordered_acn) * box_w
    ax.text(
        end + 0.2,
        box_h / 2.0,
        r"$\mathbb{N}$-CNA: fill A, hatch B",
        fontsize=label_fontsize,
        ha="left",
        va="center",
    )
    ax.set_xlim(0.0, end + 4.0)
    ax.set_ylim(-0.5, box_h + 0.2)
    ax.set_aspect("auto")

    return ax
