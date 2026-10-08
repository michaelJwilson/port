"""One format for the spatial pages: a row of panels, one per slice or map (T- #791).

Panels `PAPER_WIDTH` across in all, `PANEL_GAP` apart, no ticks or spines,
equal aspect, left titles, shared region dashed in `MUTED`, one key under the
first panel.
"""

from __future__ import annotations

from typing import Any

from port.extensions.figure_style import INK, MIN_FONT_SIZE, MUTED, PAPER_WIDTH

__all__ = [
    "KEY_ROOM",
    "PANEL_GAP",
    "TITLE_ROOM",
    "format_panel",
    "overlap_box",
    "panel_row",
    "row_height",
    "spatial_key",
]

PANEL_GAP = 0.08
"""Inches between adjacent panels."""

TITLE_ROOM = 0.2
"""Inches over the panels for their titles."""

KEY_ROOM = 0.3
"""Inches under the panels for the key's row."""


def panel_row(
    n_panels: int,
    aspect: float,
    width: float | None = None,
    *,
    figure: Any = None,
    height: float | None = None,
    top: float = 0.0,
    left: float = 0.0,
) -> tuple[Any, list[Any]]:
    """`n_panels` axes in a row `width` inches across, each `aspect` (height/width).

    New page unless `figure` is given; `height` caps the row (titles and key
    included), `top` inches are left clear above, and the row starts `left` in.
    """
    import matplotlib.pyplot as plt

    width = PAPER_WIDTH if width is None else width
    side = (width - (n_panels - 1) * PANEL_GAP) / n_panels
    tall = side * aspect
    if height is not None and tall > height - top - TITLE_ROOM - KEY_ROOM:
        tall = height - top - TITLE_ROOM - KEY_ROOM
        side = tall / aspect
    total = top + TITLE_ROOM + tall + KEY_ROOM
    if figure is None:
        figure = plt.figure(figsize=(width, total), dpi=300, facecolor="white")
        box_w, box_h = width, total
    else:
        dpi = figure.get_figure(root=True).dpi
        box_w, box_h = figure.bbox.width / dpi, figure.bbox.height / dpi
    bottom = box_h - top - TITLE_ROOM - tall
    axes = [
        figure.add_axes(
            (
                (left + k * (side + PANEL_GAP)) / box_w,
                bottom / box_h,
                side / box_w,
                tall / box_h,
            )
        )
        for k in range(n_panels)
    ]
    return figure, axes


def row_height(n_panels: int, aspect: float, width: float | None = None) -> float:
    """Inches `panel_row` takes for `n_panels` at `aspect` across `width`, titles and key included."""
    width = PAPER_WIDTH if width is None else width
    side = (width - (n_panels - 1) * PANEL_GAP) / n_panels
    return TITLE_ROOM + side * aspect + KEY_ROOM


def format_panel(
    ax: Any,
    title: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    *,
    fontsize: float = MIN_FONT_SIZE,
) -> None:
    """A spatial panel's frame: limits, equal aspect, no ticks or spines, and
    `title` at its left in `INK`."""
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(title, loc="left", fontsize=fontsize, color=INK, pad=2.0)


def overlap_box(
    ax: Any, corner: tuple[float, float], width: float, height: float
) -> None:
    """The region the slices share, dashed in `MUTED`, over what the panel draws."""
    from matplotlib.patches import Rectangle

    ax.add_patch(
        Rectangle(corner, width, height, fill=False, linestyle="--",
                  edgecolor=MUTED, linewidth=0.8, zorder=3)
    )  # fmt: skip


def spatial_key(
    ax: Any,
    names: list[str],
    colours: list[str],
    *,
    marker: str = "o",
    fontsize: float = MIN_FONT_SIZE,
) -> None:
    """One key for the row: `names` against `colours` in one row under `ax`,
    anchored to it, so `fit_to_content` keeps it (PR- #715)."""
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], marker=marker, linestyle="", markersize=5, color=colour,
               label=name)
        for name, colour in zip(names, colours, strict=True)
    ]  # fmt: skip
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.0),
              ncol=len(handles), frameon=False, fontsize=fontsize,
              handletextpad=0.2, columnspacing=1.0, borderaxespad=0.2)  # fmt: skip
