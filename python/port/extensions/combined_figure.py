r"""One page from a run: genome, copy numbers, clones and the slide (#309).

Four panels at a text column's width, `llncs`'s 122 mm (#280, #339), composed with matplotlib
subfigures so the page is drawn once, at its printed size, and included at
`width=\linewidth` with nothing scaled:

- **(a)** `clones_genomic`: RDR and BAF along the genome per clone, full
  width, drawn by `port.patch.plot_genomic` into its subfigure;
- **(b)** `copy_number_profile`: the integer copies per clone, full width,
  drawn by `cnaster` into an axis it is handed;
- **(c)** `clones_spatial`: the fitted clone of each spot, tiled by
  `port.patch.plotting.spatial`;
- **(d)** the H&E slide, as `cnaster.he.get_he_image` reads it.

(c) and (d) sit side by side at 0.48 of the width each and share the spot
coordinates, so a clone boundary in (c) reads against the tissue in (d).

**What is drawn is what the run drew.** `recording` keeps the arguments of
the run's last call to each of the three plotting functions -- for (a), the
one with integer copies, which is `clones_genomic.pdf` -- so the page is a
re-drawing of the run's own figures rather than a second derivation of them.

**Text is set once, for the page.** `cnaster`'s helpers hardcode 6 to 12
pt, sized for a 20 in page; at 4.80 in the larger are half an axis.
After drawing, every text is set to `FONT_SIZE`, (a)'s to a point under it,
and the panel labels above both.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

FONT_SIZE = 6.0
"""Every text on the page, in points, bar the panel labels and (a)."""

GENOMIC_FONT_SIZE = FONT_SIZE - 1.0
"""(a)'s text: a point under the rest, for the statistics line to fit the
gap between clones."""

LABEL_SIZE = 8.0
"""The panel labels, (a) to (d)."""

SIDE = 0.48
"""(c) and (d), as a fraction of the width: the layout's minipages."""

SCALE = 0.75 * 0.85
"""(c) and (d) within their minipages, centred, so the section does not
outweigh the genome above it: 0.75, then a further 15 per cent."""


@dataclass
class Call:
    """One plotting call's arguments."""

    args: tuple[Any, ...]
    kwargs: dict[str, Any]


@dataclass
class Recorded:
    """The run's last call to each function the page redraws."""

    genomic: Call | None = None
    spatial: Call | None = None
    profile: Call | None = None
    calls: dict[str, int] = field(default_factory=dict)


@contextlib.contextmanager
def recording() -> Iterator[Recorded]:
    """Keep the arguments of the run's plotting calls for the block.

    Wraps `port.patch.plot_genomic.plot_clones_genomic` and
    `port.patch.plotting.plot_clones_spatial` where the swap tables name them, so
    the swaps `run_cnaster_port` installs are the wrappers, and `cnaster`'s
    `plot_copy_number_profile`, which nothing swaps, where the script binds
    it. Each wrapper calls through, so the run's figures are
    unchanged. Enter before the run: `port.pipeline.patched` resolves the
    replacement when it installs it.
    """
    import cnaster.scripts.run_cnaster as script

    import port.patch.plot_genomic as genomic
    import port.patch.plotting as spatial

    recorded = Recorded()
    undo: list[tuple[Any, str, Any]] = []

    def wrap(module: Any, name: str, slot: str, keep: Any = None) -> None:
        original = getattr(module, name)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            recorded.calls[slot] = recorded.calls.get(slot, 0) + 1

            if keep is None or keep(kwargs):
                setattr(recorded, slot, Call(args, dict(kwargs)))

            return original(*args, **kwargs)

        undo.append((module, name, original))
        setattr(module, name, wrapper)

    try:
        wrap(genomic, "plot_clones_genomic", "genomic", lambda kw: "df_cnv" in kw)
        wrap(spatial, "plot_clones_spatial", "spatial")
        wrap(script, "plot_copy_number_profile", "profile")
        yield recorded
    finally:
        for module, name, original in reversed(undo):
            setattr(module, name, original)


def slide_image(frame: Any) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """The image `get_he_image` returned as rows, and where `plot_he` puts it.

    `get_he_image(pos=None)` returns one row per pixel, with `array_row`,
    `array_col` and `x = array_col / scalef`, `y = array_row / scalef`.
    `plot_he` scatters each at `(x, -y)`; the extent here places the image
    on the same coordinates, which are the spots' (#309), at one draw call
    rather than one marker per pixel.
    """
    rows = frame["array_row"].to_numpy()
    columns = frame["array_col"].to_numpy()
    image = np.zeros((rows.max() + 1, columns.max() + 1, 3))
    image[rows, columns] = frame[["red", "green", "blue"]].to_numpy()

    if image.max() > 1.0:
        image /= 255.0

    x_scale = float(frame["x"].to_numpy()[columns > 0][0] / columns[columns > 0][0])
    y_scale = float(frame["y"].to_numpy()[rows > 0][0] / rows[rows > 0][0])
    height, width = image.shape[:2]
    extent = (
        -0.5 * x_scale,
        (width - 0.5) * x_scale,
        -(height - 0.5) * y_scale,
        0.5 * y_scale,
    )

    return np.clip(image, 0.0, 1.0), extent


def _set_text(panel: Any, size: float) -> None:
    """Every text in `panel` at `size`: one size per panel, not a cap."""
    from matplotlib.text import Text

    for text in panel.findobj(Text):
        text.set_fontsize(size)


def _fit_tracks(panel: Any) -> None:
    """The genomic tracks' furniture, at a track a third of an inch tall.

    Upstream labels every integer of RDR and every 0.2 of BAF, which at this
    height is a stack of overlapping numbers: only the two ends are kept,
    and the gridlines stay. The clone name is moved clear of the axis label,
    and the legend's 10 pt markers are set to the page's.

    The statistics line and the legend share one line above each clone's
    RDR track, in the gap row `clone_axes` leaves between clones, and are
    kept out of the layout: counted in it, each pushes every track apart by
    its own height.
    """
    for ax in panel.axes:
        ticks = ax.get_yticks()

        if ticks.size > 2:
            ends = [ticks[0], ticks[-1]]
            ax.set_yticks(ends, [f"{tick:.0f}" for tick in ends])

            # NB inside the track's height: a label centred on the edge
            #    overhangs it, and the layout pads every track to make room.
            bottom, top = ax.get_yticklabels()
            bottom.set_verticalalignment("bottom")
            top.set_verticalalignment("top")

        ax.tick_params(length=2, pad=1)

        for text in ax.texts:
            if text.get_rotation() == 90.0:
                # NB centred on the RDR/BAF boundary: counted in the layout,
                #    it opens a gap between the two tracks of one clone.
                text.set_x(-0.055)
                text.set_in_layout(False)
            elif text.get_rotation() == 0.0:
                text.set_y(1.0)
                text.set_in_layout(False)

        legend = ax.get_legend()

        if legend is not None:
            # NB redrawn on the statistics line, right-aligned and unpadded:
            #    upstream's anchor, sized for a 3.2 in track, lands on the
            #    track above.
            handles = legend.legend_handles
            labels = [text.get_text() for text in legend.get_texts()]

            for handle in handles:
                handle.set_markersize(3)

            legend.remove()
            ax.legend(
                handles,
                labels,
                loc="lower right",
                bbox_to_anchor=(1.0, 1.0),
                ncol=len(labels),
                frameon=False,
                borderpad=0.0,
                borderaxespad=0.0,
                handletextpad=0.2,
                columnspacing=1.0,
                fontsize=FONT_SIZE,
            ).set_in_layout(False)


def _centred(panel: Any) -> Any:
    """One axis, `SCALE` of the panel's width, centred in it."""
    margin = (1.0 - SCALE) / 2
    grid = panel.add_gridspec(1, 3, width_ratios=(margin, SCALE, margin))

    return panel.add_subplot(grid[0, 1])


def combined_figure(
    recorded: Recorded,
    he_frame: Any,
    width: float | None = None,
) -> Any:
    """Compose (a) to (d) on one page `width` inches wide; no caption."""
    import matplotlib.pyplot as plt
    from matplotlib.layout_engine import ConstrainedLayoutEngine

    # NB `port`'s profile directly: the page is drawn after the run, when
    #    `FIGURE_SWAPS` has been restored and `cnaster`'s names are its own.
    from port.patch.plot_copy_number_profile import (
        plot_ascn_legend,
        plot_copy_number_profile,
    )
    from port.patch.plot_genomic import plot_clones_genomic
    from port.patch.plotting.genomic import PAPER_WIDTH
    from port.patch.plotting.spatial import draw_clones_spatial

    if recorded.genomic is None or recorded.spatial is None or recorded.profile is None:
        msg = f"the run made {recorded.calls}; the page needs all three"
        raise ValueError(msg)

    width = PAPER_WIDTH if width is None else width
    genomic = recorded.genomic
    n_clones = len(np.unique(genomic.kwargs["res_combine"]["new_assignment"]))
    # NB one profile row per clone rather than two halves, and the height
    #    that frees goes to (a), whose tracks are the densest on the page.
    # NB (c)'s legend sits below its tiles and out of the layout, so its row
    #    is reserved here: at 0.2 in it ran 0.107 in off the page at 4.80 in
    #    (#339), where a tight bounding box had hidden it by growing the page.
    heights = (0.68 * n_clones, 0.12 * n_clones + 0.55, SCALE * SIDE * width + 0.45)

    # NB no space between axes beyond what `clone_axes`' gap rows give.
    figure = plt.figure(
        figsize=(width, sum(heights)),
        dpi=300,
        facecolor="white",
        layout=ConstrainedLayoutEngine(h_pad=0.01, hspace=0.0),
    )
    rows: Any = figure.subfigures(3, 1, height_ratios=heights, hspace=0.02)
    top, middle = rows[0], rows[1]
    sides: Any = rows[2].subfigures(
        1, 2, width_ratios=(SIDE, SIDE), wspace=(1.0 - 2 * SIDE) / SIDE
    )
    left, right = sides[0], sides[1]

    plot_clones_genomic(
        *genomic.args,
        **{
            **genomic.kwargs,
            "figure": top,
            "pointsize": 0.4,
            "linewidth": 0.3,
            "chrtext_shift": -0.9,
        },
    )
    _fit_tracks(top)

    profile_ax, legend_ax = middle.subplots(2, 1, height_ratios=(1.0, 0.3))
    plot_copy_number_profile(recorded.profile.args[0], ax=profile_ax)
    # NB upstream's clone names are vertical, which on a row 0.3 in tall is
    #    longer than the row: set level, they take width the page has.
    profile_ax.tick_params(axis="y", which="major", pad=9)

    for text in profile_ax.get_yticklabels():
        text.set_rotation(0)
        text.set_horizontalalignment("right")
    # NB `cnaster` adds its legend at fixed page coordinates, which a layout
    #    engine does not manage; it is redrawn into an axis that is managed.
    middle.axes[-1].remove()
    plot_ascn_legend(legend_ax, label_fontsize=FONT_SIZE)

    coords, assignment = recorded.spatial.args[:2]
    spatial_ax = _centred(left)
    draw_clones_spatial(
        spatial_ax,
        np.asarray(coords),
        assignment,
        recorded.spatial.kwargs.get("single_tumor_prop"),
    )

    # NB out of the layout, below the tiles, so (c) and (d) are one size.
    spatial_ax.get_legend().set_in_layout(False)

    image, extent = slide_image(he_frame)
    slide_ax = _centred(right)
    slide_ax.imshow(image, extent=extent, interpolation="none")
    # NB the section (c) shows, so a boundary sits at the same place in both.
    slide_ax.set_xlim(spatial_ax.get_xlim())
    slide_ax.set_ylim(spatial_ax.get_ylim())
    slide_ax.set_aspect("equal")
    slide_ax.axis("off")

    _set_text(top, GENOMIC_FONT_SIZE)

    for panel in (middle, left, right):
        _set_text(panel, FONT_SIZE)

    # NB (b)'s chromosome names at (a)'s size: at 4.80 in the short
    #    chromosomes' rotated names touch at 6 pt (#339).
    for text in profile_ax.get_xticklabels():
        text.set_fontsize(GENOMIC_FONT_SIZE)

    # NB as titles, so the layout engine reserves their space.
    for panel, label in zip((top, middle, left, right), "abcd", strict=True):
        panel.suptitle(
            f"({label})", x=0.0, ha="left", fontsize=LABEL_SIZE, fontweight="bold"
        )

    return figure
