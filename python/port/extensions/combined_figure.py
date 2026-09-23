r"""One page from a run: genome, copy numbers, clones and the slide (#309).

Four panels at a text column's width, `llncs`'s 122 mm (#280, #339), composed with matplotlib
subfigures so the page is drawn once, at its printed size, and included at
`width=\linewidth` with nothing scaled:

- **(a)** the H&E slide, as `cnaster.he.get_he_image` reads it;
- **(b)** `clones_spatial`: the fitted clone of each spot, tiled by
  `port.patch.plotting.spatial`, keyed in one column to its right;
- **(c)** `clones_genomic`: RDR and BAF along the genome per clone, full
  width, drawn by `port.patch.plot_genomic` into its subfigure;
- **(d)** `copy_number_profile`: the integer copies per clone, full width,
  drawn by `port.patch.plot_copy_number_profile`, which names the
  chromosomes for both.

(a) and (b) sit across the top, square, and share the spot coordinates, so a
clone boundary in (b) reads against the tissue in (a). Clones are named as
the paper names them, $m_N$ for the normal and $m_1$, $m_2$, ... for the rest.

**One left edge and one right edge.** After one pass of the layout the page
is frozen and placed by hand (`_place`): the slide, the tracks, the profile
and its legend start on one left edge, and the clones, tracks and profile
end on one right edge, so a chromosome boundary in (c) is over the same
boundary in (d). Every line is `PROFILE_LINEWIDTH` wide. Each
letter sits over its panel, on its leftmost text: (a) and (b)'s extent
ticks, and for (c) and (d) the left column `NAME_INSET` in, where (d)'s
clone names and (c)'s RDR and BAF labels start.

**What is drawn is what the run drew.** `recording` keeps the arguments of
the run's last call to each of the three plotting functions -- for (c), the
one with integer copies, which is `clones_genomic.pdf` -- so the page is a
re-drawing of the run's own figures rather than a second derivation of them.

**Text is set once, for the page.** `cnaster`'s helpers hardcode 6 to 12
pt, sized for a 20 in page; at 4.80 in the larger are half an axis.
After drawing, every text is set to `FONT_SIZE`, the letters (a) to (d)
included.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

import numpy as np

from port.patch.plot_copy_number_profile import LINEWIDTH as PROFILE_LINEWIDTH

FONT_SIZE = 7.0
"""Every text on the page, in points (#339)."""

GENOMIC_FONT_SIZE = FONT_SIZE
"""The genomic panel's text, at the page's size."""

LABEL_SIZE = FONT_SIZE
"""The panel letters, at the page's size and not bold."""

SIDE = 0.36
"""(c) and (d), each at most this fraction of the width: square, side by side,
shrunk if (d)'s key or the page's height needs it."""

LABEL_GAP = 2.0
"""Points between a label and what it labels: (d)'s clone names and its
axis, (b)'s key and (b)."""

GAP_CLOSED = 0.5
"""The fraction of the white closed between (c)'s clones, and between (c)
and (d)'s key."""

NAME_INSET = 3 * LABEL_GAP
"""Points from the page's left edge to the left column: (d)'s clone names,
(c)'s RDR and BAF labels, and the letters of (c) and (d)."""


SPATIAL_GAP = 0.17
"""Inches between (c) and (d): twice the 0.084 the layout gave, widened if
(d)'s letter needs more."""

FOOT = 0.4
"""Inches of slack under the layout, trimmed off at the end."""

LEGEND_BOX = 0.2
"""Inches, one box of (d)'s legend."""

LEGEND_ROW = 0.24
"""(d)'s legend row against its profile's height, 1.0: 20% under the 0.3
it had, the difference given to the profile's rows."""

PROFILE_ROWS = 0.8
"""(d)'s axis against the height it had: its rows 20% shorter, the
difference given to (c)'s tracks."""

TOP_LINE = 0.1
"""Inches above the genomic panel for the top clone's statistics line."""

SPATIAL_SCALE = 0.8
"""The slide and clone panels, as a fraction of the largest that fits."""

SPATIAL_INSET = 0.9
"""Each panel within its box: centred, the rest of the box a white border."""

SPATIAL_ROW = 0.9
"""Their row's height, as a fraction of that largest: 0.1 of it left as white
space under the panels."""


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


def _fractions(ax: Any, handles: Any, labels: list[str], columns: int) -> Any:
    """A clone's state fractions on its statistics line, justified right,
    in `columns` columns."""
    from matplotlib.legend_handler import HandlerTuple

    legend = ax.legend(
        handles,
        labels,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=columns,
        # NB a tuple of dots side by side: the states behind one entry.
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.1)},
        frameon=False,
        borderpad=0.0,
        borderaxespad=0.0,
        handletextpad=0.2,
        columnspacing=0.6,
        fontsize=FONT_SIZE,
    )
    legend.set_in_layout(False)
    ax._port_fractions = (list(handles), list(labels))
    return legend


def _colour_by_state(top: Any, genomic: Any) -> None:
    """(c)'s points coloured by the fitted, continuous HMM state, each key
    entry that state's share of the clone's bins and the pair it decodes to.

    `clones_genomic` colours by the decoded `(A, B)`, which merges states
    that decode to one pair; here each fitted state keeps its own colour, and
    the key still reads in integer copies. Only the page changes: the run's
    own `clones_genomic.pdf` is `cnaster`'s colouring.
    """
    import matplotlib.colors as mcolors
    import seaborn as sns  # type: ignore[import-untyped]
    from matplotlib.collections import LineCollection, PathCollection
    from matplotlib.lines import Line2D

    from port.patch.plot_genomic import clone_groups, clone_path

    res_combine = genomic.kwargs["res_combine"]
    df_cnv = genomic.kwargs.get("df_cnv")
    n_obs = int(np.asarray(genomic.args[1]).shape[0])
    n_states = np.asarray(res_combine["new_log_mu"]).shape[0]
    palette = [mcolors.to_rgba(c) for c in sns.color_palette("deep", n_states)]
    labels, _ = clone_groups(res_combine, None)
    axes = list(top.axes)
    per_clone = len(axes) // len(labels)

    for clone, label in enumerate(labels):
        path = clone_path(res_combine, clone, n_obs)
        colours = np.array([palette[k] for k in path])

        for ax in axes[per_clone * clone : per_clone * (clone + 1)]:
            for collection in ax.collections:
                if isinstance(collection, PathCollection):
                    collection.set_facecolor([tuple(c) for c in colours])
                elif isinstance(collection, LineCollection) and len(
                    collection.get_segments()
                ) == len(colours):
                    collection.set_color([tuple(c) for c in colours])

        # NB one entry per decoded pair: the states decoding to it share the
        #    entry, their shares summed, a dot of each state's colour.
        decoded: dict[tuple[int, int] | None, list[int]] = {}

        for k in np.unique(path):
            pair: tuple[int, int] | None = None

            if df_cnv is not None:
                a = df_cnv[f"clone{label} A"].to_numpy()[path == k]
                b = df_cnv[f"clone{label} B"].to_numpy()[path == k]
                pairs, counts = np.unique(
                    np.stack([a, b], axis=1), axis=0, return_counts=True
                )
                major = pairs[np.argmax(counts)]
                pair = (int(major[0]), int(major[1]))

            decoded.setdefault(pair, []).append(int(k))

        entries: list[Any] = []
        texts: list[str] = []

        for pair, states in decoded.items():
            share = 100.0 * float(np.isin(path, states).mean())
            dots = tuple(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=palette[k],
                    markersize=3,
                )
                for k in states
            )
            entries.append(dots)
            suffix = "" if pair is None else f" ({pair[0]}, {pair[1]})"
            texts.append(f"{share:.1f}%{suffix}")

        anchor = axes[per_clone * clone]
        if anchor.get_legend() is not None:
            anchor.get_legend().remove()
        _fractions(anchor, entries, texts, len(texts))


def _fit_tracks(panel: Any) -> None:
    """The genomic tracks' furniture, at a track a third of an inch tall.

    Upstream labels every integer of RDR and every 0.2 of BAF, which at this
    height is a stack of overlapping numbers: the two ends and the middle
    are kept, at one decimal as (a) and (b) are, and the gridlines stay. The
    clone name is moved clear of the axis label, and the legend's 10 pt
    markers are set to the page's.

    The statistics line and the legend share one line above each clone's
    RDR track, in the gap row `clone_axes` leaves between clones, and are
    kept out of the layout: counted in it, each pushes every track apart by
    its own height.
    """
    for ax in panel.axes:
        ticks = ax.get_yticks()

        if ticks.size > 2:
            ends = [ticks[0], (ticks[0] + ticks[-1]) / 2, ticks[-1]]
            ax.set_yticks(ends, [f"{tick:.1f}" for tick in ends])

            # NB inside the track's height: a label centred on the edge
            #    overhangs it, and the layout pads every track to make room.
            bottom, _, top = ax.get_yticklabels()
            bottom.set_verticalalignment("bottom")
            top.set_verticalalignment("top")

        ax.tick_params(length=2, pad=1)
        # NB the RDR and BAF labels a point off their ticks, not 4, and
        #    without the blank line `cnaster` puts before each.
        ax.yaxis.labelpad = 1.0
        ax.set_ylabel(ax.get_ylabel().strip())

        names = [t for t in ax.texts if t.get_rotation() == 90.0]
        stats = [t for t in ax.texts if t.get_rotation() == 0.0]

        # NB the clone's name heads its statistics line, above the RDR track,
        #    rather than standing vertically beside the tracks, where it ran
        #    into the RDR and BAF labels at 4.80 in (#339).
        for name in names:
            if stats:
                # NB the name alone: at 11 pt the spot and UMI counts ran
                #    past the page's width.
                stats[0].set_text(clone_symbol(name.get_text()))
            name.set_visible(False)
            name.set_in_layout(False)

        for text in ax.texts:
            if text.get_rotation() == 0.0:
                text.set_y(1.0)
                text.set_in_layout(False)
            if text.get_text().startswith("chr"):
                # NB the profile below names the chromosomes on the same
                #    edges, so the tracks drop theirs and take the height.
                text.set_visible(False)
                text.set_in_layout(False)

        # NB frames at the width of the chromosome boundaries and of the
        #    profile's outlines, one line weight on the page.
        for spine in ax.spines.values():
            spine.set_linewidth(PROFILE_LINEWIDTH)

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
            _fractions(ax, handles, labels, len(labels))


ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def clone_symbol(label: str) -> str:
    """`cnaster`'s "Clone 0", "Clone III" as the paper's $m_N$, $m_3$.

    Clone 0 is the normal clone, `N`; the rest are numbered in arabic.
    """
    word = label.split()[-1]

    if word.isdigit():
        number = int(word)
    else:
        values = [ROMAN[c] for c in word.upper()]
        number = sum(
            -v if k + 1 < len(values) and v < values[k + 1] else v
            for k, v in enumerate(values)
        )

    if number == 0:
        return r"$m_N$"

    return rf"$m_{number}$" if number < 10 else rf"$m_{{{number}}}$"


def _clone_key(ax: Any, clone_ids: Any, colours: list[str]) -> None:
    """The clones in one column left of `ax`, the key's bottom on the axis's."""
    from cnaster.utils import cast_clone_label
    from matplotlib.lines import Line2D

    entries = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=colour, markersize=5)
        for colour in colours
    ]
    ax.legend(
        entries,
        [clone_symbol(cast_clone_label(clone)) for clone in clone_ids],
        ncol=1,
        loc="lower right",
        bbox_to_anchor=(-0.04, 0.0),
        frameon=False,
        borderpad=0.0,
        handlelength=0.8,
        handletextpad=0.3,
        borderaxespad=0.0,
        fontsize=FONT_SIZE,
    ).set_in_layout(False)


def _extents(ax: Any, coords: np.ndarray) -> None:
    """A perimeter like every other panel's, and the spots' first, middle and
    last coordinate on each axis as its only ticks: the section's extent.

    `y` is drawn as `-y`, so its ticks carry the coordinate, not the height.
    """
    from port.patch.plot_copy_number_profile import LINEWIDTH as PROFILE_LINEWIDTH

    ax.axis("on")
    x0, x1 = float(coords[:, 0].min()), float(coords[:, 0].max())
    y0, y1 = float(coords[:, 1].min()), float(coords[:, 1].max())
    x = (x0, (x0 + x1) / 2, x1)
    y = (y0, (y0 + y1) / 2, y1)
    ax.set_xticks(x, [f"{v:.1f}" for v in x])
    ax.set_yticks([-v for v in y], [f"{v:.1f}" for v in y])
    ax.tick_params(length=2, pad=1, width=PROFILE_LINEWIDTH)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(PROFILE_LINEWIDTH)
        spine.set_edgecolor("black")


def _set_x(
    ax: Any, x0: float, x1: float, y0: float | None = None, height: float | None = None
) -> None:
    """Move `ax` to `x0..x1` inches across the page, and optionally `y0` up
    and `height` tall, in its own (sub)figure's coordinates."""
    from matplotlib.transforms import Bbox

    figure = ax.get_figure(root=True)
    here = ax.get_window_extent(figure.canvas.get_renderer())
    dpi = figure.dpi
    bottom = here.y0 if y0 is None else y0 * dpi
    top = here.y1 if height is None else bottom + height * dpi
    parent = ax.get_figure().transSubfigure.inverted()
    (a, b), (c, d) = parent.transform([(x0 * dpi, bottom), (x1 * dpi, top)])
    ax.set_position(Bbox([[a, b], [c, d]]))


def _move(ax: Any, up: float) -> None:
    """Move `ax` `up` inches, its width and height kept."""
    dpi = ax.get_figure(root=True).dpi
    here = ax.get_window_extent(ax.get_figure(root=True).canvas.get_renderer())
    _set_x(ax, here.x0 / dpi, here.x1 / dpi, here.y0 / dpi + up, here.height / dpi)


def _trim(figure: Any, bottom: float, top: float = 0.0) -> None:
    """Cut `bottom` inches off the page's foot and `top` off its head, every
    axis kept where it is measured from the foot, less `bottom`.

    Only ever cuts: the page is drawn with slack at its foot.
    """
    if min(bottom, top) < 0.0:
        msg = f"the page is short by {-min(bottom, top):.3f} in; give it more slack"
        raise ValueError(msg)

    renderer = figure.canvas.get_renderer()
    dpi = figure.dpi
    width, height = figure.get_size_inches()
    # NB frozen: an axis's extent is a live transform of the page, so an
    #    unfrozen one reads the resized page and moves each axis twice.
    kept = [(ax, ax.get_window_extent(renderer).frozen()) for ax in figure.get_axes()]
    figure.set_size_inches(width, height - bottom - top)
    figure.canvas.draw()

    for ax, box in kept:
        _set_x(ax, box.x0 / dpi, box.x1 / dpi, box.y0 / dpi - bottom, box.height / dpi)


def _place(
    figure: Any,
    top: Any,
    profile_ax: Any,
    legend_ax: Any,
    slide_ax: Any,
    spatial_ax: Any,
) -> None:
    """Every panel on one left edge and one right edge, and the letters beside them.

    Run once the layout is drawn and frozen, when every extent is known:

    - the common left edge is as far out as the furniture left of it
      allows -- (c)'s RDR and BAF labels, (d)'s clone names, (a)'s extent
      ticks -- each a `LABEL_GAP` in from the page, the names left-aligned;
    - (a)'s tracks, (b)'s axis and its legend share that left edge, and a
      right edge pulled in until no chromosome name runs off the page, so
      bin `i` of (b) is under bin `i` of (a);
    - (a) starts on the left edge and (b) ends on the right, each drawn at
      `SPATIAL_INSET` of its box and centred in it;
    - each letter over its panel, on its leftmost text, and the page's head
      a `LABEL_GAP` over the letters.
    """
    from matplotlib.transforms import blended_transform_factory

    from port.patch.plot_copy_number_profile import plot_ascn_legend

    renderer = figure.canvas.get_renderer()
    dpi = figure.dpi
    width, height = figure.get_size_inches()
    gap = LABEL_GAP / 72.0

    letters = [figure.text(0.0, 0.0, f"({k})", fontsize=LABEL_SIZE) for k in "abcd"]
    names = profile_ax.get_yticklabels()
    widest = max(t.get_window_extent(renderer).width for t in names) / dpi

    # NB one left column `NAME_INSET` in: (d)'s names and (c)'s RDR and BAF
    #    labels start there, and the left edge is as close as what sits
    #    between the column and the axes allows -- a label, a `gap`, the
    #    ticks -- or (a)'s extent ticks, whichever is wider.
    tracks = list(top.axes)
    column = NAME_INSET / 72.0
    labelled = [ax for ax in tracks if ax.get_ylabel()]
    label = max(ax.yaxis.label.get_window_extent(renderer).width for ax in labelled)
    ticks_width = max(
        t.get_window_extent(renderer).width
        for ax in tracks
        for t in ax.get_yticklabels()
        if t.get_text()
    )
    furniture = label / dpi + gap + ticks_width / dpi + 3.0 / 72.0
    extent = (
        max(t.get_window_extent(renderer).width for t in slide_ax.get_yticklabels())
        / dpi
        + 3.0 / 72.0
    )
    left = column + max(furniture, widest + gap, extent)
    right = max(ax.get_window_extent(renderer).x1 for ax in tracks) / dpi

    chromosomes = list(profile_ax.get_xticklabels())

    for _ in range(3):
        for ax in [*tracks, profile_ax, legend_ax]:
            _set_x(ax, left, right)

        figure.canvas.draw()
        overrun = (
            max(t.get_window_extent(renderer).x1 for t in chromosomes) / dpi - width
        )

        if overrun <= 0.0:
            break

        right -= overrun + gap

    # NB each clone's state fractions follow its name, so a clone with many
    #    states wraps its key onto rows rather than running off the page.
    for ax in tracks:
        legend = ax.get_legend()

        # NB a legend lays out its box once, so a narrower one is a new one.
        while (
            legend is not None
            and legend._ncols > 1
            and legend.get_window_extent(renderer).x1 / dpi > right
        ):
            handles, labels = ax._port_fractions
            columns = legend._ncols - 1
            legend.remove()
            legend = _fractions(ax, handles, labels, columns)
            figure.canvas.draw()

    for text in names:
        text.set_horizontalalignment("left")

    profile_ax.tick_params(
        axis="y", which="major", pad=(left - column) * 72.0, length=0
    )

    # NB each RDR and BAF label's left on the column; a y label is anchored
    #    on its side nearest the axis, so it is set its own width right of it.
    for ax in labelled:
        at = column + ax.yaxis.label.get_window_extent(renderer).width / dpi
        ax.yaxis.set_label_coords(
            at,
            0.5,
            transform=blended_transform_factory(figure.dpi_scale_trans, ax.transAxes),
        )

    plot_ascn_legend(
        legend_ax,
        box_w=LEGEND_BOX,
        box_h=0.8,
        tick_len=0.1,
        label_fontsize=FONT_SIZE,
        span=right - left,
        title_on_edge=True,
    )

    # NB (a) and (b) square and equal across the top: the slide on the
    #    left edge, the clones' key in one column on the right edge, its
    #    bottom on theirs, and the clones two `LABEL_GAP` left of it. The largest that fits across is
    #    `fit`; their boxes are `SPATIAL_SCALE` of it, each panel
    #    `SPATIAL_INSET` of its box and centred there, in a row
    #    `SPATIAL_ROW` of it tall, and everything below moves up to meet it.
    figure.canvas.draw()
    key = spatial_ax.get_legend().get_window_extent(renderer)
    # NB (b)'s extent ticks on its left, between it and the key.
    ticks = (
        max(t.get_window_extent(renderer).width for t in spatial_ax.get_yticklabels())
        / dpi
        + 3.0 / 72.0
    )
    key_width = key.width / dpi + ticks + 2 * gap
    raised = max(t.get_window_extent(renderer).height for t in letters) / dpi + gap
    # NB a letter's height above the top row, for (a) and (b).
    ceiling = height - raised - gap
    fit = min(SIDE * width, (right - left - key_width - SPATIAL_GAP) / 2)
    side = SPATIAL_SCALE * fit
    inset = (1.0 - SPATIAL_INSET) * side / 2
    drawn = SPATIAL_INSET * side
    _set_x(slide_ax, left + inset, left + side - inset, ceiling - side + inset, drawn)
    # NB the key on the right edge, (b) a `gap` left of it.
    clones_right = right - key.width / dpi - 2 * gap
    _set_x(
        spatial_ax,
        clones_right - side + inset,
        clones_right - inset,
        ceiling - side + inset,
        drawn,
    )
    anchor = (right - (clones_right - side + inset)) / drawn
    spatial_ax.get_legend().set_bbox_to_anchor(
        (anchor, 0.0), transform=spatial_ax.transAxes
    )

    stats = [t for t in tracks[0].texts if t.get_visible()]
    lower = [*tracks, profile_ax, legend_ax]
    figure.canvas.draw()
    first = max(t.get_window_extent(renderer).y1 for t in stats) / dpi
    # NB under (a) and (b)'s extent ticks, where they reach below the row.
    floor = min(
        ceiling - SPATIAL_ROW * fit,
        min(ax.get_tightbbox(renderer).y0 for ax in (slide_ax, spatial_ax)) / dpi - gap,
    )
    lift = (floor - 3 * gap) - first

    for ax in lower:
        here = ax.get_window_extent(renderer)
        _set_x(
            ax, here.x0 / dpi, here.x1 / dpi, here.y0 / dpi + lift, here.height / dpi
        )

    # NB the white between clones, from a clone's statistics line to the
    #    BAF axis over it, halved: each clone moves up by half of it for
    #    every clone above, and (d) with the last.
    figure.canvas.draw()
    rows = [tracks[k : k + 2] for k in range(0, len(tracks), 2)]

    def head(ax: Any) -> float:
        texts = [t for t in ax.texts if t.get_visible()]
        legend = ax.get_legend()
        return float(
            max(
                [t.get_window_extent(renderer).y1 for t in texts]
                + (
                    [legend.get_window_extent(renderer).y1]
                    if legend is not None
                    else []
                )
            )
        )

    between = [
        (upper[-1].get_window_extent(renderer).y0 - head(lower_row[0])) / dpi
        for upper, lower_row in pairwise(rows)
    ]
    shift = 0.0

    for row, closed in zip(rows[1:], between, strict=True):
        shift += GAP_CLOSED * closed
        for ax in row:
            _move(ax, shift)

    # NB and the white between (c)'s last axis and (d)'s key, which (d)'s
    #    letter no longer sits in: it moves to the left column, level with
    #    the key.
    figure.canvas.draw()
    key_top = max(
        a.get_window_extent(renderer).y1 for a in [*legend_ax.patches, *legend_ax.texts]
    )
    last = min(
        [rows[-1][-1].get_window_extent(renderer).y0]
        + [
            t.get_window_extent(renderer).y0
            for t in rows[-1][-1].get_yticklabels()
            if t.get_text()
        ]
    )

    # NB the white as it was includes the letter's row, `raised`, which the
    #    layout no longer leaves; closed from that, not from what is left.
    white = last / dpi - shift - key_top / dpi
    for ax in (profile_ax, legend_ax):
        _move(ax, shift + white - (1.0 - GAP_CLOSED) * (white + raised))

    # NB the page ends at its lowest text, `gap` under it.
    figure.canvas.draw()
    lowest = (
        min(
            t.get_window_extent(renderer).y0
            for ax in lower
            for t in [*ax.texts, *(ax.get_xticklabels() if ax.axison else [])]
            if t.get_visible() and t.get_text()
        )
        / dpi
    )
    _trim(figure, lowest - gap)

    # NB each letter just above its panel and on its panel's leftmost text:
    #    (a) and (b)'s extent ticks, (c) and (d)'s left column. Placed, the
    #    page's head is cut to a `gap` over them, and they are placed again.
    heads = (
        min(t.get_window_extent(renderer).x0 for t in slide_ax.get_yticklabels()),
        min(t.get_window_extent(renderer).x0 for t in spatial_ax.get_yticklabels()),
        column * dpi,
        column * dpi,
    )

    def letter_positions() -> None:
        figure.canvas.draw()
        height = figure.get_size_inches()[1]

        for text, ax, x in zip(
            letters, (slide_ax, spatial_ax, tracks[0], legend_ax), heads, strict=True
        ):
            box = ax.get_window_extent(renderer)
            # NB over (c)'s statistics line and (a) and (b)'s top tick.
            above = [*ax.texts, *(ax.get_yticklabels() if ax.axison else [])]
            top = max(
                [box.y1]
                + [
                    t.get_window_extent(renderer).y1
                    for t in above
                    if t.get_visible() and t.get_text()
                ]
            )
            text.set_position((x / dpi / width, (top / dpi + gap / 2) / height))
            text.set_verticalalignment("bottom")

        # NB (d)'s level with its key's first title, in the column beside it.
        title = legend_ax.texts[0].get_window_extent(renderer)
        letters[3].set_position(
            (column / width, (title.y0 + title.y1) / 2 / dpi / height)
        )
        letters[3].set_verticalalignment("center")

    letter_positions()
    figure.canvas.draw()
    highest = max(t.get_window_extent(renderer).y1 for t in letters) / dpi
    _trim(figure, 0.0, figure.get_size_inches()[1] - highest - gap)
    letter_positions()

    # NB a legend's box is not its anchor to the point: moved, on the page
    #    as it ends, by what it falls short of the right edge.
    figure.canvas.draw()
    short = right - spatial_ax.get_legend().get_window_extent(renderer).x1 / dpi
    spatial_ax.get_legend().set_bbox_to_anchor(
        (anchor + short / drawn, 0.0), transform=spatial_ax.transAxes
    )


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
        plot_copy_number_profile,
    )
    from port.patch.plot_genomic import plot_clones_genomic
    from port.patch.plotting.genomic import PAPER_WIDTH
    from port.patch.plotting.spatial import draw_clones_spatial, spot_colours

    if recorded.genomic is None or recorded.spatial is None or recorded.profile is None:
        msg = f"the run made {recorded.calls}; the page needs all three"
        raise ValueError(msg)

    width = PAPER_WIDTH if width is None else width
    genomic = recorded.genomic
    n_clones = len(np.unique(genomic.kwargs["res_combine"]["new_assignment"]))
    # NB one profile row per clone rather than two halves, and the height
    #    that frees goes to (a), whose tracks are the densest on the page.
    # NB (b)'s rows taller than (a)'s tracks, and (c)/(d) square at `SIDE`,
    #    their key beside them rather than below, so nothing runs off the
    #    page at 4.80 in; no row for the letters, which sit in the margin
    #    (#339).
    # NB (d)'s axis at `PROFILE_ROWS` of the height it had, its legend row
    #    unchanged, and the difference given to (c)'s tracks.
    middle_height = 0.27 * n_clones + 0.45
    profile_height = middle_height / (1.0 + LEGEND_ROW)
    freed = (1.0 - PROFILE_ROWS) * profile_height
    heights = (
        SIDE * width + TOP_LINE + 0.05,
        1.05 * n_clones + freed,
        middle_height - freed,
    )

    # NB no space between axes beyond what `clone_axes`' gap rows give.
    figure = plt.figure(
        figsize=(width, sum(heights) + FOOT),
        dpi=300,
        facecolor="white",
        # NB `FOOT` of slack under the layout, which `_place` trims off once
        #    it has moved the lower panels down for (d)'s letter.
        layout=ConstrainedLayoutEngine(
            h_pad=0.01,
            hspace=0.0,
            rect=(
                0.0,
                FOOT / (sum(heights) + FOOT),
                1.0,
                1.0 - FOOT / (sum(heights) + FOOT),
            ),
        ),
    )
    rows: Any = figure.subfigures(3, 1, height_ratios=heights, hspace=0.02)
    spatial_row, top, middle = rows[0], rows[1], rows[2]

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
    _colour_by_state(top, genomic)

    # NB the phase and copy key above the profile, under (d)'s letter.
    legend_ax, profile_ax = middle.subplots(
        2, 1, height_ratios=(LEGEND_ROW, PROFILE_ROWS)
    )
    plot_copy_number_profile(recorded.profile.args[0], ax=profile_ax)

    profile_ax.set_yticklabels(
        [clone_symbol(t.get_text()) for t in profile_ax.get_yticklabels()]
    )

    for text in profile_ax.get_yticklabels():
        text.set_rotation(0)
    # NB `cnaster` adds its legend at fixed page coordinates, which a layout
    #    engine does not manage; it is redrawn into an axis that is managed.
    middle.axes[-1].remove()
    legend_ax.axis("off")

    coords, assignment = recorded.spatial.args[:2]
    slide_ax, spatial_ax = spatial_row.subplots(1, 2)
    draw_clones_spatial(
        spatial_ax,
        np.asarray(coords),
        assignment,
        recorded.spatial.kwargs.get("single_tumor_prop"),
    )
    spatial_ax.get_legend().remove()
    _, clone_ids, colours = spot_colours(assignment)
    _clone_key(spatial_ax, clone_ids, colours)

    image, extent = slide_image(he_frame)
    slide_ax.imshow(image, extent=extent, interpolation="none")
    # NB the section (b) shows, so a boundary sits at the same place in both.
    slide_ax.set_xlim(spatial_ax.get_xlim())
    slide_ax.set_ylim(spatial_ax.get_ylim())
    slide_ax.set_aspect("equal")

    for ax in (slide_ax, spatial_ax):
        _extents(ax, np.asarray(coords))
    # NB an equal-aspect axis shrinks inside its box; held to the page's edges.
    slide_ax.set_anchor("W")
    spatial_ax.set_anchor("E")

    _set_text(top, GENOMIC_FONT_SIZE)

    for panel in (middle, spatial_row):
        _set_text(panel, FONT_SIZE)

    # NB (b)'s chromosome names at (a)'s size: at 4.80 in the short
    #    chromosomes' rotated names touch at 6 pt (#339).
    for text in profile_ax.get_xticklabels():
        text.set_fontsize(GENOMIC_FONT_SIZE)
    # NB a point lower than the profile's own -5 pt, clear of its bottom edge.
    profile_ax.tick_params(axis="x", pad=-4)

    # NB laid out once and frozen, then placed on the page by hand.
    figure.canvas.draw()
    figure.set_layout_engine("none")
    _place(figure, top, profile_ax, legend_ax, slide_ax, spatial_ax)

    return figure
