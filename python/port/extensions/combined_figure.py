r"""Two figures from a run, at `llncs`'s text width (#309, #339).

Drawn once at their printed size, 122 mm wide, and included at
`width=\linewidth` with nothing scaled:

- `genomic_figure`, the text block's height: **(a)** `clones_genomic`, RDR
  and BAF along the genome per clone, drawn by `port.patch.plot_genomic`;
  **(b)** `copy_number_profile`, the integer copies per clone, drawn by
  `port.patch.plot_copy_number_profile`, which names the chromosomes for
  both, under its mirror and copy-number key.
- `spatial_figure`, about a quarter of that: **(a)** the H&E slide, as
  `cnaster.he.get_he_image` reads it; **(b)** `clones_spatial`, the fitted
  clone of each spot tiled by `port.patch.plotting.spatial`, keyed on the
  right edge. Square and on the spot coordinates, so a clone boundary in (b)
  reads against the tissue in (a).

Clones are named as the paper names them, $m_N$ for the normal and $m_1$,
$m_2$, ... for the rest.

**One left edge and one right edge.** After one pass of the layout each page
is frozen and placed by hand: the genomic tracks, the profile and its key
share a left and a right edge, so a chromosome boundary in (a) is over the
same boundary in (b). Every line is `PROFILE_LINEWIDTH` wide. Each letter
sits on its panel's leftmost text or edge: the slide's extent ticks, the
clones' left edge -- their rows are the slide's, so only the slide labels
them -- and the genomic figure's left column `NAME_INSET` in, where the
clone names and the RDR and BAF labels start.

**What is drawn is what the run drew.** `recording` keeps the arguments of
the run's last call to each of the three plotting functions -- for the
tracks, the one with integer copies, which is `clones_genomic.pdf` -- so the
figures re-draw the run's own rather than derive them a second time.

**Text is set once, for the page.** `cnaster`'s helpers hardcode 6 to 12
pt, sized for a 20 in page; every text is set to `FONT_SIZE`.
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

LABEL_SIZE = FONT_SIZE
"""The panel letters, at the page's size and not bold."""

LABEL_GAP = 2.0
"""Points between a label and what it labels: the profile's clone names and
its axis, the spatial key and the clones."""

GAP_CLOSED = 0.5
"""The fraction of the white closed between the genomic figure's clones, and
between its tracks and the profile's key."""

NAME_INSET = 3 * LABEL_GAP
"""Points from the genomic figure's left edge to its left column: the
profile's clone names, the RDR and BAF labels, and the letters."""

SPATIAL_GAP = 0.17
"""Inches between the slide and the clones' extent ticks."""

TEXT_HEIGHT = 193.0 / 25.4
"""`llncs`'s `\\textheight`, 193 mm in inches: the genomic figure's height."""

FOOT = 0.4
"""Inches of slack under the layout, trimmed off at the end."""

LEGEND_BOX = 0.2
"""Inches, one box of the profile's key."""

LEGEND_ROW = 0.24
"""The profile's key row against the profile's base height, 1.0."""

PROFILE_ROWS = 0.8
"""The profile's axis against its base height: its rows 20% shorter, the
difference given to the tracks."""

TOP_LINE = 0.1
"""Inches above the tracks for the top clone's statistics line."""


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


def _put(
    ax: Any,
    x0: float | None = None,
    x1: float | None = None,
    y0: float | None = None,
    height: float | None = None,
) -> None:
    """Set `ax`'s box in inches on the page; what is not given is kept."""
    from matplotlib.transforms import Bbox

    figure = ax.get_figure(root=True)
    dpi = figure.dpi
    here = ax.get_window_extent(figure.canvas.get_renderer())
    left = here.x0 if x0 is None else x0 * dpi
    right = here.x1 if x1 is None else x1 * dpi
    bottom = here.y0 if y0 is None else y0 * dpi
    top = here.y1 if height is None else bottom + height * dpi
    parent = ax.get_figure().transSubfigure.inverted()
    (a, b), (c, d) = parent.transform([(left, bottom), (right, top)])
    ax.set_position(Bbox([[a, b], [c, d]]))


def _up(ax: Any, inches: float) -> None:
    """Move `ax` up by `inches`, its size kept."""
    dpi = ax.get_figure(root=True).dpi
    here = ax.get_window_extent(ax.get_figure(root=True).canvas.get_renderer())
    _put(ax, here.x0 / dpi, here.x1 / dpi, here.y0 / dpi + inches, here.height / dpi)


def _inches(figure: Any, artists: Any, edge: str) -> list[float]:
    """Each visible, non-empty artist's `edge` (`x0`, `x1`, `y0`, `y1`),
    in inches from the page's foot or left."""
    renderer = figure.canvas.get_renderer()
    return [
        getattr(a.get_window_extent(renderer), edge) / figure.dpi
        for a in artists
        if a.get_visible() and (not hasattr(a, "get_text") or a.get_text())
    ]


def _cut(
    figure: Any, foot: float, letters: list[tuple[Any, float, float, str]]
) -> None:
    """Cut `foot` inches off the page, set each letter, and cut the head to
    a `LABEL_GAP` over the highest.

    Each letter is `(text, x, y, alignment)`, in inches on the uncut page:
    a letter is set as a fraction of the page, so it is set after each cut.
    Every axis keeps its place measured from the foot, less the cut.
    """
    renderer = figure.canvas.get_renderer()
    dpi = figure.dpi
    width = figure.get_size_inches()[0]

    def resize(below: float, above: float) -> None:
        if min(below, above) < 0.0:
            msg = (
                f"the page is short by {-min(below, above):.3f} in; give it more slack"
            )
            raise ValueError(msg)

        # NB frozen: an axis's extent is a live transform of the page, so an
        #    unfrozen one reads the resized page and moves each axis twice.
        kept = [
            (ax, ax.get_window_extent(renderer).frozen()) for ax in figure.get_axes()
        ]
        height = figure.get_size_inches()[1]
        figure.set_size_inches(width, height - below - above)
        figure.canvas.draw()

        for ax, box in kept:
            _put(ax, box.x0 / dpi, box.x1 / dpi, box.y0 / dpi - below, box.height / dpi)

    def place() -> None:
        height = figure.get_size_inches()[1]

        for text, x, y, va in letters:
            text.set_position((x / width, (y - foot) / height))
            text.set_verticalalignment(va)

        figure.canvas.draw()

    # NB the foot, then the head over the letters as drawn: one cut, or a
    #    head reckoned rather than measured, moves each axis through other
    #    floats, enough to shift an image's edge across a pixel.
    resize(foot, 0.0)
    place()
    highest = max(_inches(figure, [text for text, *_ in letters], "y1"))
    resize(0.0, figure.get_size_inches()[1] - highest - LABEL_GAP / 72.0)
    place()


def _left_column(
    figure: Any, tracks: list[Any], profile_ax: Any
) -> tuple[float, float]:
    """The left column, `NAME_INSET` in, and the axes' common left edge.

    (b)'s names and (a)'s RDR and BAF labels start on the column, and the
    left edge is as close as what sits between the column and the axes
    allows -- a label, a `gap`, the ticks -- or the widest name and a `gap`.
    """
    renderer = figure.canvas.get_renderer()
    dpi = figure.dpi
    gap = LABEL_GAP / 72.0
    column = NAME_INSET / 72.0
    widest = max(
        t.get_window_extent(renderer).width for t in profile_ax.get_yticklabels()
    )
    label = max(
        ax.yaxis.label.get_window_extent(renderer).width
        for ax in tracks
        if ax.get_ylabel()
    )
    ticks = max(
        t.get_window_extent(renderer).width
        for ax in tracks
        for t in ax.get_yticklabels()
        if t.get_text()
    )
    furniture = label / dpi + gap + ticks / dpi + 3.0 / 72.0
    return column, column + max(furniture, widest / dpi + gap)


def _place_genomic(figure: Any, top: Any, profile_ax: Any, legend_ax: Any) -> None:
    """(a)'s tracks and (b)'s profile on one left and one right edge.

    Run once the layout is drawn and frozen, when every extent is known:

    - the left edge is `_left_column`'s, and the right edge is pulled in
      until no chromosome name runs off the page, so bin `i` of (b) is under
      bin `i` of (a);
    - the white between clones, and between (a) and (b)'s key, is closed by
      `GAP_CLOSED`;
    - (a)'s letter over its first statistics line on the column, (b)'s in
      the column level with its key, and the page cut to its text, a
      `LABEL_GAP` clear at the head and foot.
    """
    from matplotlib.transforms import blended_transform_factory

    from port.patch.plot_copy_number_profile import plot_ascn_legend

    renderer = figure.canvas.get_renderer()
    dpi = figure.dpi
    width, height = figure.get_size_inches()
    gap = LABEL_GAP / 72.0
    tracks = list(top.axes)
    letters = [figure.text(0.0, 0.0, f"({k})", fontsize=LABEL_SIZE) for k in "ab"]
    column, left = _left_column(figure, tracks, profile_ax)
    right = max(ax.get_window_extent(renderer).x1 for ax in tracks) / dpi
    chromosomes = list(profile_ax.get_xticklabels())

    for _ in range(3):
        for ax in [*tracks, profile_ax, legend_ax]:
            _put(ax, left, right)

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

    for text in profile_ax.get_yticklabels():
        text.set_horizontalalignment("left")

    profile_ax.tick_params(
        axis="y", which="major", pad=(left - column) * 72.0, length=0
    )

    # NB each RDR and BAF label's left on the column; a y label is anchored
    #    on its side nearest the axis, so it is set its own width right of it.
    for ax in tracks:
        if ax.get_ylabel():
            at = column + ax.yaxis.label.get_window_extent(renderer).width / dpi
            ax.yaxis.set_label_coords(
                at,
                0.5,
                transform=blended_transform_factory(
                    figure.dpi_scale_trans, ax.transAxes
                ),
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

    # NB one measurement, then every move at once: the first statistics
    #    line a letter's height and two `gap` under the head; the white
    #    between clones, from a clone's statistics line to the BAF axis over
    #    it, closed by `GAP_CLOSED`, each clone moving by that much of every
    #    gap above it; and the white between (a)'s last axis and (b)'s key,
    #    counted with a letter's row, `raised`, which (b)'s letter sat in
    #    before it moved to the column beside the key.
    figure.canvas.draw()
    raised = max(t.get_window_extent(renderer).height for t in letters) / dpi + gap
    stats = [t for t in tracks[0].texts if t.get_visible()]
    lift = height - raised - gap - max(_inches(figure, stats, "y1"))
    rows = [tracks[k : k + 2] for k in range(0, len(tracks), 2)]
    between = [
        _inches(figure, [upper[-1]], "y0")[0]
        - max(
            _inches(figure, lower[0].texts, "y1")
            + _inches(
                figure, [lower[0].get_legend()] if lower[0].get_legend() else [], "y1"
            )
        )
        for upper, lower in pairwise(rows)
    ]
    shifts = np.concatenate([[0.0], np.cumsum(GAP_CLOSED * np.asarray(between))])
    white = min(
        _inches(figure, [rows[-1][-1], *rows[-1][-1].get_yticklabels()], "y0")
    ) - max(_inches(figure, [*legend_ax.patches, *legend_ax.texts], "y1"))

    for row, shift in zip(rows, shifts, strict=True):
        for ax in row:
            _up(ax, lift + shift)

    for ax in (profile_ax, legend_ax):
        _up(ax, lift + shifts[-1] + white - (1.0 - GAP_CLOSED) * (white + raised))

    # NB (a)'s letter over its first statistics line, (b)'s level with its
    #    key's first title, both on the column; the foot a `gap` under the
    #    lowest text.
    figure.canvas.draw()
    lowest = min(
        _inches(
            figure,
            [
                t
                for ax in [*tracks, profile_ax, legend_ax]
                for t in [*ax.texts, *(ax.get_xticklabels() if ax.axison else [])]
            ],
            "y0",
        )
    )
    title = legend_ax.texts[0].get_window_extent(renderer)
    _cut(
        figure,
        lowest - gap,
        [
            (letters[0], column, max(_inches(figure, stats, "y1")) + gap / 2, "bottom"),
            (letters[1], column, (title.y0 + title.y1) / 2 / dpi, "center"),
        ],
    )


def _page(width: float, height: float, rect: tuple[float, float, float, float]) -> Any:
    """A white page `width` by `height` inches at 300 dpi, laid out once in
    `rect` by the constrained engine."""
    import matplotlib.pyplot as plt
    from matplotlib.layout_engine import ConstrainedLayoutEngine

    return plt.figure(
        figsize=(width, height),
        dpi=300,
        facecolor="white",
        layout=ConstrainedLayoutEngine(h_pad=0.01, hspace=0.0, rect=rect),
    )


def _genomic_page(genomic: Call, profile: Call, width: float, scale: float) -> Any:
    """The genomic figure with its tracks and profile rows `scale` times
    their base height."""
    # NB `port`'s profile directly: the page is drawn after the run, when
    #    `FIGURE_SWAPS` has been restored and `cnaster`'s names are its own.
    from port.patch.plot_copy_number_profile import plot_copy_number_profile
    from port.patch.plot_genomic import plot_clones_genomic

    n_clones = len(np.unique(genomic.kwargs["res_combine"]["new_assignment"]))
    # NB the profile's base height, its key `LEGEND_ROW` of it and fixed,
    #    its rows `PROFILE_ROWS` of it, the rest given to (a)'s tracks.
    base = (0.27 * n_clones + 0.45) / (1.0 + LEGEND_ROW)
    key, profile_rows = LEGEND_ROW * base, PROFILE_ROWS * base
    heights = (
        scale * (1.05 * n_clones + base - profile_rows) + TOP_LINE,
        scale * profile_rows + key,
    )
    total = sum(heights) + FOOT
    figure = _page(width, total, (0.0, FOOT / total, 1.0, 1.0 - FOOT / total))
    rows: Any = figure.subfigures(2, 1, height_ratios=heights, hspace=0.02)
    top, middle = rows[0], rows[1]

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

    # NB the mirror and copy key above the profile.
    legend_ax, profile_ax = middle.subplots(
        2,
        1,
        height_ratios=(key, scale * profile_rows),
    )
    plot_copy_number_profile(profile.args[0], ax=profile_ax)
    profile_ax.set_yticklabels(
        [clone_symbol(t.get_text()) for t in profile_ax.get_yticklabels()]
    )

    for text in profile_ax.get_yticklabels():
        text.set_rotation(0)
    # NB `cnaster` adds its legend at fixed page coordinates, which a layout
    #    engine does not manage; it is redrawn into an axis that is managed.
    middle.axes[-1].remove()
    legend_ax.axis("off")

    for panel in (top, middle):
        _set_text(panel, FONT_SIZE)
    # NB a point lower than the profile's own -5 pt, clear of its bottom edge.
    profile_ax.tick_params(axis="x", pad=-4)

    # NB laid out once and frozen, then placed on the page by hand.
    figure.canvas.draw()
    figure.set_layout_engine("none")
    _place_genomic(figure, top, profile_ax, legend_ax)
    return figure


@contextlib.contextmanager
def page_style() -> Iterator[None]:
    """Matplotlib's own defaults for the block, whatever the run has set.

    `cnaster.plotting` sets `font.family` to a serif face when it is
    imported, and `cnaster`'s plots set seaborn's theme when they run, so
    without this a figure depended on what had been imported or drawn before
    it (#342). A text or line takes most of its style when it is made, and
    some -- the hatch's weight among them -- when it is drawn, so both the
    build and the write are held: `genomic_figure` and `spatial_figure` hold
    their own, and a caller writing one holds it around the write.
    """
    import matplotlib as mpl

    with mpl.rc_context():
        mpl.style.use("default")
        yield


def _styled(build: Any) -> Any:
    """`build` under `page_style`."""
    import functools

    @functools.wraps(build)
    def styled(*args: Any, **kwargs: Any) -> Any:
        with page_style():
            return build(*args, **kwargs)

    return styled


@_styled
def genomic_figure(
    recorded: Recorded, width: float | None = None, height: float = TEXT_HEIGHT
) -> Any:
    """(a) `clones_genomic` over (b) `copy_number_profile`, `width` by
    `height` inches, the text block's by default; no caption.

    Everything but the tracks and the profile's rows is fixed in points, so
    the two are scaled together until the page is `height` tall: from the
    base scale, then by the secant through two pages, to 0.005 in.
    """
    from port.patch.plotting.genomic import PAPER_WIDTH

    if recorded.genomic is None or recorded.profile is None:
        msg = f"the run made {recorded.calls}; the genomic figure needs both"
        raise ValueError(msg)

    width = PAPER_WIDTH if width is None else width
    genomic, profile = recorded.genomic, recorded.profile
    scales = [1.0]
    heights = [_genomic_page(genomic, profile, width, 1.0).get_size_inches()[1]]
    scales.append(height / heights[0])

    for _ in range(4):
        figure = _genomic_page(genomic, profile, width, scales[-1])
        heights.append(figure.get_size_inches()[1])

        if abs(heights[-1] - height) < 0.005:
            return figure

        slope = (heights[-1] - heights[-2]) / (scales[-1] - scales[-2])
        scales.append(scales[-1] + (height - heights[-1]) / slope)

    msg = f"the genomic figure is {heights[-1]:.3f} in at best, not {height:.3f}"
    raise ValueError(msg)


def _place_spatial(figure: Any, slide_ax: Any, spatial_ax: Any) -> None:
    """(a) the slide on the left, (b)'s key on the right edge and (b) left of
    it, square and as large as fits across; each letter over its panel's
    top-left, on (a)'s extent ticks, and the page cut to its text."""
    renderer = figure.canvas.get_renderer()
    dpi = figure.dpi
    width, height = figure.get_size_inches()
    gap = LABEL_GAP / 72.0
    letters = [figure.text(0.0, 0.0, f"({k})", fontsize=LABEL_SIZE) for k in "ab"]
    figure.canvas.draw()

    def ticks(ax: Any) -> float:
        """The tick labels' width, if shown, and the tick and its pad."""
        widths = [
            t.get_window_extent(renderer).width
            for t in ax.get_yticklabels()
            if t.get_visible()
        ]
        return float(max(widths, default=0.0) / dpi + 3.0 / 72.0)

    key = spatial_ax.get_legend()
    left = NAME_INSET / 72.0 + ticks(slide_ax)
    right = width - gap
    clones_right = right - key.get_window_extent(renderer).width / dpi - 2 * gap
    side = (clones_right - left - SPATIAL_GAP - ticks(spatial_ax)) / 2
    bottom = height - side - 1.0
    _put(slide_ax, left, left + side, bottom, side)
    _put(spatial_ax, clones_right - side, clones_right, bottom, side)
    anchor = (right - (clones_right - side)) / side
    key.set_bbox_to_anchor((anchor, 0.0), transform=spatial_ax.transAxes)

    figure.canvas.draw()
    lowest = min(ax.get_tightbbox(renderer).y0 for ax in (slide_ax, spatial_ax)) / dpi
    _cut(
        figure,
        lowest - gap,
        [
            (
                text,
                min(_inches(figure, [ax, *ax.get_yticklabels()], "x0")),
                max(_inches(figure, [ax, *ax.get_yticklabels()], "y1")) + gap / 2,
                "bottom",
            )
            for text, ax in zip(letters, (slide_ax, spatial_ax), strict=True)
        ],
    )

    # NB a legend's box is not its anchor to the point: moved, on the page
    #    as it ends, by what it falls short of the right edge.
    figure.canvas.draw()
    short = right - key.get_window_extent(renderer).x1 / dpi
    key.set_bbox_to_anchor((anchor + short / side, 0.0), transform=spatial_ax.transAxes)


def integer_labels(assignment: Any, df_cnv: Any) -> Any:
    """`assignment`'s "clone {c}" labels, each clone named by its integer
    copy profile in `df_cnv` (`port.extensions.outputs.integer_clones`):
    clones that decode alike at every bin take the smallest id among them."""
    from port.extensions.outputs import integer_clones

    merged = integer_clones(df_cnv)

    def name(label: Any) -> Any:
        if not isinstance(label, str) or label.split()[-1] not in merged:
            return label
        return f"clone {merged[label.split()[-1]]}"

    return assignment.map(name)


@_styled
def spatial_figure(
    recorded: Recorded,
    he_frame: Any,
    width: float | None = None,
    labels: str = "integer",
) -> Any:
    """(a) the H&E slide and (b) `clones_spatial`, square and as large as fit
    across `width` inches, (b) keyed on the right edge; no caption.

    `labels` is "integer", the default, for clones named by their integer
    copy profile (#344) -- which needs the run's profile call -- or
    "continuous" for the fit's own clones.
    """
    import matplotlib.pyplot as plt

    from port.patch.plotting.genomic import PAPER_WIDTH
    from port.patch.plotting.spatial import draw_clones_spatial, spot_colours

    if recorded.spatial is None:
        msg = f"the run made {recorded.calls}; the spatial figure needs its clones"
        raise ValueError(msg)

    width = PAPER_WIDTH if width is None else width
    # NB drawn on a page taller than it needs, and cut to its text.
    figure = plt.figure(figsize=(width, width), dpi=300, facecolor="white")
    slide_ax = figure.add_axes((0.0, 0.0, 0.4, 0.4))
    spatial_ax = figure.add_axes((0.5, 0.0, 0.4, 0.4))

    coords, assignment = recorded.spatial.args[:2]

    if labels == "integer":
        if recorded.profile is None:
            msg = "integer clone labels need the run's copy_number_profile call"
            raise ValueError(msg)
        assignment = integer_labels(assignment, recorded.profile.args[0])
    elif labels != "continuous":
        msg = f'labels is "integer" or "continuous", not {labels!r}'
        raise ValueError(msg)
    draw_clones_spatial(
        spatial_ax,
        np.asarray(coords),
        assignment,
        recorded.spatial.kwargs.get("single_tumor_prop"),
    )
    # NB `draw_clones_spatial`'s key, a row under the tiles, is redrawn as a
    #    column beside them.
    upstream_key = spatial_ax.get_legend()
    if upstream_key is not None:
        upstream_key.remove()
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
    # NB (b) shares (a)'s rows, so its row labels are (a)'s; its ticks stay.
    spatial_ax.tick_params(axis="y", labelleft=False)

    _set_text(figure, FONT_SIZE)
    _place_spatial(figure, slide_ax, spatial_ax)
    return figure
