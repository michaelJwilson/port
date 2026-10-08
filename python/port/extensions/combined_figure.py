r"""Two figures from a run, and both on one page, at the paper's text width (#309, #339, T- #740).

`genomic_figure`: (a) copy-number profile, (b) RDR/BAF tracks per clone.
`spatial_figure`: (a) H&E slide, (b) fitted clones per spot. `combined_figure`
stacks them in `PANELS` order (PR- #715). Drawn at `PAPER_WIDTH`, text at
`FONT_SIZE`, from the run's own calls kept by `recording`. Clones are named
$m_N$ (normal) and $m_1$, $m_2$, ...
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any, NamedTuple

import numpy as np

from port.extensions.figure_style import (
    CAPTION_ROOM,
    MIN_FONT_SIZE,
    TEXT_HEIGHT,
    TRACK_FONT_SIZE,
)
from port.patch.plot_copy_number_profile import KEY_GROWTH
from port.patch.plot_copy_number_profile import LINEWIDTH as PROFILE_LINEWIDTH

FONT_SIZE = MIN_FONT_SIZE
"""Every text on the page, in points (#743)."""

LABEL_SIZE = FONT_SIZE
"""The panel letters, at the page's size and not bold."""

LABEL_GAP = 2.0
"""Points between a label and what it labels."""

GAP_CLOSED = 0.5
"""The fraction of white closed between the genomic figure's clones, and above its tracks."""

NAME_INSET = 3 * LABEL_GAP
"""Points from the genomic figure's left edge to its left column of names, labels and letters."""

SPATIAL_GAP = 0.17
"""Inches between the slide and the clones' extent ticks."""

FOOT = 1.0
"""Inches of slack under the layout, trimmed off at the end (T- #740, T- #771)."""

LEGEND_BOX = 0.2
"""Inches, one box of the profile's key."""

LEGEND_ROW = 0.24 * KEY_GROWTH
"""The profile's key row against the profile's base height (PR- #715)."""

PROFILE_ROWS = 0.8
"""The profile's axis against its base height; the difference goes to the tracks."""

PANELS = ("clones", "profile", "tracks")
"""The panels of `combined_figure` and `truth_combined_figure`, top to bottom (PR- #715)."""

TOP_LINE = 0.1
"""Inches above the tracks for the top clone's statistics line."""


class Call(NamedTuple):
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
    """Keep the arguments of the run's plotting calls for the block; enter before `patched`.

    Wrappers call through, so the run's figures are unchanged.
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
    """The image `get_he_image` returned as rows, and its extent on the spots' coordinates `(x, -y)` (#309)."""
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


HE_CLASSES = 4
"""`cnaster.he.get_he_image`'s `num_labels`, the gray-level classes `run_cnaster` reads."""

HE_PALETTE = "mako"
"""The H&E classes' palette, distinct from the clones' `rocket` (T- #771)."""


def he_classes(
    spaceranger_dir: str, coords: np.ndarray, num_labels: int = HE_CLASSES
) -> np.ndarray:
    """Each spot's H&E class, `1..num_labels` darkest to brightest, via `port.patch.he.he_image` (#311, T- #771)."""
    import pandas as pd

    from port.patch.he import he_image

    spots = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]})
    frame = he_image(spaceranger_dir, res="hires", pos=spots, num_labels=num_labels)

    if "label" not in frame.columns:
        msg = f"no H&E image under {spaceranger_dir}/spatial/"
        raise ValueError(msg)

    return np.asarray(frame["label"].to_numpy(), dtype=np.int64)


def set_font_size(panel: Any, size: float) -> None:
    """Every text in `panel` at `size`: one size per panel, not a cap."""
    from matplotlib.text import Text

    for text in panel.findobj(Text):
        text.set_fontsize(size)


def _fractions(ax: Any, handles: Any, labels: list[str], columns: int) -> Any:
    """A clone's state fractions on its statistics line, right-justified, in `columns` columns."""
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
    """(c)'s points coloured by fitted HMM state; key entries are shares and decoded pairs (page only)."""
    import matplotlib.colors as mcolors
    import seaborn as sns  # type: ignore[import-untyped]
    from matplotlib.collections import LineCollection, PathCollection
    from matplotlib.lines import Line2D

    from port.patch.plot_genomic import clone_groups, fitted_clone_path

    res_combine = genomic.kwargs["res_combine"]
    df_cnv = genomic.kwargs.get("df_cnv")
    n_obs = int(np.asarray(genomic.args[1]).shape[0])
    n_states = np.asarray(res_combine["new_log_mu"]).shape[0]
    palette = [mcolors.to_rgba(c) for c in sns.color_palette("deep", n_states)]
    labels, _ = clone_groups(res_combine, None)
    axes = list(top.axes)
    per_clone = len(axes) // len(labels)

    for clone, label in enumerate(labels):
        path = fitted_clone_path(res_combine, clone, n_obs)
        colours = np.array([palette[k] for k in path])

        for ax in axes[per_clone * clone : per_clone * (clone + 1)]:
            for collection in ax.collections:
                if isinstance(collection, PathCollection):
                    collection.set_facecolor([tuple(c) for c in colours])
                elif isinstance(collection, LineCollection) and len(
                    collection.get_segments()
                ) == len(colours):
                    collection.set_color([tuple(c) for c in colours])

        # NB one entry per decoded pair, a dot per state decoding to it.
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


def fit_track_furniture(panel: Any) -> None:
    """The genomic tracks' furniture, at a track a third of an inch tall.

    Labels only the y ticks' ends (#743), moves the clone name and legend to
    the statistics line above each RDR track, out of the layout.
    """
    for ax in panel.axes:
        ticks = ax.get_yticks()

        if ticks.size > 2:
            ends = [ticks[0], ticks[-1]]
            ax.set_yticks(ends, [f"{tick:.1f}" for tick in ends])

            # NB inside the track, or the layout pads every track for the overhang.
            bottom, top = ax.get_yticklabels()
            bottom.set_verticalalignment("bottom")
            top.set_verticalalignment("top")

        ax.tick_params(length=2, pad=1)
        ax.yaxis.labelpad = 1.0
        ax.set_ylabel(ax.get_ylabel().strip())

        names = [t for t in ax.texts if t.get_rotation() == 90.0]
        stats = [t for t in ax.texts if t.get_rotation() == 0.0]

        # NB the clone's name heads its statistics line (#339).
        for name in names:
            if stats:
                stats[0].set_text(clone_symbol(name.get_text()))
            name.set_visible(False)
            name.set_in_layout(False)

        for text in ax.texts:
            if text.get_rotation() == 0.0:
                text.set_y(1.0)
                text.set_in_layout(False)
            if text.get_text().startswith("chr"):
                # NB the last track names the contigs (`name_contigs`).
                text.set_visible(False)
                text.set_in_layout(False)

        for spine in ax.spines.values():
            spine.set_linewidth(PROFILE_LINEWIDTH)

        legend = ax.get_legend()

        if legend is not None:
            # NB redrawn on the statistics line: upstream's anchor lands on the track above.
            handles = legend.legend_handles
            labels = [text.get_text() for text in legend.get_texts()]

            for handle in handles:
                handle.set_markersize(3)

            legend.remove()
            _fractions(ax, handles, labels, len(labels))


ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def clone_symbol(label: str) -> str:
    """`cnaster`'s "Clone 0", "Clone III" as the paper's $m_N$, $m_3$ (0 is the normal)."""
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


def clone_order(ids: Any) -> list[str]:
    """`ids` in the fitted clones' index order, normal (0) first, non-integers after by name (PR- #715)."""

    def index(clone: Any) -> tuple[bool, int, str]:
        tail = str(clone).split()[-1]
        return (not tail.isdigit(), int(tail) if tail.isdigit() else 0, tail)

    return sorted((str(c) for c in ids), key=index)


def _clone_key(ax: Any, names: list[str], colours: list[str]) -> None:
    """The clones' key in one column, `names` against `colours`; placed by `_place_spatial`."""
    from matplotlib.lines import Line2D

    entries = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=colour, markersize=5)
        for colour in colours
    ]
    ax.legend(
        entries,
        names,
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
    """A perimeter, and the spots' first, middle and last coordinate per axis as ticks; `y` drawn as `-y`."""
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


def _frame(ax: Any, x: tuple[float, float], y: tuple[float, float]) -> None:
    """Left and bottom spines bound to the data's `y` and `x`, no top or right; images clipped to it (PR- #715)."""
    from matplotlib.patches import Rectangle

    for image in ax.get_images():
        image.set_clip_path(
            Rectangle((x[0], y[0]), x[1] - x[0], y[1] - y[0], transform=ax.transData)
        )
    ax.spines["left"].set_bounds(*y)
    ax.spines["bottom"].set_bounds(*x)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def place_in_inches(
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
    place_in_inches(
        ax, here.x0 / dpi, here.x1 / dpi, here.y0 / dpi + inches, here.height / dpi
    )


def _inches(figure: Any, artists: Any, edge: str) -> list[float]:
    """Each visible, non-empty artist's `edge` (`x0`, `x1`, `y0`, `y1`), in inches from the page's foot or left."""
    renderer = figure.canvas.get_renderer()
    return [
        getattr(a.get_window_extent(renderer), edge) / figure.dpi
        for a in artists
        if a.get_visible() and (not hasattr(a, "get_text") or a.get_text())
    ]


def _cut(
    figure: Any,
    foot: float,
    letters: list[tuple[Any, float, float, str]],
    head: Any = (),
) -> None:
    """Cut `foot` inches off the page, set each letter, and cut the head to a `LABEL_GAP` over the highest text.

    Each letter is `(text, x, y, alignment)`, in inches on the uncut page.
    """
    renderer = figure.canvas.get_renderer()
    dpi = figure.dpi
    width = figure.get_size_inches()[0]

    def resize(below: float, above: float) -> None:
        # NB a head already at its cut reads a float's width short of it.
        below, above = (0.0 if -1e-6 < v < 0.0 else v for v in (below, above))
        if min(below, above) < 0.0:
            msg = (
                f"the page is short by {-min(below, above):.3f} in; give it more slack"
            )
            raise ValueError(msg)

        # NB frozen, or the extent reads the resized page and moves each axis twice.
        kept = [
            (ax, ax.get_window_extent(renderer).frozen()) for ax in figure.get_axes()
        ]
        height = figure.get_size_inches()[1]
        figure.set_size_inches(width, height - below - above)
        figure.canvas.draw()

        for ax, box in kept:
            place_in_inches(
                ax, box.x0 / dpi, box.x1 / dpi, box.y0 / dpi - below, box.height / dpi
            )

    def place() -> None:
        height = figure.get_size_inches()[1]

        for text, x, y, va in letters:
            text.set_position((x / width, (y - foot) / height))
            text.set_verticalalignment(va)

        figure.canvas.draw()

    # NB foot then measured head: one reckoned cut can shift an image edge a pixel.
    resize(foot, 0.0)
    place()
    highest = max(_inches(figure, [*(text for text, *_ in letters), *head], "y1"))
    resize(0.0, figure.get_size_inches()[1] - highest - LABEL_GAP / 72.0)
    place()


def _left_column(
    figure: Any, tracks: list[Any], profile_ax: Any
) -> tuple[float, float]:
    """The left column, `NAME_INSET` in, and the nearest common left edge the labels and ticks allow."""
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


def _place_genomic(
    figure: Any,
    top: Any,
    profile_ax: Any,
    legend_ax: Any,
    contigs: tuple[list[float], list[str]],
) -> None:
    """(a)'s profile over (b)'s tracks, on one left and one right edge, once the layout is frozen.

    The right edge is pulled in until no contig name overruns; keys, rows
    and letters are stacked `LABEL_GAP` apart, and the page cut to its text.
    """
    from matplotlib.transforms import blended_transform_factory

    from port.extensions.genomic_axis import name_contigs
    from port.patch.plot_copy_number_profile import plot_ascn_legend

    renderer = figure.canvas.get_renderer()
    dpi = figure.dpi
    width, height = figure.get_size_inches()
    gap = LABEL_GAP / 72.0
    tracks = list(top.axes)
    foot_ax = tracks[-1]
    letters = [figure.text(0.0, 0.0, f"({k})", fontsize=LABEL_SIZE) for k in "ab"]
    column, left = _left_column(figure, tracks, profile_ax)
    right = max(ax.get_window_extent(renderer).x1 for ax in tracks) / dpi

    for _ in range(3):
        for ax in [*tracks, profile_ax, legend_ax]:
            place_in_inches(ax, left, right)

        figure.canvas.draw()
        name_contigs(foot_ax, *contigs, size=FONT_SIZE)
        names = [t for t in foot_ax.texts if t.get_gid() == "contig"]
        overrun = max(t.get_window_extent(renderer).x1 for t in names) / dpi - width

        if overrun <= 0.0:
            break

        right -= overrun + gap

    name_contigs(foot_ax, *contigs, size=FONT_SIZE)

    # NB a clone with many states wraps its key onto rows.
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

    # NB a y label anchors on its axis side, so offset it by its own width.
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
    )

    # NB key a `gap` under the head; white between clones closed by `GAP_CLOSED`;
    #    tracks a letter's row and a `gap` under the profile.
    figure.canvas.draw()
    raised = max(t.get_window_extent(renderer).height for t in letters) / dpi + gap
    key = [*legend_ax.patches, *legend_ax.texts]
    lift = height - gap - max(_inches(figure, key, "y1"))

    for ax in (profile_ax, legend_ax):
        _up(ax, lift)

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

    for row, shift in zip(rows, shifts, strict=True):
        for ax in row:
            _up(ax, shift)

    figure.canvas.draw()
    head = tracks[0]
    stats = [t for t in head.texts if t.get_visible()]
    first = max(
        _inches(
            figure, [*stats, *([head.get_legend()] if head.get_legend() else [])], "y1"
        )
    )
    under = min(_inches(figure, [profile_ax, *profile_ax.get_yticklabels()], "y0"))
    drop = under - raised - gap - first

    for ax in tracks:
        _up(ax, drop)

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
            (letters[0], column, (title.y0 + title.y1) / 2 / dpi, "center"),
            (letters[1], column, max(_inches(figure, stats, "y1")) + gap / 2, "bottom"),
        ],
        head=key,
    )


def _page(width: float, height: float, rect: tuple[float, float, float, float]) -> Any:
    """A white page `width` by `height` inches at 300 dpi, constrained layout in `rect`."""
    import matplotlib.pyplot as plt
    from matplotlib.layout_engine import ConstrainedLayoutEngine

    return plt.figure(
        figsize=(width, height),
        dpi=300,
        facecolor="white",
        layout=ConstrainedLayoutEngine(h_pad=0.01, hspace=0.0, rect=rect),
    )


def _genomic_page(
    genomic: Call, profile: Call, width: float, scale: float, metric: bool = False
) -> Any:
    """The genomic figure, tracks and profile rows at `scale`; on `metric`, CNAs widened (T- #683)."""
    # NB `port`'s plotters directly: `FIGURE_SWAPS` is restored after the run.
    from matplotlib.ticker import NullLocator

    from port.extensions.genomic_axis import (
        GenomicAxis,
        Ticks,
        altered_bins,
        disclose,
        resolve,
    )
    from port.patch.plot_copy_number_profile import plot_copy_number_profile
    from port.patch.plot_genomic import plot_clones_genomic

    n_clones = len(np.unique(genomic.kwargs["res_combine"]["new_assignment"]))
    base = (0.27 * n_clones + 0.45) / (1.0 + LEGEND_ROW)
    key, profile_rows = LEGEND_ROW * base, PROFILE_ROWS * base
    heights = (
        scale * (1.05 * n_clones + base - profile_rows) + TOP_LINE,
        scale * profile_rows + key,
    )
    total = sum(heights) + FOOT
    figure = _page(width, total, (0.0, FOOT / total, 1.0, 1.0 - FOOT / total))
    rows: Any = figure.subfigures(2, 1, height_ratios=heights[::-1], hspace=0.02)
    middle, top = rows[0], rows[1]

    # NB one axis for (a) and (b) (T- #683).
    df_cnv = genomic.kwargs["df_cnv"]
    genome = resolve(Ticks(), df_cnv, len(df_cnv))

    # NB no Mb numbers, as on the truth page (PR- #715).
    if genome is not None:
        genome = GenomicAxis.of_table(
            df_cnv, altered_bins(df_cnv) if metric else None, labels=False
        )

    legend_ax, profile_ax = middle.subplots(
        2,
        1,
        height_ratios=(key, scale * profile_rows),
    )
    # NB rows in the fitted clones' index order, as (c)'s tracks are (PR- #715).
    frame = profile.args[0]
    ids = [c[len("clone") : -len(" A")] for c in frame.columns if c.endswith(" A")]
    plot_copy_number_profile(frame, ax=profile_ax, axis=genome, rows=clone_order(ids))
    profile_ax.set_yticklabels(
        [clone_symbol(t.get_text()) for t in profile_ax.get_yticklabels()]
    )

    for text in profile_ax.get_yticklabels():
        text.set_rotation(0)
    # NB `cnaster`'s legend sits at fixed page coordinates; redrawn into a managed axis.
    middle.axes[-1].remove()
    legend_ax.axis("off")
    # NB contigs are named under the last track only (PR- #715).
    contigs = (
        [float(x) for x in profile_ax.get_xticks()],
        [t.get_text() for t in profile_ax.get_xticklabels()],
    )
    profile_ax.set_xticks([])
    profile_ax.xaxis.set_minor_locator(NullLocator())

    plot_clones_genomic(
        *genomic.args,
        **{
            **genomic.kwargs,
            "figure": top,
            "pointsize": 0.4,
            "linewidth": 0.3,
            "chrtext_shift": -0.9,
            "axis": genome,
        },
    )
    fit_track_furniture(top)
    _colour_by_state(top, genomic)

    for ax in top.axes[:-1]:
        ax.xaxis.set_minor_locator(NullLocator())

    set_font_size(top, TRACK_FONT_SIZE)
    set_font_size(middle, FONT_SIZE)

    figure.canvas.draw()
    figure.set_layout_engine("none")
    _place_genomic(figure, top, profile_ax, legend_ax, contigs)
    disclose(figure, genome)
    return figure


@contextlib.contextmanager
def page_style() -> Iterator[None]:
    """Matplotlib's defaults and the stated face for the block, whatever `cnaster` has set (#342).

    Hold it around both the build and the write.
    """
    import matplotlib as mpl

    from port.extensions.figure_style import figure_rc

    with mpl.rc_context():
        mpl.style.use("default")
        mpl.rcParams.update(figure_rc())
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
    recorded: Recorded,
    width: float | None = None,
    height: float = TEXT_HEIGHT - CAPTION_ROOM,
    *,
    metric: bool = False,
    labels: str = "integer",
) -> Any:
    """(a) `copy_number_profile` over (b) `clones_genomic`, `width` by `height` inches; no caption.

    `labels`: "integer" (merged clones, `integer_recorded`, #745) or
    "continuous". On `metric`, altered bins drawn wider (T- #683). Tracks and
    profile rows are scaled by secant until the page is `height` to 0.005 in.
    """
    from port.extensions.figure_style import PAPER_WIDTH

    if recorded.genomic is None or recorded.profile is None:
        msg = f"the run made {recorded.calls}; the genomic figure needs both"
        raise ValueError(msg)

    width = PAPER_WIDTH if width is None else width
    if labels == "integer":
        recorded = integer_recorded(recorded)
    elif labels != "continuous":
        msg = f'labels is "integer" or "continuous", not {labels!r}'
        raise ValueError(msg)
    genomic, profile = recorded.genomic, recorded.profile
    if genomic is None or profile is None:  # invariant
        msg = "expected the genomic and profile calls"
        raise AssertionError(msg)
    scales = [1.0]
    heights = [_genomic_page(genomic, profile, width, 1.0, metric).get_size_inches()[1]]
    scales.append(height / heights[0])

    for _ in range(4):
        figure = _genomic_page(genomic, profile, width, scales[-1], metric)
        heights.append(figure.get_size_inches()[1])

        if abs(heights[-1] - height) < 0.005:
            return figure

        slope = (heights[-1] - heights[-2]) / (scales[-1] - scales[-2])
        scales.append(scales[-1] + (height - heights[-1]) / slope)

    msg = f"the genomic figure is {heights[-1]:.3f} in at best, not {height:.3f}"
    raise ValueError(msg)


def _place_spatial(
    figure: Any, slide_ax: Any, spatial_ax: Any, most: float | None = None
) -> float:
    """(a) the slide, (b) the clones and (b)'s key, left to right, in square footprints (PR- #715).

    Each side as large as fits across, at most `most` inches; letters over
    each panel and the page cut to its text. Returns the side, in inches.
    """
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
    # NB square footprints with limits from the spots' left and bottom edges,
    #    so both axes share one scale (PR- #715).
    side = (clones_right - left - SPATIAL_GAP - ticks(spatial_ax)) / 2
    side = side if most is None else min(side, most)
    wide = side
    (x0, x1), (y0, y1) = spatial_ax.get_xlim(), spatial_ax.get_ylim()
    span = max(x1 - x0, y1 - y0)
    for ax in (slide_ax, spatial_ax):
        ax.set_xlim(x0, x0 + span)
        ax.set_ylim(y0, y0 + span)
        ax.set_aspect("equal", adjustable="box")
        _frame(ax, (x0, x1), (y0, y1))
    # NB squares smaller than fit across leave the row centred on the page.
    left += (clones_right - left - SPATIAL_GAP - ticks(spatial_ax) - 2 * side) / 2
    bottom = height - side - 1.0
    place_in_inches(slide_ax, left, left + wide, bottom, side)
    start = left + wide + SPATIAL_GAP + ticks(spatial_ax)
    place_in_inches(spatial_ax, start, start + wide, bottom, side)
    clones_right = start + wide
    right = clones_right + 2 * gap + key.get_window_extent(renderer).width / dpi
    anchor = (right - start) / wide
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

    # NB a legend's box is not exactly its anchor: correct by the shortfall.
    figure.canvas.draw()
    short = right - key.get_window_extent(renderer).x1 / dpi
    key.set_bbox_to_anchor((anchor + short / wide, 0.0), transform=spatial_ax.transAxes)
    return float(side)


def integer_recorded(recorded: Recorded) -> Recorded:
    """`recorded` with its clones the run's integer clones, numbered anew (#344, #518, #745).

    Clones merged as `write_outputs` merges them are one, with summed counts
    and the smallest id's profile and path; survivors are numbered 0, 1, ...
    in index order. Idempotent.
    """
    import pandas as pd

    from port.extensions.outputs import installed_keys, integer_clones, merge_agreement
    from port.patch._clone_paths import clone_path

    if recorded.profile is None:
        msg = "integer clones need the run's copy_number_profile call"
        raise ValueError(msg)

    groups = integer_clones(recorded.profile.args[0], merge_agreement(installed_keys()))
    kept = clone_order(set(groups.values()))
    number = {old: str(kept.index(group)) for old, group in groups.items()}

    def frame(table: Any) -> Any:
        keep = [
            c
            for c in table.columns
            if not c.startswith("clone") or c.split()[0][len("clone") :] in kept
        ]

        def name(column: str) -> str:
            head, _, tail = column.partition(" ")
            if not head.startswith("clone"):
                return column
            return f"clone{number[head[len('clone') :]]} {tail}"

        return table[keep].rename(columns=name)

    def label(value: Any) -> Any:
        if isinstance(value, str) and value.split()[-1] in number:
            return f"{value.rsplit(' ', 1)[0]} {number[value.split()[-1]]}"
        return value

    merged = Recorded(calls=dict(recorded.calls))
    profile = recorded.profile
    merged.profile = Call((frame(profile.args[0]), *profile.args[1:]), profile.kwargs)

    if recorded.spatial is not None:
        coords, assignment, *rest = recorded.spatial.args
        merged.spatial = Call(
            (coords, pd.Series(assignment).map(label), *rest), recorded.spatial.kwargs
        )

    if recorded.genomic is not None:
        kwargs = dict(recorded.genomic.kwargs)
        fit = dict(kwargs["res_combine"])
        n_obs = int(np.asarray(recorded.genomic.args[1]).shape[0])
        fitted = [str(c) for c in np.sort(np.unique(fit["new_assignment"]))]
        # NB either layout: a column per clone, or concatenated.
        pred = np.asarray(fit["pred_cnv"])
        index = [fitted.index(c) for c in kept]
        fit["pred_cnv"] = (
            pred[:, index]
            if pred.ndim == 2 and pred.shape[1] > 1
            else np.concatenate([clone_path(pred, i, n_obs) for i in index])
        )
        fit["new_assignment"] = np.array(
            [int(number[str(c)]) for c in np.asarray(fit["new_assignment"])]
        )
        shift = fit.get("new_log_mu_shift")
        if shift is not None and np.size(shift) == len(fitted):
            shift = np.asarray(shift).reshape(-1)
            fit["new_log_mu_shift"] = shift[index]
        kwargs["res_combine"] = fit
        if kwargs.get("df_cnv") is not None:
            kwargs["df_cnv"] = frame(kwargs["df_cnv"])
        merged.genomic = Call(recorded.genomic.args, kwargs)

    return merged


def he_segmentation_figure(
    coords: np.ndarray, he_frame: Any, he_labels: np.ndarray, width: float | None = None
) -> Any:
    """(a) the slide and (b) each spot's H&E class in `HE_PALETTE`, keyed darkest first (T- #771, T- #791)."""
    import pandas as pd

    from port.extensions.figure_style import fit_to_content
    from port.extensions.spatial_page import format_panel, panel_row, spatial_key
    from port.patch.plotting.spatial import draw_clones_spatial, spot_colours

    coords = np.asarray(coords)
    classes = pd.Series([f"H&E {int(c)}" for c in he_labels])
    figure, (slide_ax, class_ax) = panel_row(2, 1.0, width)
    # NB no upstream key: it reads an integer clone id from each label.
    draw_clones_spatial(
        class_ax, coords, classes, None, palette=HE_PALETTE, legend=False
    )
    _, names, colours = spot_colours(classes, palette=HE_PALETTE)
    image, extent = slide_image(he_frame)
    slide_ax.imshow(image, extent=extent, interpolation="none")
    # NB the section (b) shows, so a boundary sits at the same place in both.
    limits = class_ax.get_xlim(), class_ax.get_ylim()
    format_panel(slide_ax, "H&E", *limits, fontsize=FONT_SIZE)
    format_panel(class_ax, "H&E class", *limits, fontsize=FONT_SIZE)
    spatial_key(class_ax, [str(n) for n in names], list(colours), marker="s",
                fontsize=FONT_SIZE)  # fmt: skip
    fit_to_content(figure)
    return figure


def _draw_spatial(
    figure: Any,
    recorded: Recorded,
    he_frame: Any,
    labels: str = "integer",
) -> tuple[Any, Any]:
    """The slide and the clones on `figure`, drawn but not placed; `labels` "integer" (#344, #745) or "continuous"."""
    from cnaster.utils import cast_clone_label

    from port.patch.plotting.spatial import draw_clones_spatial, spot_colours

    if recorded.spatial is None:  # invariant
        msg = "expected recorded.spatial is not None"
        raise AssertionError(msg)
    coords, assignment = recorded.spatial.args[:2]

    if labels == "integer":
        coords, assignment = integer_recorded(recorded).spatial.args[:2]  # type: ignore[union-attr]
    elif labels != "continuous":
        msg = f'labels is "integer" or "continuous", not {labels!r}'
        raise ValueError(msg)

    slide_ax = figure.add_axes((0.0, 0.0, 0.4, 0.4))
    spatial_ax = figure.add_axes((0.5, 0.0, 0.4, 0.4))
    draw_clones_spatial(
        spatial_ax,
        np.asarray(coords),
        assignment,
        recorded.spatial.kwargs.get("single_tumor_prop"),
    )
    # NB the key is redrawn as a column beside the tiles.
    upstream_key = spatial_ax.get_legend()
    if upstream_key is not None:
        upstream_key.remove()
    _, clone_ids, colours = spot_colours(assignment)
    keyed = dict(zip((str(c) for c in clone_ids), colours, strict=True))
    order = clone_order(keyed)
    names = [clone_symbol(cast_clone_label(f"clone {c.split()[-1]}")) for c in order]
    _clone_key(spatial_ax, names, [keyed[c] for c in order])

    image, extent = slide_image(he_frame)
    slide_ax.imshow(image, extent=extent, interpolation="none")
    # NB the section (b) shows, so a boundary sits at the same place in both.
    slide_ax.set_xlim(spatial_ax.get_xlim())
    slide_ax.set_ylim(spatial_ax.get_ylim())
    slide_ax.set_aspect("equal")

    for ax in (slide_ax, spatial_ax):
        _extents(ax, np.asarray(coords))
    # NB (b) shares (a)'s rows, so only (a) labels them.
    spatial_ax.tick_params(axis="y", labelleft=False)

    return slide_ax, spatial_ax


@_styled
def spatial_figure(
    recorded: Recorded,
    he_frame: Any,
    width: float | None = None,
    labels: str = "integer",
    *,
    he_labels: np.ndarray | None = None,
) -> Any:
    """(a) the H&E slide and (b) `clones_spatial`, on a "third" page `width` across (T- #740); no caption.

    `labels`: "integer" (#344) or "continuous". With `he_labels`, returns
    `he_segmentation_figure` instead (T- #771).
    """
    import matplotlib.pyplot as plt

    from port.extensions.figure_style import PAPER_WIDTH, page_size

    if recorded.spatial is None:
        msg = f"the run made {recorded.calls}; the spatial figure needs its clones"
        raise ValueError(msg)

    if he_labels is not None:
        return he_segmentation_figure(
            recorded.spatial.args[0], he_frame, he_labels, width
        )

    width = PAPER_WIDTH if width is None else width
    tallest = page_size("third")[1]
    most = None

    # NB redrawn once with smaller squares if the page overruns a "third" (T- #740).
    for attempt in range(2):
        figure = plt.figure(figsize=(width, width), dpi=300, facecolor="white")
        slide_ax, spatial_ax = _draw_spatial(figure, recorded, he_frame, labels)

        set_font_size(figure, FONT_SIZE)
        side = _place_spatial(figure, slide_ax, spatial_ax, most)
        over = float(figure.get_size_inches()[1]) - tallest
        if over <= 0.0 or attempt == 1:
            break
        plt.close(figure)
        most = side - over

    return figure


@_styled
def combined_figure(
    recorded: Recorded,
    he_frame: Any,
    width: float | None = None,
    height: float = TEXT_HEIGHT - CAPTION_ROOM,
    labels: str = "integer",
    *,
    metric: bool = False,
) -> Any:
    """The spatial figure, (a), over the genomic one, (b) and (c), on one page `height` tall (T- #733, PR- #715).

    `labels` as `spatial_figure`; `metric` as `genomic_figure`.
    """
    import matplotlib.pyplot as plt

    spatial = spatial_figure(recorded, he_frame, width, labels)
    above = float(spatial.get_size_inches()[1])
    figure = genomic_figure(
        recorded, width, height - above, metric=metric, labels=labels
    )
    dpi = figure.dpi
    wide, tall = figure.get_size_inches()

    # NB grow the genomic page at its head; axes keep their place from the foot.
    kept = [
        (ax, ax.get_window_extent(figure.canvas.get_renderer()).frozen())
        for ax in figure.get_axes()
    ]
    letters = [(text, text.get_position()) for text in figure.texts]
    figure.set_size_inches(wide, tall + above)
    figure.canvas.draw()

    for ax, box in kept:
        place_in_inches(ax, box.x0 / dpi, box.x1 / dpi, box.y0 / dpi, box.height / dpi)
    for (text, (x, y)), letter in zip(letters, "bc", strict=True):
        text.set_position((x, y * tall / (tall + above)))
        text.set_text(f"({letter})")

    # NB redraw the maps above, where the spatial page put them, `tall` higher.
    slide_ax, spatial_ax = _draw_spatial(figure, recorded, he_frame, labels)
    for ax in (slide_ax, spatial_ax):
        set_font_size(ax, FONT_SIZE)
    figure.canvas.draw()
    source = spatial.canvas.get_renderer()
    old_axes = spatial.get_axes()[:2]

    # NB one left edge for the page (#745).
    left = min(box.x0 for ax, box in kept if ax.axison) / dpi
    shift = left - old_axes[0].get_window_extent(source).x0 / dpi

    # NB copy the square limits, or each axis shrinks to the spots' extent.
    for new, old in zip((slide_ax, spatial_ax), old_axes, strict=True):
        new.set_xlim(old.get_xlim())
        new.set_ylim(old.get_ylim())
        new.set_aspect("equal", adjustable="box")
        _frame(new, old.spines["bottom"].get_bounds(), old.spines["left"].get_bounds())
        box = old.get_window_extent(source)
        place_in_inches(
            new,
            box.x0 / dpi + shift,
            box.x1 / dpi + shift,
            box.y0 / dpi + tall,
            box.height / dpi,
        )

    old_key, new_key = old_axes[1].get_legend(), spatial_ax.get_legend()
    anchor = old_key.get_bbox_to_anchor()
    axis = old_axes[1].get_window_extent(source)
    new_key.set_bbox_to_anchor(
        ((anchor.x0 - axis.x0) / axis.width, (anchor.y0 - axis.y0) / axis.height),
        transform=spatial_ax.transAxes,
    )

    # NB slide and clones are one panel, (a), lettered in (b)'s column (#745).
    column = letters[0][0].get_position()[0]
    for text in spatial.texts[:1]:
        _, y = text.get_position()
        figure.text(
            column,
            (y * above + tall) / (tall + above),
            text.get_text(),
            fontsize=LABEL_SIZE,
            verticalalignment=text.get_verticalalignment(),
        )

    plt.close(spatial)
    figure.canvas.draw()
    return figure


@_styled
def plot_clones_genomic_he(
    recorded: Recorded,
    he_labels: np.ndarray,
    width: float | None = None,
    height: float | None = None,
) -> Any:
    """RDR and BAF along the genome, one pair of tracks per H&E class, spots pseudobulked per class (T- #771).

    No fitted levels. `width` by `height` inches, a "full" page by default.
    """
    import matplotlib.pyplot as plt

    from port.extensions.figure_style import page_size
    from port.extensions.genomic_axis import name_contigs
    from port.patch.plot_genomic import plot_clones_genomic

    if recorded.genomic is None:
        msg = "the H&E tracks need the run's clones_genomic call"
        raise ValueError(msg)

    full_width, full_height = page_size("full")
    width = full_width if width is None else width
    height = full_height if height is None else height

    labels = np.asarray(he_labels)
    classes = np.unique(labels)
    groups = [np.flatnonzero(labels == c) for c in classes]

    figure = plt.figure(figsize=(width, height), dpi=300, facecolor="white")
    arguments = recorded.genomic.args[:4]
    # NB display options only: fitted levels are per clone, not per class.
    keywords: dict[str, Any] = {
        key: value
        for key, value in recorded.genomic.kwargs.items()
        if key
        in {"rdr_ylim", "plot_baf_errors", "plot_rdr_errors", "known_nb_baseline"}
    }
    keywords |= {
        "clone_index": groups,
        "figure": figure,
        "pointsize": 0.4,
        "linewidth": 0.3,
    }
    plot_clones_genomic(*arguments, **keywords)

    fit_track_furniture(figure)
    tracks = [ax for ax in figure.get_axes() if ax.get_visible()]
    per_class = len(tracks) // len(groups)
    for k, (label, group) in enumerate(zip(classes, groups, strict=True)):
        stats = [t for t in tracks[per_class * k].texts if t.get_rotation() == 0.0]
        if stats:
            stats[0].set_text(f"H&E {label} ({group.size} spots)")

    set_font_size(figure, FONT_SIZE)
    lengths = np.asarray(arguments[0])
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(float)
    # NB `cnaster`'s names, hidden by `fit_track_furniture`.
    names = [
        t.get_text()
        for t in sorted(tracks[-1].texts, key=lambda t: float(t.get_position()[0]))
        if t.get_text().startswith("chr")
    ]
    name_contigs(tracks[-1], starts.tolist(), names, size=FONT_SIZE)

    set_font_size(figure, FONT_SIZE)
    return figure
