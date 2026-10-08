r"""What a realization planted, on one page at `combined.pdf`'s size and type.

    python -m port.sim.truth_figure sim/generated/<name>/r<k> [OUT.pdf]

Top to bottom, at the paper's text width and its height less `CAPTION_ROOM`
(T- #733, T- #740), at `FONT_SIZE` throughout, as
`port.extensions.combined_figure` sets an estimate:

- **(a)** the clones' tree, each event at its time (`analysis.draw_tree`),
  above `analysis.MANY_EVENTS` events without them, its edges and whole
  barcodes alone (PR- #701);
- **(b)** each clone's planted `(A, B)`, drawn by `port`'s profile plotter
  under its mirror and copy-number key, the rows `combined.pdf` draws;
- **(c)** RDR and BAF along the genome per true clone
  (`analysis.genomic_truth`), drawn by `plot_clones_genomic`, each clone
  named with its barcode from (a). A barcode is shown whole in (a) and as
  `analysis.shown` cuts it in (c).

(b) and (c) share one `port.extensions.genomic_axis.GenomicAxis`, its Mb
unlabelled (T- #683, PR- #701). The page's last track alone carries the
10 Mb marks, outward, and every contig's name, staggered where adjacent
contigs are short (`genomic_axis.name_contigs`); every track and (b) keep
the chromosome boundaries.

With `spatial`, (c) is the true clone of each spot per slice in place of
the tracks (T- #791), and the phase track moves under (b), where it names
the contigs and carries the 10 Mb marks (T- #794).

(b) and (c) share one left and one right edge, so a chromosome boundary is
at one place on the page in both. Clones
are named as the paper names them, $m_N$ for the normal. The phase and the
count laws stay in their own figures, `phase.png` and `coverage.png`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from port.extensions.figure_style import CAPTION_ROOM, TEXT_HEIGHT
from port.patch.plot_copy_number_profile import KEY_GROWTH
from port.sim.analysis import Realization, clone_name, read

__all__ = ["truth_combined_figure", "write_truth_combined"]

KEY = (0.05, 0.2 * KEY_GROWTH)
"""Inches: (b)'s gap between its key and its rows, and the key's height,
0.2 before the mirror swatches stacked (PR- #715)."""

ROWS = (0.04, 0.42)
"""Inches: (b)'s foot under its rows, unlabelled (PR- #701), and the rows' height."""

KEY_TOP = 0.075
"""Inches over (b)'s key: with (a)'s foot, the white between (a) and (b) (PR- #701)."""

HEIGHTS = {
    "tree": 0.9,
    "profile": sum(ROWS) + sum(KEY) + KEY_TOP,
    "genomic": TEXT_HEIGHT - CAPTION_ROOM - 0.9 - (sum(ROWS) + sum(KEY) + KEY_TOP),
}
"""Inches per panel, summing to `TEXT_HEIGHT` less `CAPTION_ROOM` (T- #733);
(a) 0.9, 10% under its 1.0 before PR- #701, (b) fixed, (c) the rest."""

CONTIG_FOOT = 0.3
"""Inches under (b)'s rows for the contig names where (b) is the page's last
genome panel: the spatial variant (T- #791)."""


PHASE_TRACK = 0.65
"""Inches: the spatial variant's phase track under (b), as tall as the RDR
and BAF variant's tracks draw on `dev_tree_1s_easy` r0 (`7ba9b01f`), 0.654 in (T- #794)."""

PHASE_GAP = 0.06
"""Inches between (b)'s rows and the spatial variant's phase track."""

LETTER_ROOM = 0.15
"""Inches over the spatial variant's (c) titles for the panel letter."""


def heights(spatial: float | None = None) -> dict[str, float]:
    """`HEIGHTS`, or with `spatial`, (c)'s row height (`spatial_page.row_height`),
    (b) taller by its phase track and `CONTIG_FOOT` (T- #794) and (c) that row
    under `LETTER_ROOM`, no taller than the rest of the page: the page as tall
    as it draws (T- #791)."""
    if spatial is None:
        return dict(HEIGHTS)
    under = PHASE_GAP + PHASE_TRACK + CONTIG_FOOT
    room = HEIGHTS["genomic"] - under
    return {
        "tree": HEIGHTS["tree"],
        "profile": HEIGHTS["profile"] + under,
        "spatial": min(room, LETTER_ROOM + spatial),
    }


LEFT = 0.42
"""Inches from the page's left edge to the genome panels' axes: room for the RDR and BAF labels."""

RIGHT = 0.06
"""Inches of the page right of the genome panels, before the last chromosome's name is fitted."""


def _symbol(r: Realization) -> Any:
    return lambda clone: clone_name(clone, r.clones)


def truth_combined_figure(
    r: Realization,
    width: float | None = None,
    *,
    metric: bool = False,
    spatial: bool = False,
) -> Any:
    """The three panels on one page, `width` wide (the paper's text width by default) and
    `TEXT_HEIGHT` less `CAPTION_ROOM` tall (T- #733); on `metric`, the planted CNAs drawn wider (T- #683).

    With `spatial`, (c) is each spot's true clone per slice in the spatial
    pages' format (`analysis.draw_spatial`, `port.extensions.spatial_page`)
    in place of RDR and BAF per clone (T- #791); (b) carries the phase track
    under its rows, which names the contigs and carries the genome's marks
    (T- #794).
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    from port.extensions.combined_figure import (
        FONT_SIZE,
        LABEL_GAP,
        LABEL_SIZE,
        LEGEND_BOX,
        fit_track_furniture,
        page_style,
        place_in_inches,
        set_font_size,
    )
    from port.extensions.figure_style import PAPER_WIDTH, TRACK_FONT_SIZE
    from port.extensions.genomic_axis import disclose, name_contigs
    from port.patch.plot_copy_number_profile import (
        plot_ascn_legend,
        plot_copy_number_profile,
    )
    from port.patch.plot_genomic import plot_clones_genomic
    from port.sim.analysis import (
        binned_axis,
        binned_profile,
        draw_tree,
        genomic_truth,
        shown,
        tree,
    )

    width = PAPER_WIDTH if width is None else width
    symbol = _symbol(r)
    genome = binned_axis(r, metric=metric, labels=False)
    tall_of = heights(_spatial_row(r, width - LEFT - RIGHT) if spatial else None)
    foot = ROWS[0] + (CONTIG_FOOT + PHASE_TRACK + PHASE_GAP if spatial else 0.0)

    with page_style():
        figure: Any = plt.figure(
            figsize=(width, sum(tall_of.values())), dpi=300, facecolor="white"
        )
        panels: Any = figure.subfigures(
            len(tall_of), 1, height_ratios=list(tall_of.values()), hspace=0.0
        )
        tree_fig, profile_fig, genomic_fig = panels

        # (a)
        tree_ax = tree_fig.add_axes((0.02, 0.02, 0.96, 0.88))
        draw_tree(tree_ax, r, event_size=FONT_SIZE, node_size=FONT_SIZE,
                  dot=18.0, name=symbol, ancestors=False, edges=True)  # fmt: skip

        # (b): the key, then the rows, as `combined.pdf` draws its profile.
        tall = tall_of["profile"]
        legend_ax = profile_fig.add_axes(
            (0.0, (foot + ROWS[1] + KEY[0]) / tall, 1.0, KEY[1] / tall)
        )
        profile_ax = profile_fig.add_axes((0.0, foot / tall, 1.0, ROWS[1] / tall))
        # NB rows in (a)'s order, `r.clones`', as (c)'s tracks are (PR- #701).
        plot_copy_number_profile(binned_profile(r), ax=profile_ax, axis=genome,
                                 rows=[str(k) for k in range(len(r.clones))])  # fmt: skip
        # NB the plotter's own key, at fixed page coordinates, is redrawn into
        #    `legend_ax`, which is placed with the rows.
        profile_fig.axes[-1].remove()
        profile_ax.set_yticklabels(
            [symbol_of(t.get_text()) for t in profile_ax.get_yticklabels()]
        )
        for text in profile_ax.get_yticklabels():
            text.set_rotation(0)
        # NB the chromosome names move to the page's last track; (b) keeps
        #    its boundaries, and neither names nor marks (PR- #701).
        starts = list(profile_ax.get_xticks())
        contigs = [t.get_text() for t in profile_ax.get_xticklabels()]
        profile_ax.set_xticks([])
        profile_ax.xaxis.set_minor_locator(NullLocator())

        # (c)
        if spatial:
            # NB the phase track under (b)'s rows, as the RDR and BAF variant
            #    draws it in (c) (T- #794); the slices drawn once (b)'s edges
            #    are fitted, across them (below)
            phase_ax = profile_fig.add_axes(
                (0.0, CONTIG_FOOT / tall, 1.0, PHASE_TRACK / tall)
            )
            _phase_track(phase_ax, r, genome)
            _end_ticks(phase_ax)
            tracks: list[list[Any]] = [[phase_ax]]
            bottom_ax = phase_ax
            bottom_ax.set_xticks([])
        else:
            g = genomic_truth(r)
            plot_clones_genomic(g.lengths, g.counts, g.expected, g.trials,
                                clone_index=g.groups, figure=genomic_fig,
                                pointsize=0.4, linewidth=0.3, chrtext_shift=-0.9,
                                axis=genome)  # fmt: skip
            fit_track_furniture(genomic_fig)
            # NB the normal clone's pair gives way to the phase track, in its place,
            #    one track tall as an RDR track is (#745)
            tracks = [
                list(genomic_fig.axes[k : k + 2])
                for k in range(0, len(genomic_fig.axes), 2)
            ]
            normal = next(i for i, (rdr, _) in enumerate(tracks)
                          if any(t.get_text().startswith(symbol("normal")) for t in rdr.texts))  # fmt: skip
            # NB drawn into the normal clone's RDR axes, so (c)'s axes stay in page order
            phase_ax, baf_ax = tracks[normal]
            phase_ax.cla()
            _phase_track(phase_ax, r, genome)
            baf_ax.remove()
            tracks[normal] = [phase_ax]
            # NB each clone's name followed by its barcode, as (a) sets it.
            barcode = tree(r).barcode
            named = {symbol(clone): clone for clone in r.clones}
            for ax in genomic_fig.axes:
                for text in ax.texts:
                    clone = named.get(text.get_text())
                    if clone is not None and text.get_visible():
                        text.set_text(f"{symbol(clone)} ({shown(barcode[clone])})")
            for ax in genomic_fig.axes:
                _end_ticks(ax)
                # NB `cnaster`'s names, 10 pt at 45 degrees, give way to the
                #    last track's ticks; the marks stay on the last track alone.
                for text in ax.texts:
                    if text.get_text().startswith("chr"):
                        text.set_visible(False)
                if ax is not tracks[-1][-1]:
                    ax.xaxis.set_minor_locator(NullLocator())
            bottom_ax = tracks[-1][-1]
            bottom_ax.set_xticks([])

        # NB the tracks at `TRACK_FONT_SIZE`, as `combined_figure` sets its own (#743)
        for panel in (tree_fig, profile_fig):
            set_font_size(panel, FONT_SIZE)
        for group in tracks if spatial else ():
            set_font_size(group[0], TRACK_FONT_SIZE)
        set_font_size(genomic_fig, FONT_SIZE if spatial else TRACK_FONT_SIZE)
        figure.canvas.draw()

        # NB one left and one right edge for the key, the rows and each
        #    track; the right pulled in until no contig name runs off the
        #    page, as `combined_figure` fits its own. Every contig is named,
        #    staggered where adjacent ones are short (`name_contigs`).
        renderer = figure.canvas.get_renderer()
        right = width - RIGHT
        for _ in range(3):
            for ax in [
                legend_ax,
                profile_ax,
                *(ax for group in tracks for ax in group),
            ]:
                place_in_inches(ax, LEFT, right)
            figure.canvas.draw()
            name_contigs(bottom_ax, starts, contigs, size=FONT_SIZE)
            names = [t for t in bottom_ax.texts if t.get_gid() == "contig"]
            overrun = (
                max(t.get_window_extent(renderer).x1 for t in names) / figure.dpi
                - width
            )
            if overrun <= 0.0:
                break
            right -= overrun + LABEL_GAP / 72.0

        names_foot = name_contigs(bottom_ax, starts, contigs, size=FONT_SIZE)
        if not spatial:
            _stack_tracks(genomic_fig, names_foot, tracks)
        if spatial:
            # NB (c)'s row across (b)'s genome axis: one left and one right edge
            _spatial_panel(genomic_fig, r, LEFT, right, tall_of["spatial"])
            set_font_size(genomic_fig, FONT_SIZE)
        place_in_inches(tree_ax, LEFT, right)
        figure.canvas.draw()
        _fit_tree(tree_ax)
        plot_ascn_legend(legend_ax, box_w=LEGEND_BOX, box_h=0.8, tick_len=0.1,
                         label_fontsize=FONT_SIZE, span=right - LEFT)  # fmt: skip

        for panel, letter in zip(panels, "abc", strict=True):
            panel.text(0.0, 1.0, f"({letter})", fontsize=LABEL_SIZE, ha="left",
                       va="top")  # fmt: skip
        figure.canvas.draw()
    disclose(figure, genome)
    return figure


def _phase_track(ax: Any, r: Realization, genome: Any) -> None:
    """The phase switches per Mb on `ax`, on the page's genome axis, its top 1
    (`analysis.draw_phase`), in the tracks' furniture (`fit_track_furniture`):
    ticks 2 pt, the label a point off them (#745)."""
    from port.sim.analysis import draw_phase

    draw_phase(ax, r, genome, rate_size=None, ylim=1.0)
    ax.tick_params(length=2, pad=1)
    ax.yaxis.labelpad = 1.0
    ax.set_ylabel("Switches / Mb")
    # NB `cnaster`'s names give way to the page's own (`name_contigs`)
    for text in ax.texts:
        if text.get_text().startswith("chr"):
            text.set_visible(False)


def _end_ticks(ax: Any) -> None:
    """A track's y ticks at its two ends alone, each label inside the axis,
    its label upright on its left."""
    ticks = ax.get_yticks()
    if ticks.size > 2:
        ends = [ticks[0], ticks[-1]]
        ax.set_yticks(ends, [f"{tick:.1f}" for tick in ends])
        bottom, top = ax.get_yticklabels()
        bottom.set_verticalalignment("bottom")
        top.set_verticalalignment("top")
    ax.set_ylabel(ax.get_ylabel(), rotation=90, ha="center", va="bottom")


def _spatial_row(r: Realization, width: float) -> float:
    """Inches (c)'s row of slices takes across `width`, as `_spatial_panel` draws it."""
    from port.extensions.spatial_page import row_height
    from port.sim.analysis import slice_frame

    frame = slice_frame(r)
    return float(row_height(len(frame.slices), frame.aspect, width))


def _spatial_panel(
    panel: Any, r: Realization, left: float, right: float, height: float
) -> None:
    """(c) of the spatial variant: each slice's true clones in the spatial
    pages' format, keyed by the paper's names, the key under the first slice."""
    from port.extensions.spatial_page import panel_row, spatial_key
    from port.sim.analysis import clone_colour, draw_spatial, slice_frame

    frame = slice_frame(r)
    _, axes = panel_row(
        len(frame.slices),
        frame.aspect,
        right - left,
        figure=panel,
        height=height,
        top=LETTER_ROOM,
        left=left,
    )
    present = draw_spatial(axes, r, size=4.0)
    spatial_key(axes[0], [clone_name(c, r.clones) for c in present],
                [clone_colour(c, r.clones) for c in present])  # fmt: skip


TRACK_GAP = 0.02
"""Inches between a clone's RDR and BAF tracks."""

STATS_ROW = 0.13
"""Inches above each clone's RDR track for its name and state line."""


def _stack_tracks(panel: Any, foot: float, tracks: list[list[Any]]) -> None:
    """(c)'s tracks filling its panel over `foot` inches for the contig
    names: per clone, its name row, RDR, then BAF; the phase track, in the
    normal clone's place, one track tall as an RDR track is (#745).

    `plot_clones_genomic` spaces the tracks for its own page, which in a
    subfigure leaves white at the head and foot; the tracks take it.
    """
    from port.extensions.combined_figure import place_in_inches

    dpi = panel.get_figure(root=True).dpi
    box = panel.bbox
    top, bottom = box.y1 / dpi - 0.02, box.y0 / dpi + foot
    count = sum(len(group) for group in tracks)
    height = (top - bottom - len(tracks) * (STATS_ROW + TRACK_GAP)) / count
    y = top
    for group in tracks:
        y -= STATS_ROW
        for ax in group:
            y -= height
            place_in_inches(ax, y0=y, height=height)
        y -= TRACK_GAP


NAME_GAP = 4.0
"""Points between a node's name and the barcode on its edge, at the closest."""


def _fit_tree(ax: Any) -> None:
    """(a)'s x limits set so the tree fills the genome panels' width up to its barcodes.

    The root, unlabelled at event time 0, sits `NAME_GAP` in from the axis's
    left edge; each leaf's barcode ends on its right (`draw_tree(edges=True)`),
    and the tree is stretched until the leaf whose name comes closest to its
    barcode is `NAME_GAP` from it.
    """
    figure = ax.get_figure(root=True)
    renderer = figure.canvas.get_renderer()
    dpi = figure.dpi
    box = ax.get_window_extent(renderer)
    left, right = box.x0 / dpi, box.x1 / dpi
    gap = NAME_GAP / 72.0

    def inches(text: Any) -> float:
        return float(text.get_window_extent(renderer).width / dpi)

    def offset(text: Any) -> float:
        return abs(float(text.xyann[0])) / 72.0

    leaves = [t for t in ax.texts if t.get_gid() == "name"]
    barcodes = {
        round(float(t.get_position()[1]), 6): t
        for t in ax.texts
        if t.get_gid() == "barcode"
    }
    x0 = 0.0
    start = left + gap
    # NB per leaf, the most inches per unit of event time it allows.
    scale = min(
        (
            right
            - inches(barcodes[round(float(leaf.xy[1]), 6)])
            - gap
            - inches(leaf)
            - offset(leaf)
            - start
        )
        / (float(leaf.xy[0]) - x0)
        for leaf in leaves
    )
    ax.set_xlim(x0 - (start - left) / scale, x0 + (right - start) / scale)


def symbol_of(label: str) -> str:
    """A profile row's `cnaster` numeral as the paper's $m$."""
    from port.extensions.combined_figure import clone_symbol

    return str(clone_symbol(label))


def simulated_tree_figure(r: Realization, width: float | None = None) -> Any:
    """The simulated clone tree, `width` wide, at any event count: `truth_combined_figure`'s panel (a) alone (T- #660, PR- #701)."""
    import matplotlib.pyplot as plt

    from port.extensions.combined_figure import FONT_SIZE, page_style, place_in_inches
    from port.extensions.figure_style import PAPER_WIDTH
    from port.sim.analysis import draw_tree

    width = PAPER_WIDTH if width is None else width
    with page_style():
        figure: Any = plt.figure(
            figsize=(width, HEIGHTS["tree"]), dpi=300, facecolor="white"
        )
        ax = figure.add_axes((0.02, 0.02, 0.96, 0.88))
        draw_tree(ax, r, event_size=FONT_SIZE, node_size=FONT_SIZE, dot=18.0,
                  name=_symbol(r), ancestors=False, edges=True)  # fmt: skip
        place_in_inches(ax, LEFT, width - 0.05)
        figure.canvas.draw()
        _fit_tree(ax)
    return figure


def write_truth_combined(r: Realization, out: Path) -> Path:
    """`truth_combined_figure` to `out` as a PDF with no creation date, so it reproduces byte for byte."""
    import matplotlib.pyplot as plt

    from port.extensions.combined_figure import page_style

    figure = truth_combined_figure(r)
    out.parent.mkdir(parents=True, exist_ok=True)
    with page_style():
        figure.savefig(out, dpi=300, facecolor="white",
                       metadata={"CreationDate": None, "Producer": None})  # fmt: skip
    plt.close(figure)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("realization", type=Path)
    parser.add_argument("out", type=Path, nargs="?")
    arguments = parser.parse_args(argv)
    r = read(arguments.realization)
    out = arguments.out or r.path / "qa" / "truth_combined.pdf"
    print(write_truth_combined(r, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
