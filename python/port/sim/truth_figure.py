r"""What a realization planted, on one page at `combined.pdf`'s size and type.

    python -m port.sim.truth_figure sim/generated/<name>/r<k> [OUT.pdf]

Top to bottom, at `llncs`'s text width and height, 7 pt throughout, as
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
10 Mb marks and the chromosome names; every track and (b) keep the
chromosome boundaries.

The true clone of each spot is not drawn here: `analysis.plot_spatial` draws
it as its own figure, `truth/spatial.png` (T- #660).

(b) and (c) share one left and one right edge, so a chromosome boundary is
at one place on the page in both. Clones
are named as the paper names them, $m_N$ for the normal. The phase and the
count laws stay in their own figures, `phase.png` and `coverage.png`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from port.sim.analysis import Realization, display, read

__all__ = ["truth_combined_figure", "write_truth_combined"]

KEY = (0.05, 0.2)
"""Inches: (b)'s gap between its key and its rows, and the key's height."""

ROWS = (0.04, 0.42)
"""Inches: (b)'s foot under its rows, unlabelled (PR- #701), and the rows' height."""

KEY_TOP = 0.075
"""Inches over (b)'s key: with (a)'s foot, the white between (a) and (b) (PR- #701)."""

HEIGHTS = {
    "tree": 0.9,
    "profile": sum(ROWS) + sum(KEY) + KEY_TOP,
    "genomic": 7.6 - 0.9 - (sum(ROWS) + sum(KEY) + KEY_TOP),
}
"""Inches per panel, summing to `TEXT_HEIGHT` less its rounding; (a) 0.9,
10% under its 1.0 before PR- #701."""

LEFT = 0.42
"""Inches from the page's left edge to the genome panels' axes: room for the RDR and BAF labels."""

RIGHT = 0.06
"""Inches of the page right of the genome panels, before the last chromosome's name is fitted."""


def _symbol(r: Realization) -> Any:
    from port.extensions.combined_figure import clone_symbol

    return lambda clone: clone_symbol(display(clone, r.clones))


def truth_combined_figure(
    r: Realization, width: float | None = None, *, metric: bool = False
) -> Any:
    """The three panels on one page, `width` wide (`llncs`'s by default) and
    `TEXT_HEIGHT` tall; on `metric`, the planted CNAs drawn wider (T- #683)."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    from port.extensions.combined_figure import (
        FONT_SIZE,
        LABEL_GAP,
        LABEL_SIZE,
        LEGEND_BOX,
        TEXT_HEIGHT,
        _fit_tracks,
        _put,
        _set_text,
        page_style,
    )
    from port.extensions.figure_style import PAPER_WIDTH
    from port.extensions.genomic_axis import disclose
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

    with page_style():
        figure: Any = plt.figure(
            figsize=(width, TEXT_HEIGHT), dpi=300, facecolor="white"
        )
        panels: Any = figure.subfigures(
            len(HEIGHTS), 1, height_ratios=list(HEIGHTS.values()), hspace=0.0
        )
        tree_fig, profile_fig, genomic_fig = panels

        # (a)
        tree_ax = tree_fig.add_axes((0.02, 0.02, 0.96, 0.88))
        draw_tree(tree_ax, r, event_size=FONT_SIZE, node_size=FONT_SIZE,
                  dot=18.0, name=symbol, ancestors=False, edges=True)  # fmt: skip

        # (b): the key, then the rows, as `combined.pdf` draws its profile.
        tall = HEIGHTS["profile"]
        legend_ax = profile_fig.add_axes(
            (0.0, (sum(ROWS) + KEY[0]) / tall, 1.0, KEY[1] / tall)
        )
        profile_ax = profile_fig.add_axes((0.0, ROWS[0] / tall, 1.0, ROWS[1] / tall))
        plot_copy_number_profile(binned_profile(r), ax=profile_ax, axis=genome)
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
        g = genomic_truth(r)
        plot_clones_genomic(g.lengths, g.counts, g.expected, g.trials,
                            clone_index=g.groups, figure=genomic_fig,
                            pointsize=0.4, linewidth=0.3, chrtext_shift=-0.9,
                            axis=genome)  # fmt: skip
        _fit_tracks(genomic_fig)
        # NB each clone's name followed by its barcode, as (a) sets it.
        barcode = tree(r).barcode
        named = {symbol(clone): clone for clone in r.clones}
        for ax in genomic_fig.axes:
            for text in ax.texts:
                clone = named.get(text.get_text())
                if clone is not None and text.get_visible():
                    text.set_text(f"{symbol(clone)} ({shown(barcode[clone])})")
        for ax in genomic_fig.axes:
            ticks = ax.get_yticks()
            if ticks.size > 2:
                ends = [ticks[0], ticks[-1]]
                ax.set_yticks(ends, [f"{tick:.1f}" for tick in ends])
                bottom, top = ax.get_yticklabels()
                bottom.set_verticalalignment("bottom")
                top.set_verticalalignment("top")
            ax.set_ylabel(ax.get_ylabel(), rotation=90, ha="center", va="bottom")
            # NB `cnaster`'s names, 10 pt at 45 degrees, give way to the
            #    last track's ticks; the marks stay on the last track alone.
            for text in ax.texts:
                if text.get_text().startswith("chr"):
                    text.set_visible(False)
            if ax is not genomic_fig.axes[-1]:
                ax.xaxis.set_minor_locator(NullLocator())
        bottom_ax = genomic_fig.axes[-1]
        bottom_ax.set_xticks(starts, contigs, rotation=45, ha="left")
        bottom_ax.tick_params(axis="x", which="major", bottom=False,
                              labelbottom=True, pad=1.0)  # fmt: skip

        for panel in panels:
            _set_text(panel, FONT_SIZE)
        figure.canvas.draw()

        # NB one left and one right edge for the key, the rows and each
        #    track; the right pulled in until no chromosome name runs off
        #    the page, as `combined_figure` fits its own.
        renderer = figure.canvas.get_renderer()
        right = width - RIGHT
        for _ in range(3):
            for ax in [legend_ax, profile_ax, *genomic_fig.axes]:
                _put(ax, LEFT, right)
            figure.canvas.draw()
            names = [t for t in bottom_ax.get_xticklabels() if t.get_text()]
            overrun = (
                max(t.get_window_extent(renderer).x1 for t in names) / figure.dpi
                - width
            )
            if overrun <= 0.0:
                break
            right -= overrun + LABEL_GAP / 72.0

        _stack_tracks(genomic_fig)
        _put(tree_ax, LEFT, right)
        figure.canvas.draw()
        _fit_tree(tree_ax)
        plot_ascn_legend(legend_ax, box_w=LEGEND_BOX, box_h=0.8, tick_len=0.1,
                         label_fontsize=FONT_SIZE, span=right - LEFT)  # fmt: skip
        _thin(bottom_ax.get_xticklabels(), renderer)

        for panel, letter in zip(panels, "abc", strict=True):
            panel.text(0.0, 1.0, f"({letter})", fontsize=LABEL_SIZE, ha="left",
                       va="top")  # fmt: skip
        figure.canvas.draw()
    disclose(figure, genome)
    return figure


TRACK_GAP = 0.02
"""Inches between a clone's RDR and BAF tracks."""

STATS_ROW = 0.13
"""Inches above each clone's RDR track for its name and state line."""

CONTIG_ROW = 0.27
"""Inches under the last track for its chromosome names, at 45 degrees (PR- #701)."""


def _stack_tracks(panel: Any) -> None:
    """(c)'s tracks filling its panel: per clone, its name row, RDR, then BAF.

    `plot_clones_genomic` spaces the tracks for its own page, which in a
    subfigure leaves white at the head and foot; the tracks take it.
    """
    from port.extensions.combined_figure import _put

    dpi = panel.get_figure(root=True).dpi
    box = panel.bbox
    top, bottom = box.y1 / dpi - 0.02, box.y0 / dpi + CONTIG_ROW
    tracks = list(panel.axes)
    clones = len(tracks) // 2
    height = (top - bottom - clones * (STATS_ROW + TRACK_GAP)) / len(tracks)
    y = top
    for k in range(clones):
        y -= STATS_ROW
        for ax in tracks[2 * k : 2 * k + 2]:
            y -= height
            _put(ax, y0=y, height=height)
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


def _thin(labels: Any, renderer: Any) -> None:
    """Hide each label that overlaps the last one kept, left to right: the small chromosomes' names."""
    kept = None
    for label in labels:
        box = label.get_window_extent(renderer)
        if kept is not None and box.overlaps(kept):
            label.set_visible(False)
            continue
        kept = box


def symbol_of(label: str) -> str:
    """A profile row's `cnaster` numeral as the paper's $m$."""
    from port.extensions.combined_figure import clone_symbol

    return str(clone_symbol(label))


def simulated_tree_figure(r: Realization, width: float | None = None) -> Any:
    """The simulated clone tree, `width` wide, at any event count: `truth_combined_figure`'s panel (a) alone (T- #660, PR- #701)."""
    import matplotlib.pyplot as plt

    from port.extensions.combined_figure import FONT_SIZE, _put, page_style
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
        _put(ax, LEFT, width - 0.05)
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
