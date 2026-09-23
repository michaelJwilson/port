"""`port.extensions.combined_figure`: one page from a run's figures (#309).

Two claims a reader relies on. **Drawing into a subfigure changes the layout
and nothing drawn**: every point, segment and colour of (a) is the one the
standalone `clones_genomic` page carries. **Recording does not touch the
run**: each wrapper calls through, returns what it wraps, and is removed on
exit. The page itself is checked for what #280 asks of it -- a text column
wide, no text above the cap -- which is `smoke`: it says the page composes,
not that anything on it is right.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.test_plot_genomic_patch import _drawn, _instance, _integer_copies


def _genomic_arguments() -> tuple[tuple[Any, ...], dict[str, Any]]:
    instance = _instance()
    keywords = {
        "res_combine": instance["result"],
        "df_cnv": _integer_copies(instance["rng"], 24, 3),
    }

    return instance["arguments"], keywords


@pytest.mark.patch
def test_a_subfigure_draws_what_the_standalone_page_draws(cnaster_config: None) -> None:
    """Every array of (a), bitwise, against the page `clones_genomic.pdf` is."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from port.patch.plot_genomic import plot_clones_genomic

    arguments, keywords = _genomic_arguments()
    page = plot_clones_genomic(*arguments, **keywords)

    host = plt.figure(figsize=(6.5, 4.0), layout="constrained")
    panels: Any = host.subfigures(2, 1)
    top = panels[0]
    plot_clones_genomic(*arguments, **{**keywords, "figure": top})

    ours, theirs = _drawn(host), _drawn(page)

    assert len(ours) == len(theirs)

    for index, (mine, reference) in enumerate(zip(ours, theirs, strict=True)):
        np.testing.assert_array_equal(mine, reference, err_msg=f"array {index}")


@pytest.mark.infra
def test_recording_calls_through_and_restores() -> None:
    """A recorded call returns the wrapped function's figure; the names return."""
    import matplotlib as mpl

    mpl.use("Agg")
    import cnaster.scripts.run_cnaster as script
    import port.patch.plot_genomic as genomic
    import port.patch.plotting as spatial
    from port.extensions.combined_figure import recording

    before = (
        genomic.plot_clones_genomic,
        spatial.plot_clones_spatial,
        script.plot_copy_number_profile,
    )
    coords = np.column_stack([np.arange(4.0), np.zeros(4)])
    assignment = pd.Series(["clone 0", "clone 1"] * 2)

    with recording() as recorded:
        figure = spatial.plot_clones_spatial(coords, assignment)

    assert figure.axes[0].collections, "the wrapped function drew nothing"
    assert recorded.spatial is not None
    assert recorded.spatial.args[0] is coords
    assert recorded.calls == {"spatial": 1}
    assert (
        genomic.plot_clones_genomic,
        spatial.plot_clones_spatial,
        script.plot_copy_number_profile,
    ) == before


@pytest.mark.smoke
def test_the_page_is_a_column_wide_with_capped_text(
    cnaster_config: None, tmp_path: Path
) -> None:
    """`llncs`'s 122 mm wide, four panels labelled (a) to (d), no text over 6 pt else."""
    import matplotlib as mpl

    mpl.use("Agg")
    from cnaster.he import get_he_image
    from matplotlib.text import Text
    from port.extensions.combined_figure import (
        FONT_SIZE,
        LABEL_SIZE,
        Call,
        Recorded,
        combined_figure,
    )

    from tests.fixtures import clone_bands
    from tests.he_slide import mock_he, write_he_slide

    arguments, keywords = _genomic_arguments()
    n_spots = arguments[1].shape[2]
    rows, columns = np.unravel_index(np.arange(n_spots), (3, 3))
    coords = np.column_stack([rows, columns]).astype(float)
    assignment = pd.Series(
        [f"clone {c}" for c in keywords["res_combine"]["new_assignment"]]
    )

    write_he_slide(mock_he(clone_bands(3, 3, 3), (3, 3), seed=1), tmp_path)
    frame = get_he_image(str(tmp_path), pos=None)

    recorded = Recorded(
        genomic=Call(arguments, keywords),
        spatial=Call((coords, assignment), {}),
        profile=Call((keywords["df_cnv"].assign(START=0, END=1),), {}),
    )
    figure = combined_figure(recorded, frame)

    assert figure.get_size_inches()[0] * 25.4 == pytest.approx(122.0)

    titles = [text.get_text() for text in figure.texts]
    assert titles == ["(a)", "(b)", "(c)", "(d)"]

    sizes = [
        text.get_fontsize()
        for text in figure.findobj(Text)
        if text.get_text() and text.get_text() not in titles
    ]
    assert max(sizes) <= FONT_SIZE == LABEL_SIZE


@pytest.mark.infra
def test_the_page_is_written_at_the_text_width_with_nothing_past_it(
    cnaster_config: None, tmp_path: Path
) -> None:
    """The PDF's MediaBox is 122 mm to 0.1 pt, and every legend is on the page.

    `llncs` fixes `\\textwidth` at 122 mm, so a page written wider is scaled
    down by `\\includegraphics[width=\\linewidth]` and its 6 pt text shrinks
    with it. At a tight bounding box the page grew to 6.66 in at 6.5 (#339),
    which also hid (c)'s legend running off the bottom.
    """
    import re

    import matplotlib as mpl

    mpl.use("Agg")
    from cnaster.he import get_he_image
    from port.extensions.combined_figure import Call, Recorded, combined_figure
    from port.patch.utils import write_fig

    from tests.fixtures import clone_bands
    from tests.he_slide import mock_he, write_he_slide

    arguments, keywords = _genomic_arguments()
    n_spots = arguments[1].shape[2]
    rows, columns = np.unravel_index(np.arange(n_spots), (3, 3))
    coords = np.column_stack([rows, columns]).astype(float)
    assignment = pd.Series(
        [f"clone {c}" for c in keywords["res_combine"]["new_assignment"]]
    )
    write_he_slide(mock_he(clone_bands(3, 3, 3), (3, 3), seed=1), tmp_path)
    recorded = Recorded(
        genomic=Call(arguments, keywords),
        spatial=Call((coords, assignment), {}),
        profile=Call((keywords["df_cnv"].assign(START=0, END=1),), {}),
    )
    figure = combined_figure(recorded, get_he_image(str(tmp_path), pos=None))

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    page = figure.bbox

    # NB the legends one by one: one kept out of the layout is not in the
    #    figure's tight box, and (c)'s ran off the page that way at 4.80 in.
    extents = [figure.get_tightbbox(renderer)]
    extents += [
        ax.get_legend().get_window_extent(renderer)
        for ax in figure.axes
        if ax.get_legend() is not None
    ]

    for extent in extents[1:]:
        assert extent.y0 >= page.y0 - 0.5, (
            f"{extent.y0 - page.y0:.1f} px below the page"
        )
        assert extent.x1 <= page.x1 + 0.5, f"{extent.x1 - page.x1:.1f} px past the page"

    path = tmp_path / "combined.pdf"
    write_fig(str(path), figure, bbox_inches=None)
    box = re.search(rb"/MediaBox\s*\[\s*[\d.]+\s+[\d.]+\s+([\d.]+)", path.read_bytes())

    assert box is not None
    assert float(box.group(1)) == pytest.approx(122.0 / 25.4 * 72.0, abs=0.1)


def _page(tmp_path: Path, n_clones: int = 3) -> Any:
    from cnaster.he import get_he_image
    from port.extensions.combined_figure import Call, Recorded, combined_figure

    from tests.fixtures import clone_bands
    from tests.he_slide import mock_he, write_he_slide

    arguments, keywords = _genomic_arguments()
    n_spots = arguments[1].shape[2]
    rows, columns = np.unravel_index(np.arange(n_spots), (3, 3))
    coords = np.column_stack([rows, columns]).astype(float)
    assignment = pd.Series([f"clone {k % n_clones}" for k in range(n_spots)])
    write_he_slide(mock_he(clone_bands(3, 3, 3), (3, 3), seed=1), tmp_path)

    return combined_figure(
        Recorded(
            genomic=Call(arguments, keywords),
            spatial=Call((coords, assignment), {}),
            profile=Call((keywords["df_cnv"].assign(START=0, END=1),), {}),
        ),
        get_he_image(str(tmp_path), pos=None),
    )


@pytest.mark.infra
def test_b_spans_a_and_its_clone_names_start_in_one_column(
    cnaster_config: None, tmp_path: Path
) -> None:
    """The profile's axis has the tracks' left and right edges to 1.5 px, so bin
    `i` is under bin `i`; its names are left-aligned on the column
    `NAME_INSET` from the page's edge, clear of the axis."""
    import matplotlib as mpl

    mpl.use("Agg")
    from port.extensions.combined_figure import LABEL_GAP, NAME_INSET

    figure = _page(tmp_path)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    top, middle = figure.subfigs[1], figure.subfigs[2]

    tracks = [ax.get_window_extent(renderer) for ax in top.axes]
    profile = middle.axes[1].get_window_extent(renderer)

    assert profile.x0 == pytest.approx(min(b.x0 for b in tracks), abs=1.5)
    assert profile.x1 == pytest.approx(max(b.x1 for b in tracks), abs=1.5)

    names = [t.get_window_extent(renderer) for t in middle.axes[1].get_yticklabels()]
    lefts = [name.x0 for name in names]

    assert max(lefts) - min(lefts) < 1.0
    gap = LABEL_GAP / 72.0 * figure.dpi
    assert profile.x0 - max(name.x1 for name in names) >= gap - 1.0

    # NB on the left column: the letters sit over the panels.
    column = NAME_INSET / 72.0 * figure.dpi
    assert min(lefts) == pytest.approx(column, abs=1.5)


@pytest.mark.infra
def test_the_top_row_is_slide_clones_key_edge_to_edge(
    cnaster_config: None, tmp_path: Path
) -> None:
    """(a) the slide on the left edge, centred in a box `SPATIAL_INSET`
    larger; (b)'s key one column on the right edge, its bottom on (b)'s, (b)
    left of it; each clone named $m$."""
    import matplotlib as mpl

    mpl.use("Agg")
    from port.extensions.combined_figure import SPATIAL_INSET

    figure = _page(tmp_path, n_clones=3)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    slide, clones = figure.subfigs[0].axes
    tracks = figure.subfigs[1].axes

    here = slide.get_window_extent(renderer)
    tiles = clones.get_window_extent(renderer)
    key = clones.get_legend()
    box = key.get_window_extent(renderer)

    assert slide.get_images(), "(a) is the slide"
    border = (1.0 - SPATIAL_INSET) / 2 / SPATIAL_INSET
    assert here.x0 - border * here.width == pytest.approx(
        min(ax.get_window_extent(renderer).x0 for ax in tracks), abs=1.5
    )
    assert box.x1 == pytest.approx(
        max(ax.get_window_extent(renderer).x1 for ax in tracks), abs=1.5
    )
    assert here.x1 <= tiles.x0
    assert tiles.x1 <= box.x0
    assert box.y0 == pytest.approx(tiles.y0, abs=1.0)
    assert [t.get_text() for t in key.get_texts()] == ["$m_N$", "$m_1$", "$m_2$"]

    lefts = {round(t.get_window_extent(renderer).x0) for t in key.get_texts()}
    assert len(lefts) == 1, "one column"


@pytest.mark.infra
def test_the_letters_sit_over_their_panels_on_its_leftmost_text(
    cnaster_config: None, tmp_path: Path
) -> None:
    """(a) to (c)'s bottom on its panel's top, or on the text over it, to one
    `LABEL_GAP`, and (d) level with its key; each letter's left on (a) and
    (b)'s extent ticks, or on (c) and (d)'s left column, which (c)'s labels
    and (d)'s names start on; the page's head
    a `LABEL_GAP` over the letters; no text off the page, (d)'s legend on
    (c)'s edges."""
    import matplotlib as mpl

    mpl.use("Agg")
    from port.extensions.combined_figure import LABEL_GAP, NAME_INSET

    figure = _page(tmp_path)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    page = figure.bbox
    gap = LABEL_GAP / 72.0 * figure.dpi
    column = NAME_INSET / 72.0 * figure.dpi
    tracks = figure.subfigs[1].axes
    left = min(ax.get_window_extent(renderer).x0 for ax in tracks)
    right = max(ax.get_window_extent(renderer).x1 for ax in tracks)

    slide, clones = figure.subfigs[0].axes
    legend_ax, profile = figure.subfigs[2].axes
    edges = (
        min(t.get_window_extent(renderer).x0 for t in slide.get_yticklabels()),
        min(t.get_window_extent(renderer).x0 for t in clones.get_yticklabels()),
        column,
        column,
    )

    for text, ax, edge in zip(
        figure.texts[:3], (slide, clones, tracks[0]), edges[:3], strict=True
    ):
        box = text.get_window_extent(renderer)
        heads = [
            t.get_window_extent(renderer).y1
            for t in [*ax.texts, *(ax.get_yticklabels() if ax.axison else [])]
            if t.get_visible() and t.get_text()
        ]
        above = max([ax.get_window_extent(renderer).y1, *heads])

        assert box.x0 == pytest.approx(edge, abs=1.5), text.get_text()
        assert box.y0 >= above - 0.5, text.get_text()
        assert box.y0 <= above + gap, text.get_text()

    # NB (d)'s in the column, level with its key's first title.
    letter = figure.texts[3].get_window_extent(renderer)
    title = legend_ax.texts[0].get_window_extent(renderer)
    assert letter.x0 == pytest.approx(edges[3], abs=1.5)
    assert (letter.y0 + letter.y1) / 2 == pytest.approx(
        (title.y0 + title.y1) / 2, abs=1.0
    )

    labels = [ax.yaxis.label.get_window_extent(renderer).x0 for ax in tracks]
    names = [t.get_window_extent(renderer).x0 for t in profile.get_yticklabels()]
    assert labels + names == pytest.approx([column] * len(labels + names), abs=1.5)

    highest = max(t.get_window_extent(renderer).y1 for t in figure.texts)
    assert page.y1 - highest == pytest.approx(gap, abs=1.5)

    for panel in figure.subfigs:
        for ax in panel.axes:
            ticks = [*ax.get_xticklabels(), *ax.get_yticklabels()] if ax.axison else []

            for text in [*ax.texts, *ticks]:
                if text.get_visible() and text.get_text():
                    box = text.get_window_extent(renderer)
                    assert box.x0 >= page.x0 - 0.5, text.get_text()
                    assert box.x1 <= page.x1 + 0.5, text.get_text()

    legend = figure.subfigs[2].axes[0].get_window_extent(renderer)
    assert legend.x0 == pytest.approx(left, abs=1.0)
    assert legend.x1 == pytest.approx(right, abs=1.0)
