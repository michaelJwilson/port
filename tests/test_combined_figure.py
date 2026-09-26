"""`port.extensions.combined_figure`: two figures from a run (#309, #339).

Two claims a reader relies on. **Drawing into a subfigure changes the layout
and nothing drawn**: every point, segment and colour of the tracks is the
one the standalone `clones_genomic` page carries. **Recording does not touch
the run**: each wrapper calls through, returns what it wraps, and is removed
on exit. The pages are checked for what #280 and #339 ask of them -- a text
column wide, one text size, panels on shared edges -- which is `smoke` and
`infra`: they say the pages compose, not that anything on them is right.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.adapters import drawn
from tests.fixtures import genomic_plot_instance, integer_copies


def _genomic_arguments() -> tuple[tuple[Any, ...], dict[str, Any]]:
    instance = genomic_plot_instance()
    keywords = {
        "res_combine": instance["result"],
        "df_cnv": integer_copies(instance["rng"], 24, 3),
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

    ours, theirs = drawn(host), drawn(page)

    assert len(ours) == len(theirs)

    for index, (mine, reference) in enumerate(zip(ours, theirs, strict=True)):
        np.testing.assert_array_equal(mine, reference, err_msg=f"array {index}")


@pytest.mark.smoke
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


def _recorded(tmp_path: Path, n_clones: int = 3) -> tuple[Any, Any]:
    """A run's three recorded calls on the 3 by 3 fixture, and its slide."""
    from cnaster.he import get_he_image
    from port.extensions.combined_figure import Call, Recorded

    from tests.fixtures import clone_bands
    from tests.he_slide import mock_he, write_he_slide

    arguments, keywords = _genomic_arguments()
    n_spots = arguments[1].shape[2]
    rows, columns = np.unravel_index(np.arange(n_spots), (3, 3))
    coords = np.column_stack([rows, columns]).astype(float)
    assignment = pd.Series([f"clone {k % n_clones}" for k in range(n_spots)])
    write_he_slide(mock_he(clone_bands(3, 3, 3), (3, 3), seed=1), tmp_path)
    recorded = Recorded(
        genomic=Call(arguments, keywords),
        spatial=Call((coords, assignment), {}),
        profile=Call((keywords["df_cnv"].assign(START=0, END=1),), {}),
    )

    return recorded, get_he_image(str(tmp_path), pos=None)


def _figures(tmp_path: Path) -> tuple[Any, Any]:
    import matplotlib as mpl

    mpl.use("Agg")
    from port.extensions.combined_figure import genomic_figure, spatial_figure

    recorded, frame = _recorded(tmp_path)
    genomic, spatial = genomic_figure(recorded), spatial_figure(recorded, frame)

    for figure in (genomic, spatial):
        figure.canvas.draw()

    return genomic, spatial


def _texts(figure: Any) -> list[Any]:
    from matplotlib.text import Text

    return [
        t
        for t in figure.findobj(Text)
        if t.get_visible() and t.get_text() and t.get_figure(root=True) is figure
    ]


@pytest.mark.smoke
@pytest.mark.merge
def test_each_figure_is_a_column_wide_with_one_text_size(
    cnaster_config: None, tmp_path: Path
) -> None:
    """`llncs`'s 122 mm wide, the genomic figure its 193 mm less
    `CAPTION_ROOM` tall to 0.005 in, each lettered (a) and (b), and no text
    over `FONT_SIZE`."""
    from port.extensions.combined_figure import CAPTION_ROOM, FONT_SIZE, TEXT_HEIGHT

    genomic, spatial = _figures(tmp_path)

    for figure in (genomic, spatial):
        assert figure.get_size_inches()[0] * 25.4 == pytest.approx(122.0)
        assert [t.get_text() for t in figure.texts] == ["(a)", "(b)"]
        assert max(t.get_fontsize() for t in _texts(figure)) <= FONT_SIZE

    assert genomic.get_size_inches()[1] == pytest.approx(
        TEXT_HEIGHT - CAPTION_ROOM, abs=0.005
    )
    assert spatial.get_size_inches()[1] < TEXT_HEIGHT / 3


@pytest.mark.infra
@pytest.mark.merge
def test_each_page_is_written_at_its_size_with_nothing_past_it(
    cnaster_config: None, tmp_path: Path
) -> None:
    """Each PDF's MediaBox is its figure's size to 0.1 pt, and every text and
    legend is on the page to half a pixel.

    `llncs` fixes `\\textwidth` at 122 mm, so a page written wider is scaled
    down by `\\includegraphics[width=\\linewidth]` and its text shrinks with
    it. At a tight bounding box the page grew to 6.66 in at 6.5 (#339).
    """
    import re

    from port.patch.utils import write_fig

    for name, figure in zip(("genomic", "spatial"), _figures(tmp_path), strict=True):
        renderer = figure.canvas.get_renderer()
        page = figure.bbox
        extents = [t.get_window_extent(renderer) for t in _texts(figure)]
        extents += [
            ax.get_legend().get_window_extent(renderer)
            for ax in figure.get_axes()
            if ax.get_legend() is not None
        ]

        for extent in extents:
            assert extent.y0 >= page.y0 - 0.5, name
            assert extent.y1 <= page.y1 + 0.5, name
            assert extent.x0 >= page.x0 - 0.5, name
            assert extent.x1 <= page.x1 + 0.5, name

        path = tmp_path / f"{name}.pdf"
        write_fig(str(path), figure, bbox_inches=None)
        box = re.search(
            rb"/MediaBox\s*\[\s*[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)",
            path.read_bytes(),
        )

        assert box is not None
        width, height = figure.get_size_inches() * 72.0
        assert float(box.group(1)) == pytest.approx(width, abs=0.1)
        assert float(box.group(2)) == pytest.approx(height, abs=0.1)


@pytest.mark.infra
@pytest.mark.merge
def test_the_profile_spans_the_tracks_on_one_left_column(
    cnaster_config: None, tmp_path: Path
) -> None:
    """(b)'s axis and key have (a)'s left and right edges to 1.5 px, so bin `i`
    is under bin `i`; (b)'s names and (a)'s RDR and BAF labels start on the
    column `NAME_INSET` in, clear of the axes; (a)'s letter over its first
    statistics line and (b)'s level with its key, both on the column; the
    head a `LABEL_GAP` over them."""
    from port.extensions.combined_figure import LABEL_GAP, NAME_INSET

    figure, _ = _figures(tmp_path)
    renderer = figure.canvas.get_renderer()
    gap = LABEL_GAP / 72.0 * figure.dpi
    column = NAME_INSET / 72.0 * figure.dpi
    tracks = figure.subfigs[0].axes
    key, profile = figure.subfigs[1].axes
    edges = [ax.get_window_extent(renderer) for ax in tracks]
    left, right = min(b.x0 for b in edges), max(b.x1 for b in edges)

    for ax in (profile, key):
        box = ax.get_window_extent(renderer)
        assert box.x0 == pytest.approx(left, abs=1.5)
        assert box.x1 == pytest.approx(right, abs=1.5)

    names = [t.get_window_extent(renderer) for t in profile.get_yticklabels()]
    labels = [ax.yaxis.label.get_window_extent(renderer) for ax in tracks]
    starts = [b.x0 for b in names + labels]
    assert starts == pytest.approx([column] * len(starts), abs=1.5)
    assert left - max(b.x1 for b in names) >= gap - 1.0

    first, second = (t.get_window_extent(renderer) for t in figure.texts)
    stats = max(
        t.get_window_extent(renderer).y1 for t in tracks[0].texts if t.get_visible()
    )
    title = key.texts[0].get_window_extent(renderer)
    assert first.x0 == pytest.approx(column, abs=1.5)
    assert stats - 0.5 <= first.y0 <= stats + gap
    assert second.x0 == pytest.approx(column, abs=1.5)
    assert (second.y0 + second.y1) / 2 == pytest.approx(
        (title.y0 + title.y1) / 2, abs=1.0
    )
    assert figure.bbox.y1 - first.y1 == pytest.approx(gap, abs=1.5)


@pytest.mark.infra
@pytest.mark.merge
def test_the_spatial_panels_are_square_and_keyed_on_the_right_edge(
    cnaster_config: None, tmp_path: Path
) -> None:
    """(a) the slide and (b) the clones, square and of one size; (b)'s key one
    column on the page's right edge, a `LABEL_GAP` in, its bottom on (b)'s,
    each clone named $m$; (b)'s rows labelled by (a)'s alone; each letter
    over its panel's top-left text or corner, the head a `LABEL_GAP` over
    them."""
    from port.extensions.combined_figure import LABEL_GAP

    _, figure = _figures(tmp_path)
    renderer = figure.canvas.get_renderer()
    gap = LABEL_GAP / 72.0 * figure.dpi
    slide, clones = figure.axes
    here, tiles = (ax.get_window_extent(renderer) for ax in (slide, clones))
    key = clones.get_legend()
    box = key.get_window_extent(renderer)

    assert slide.get_images(), "(a) is the slide"
    assert here.width == pytest.approx(here.height, abs=1.0)
    assert (tiles.width, tiles.height) == pytest.approx(
        (here.width, here.height), abs=1.0
    )
    assert here.x1 < tiles.x0
    assert tiles.x1 <= box.x0
    assert box.x1 == pytest.approx(figure.bbox.x1 - gap, abs=1.5)
    assert box.y0 == pytest.approx(tiles.y0, abs=1.0)
    assert [t.get_text() for t in key.get_texts()] == ["$m_N$", "$m_1$", "$m_2$"]
    assert len({round(t.get_window_extent(renderer).x0) for t in key.get_texts()}) == 1

    assert not any(t.get_visible() for t in clones.get_yticklabels()), "(a)'s rows"

    for text, ax in zip(figure.texts, (slide, clones), strict=True):
        letter = text.get_window_extent(renderer)
        edges = [ax.get_window_extent(renderer)] + [
            t.get_window_extent(renderer)
            for t in ax.get_yticklabels()
            if t.get_visible()
        ]
        assert letter.x0 == pytest.approx(min(e.x0 for e in edges), abs=1.5)
        top = max(e.y1 for e in edges)
        assert top - 0.5 <= letter.y0 <= top + gap

    highest = max(t.get_window_extent(renderer).y1 for t in figure.texts)
    assert figure.bbox.y1 - highest == pytest.approx(gap, abs=1.5)


@pytest.mark.infra
# NB too specific to run on every change (#403): it passed where it merged,
#    and runs again where this module or the lock changes, and at a release.
@pytest.mark.deprecate
def test_the_combined_page_is_the_two_figures_stacked(
    cnaster_config: None, tmp_path: Path
) -> None:
    """One page, 122 mm by 193 mm to 0.005 in, lettered (a) to (d).

    (a) and (b) sit where the spatial figure puts them, to a pixel, measured
    from the head: the page is the spatial figure over a genomic one drawn
    the rest of the height.
    """
    import matplotlib.pyplot as plt
    from port.extensions.combined_figure import (
        TEXT_HEIGHT,
        combined_figure,
        spatial_figure,
    )

    recorded, frame = _recorded(tmp_path)
    combined = combined_figure(recorded, frame)
    spatial = spatial_figure(recorded, frame)

    assert combined.get_size_inches()[0] * 25.4 == pytest.approx(122.0)
    assert combined.get_size_inches()[1] == pytest.approx(TEXT_HEIGHT, abs=0.005)
    assert sorted(t.get_text() for t in combined.texts) == ["(a)", "(b)", "(c)", "(d)"]

    placed = combined.get_axes()[-2:]
    for new, old in zip(placed, spatial.get_axes()[:2], strict=True):
        here = new.get_window_extent(combined.canvas.get_renderer())
        there = old.get_window_extent(spatial.canvas.get_renderer())
        head = combined.bbox.height - spatial.bbox.height
        np.testing.assert_allclose(
            (here.x0, here.y0 - head, here.width, here.height),
            there.bounds,
            atol=1.0,
        )

    plt.close(combined)
    plt.close(spatial)


@pytest.mark.analytic
def test_the_hatch_stripes_are_one_width() -> None:
    """B's lines are half the hatch's period measured across them, so A's
    stripes between them are as wide: 2.065 pt at 0.10 in and 35 degrees."""
    from port.patch.plot_copy_number_profile import (
        HATCH_ANGLE,
        HATCH_LINEWIDTH,
        HATCH_SPACING,
    )

    period = 72.0 * HATCH_SPACING * np.sin(np.radians(HATCH_ANGLE))

    assert pytest.approx(period - HATCH_LINEWIDTH) == HATCH_LINEWIDTH
    assert pytest.approx(2.0649, abs=1e-4) == HATCH_LINEWIDTH
