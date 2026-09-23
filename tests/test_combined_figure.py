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
        if text.get_text() not in titles
    ]
    assert max(sizes) <= FONT_SIZE < LABEL_SIZE


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
    """(b)'s axis has (a)'s left and right edges to a pixel, so bin `i` is under
    bin `i`; its names are left-aligned, `LABEL_GAP` clear of the axis."""
    import matplotlib as mpl

    mpl.use("Agg")
    from port.extensions.combined_figure import LABEL_GAP

    figure = _page(tmp_path)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    top, middle = figure.subfigs[0], figure.subfigs[1]

    tracks = [ax.get_window_extent(renderer) for ax in top.axes]
    profile = middle.axes[0].get_window_extent(renderer)

    assert profile.x0 == pytest.approx(min(b.x0 for b in tracks), abs=1.0)
    assert profile.x1 == pytest.approx(max(b.x1 for b in tracks), abs=1.0)

    names = [t.get_window_extent(renderer) for t in middle.axes[0].get_yticklabels()]
    lefts = [name.x0 for name in names]

    assert max(lefts) - min(lefts) < 1.0
    gap = profile.x0 - max(name.x1 for name in names)
    assert 0.0 < gap <= LABEL_GAP / 72.0 * figure.dpi + 1.0


@pytest.mark.infra
def test_the_slide_is_left_and_the_clone_key_is_two_columns_one_alone_on_top(
    cnaster_config: None, tmp_path: Path
) -> None:
    """(c) the slide on (a)'s left edge, (d) the clones; three clones key as one
    over two, the key's bottom on (d)'s bottom."""
    import matplotlib as mpl

    mpl.use("Agg")
    figure = _page(tmp_path, n_clones=3)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    slide, clones = figure.subfigs[2].subfigs

    assert slide.axes[0].get_images(), "(c) is the slide"
    assert (
        slide.axes[0].get_window_extent(renderer).x1
        <= clones.axes[0].get_window_extent(renderer).x0
    )

    key = clones.axes[0].get_legend()
    entries = {
        text.get_text(): text.get_window_extent(renderer)
        for text in key.get_texts()
        if text.get_text()
    }
    rows = sorted({round(box.y0) for box in entries.values()}, reverse=True)

    assert len(entries) == 3
    assert len(rows) == 2
    assert sum(round(box.y0) == rows[0] for box in entries.values()) == 1

    tracks = figure.subfigs[0].axes
    here = slide.axes[0].get_window_extent(renderer)
    tiles = clones.axes[0].get_window_extent(renderer)

    assert here.x0 == pytest.approx(
        min(ax.get_window_extent(renderer).x0 for ax in tracks), abs=1.0
    )
    assert key.get_window_extent(renderer).y0 == pytest.approx(tiles.y0, abs=1.0)


@pytest.mark.infra
def test_the_letters_sit_level_with_their_panels_in_the_margin(
    cnaster_config: None, tmp_path: Path
) -> None:
    """Each letter's top on its panel's top, left of every axis; nothing off
    the page left or right, (b)'s legend on (a)'s edges."""
    import matplotlib as mpl

    mpl.use("Agg")
    figure = _page(tmp_path)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    page = figure.bbox
    tracks = figure.subfigs[0].axes
    left = min(ax.get_window_extent(renderer).x0 for ax in tracks)
    right = max(ax.get_window_extent(renderer).x1 for ax in tracks)

    tiles = figure.subfigs[2].subfigs[1].axes[0].get_window_extent(renderer)
    # NB (d)'s letter sits between (c) and (d), the others in the margin.
    edges = (left, left, left, tiles.x0)

    for text, edge in zip(figure.texts, edges, strict=True):
        box = text.get_window_extent(renderer)
        assert box.x0 >= page.x0 - 0.5
        assert box.x1 <= edge, text.get_text()

    for panel in figure.subfigs[:2]:
        for ax in panel.axes:
            ticks = [*ax.get_xticklabels(), *ax.get_yticklabels()] if ax.axison else []

            for text in [*ax.texts, *ticks]:
                if text.get_visible() and text.get_text():
                    box = text.get_window_extent(renderer)
                    assert box.x0 >= page.x0 - 0.5, text.get_text()
                    assert box.x1 <= page.x1 + 0.5, text.get_text()

    legend = figure.subfigs[1].axes[1].get_window_extent(renderer)
    assert legend.x0 == pytest.approx(left, abs=1.0)
    assert legend.x1 == pytest.approx(right, abs=1.0)
