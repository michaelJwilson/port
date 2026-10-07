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


def _recorded(tmp_path: Path, n_clones: int = 3, tall: float = 1.0) -> tuple[Any, Any]:
    """A run's three recorded calls on the 3 by 3 fixture, its rows `tall`
    times as far apart as its columns, and its slide."""
    from cnaster.he import get_he_image
    from port.extensions.combined_figure import Call, Recorded
    from port.sim.he_slide import mock_he, write_he_slide
    from port.sim.truth import clone_bands

    arguments, keywords = _genomic_arguments()
    n_spots = arguments[1].shape[2]
    rows, columns = np.unravel_index(np.arange(n_spots), (3, 3))
    coords = np.column_stack([rows, tall * columns]).astype(float)
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
    """A text column wide, the genomic figure the text block less
    `CAPTION_ROOM` tall to 0.005 in, each lettered (a) and (b), and every text
    at `FONT_SIZE` but the tracks' at `TRACK_FONT_SIZE` (#743)."""
    from port.extensions.combined_figure import FONT_SIZE
    from port.extensions.figure_style import (
        CAPTION_ROOM,
        PAPER_WIDTH,
        TEXT_HEIGHT,
        TRACK_FONT_SIZE,
        page_size,
    )

    genomic, spatial = _figures(tmp_path)

    for figure in (genomic, spatial):
        assert figure.get_size_inches()[0] == pytest.approx(PAPER_WIDTH)
        assert [t.get_text() for t in figure.texts] == ["(a)", "(b)"]
        assert max(t.get_fontsize() for t in _texts(figure)) <= FONT_SIZE
        assert {t.get_fontsize() for t in _texts(figure)} <= {
            FONT_SIZE,
            TRACK_FONT_SIZE,
        }

    assert genomic.get_size_inches()[1] == pytest.approx(
        TEXT_HEIGHT - CAPTION_ROOM, abs=0.005
    )
    assert spatial.get_size_inches()[1] <= page_size("third")[1] + 0.005


@pytest.mark.infra
@pytest.mark.merge
def test_each_page_is_written_at_its_size_with_nothing_past_it(
    cnaster_config: None, tmp_path: Path
) -> None:
    """Each PDF's MediaBox is its figure's size to 0.1 pt, and every text and
    legend is on the page to half a pixel.

    The paper fixes `\\textwidth` at `PAPER_WIDTH`, so a page written wider is scaled
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
    """(a)'s axis and key have (b)'s left and right edges to 1.5 px, so bin `i`
    is over bin `i`; (a)'s names and (b)'s RDR and BAF labels start on the
    column `NAME_INSET` in, clear of the axes; (a)'s letter level with its
    key and (b)'s over its first statistics line, both on the column; the
    head a `LABEL_GAP` over the key; the mirror key as `mirror_key_holds` says
    (PR- #701)."""
    from port.extensions.combined_figure import LABEL_GAP, NAME_INSET

    from tests.test_plot_copy_number_profile_patch import mirror_key_holds

    figure, _ = _figures(tmp_path)
    renderer = figure.canvas.get_renderer()
    gap = LABEL_GAP / 72.0 * figure.dpi
    column = NAME_INSET / 72.0 * figure.dpi
    key, profile = figure.subfigs[0].axes
    tracks = figure.subfigs[1].axes
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
        t.get_window_extent(renderer).y1
        for t in tracks[0].texts
        if t.get_visible() and t.get_gid() is None
    )
    title = key.texts[0].get_window_extent(renderer)
    top = max(p.get_window_extent(renderer).y1 for p in [*key.patches, *key.texts])
    assert first.x0 == pytest.approx(column, abs=1.5)
    assert (first.y0 + first.y1) / 2 == pytest.approx(
        (title.y0 + title.y1) / 2, abs=1.0
    )
    assert second.x0 == pytest.approx(column, abs=1.5)
    assert stats - 0.5 <= second.y0 <= stats + gap
    assert figure.bbox.y1 - top == pytest.approx(gap, abs=1.5)
    mirror_key_holds(key, profile)


@pytest.mark.infra
@pytest.mark.merge
def test_the_spatial_panels_are_square_keyed_on_the_right_and_centred(
    cnaster_config: None, tmp_path: Path
) -> None:
    """(a) the slide and (b) the clones, of one size, each box square and its
    data at one scale on both axes to 1%; (b)'s key one
    column on the page's right edge, a `LABEL_GAP` in, its bottom on (b)'s,
    each clone named $m$; (b)'s rows labelled by (a)'s alone; each letter
    over its panel's top-left text or corner, the head a `LABEL_GAP` over
    them. The squares capped to a "third" page, the row is centred: the white
    right of the key, less a `LABEL_GAP`, is the white left of (a)'s tick
    labels, less `NAME_INSET`, to 2 px (T- #740)."""
    from port.extensions.combined_figure import LABEL_GAP, NAME_INSET

    _, figure = _figures(tmp_path)
    renderer = figure.canvas.get_renderer()
    gap = LABEL_GAP / 72.0 * figure.dpi
    slide, clones = figure.axes
    here, tiles = (ax.get_window_extent(renderer) for ax in (slide, clones))
    key = clones.get_legend()
    box = key.get_window_extent(renderer)

    assert slide.get_images(), "(a) is the slide"
    assert here.width == pytest.approx(here.height, abs=1.0)
    for ax, frame in ((slide, here), (clones, tiles)):
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        assert abs(x1 - x0) / frame.width == pytest.approx(
            abs(y1 - y0) / frame.height, rel=0.01
        )
        assert (x0, x1, y0, y1) == clones.get_xlim() + clones.get_ylim()
    assert (tiles.width, tiles.height) == pytest.approx(
        (here.width, here.height), abs=1.0
    )
    assert here.x1 < tiles.x0
    assert tiles.x1 <= box.x0
    labels = [
        t.get_window_extent(renderer)
        for t in slide.get_yticklabels()
        if t.get_visible()
    ]
    left = min(t.x0 for t in labels) - NAME_INSET / 72.0 * figure.dpi
    assert figure.bbox.x1 - gap - box.x1 == pytest.approx(left, abs=2.0)
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
def test_the_spatial_labels_are_integer_by_default_or_continuous(
    cnaster_config: None, tmp_path: Path
) -> None:
    """Two clones that decode alike at every bin are one clone under the
    default "integer" labels -- two entries, $m_N$ and $m_1$, the merged
    clone renumbered with no gap (#745) -- and stay two under "continuous":
    three keyed, as the fit found them (#344)."""
    import matplotlib as mpl

    mpl.use("Agg")
    from port.extensions.combined_figure import Call, spatial_figure

    recorded, frame = _recorded(tmp_path)
    assert recorded.profile is not None
    df_cnv = recorded.profile.args[0].copy()

    for column in ("A", "B"):
        df_cnv[f"clone2 {column}"] = df_cnv[f"clone1 {column}"]

    recorded.profile = Call((df_cnv,), {})

    def keyed(labels: str) -> list[str]:
        figure = spatial_figure(recorded, frame, labels=labels)
        return [t.get_text() for t in figure.axes[1].get_legend().get_texts()]

    assert keyed("continuous") == ["$m_N$", "$m_1$", "$m_2$"]
    assert keyed("integer") == ["$m_N$", "$m_1$"]

    with pytest.raises(ValueError, match="integer"):
        spatial_figure(recorded, frame, labels="decoded")


@pytest.mark.infra
# NB too specific to run on every change (#403): it passed where it merged,
#    and runs again where this module or the lock changes, and at a release.
@pytest.mark.deprecate
def test_the_combined_page_is_the_two_figures_stacked(
    cnaster_config: None, tmp_path: Path
) -> None:
    """One page, the text block less `CAPTION_ROOM` to 0.005 in, lettered
    (a) to (c).

    (a)'s slide and clones sit where the spatial figure puts them, to a
    pixel, measured from the head: the page is the spatial figure over a
    genomic one drawn the rest of the height.
    """
    import matplotlib.pyplot as plt
    from port.extensions.combined_figure import combined_figure, spatial_figure
    from port.extensions.figure_style import CAPTION_ROOM, PAPER_WIDTH, TEXT_HEIGHT

    recorded, frame = _recorded(tmp_path)
    combined = combined_figure(recorded, frame)
    spatial = spatial_figure(recorded, frame)

    assert combined.get_size_inches()[0] == pytest.approx(PAPER_WIDTH)
    assert combined.get_size_inches()[1] == pytest.approx(
        TEXT_HEIGHT - CAPTION_ROOM, abs=0.005
    )
    assert sorted(t.get_text() for t in combined.texts) == ["(a)", "(b)", "(c)"]

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


def panels_in_order(letters: list[Any], panels: dict[str, list[Any]]) -> list[str]:
    """`panels`' names top to bottom by their axes' tops, each lettered in
    turn: the k-th letter from the head reads `(a)`, `(b)`, ... and sits
    under the panel before its own and over the panel after it."""
    figure = next(iter(panels.values()))[0].get_figure(root=True)
    renderer = figure.canvas.get_renderer()

    def extent(axes: list[Any]) -> tuple[float, float]:
        boxes = [ax.get_window_extent(renderer) for ax in axes]
        return min(b.y0 for b in boxes), max(b.y1 for b in boxes)

    order = sorted(panels, key=lambda name: -extent(panels[name])[1])
    boxes = sorted(
        (t.get_window_extent(renderer) for t in letters), key=lambda b: -b.y1
    )
    texts = sorted(letters, key=lambda t: -t.get_window_extent(renderer).y1)

    assert [t.get_text() for t in texts] == [f"({k})" for k in "abc"[: len(order)]]
    for k, box in enumerate(boxes):
        middle = (box.y0 + box.y1) / 2
        if k > 0:
            assert middle < extent(panels[order[k - 1]])[0]
        if k + 1 < len(order):
            assert middle > extent(panels[order[k + 1]])[1]
    return order


@pytest.mark.infra
@pytest.mark.merge
def test_the_combined_page_reads_clones_profile_tracks(
    cnaster_config: None, tmp_path: Path
) -> None:
    """The run's page is (a) the slide and the clones, (b) the profile under
    its key, (c) the tracks: `PANELS`, `truth_combined`'s order (PR-
    #715)."""
    import matplotlib.pyplot as plt
    from port.extensions.combined_figure import PANELS, combined_figure

    recorded, frame = _recorded(tmp_path)
    figure = combined_figure(recorded, frame)
    profile, tracks = figure.subfigs

    assert (
        tuple(
            panels_in_order(
                list(figure.texts),
                {
                    "clones": figure.get_axes()[-2:],
                    "profile": list(profile.axes),
                    "tracks": list(tracks.axes),
                },
            )
        )
        == PANELS
    )
    plt.close(figure)


@pytest.mark.infra
def test_a_spatial_page_is_cut_to_its_axes() -> None:
    """`plot_clones_spatial` on a tall 4 by 10 section: equal x and y scale,
    the tiles' box at the section's aspect to 1%, the key wrapped no wider
    than the tiles, and the page's content
    `FIT_MARGIN` from the head and sides and `STAMP_ROOM` from the foot to
    a pixel, so no band of white is left (PR- #715)."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from port.extensions.figure_style import FIT_MARGIN, STAMP_ROOM
    from port.patch.plotting.spatial import plot_clones_spatial

    rows, columns = np.meshgrid(np.arange(4.0), np.arange(10.0))
    coords = np.column_stack([rows.ravel(), columns.ravel()])
    assignment = pd.Series([f"clone {k % 5}" for k in range(len(coords))])
    figure = plot_clones_spatial(coords, assignment)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    (ax,) = figure.axes
    box = ax.get_window_extent(renderer)
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    content = figure.get_tightbbox(renderer)
    dpi = figure.dpi
    width, height = figure.get_size_inches()

    assert ax.get_aspect() == 1.0
    assert ax.get_legend().get_window_extent(renderer).width <= box.width + 1.0
    assert box.height / box.width == pytest.approx((y1 - y0) / (x1 - x0), rel=0.01)
    assert content.x0 == pytest.approx(FIT_MARGIN, abs=1.0 / dpi)
    assert width - content.x1 == pytest.approx(FIT_MARGIN, abs=1.0 / dpi)
    assert height - content.y1 == pytest.approx(FIT_MARGIN, abs=1.0 / dpi)
    assert content.y0 == pytest.approx(STAMP_ROOM, abs=1.0 / dpi)
    plt.close(figure)


@pytest.mark.infra
@pytest.mark.merge
def test_the_combined_page_s_spatial_panels_are_square_keyed_clear_and_in_order(
    cnaster_config: None, tmp_path: Path
) -> None:
    """On the rendered page, for a section 3 times as tall as wide: (a)'s
    slide and clone map each a square footprint (1 px) with square limits,
    the spots on the left and bottom axes, the frame -- the left and bottom
    spines alone -- within the spots' extent (1 px), the clone key clear of
    both; and (a)'s key, (b)'s rows and (c)'s tracks
    name the clones in one order, `clone_order`'s (PR- #715)."""
    import matplotlib.pyplot as plt
    from port.extensions.combined_figure import (
        clone_order,
        clone_symbol,
        combined_figure,
    )

    recorded, frame = _recorded(tmp_path, tall=3.0)
    figure = combined_figure(recorded, frame)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    slide, clones = figure.get_axes()[-2:]
    key = clones.get_legend().get_window_extent(renderer)

    spots = clones.collections[0].get_datalim(clones.transData)

    for ax in (slide, clones):
        box = ax.get_window_extent(renderer)
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        assert box.width == pytest.approx(box.height, abs=1.0)
        assert abs(x1 - x0) == pytest.approx(abs(y1 - y0))
        assert not key.overlaps(box)
        # NB the spots on the left and bottom axes, the room to the right.
        assert spots.x0 == pytest.approx(x0)
        assert spots.y0 == pytest.approx(y0)
        corners = ax.transData.transform([(spots.x0, spots.y0), (spots.x1, spots.y1)])
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        # NB along its length: a spine's extent across it takes its ticks.
        (left,) = (
            ax.spines["left"]
            .get_path()
            .transformed(ax.spines["left"].get_transform())
            .get_extents()
            .intervaly.reshape(1, 2)
        )
        (bottom,) = (
            ax.spines["bottom"]
            .get_path()
            .transformed(ax.spines["bottom"].get_transform())
            .get_extents()
            .intervalx.reshape(1, 2)
        )
        assert corners[0][1] - 1.0 <= left[0] <= left[1] <= corners[1][1] + 1.0
        assert corners[0][0] - 1.0 <= bottom[0] <= bottom[1] <= corners[1][0] + 1.0

    profile, tracks = figure.subfigs
    # NB tick labels run bottom to top.
    rows = [t.get_text() for t in profile.axes[1].get_yticklabels()][::-1]
    named = [
        t.get_text()
        for ax in tracks.axes
        for t in ax.texts
        if t.get_visible() and t.get_text().startswith("$m")
    ]
    ids = [
        c[len("clone") : -len(" A")]
        for c in recorded.profile.args[0].columns
        if c.endswith(" A")
    ]
    expected = [clone_symbol(str(k)) for k in clone_order(ids)]
    keyed = [t.get_text().split(", ")[0] for t in clones.get_legend().get_texts()]

    assert rows == expected
    assert named == expected
    assert keyed == [m for m in expected if m in keyed]
    plt.close(figure)
