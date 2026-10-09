"""`port.extensions.combined_figure`: two figures from a run (#309, #339).

Subfigure tracks match the standalone `clones_genomic` page bitwise; recording wrappers
call through and are removed on exit. Page-layout checks are `smoke`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cnaster.scripts.run_cnaster as script
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import port.patch.plot_genomic as genomic
import port.patch.plotting as spatial
import pytest
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.text import Text
from port.extensions.combined_figure import (
    FONT_SIZE,
    HE_PALETTE,
    LABEL_GAP,
    NAME_INSET,
    PANELS,
    Call,
    clone_order,
    clone_symbol,
    combined_figure,
    genomic_figure,
    plot_clones_genomic_he,
    recording,
    spatial_figure,
)
from port.extensions.figure_style import (
    CAPTION_ROOM,
    FIT_MARGIN,
    PAPER_WIDTH,
    STAMP_ROOM,
    TEXT_HEIGHT,
    TRACK_FONT_SIZE,
    page_size,
)
from port.patch.plot_copy_number_profile import (
    HATCH_ANGLE,
    HATCH_LINEWIDTH,
    HATCH_SPACING,
)
from port.patch.plot_genomic import plot_clones_genomic
from port.patch.plotting.spatial import plot_clones_spatial, spot_colours
from port.patch.utils import write_fig
from port.sim.inputs import written_config

from tests.adapters import drawn
from tests.conftest import SHIPPED_EM_FTOL, cnaster_test_config
from tests.figure_checks import (
    genomic_plot_arguments,
    mirror_key_holds,
    panels_in_order,
    recorded_combined_calls,
)


@pytest.mark.patch
def test_a_subfigure_draws_what_the_standalone_page_draws(cnaster_config: None) -> None:
    """Every array of (a), bitwise, against the page `clones_genomic.pdf` is."""

    mpl.use("Agg")

    arguments, keywords = genomic_plot_arguments()
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

    mpl.use("Agg")

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


def _figures(tmp_path: Path) -> tuple[Any, Any]:
    mpl.use("Agg")

    recorded, frame = recorded_combined_calls(tmp_path)
    genomic, spatial = genomic_figure(recorded), spatial_figure(recorded, frame)

    for figure in (genomic, spatial):
        figure.canvas.draw()

    return genomic, spatial


def _texts(figure: Any) -> list[Any]:
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
    """Text-column width, genomic height to 0.005 in, letters (a)/(b), fonts at `FONT_SIZE` (#743)."""

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
    """Each PDF's MediaBox is its figure size to 0.1 pt; all text on the page to 0.5 px (#339)."""

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
    """(a) and (b) share left/right edges to 1.5 px; labels on the `NAME_INSET` column (PR- #701)."""

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
    """(a) slide and (b) clones are equal squares at one scale to 1%; key and margins to 2 px (T- #740)."""

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
    """Clones decoding alike merge under "integer" labels (#745) and stay two under "continuous" (#344)."""

    mpl.use("Agg")

    recorded, frame = recorded_combined_calls(tmp_path)
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


@pytest.mark.bug
def test_the_figure_merges_clones_at_the_runs_agreement(tmp_path: Path) -> None:
    """Two clones that agree at 23 of 24 bins (0.958) stay two at the default
    0.99 and are one at a configured `merge_agreement` of 0.9, as
    the run's `/integer_clones` merges (#749 WP0, T- #817). Before, the figure
    merged at the default whatever the run's configuration said."""

    mpl.use("Agg")

    recorded, frame = recorded_combined_calls(tmp_path)
    assert recorded.profile is not None
    df_cnv = recorded.profile.args[0].copy()
    for column in ("A", "B"):
        df_cnv[f"clone2 {column}"] = df_cnv[f"clone1 {column}"]
    df_cnv.loc[0, "clone2 A"] = df_cnv.loc[0, "clone1 A"] + 1
    assert len(df_cnv) == 24
    recorded.profile = Call((df_cnv,), {})

    def keyed(agreement: float) -> int:
        document = cnaster_test_config(tmp_path, SHIPPED_EM_FTOL, 100)
        document["int_copy_num"] = {
            **document.get("int_copy_num", {}),
            "merge_agreement": agreement,
        }
        with written_config(document):
            figure = spatial_figure(recorded, frame)
        return len(figure.axes[1].get_legend().get_texts())

    assert keyed(0.99) == 3
    assert keyed(0.9) == 2


@pytest.mark.infra
# NB too specific for every change (#403); runs where this module or the lock changes.
@pytest.mark.deprecate
def test_the_combined_page_is_the_two_figures_stacked(
    cnaster_config: None, tmp_path: Path
) -> None:
    """One page, text block less `CAPTION_ROOM` to 0.005 in, (a)-(c); (a) placed as in the spatial figure to 1 px (#745)."""

    recorded, frame = recorded_combined_calls(tmp_path)
    combined = combined_figure(recorded, frame)
    spatial = spatial_figure(recorded, frame)

    assert combined.get_size_inches()[0] == pytest.approx(PAPER_WIDTH)
    assert combined.get_size_inches()[1] == pytest.approx(
        TEXT_HEIGHT - CAPTION_ROOM, abs=0.005
    )
    assert sorted(t.get_text() for t in combined.texts) == ["(a)", "(b)", "(c)"]

    renderer = combined.canvas.get_renderer()
    placed = combined.get_axes()[-2:]
    genomic = [ax for ax in combined.get_axes()[:-2] if ax.axison]
    left = min(ax.get_window_extent(renderer).x0 for ax in genomic)
    shifts = []
    for new, old in zip(placed, spatial.get_axes()[:2], strict=True):
        here = new.get_window_extent(renderer)
        there = old.get_window_extent(spatial.canvas.get_renderer())
        head = combined.bbox.height - spatial.bbox.height
        shifts.append(here.x0 - there.x0)
        np.testing.assert_allclose(
            (here.y0 - head, here.width, here.height),
            (there.y0, there.width, there.height),
            atol=1.0,
        )
    assert shifts[0] == pytest.approx(shifts[1], abs=1.0)
    assert placed[0].get_window_extent(renderer).x0 == pytest.approx(left, abs=1.0)

    plt.close(combined)
    plt.close(spatial)


@pytest.mark.analytic
def test_the_hatch_stripes_are_one_width() -> None:
    """B's lines are half the hatch period apart: 2.065 pt at 0.10 in and 35 degrees."""

    period = 72.0 * HATCH_SPACING * np.sin(np.radians(HATCH_ANGLE))

    assert pytest.approx(period - HATCH_LINEWIDTH) == HATCH_LINEWIDTH
    assert pytest.approx(2.0649, abs=1e-4) == HATCH_LINEWIDTH


@pytest.mark.infra
@pytest.mark.merge
def test_the_combined_page_reads_clones_profile_tracks(
    cnaster_config: None, tmp_path: Path
) -> None:
    """The run's page is (a) slide and clones, (b) profile, (c) tracks, in `PANELS` order (PR- #715)."""

    recorded, frame = recorded_combined_calls(tmp_path)
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
    """`plot_clones_spatial` on a 4x10 section: equal scale, box aspect to 1%, margins to 1 px (PR- #715)."""

    mpl.use("Agg")

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
    """3:1 section: square footprints to 1 px, frame within the spots, one clone order across panels (PR- #715)."""

    recorded, frame = recorded_combined_calls(tmp_path, tall=3.0)
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


@pytest.mark.oracle
def test_each_h_and_e_class_is_its_spots_pseudobulk(
    cnaster_config: None, tmp_path: Path
) -> None:
    """`plot_clones_genomic_he`: class RDR equals summed counts over summed baseline to 1e-12 vs NumPy (T- #771)."""

    mpl.use("Agg")

    recorded, _ = recorded_combined_calls(tmp_path)
    counts, baseline = recorded.genomic.args[1][:, 0, :], recorded.genomic.args[2]
    classes = np.array([1, 1, 2, 2, 2, 3, 4, 4, 1])
    figure = plot_clones_genomic_he(recorded, classes)
    tracks = [ax for ax in figure.get_axes() if ax.get_visible()]

    assert len(tracks) == 8
    for k, label in enumerate((1, 2, 3, 4)):
        spots = np.flatnonzero(classes == label)
        expected = counts[:, spots].sum(axis=1) / baseline[:, spots].sum(axis=1)
        points = next(
            c for c in tracks[2 * k].collections if isinstance(c, PathCollection)
        )
        np.testing.assert_allclose(
            np.asarray(points.get_offsets())[:, 1], expected, rtol=1e-12
        )
        named = [t.get_text() for t in tracks[2 * k].texts if t.get_rotation() == 0.0]
        assert named[0] == f"H&E {label} ({spots.size} spots)"


@pytest.mark.smoke
@pytest.mark.merge
def test_the_h_and_e_page_tiles_each_spot_by_its_class(
    cnaster_config: None, tmp_path: Path
) -> None:
    """`spatial_figure(he_labels=)`: (b) tiles spots in `HE_PALETTE`, keyed `H&E 1..4` (T- #771)."""

    mpl.use("Agg")

    recorded, frame = recorded_combined_calls(tmp_path)
    classes = np.array([1, 1, 2, 2, 2, 3, 4, 4, 1])
    figure = spatial_figure(recorded, frame, he_labels=classes)
    tiles = next(c for c in figure.axes[1].collections if isinstance(c, PolyCollection))
    _, _, palette = spot_colours(
        pd.Series([f"H&E {c}" for c in (1, 2, 3, 4)]), palette=HE_PALETTE
    )
    _, _, clones = spot_colours(pd.Series(["clone 0", "clone 1", "clone 2"]))
    expected = np.array([mcolors.to_rgb(palette[c - 1]) for c in classes])

    np.testing.assert_allclose(np.asarray(tiles.get_facecolor())[:, :3], expected)
    assert not {mcolors.to_hex(c) for c in palette} & {
        mcolors.to_hex(c) for c in clones
    }
    key = figure.axes[1].get_legend()
    assert [t.get_text() for t in key.get_texts()] == [f"H&E {c}" for c in (1, 2, 3, 4)]
