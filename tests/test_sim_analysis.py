"""Truth figures of a `port.sim.draw` realization, checked against the draw (#452).

Profile rows keep the planted breakpoints and states; barcodes are the path events.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from port.sim.analysis import (
    PLOTS,
    STATISTICS,
    _runs,
    common_region,
    outline,
    plot,
    read,
    tree,
)
from port.sim.draw import Drawn, draw
from port.sim.fixtures import references
from port.sim.truth_figure import NAME_GAP

from tests.figure_checks import mirror_key_holds, panels_in_order
from tests.fixtures import draw_manifest


@pytest.fixture(scope="module")
def drawn(tmp_path_factory: pytest.TempPathFactory) -> Drawn:
    resources = references()
    if resources is None:
        pytest.skip("CalicoST's GRCh38_resources not found; set $PORT_GRCH38")
    return draw(
        draw_manifest("dev_tree"), tmp_path_factory.mktemp("qa"), resources=resources
    )


@pytest.mark.analytic
def test_merged_runs_tile_each_chromosome_with_the_planted_states(
    drawn: Drawn,
) -> None:
    """Runs cover every chromosome end to end; each profile row's state is its run's."""
    r = read(drawn.path)

    for clone in r.clones:
        runs = _runs(r.profile, clone)
        for chrom, length in enumerate(r.lengths, start=1):
            mine = [x for x in runs if x[0] == str(chrom)]
            assert mine[0][1] == 0
            assert mine[-1][2] == length
            assert all(a[2] == b[1] for a, b in itertools.pairwise(mine))
        for row in r.profile.itertuples(index=False):
            run = next(
                x for x in runs
                if x[0] == str(row.chr) and x[1] <= row.start and row.end <= x[2]
            )  # fmt: skip
            assert run[3:] == (
                int(getattr(row, f"{clone}_A_copy")),
                int(getattr(row, f"{clone}_B_copy")),
            )


@pytest.mark.analytic
def test_barcodes_set_the_bits_of_each_nodes_path_founder_first(drawn: Drawn) -> None:
    """A barcode has a 1 exactly at its path's events, trunk first, against the draw."""
    t = tree(read(drawn.path))

    assert set(t.barcode["normal"]) == {"0"}
    assert list(t.events["time"]) == sorted(t.events["time"])
    trunk = len(drawn.tree.edge_events["founder"])
    for leaf in drawn.tree.leaves:
        code = t.barcode[leaf]
        assert code.count("1") == len(drawn.tree.events(leaf))
        assert code[:trunk] == "1" * trunk
    assert len(set(t.barcode.values())) == len(t.barcode)
    assert all("::" in label for label in t.events["label"])


@pytest.mark.analytic
def test_the_shared_region_bounds_the_spots_both_slices_image(drawn: Drawn) -> None:
    """#454: the dashed box bounds the spots both slices image, at `(x, -y)`."""
    r = read(drawn.path)
    frames, origins, extent = [], [], None
    for k, sid in enumerate(drawn.sample_ids):
        spots = r.truth[r.truth["sample_id"] == sid]
        x = spots["y"].to_numpy() / 2.0
        y = spots["x"].to_numpy() * np.sqrt(3.0) / 2.0
        extent = np.array([x.max() - x.min(), y.max() - y.min()])
        origin = np.array(r.manifest["slice"][k]["offset"]) * extent
        origins.append(origin)
        frames.append(np.column_stack([x + origin[0], -(y + origin[1])]))
    assert extent is not None

    shared = common_region(origins, extent)
    assert shared is not None
    lower, upper = shared
    np.testing.assert_allclose(lower, [0.5 * extent[0], 0.0])
    np.testing.assert_allclose(upper, extent)
    (x0, y0), width, height = outline(lower, upper)

    for points in frames:
        inside = (points[:, 0] >= lower[0]) & (points[:, 0] <= upper[0])
        near = (points[:, 0] > lower[0] - 0.5) & (points[:, 0] < upper[0] + 0.5)
        boxed = (points[:, 0] > x0) & (points[:, 0] < x0 + width)
        boxed &= (points[:, 1] > y0) & (points[:, 1] < y0 + height)
        assert np.all(boxed[inside])
        assert not np.any(boxed[~near])


def _panels_of(figure: Any) -> list[tuple[Any, ...]]:
    """Each panel's size in inches, limits, title and dashed boxes, left to right."""
    from matplotlib.patches import Rectangle

    figure.canvas.draw()
    dpi = figure.dpi
    rows = []
    for ax in sorted(figure.axes, key=lambda a: a.get_position().x0):
        at = ax.get_window_extent()
        boxes = sorted(
            (p.get_xy(), p.get_width(), p.get_height())
            for p in ax.patches
            if isinstance(p, Rectangle) and p.get_linestyle() == "--"
        )
        rows.append((round(at.width / dpi, 2), round(at.height / dpi, 2),
                     ax.get_xlim(), ax.get_ylim(), ax.get_title(loc="left"), boxes))  # fmt: skip
    return rows


@pytest.mark.infra
def test_the_spatial_and_h_and_e_pages_share_one_format(
    drawn: Drawn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T- #791: spatial and H&E pages share one panel format; (c) spans (b) to 1 px."""
    import matplotlib.pyplot as plt
    from port.sim import analysis
    from port.sim.truth_figure import truth_combined_figure
    from port.studies.paper_figures import he_slices_figure

    r = read(drawn.path)
    caught: dict[str, Any] = {}

    def keep(fig: Any, path: Path, *, tight: bool = True, dpi: int = 150) -> Path:
        caught[path.name] = fig
        return path

    monkeypatch.setattr(analysis, "_save", keep)
    analysis.plot_spatial(r, Path("unwritten"))
    spatial = _panels_of(caught["spatial.png"])
    he = _panels_of(he_slices_figure(r))

    assert [row[4] for row in spatial] == [str(s) for s in drawn.sample_ids]
    assert he == spatial
    assert all(len(row[5]) == 1 for row in spatial)

    page = truth_combined_figure(r, spatial=True)
    page.canvas.draw()
    inset = [(ax.get_xlim(), ax.get_ylim(), ax.get_title(loc="left"))
             for ax in sorted(page.subfigs[2].axes, key=lambda a: a.get_position().x0)]  # fmt: skip
    assert inset == [row[2:5] for row in spatial]
    genome = page.subfigs[1].axes[-1].get_window_extent()
    row = sorted(
        (ax.get_window_extent() for ax in page.subfigs[2].axes), key=lambda b: b.x0
    )
    assert row[0].x0 == pytest.approx(genome.x0, abs=1.0)
    assert row[-1].x1 == pytest.approx(genome.x1, abs=1.0)
    for fig in (*caught.values(), page):
        plt.close(fig)
    plt.close("all")


@pytest.mark.infra
def test_the_spatial_variant_carries_the_phase_track_and_the_genomes_marks(
    drawn: Drawn,
) -> None:
    """T- #794: the spatial variant's phase track equals the flat one's, to 0.01 in."""
    import matplotlib.pyplot as plt
    from port.sim.truth_figure import truth_combined_figure

    r = read(drawn.path)
    pages = [truth_combined_figure(r, metric=True, spatial=s) for s in (False, True)]

    def phase(page: Any, panel: int) -> Any:
        (ax,) = [ax for ax in page.subfigs[panel].axes
                 if ax.get_ylabel() == "Switches / Mb"]  # fmt: skip
        return ax

    def last(page: Any, panel: int) -> Any:
        return min(page.subfigs[panel].axes, key=lambda ax: ax.get_position().y0)

    for page in pages:
        page.canvas.draw()
    flat, spatial = phase(pages[0], 2), phase(pages[1], 1)
    assert spatial is last(pages[1], 1)
    lines = [[np.c_[line.get_xdata(), line.get_ydata()] for line in ax.lines]
             for ax in (flat, spatial)]  # fmt: skip
    assert len(lines[1]) == len(lines[0]) > 0
    for ours, theirs in zip(*lines, strict=True):
        np.testing.assert_array_equal(ours, theirs)
    tall = [ax.get_window_extent().height / page.dpi
            for ax, page in zip((flat, spatial), pages, strict=True)]  # fmt: skip
    assert tall[1] == pytest.approx(tall[0], abs=0.01)
    bottom = last(pages[0], 2)
    np.testing.assert_array_equal(
        spatial.xaxis.get_minorticklocs(), bottom.xaxis.get_minorticklocs()
    )
    assert spatial.xaxis.get_minorticklocs().size > 0

    def names(ax: Any) -> list[str]:
        return [t.get_text() for t in ax.texts if t.get_gid() == "contig"]

    assert names(spatial) == names(bottom) != []
    for page in pages:
        plt.close(page)


@pytest.mark.analytic
def test_a_streamed_population_holds_each_statistics_mean_and_sd(
    tmp_path: Path,
) -> None:
    """Two realizations streamed: the running mean and sd are those of the two values."""
    from port.sim.analysis import Population, stream
    from port.sim.draw import draw

    resources = references()
    if resources is None:
        pytest.skip("CalicoST's GRCh38_resources not found; set $PORT_GRCH38")
    manifest = draw_manifest("dev_tree", {"sample": {"realizations": 2}})
    written = draw(manifest, tmp_path, resources=resources)

    seen = []
    population = Population.of({"switches_per_mb": STATISTICS["switches_per_mb"]})
    for r in stream(written.root):
        seen.append(STATISTICS["switches_per_mb"](r))
        population.add(r)

    m = population.moments["switches_per_mb"]
    assert m.n == 2
    np.testing.assert_allclose(m.mean, np.mean(seen), rtol=1e-12)
    np.testing.assert_allclose(m.sd, np.std(seen, ddof=1), rtol=1e-12)


@pytest.mark.smoke
def test_every_figure_is_written(drawn: Drawn) -> None:
    """One nonempty PNG per figure, plus the page (no content check)."""
    written = plot(drawn.path)

    assert len(written) == len(PLOTS) + 1
    assert all(Path(p).stat().st_size > 10_000 for p in written)
    assert np.unique([Path(p).name for p in written]).size == len(PLOTS) + 1
    assert Path(written[-1]).name == "truth_combined.pdf"


def _visible_texts(figure: Any) -> list[Any]:
    from matplotlib.text import Text

    return [t for t in figure.findobj(Text) if t.get_visible() and t.get_text().strip()]


@pytest.mark.infra
@pytest.mark.merge
def test_the_truth_page_is_combined_pdfs_page_with_everything_on_it(
    drawn: Drawn,
) -> None:
    """Page is the text block less `CAPTION_ROOM`; fonts and texts on it to 0.5 px (#743)."""
    from port.extensions.figure_style import (
        CAPTION_ROOM,
        PAPER_WIDTH,
        TEXT_HEIGHT,
        TRACK_FONT_SIZE,
    )
    from port.qa.combined_figure import FONT_SIZE
    from port.sim.truth_figure import truth_combined_figure

    figure = truth_combined_figure(read(drawn.path))
    renderer = figure.canvas.get_renderer()
    page = figure.bbox
    texts = _visible_texts(figure)
    legends = [
        legend
        for panel in figure.subfigs
        for legend in [*panel.legends, *(ax.get_legend() for ax in panel.axes)]
        if legend is not None
    ]

    assert figure.get_size_inches()[0] == pytest.approx(PAPER_WIDTH)
    assert figure.get_size_inches()[1] == pytest.approx(TEXT_HEIGHT - CAPTION_ROOM)
    assert [panel.texts[-1].get_text() for panel in figure.subfigs] == [
        f"({k})" for k in "abc"
    ]
    assert max(t.get_fontsize() for t in texts) <= FONT_SIZE
    assert {t.get_fontsize() for t in texts} <= {FONT_SIZE, TRACK_FONT_SIZE}

    for artist in [*texts, *legends]:
        extent = artist.get_window_extent(renderer)
        assert extent.x0 >= page.x0 - 0.5, artist
        assert extent.x1 <= page.x1 + 0.5, artist
        assert extent.y0 >= page.y0 - 0.5, artist
        assert extent.y1 <= page.y1 + 0.5, artist


@pytest.mark.infra
@pytest.mark.merge
def test_the_genome_panels_share_one_left_and_one_right_edge(drawn: Drawn) -> None:
    """(b)'s and (c)'s axes share one left and right edge, to 0.5 px."""
    from port.sim.truth_figure import truth_combined_figure

    figure = truth_combined_figure(read(drawn.path))
    renderer = figure.canvas.get_renderer()
    _, profile, genomic = figure.subfigs
    axes = [*profile.axes, *genomic.axes]
    boxes = [ax.get_window_extent(renderer) for ax in axes]

    # NB a pair per tumour clone, plus the phase track in the normal's place (#745)
    assert len(genomic.axes) == 2 * (len(read(drawn.path).clones) - 1) + 1
    for box in boxes:
        assert box.x0 == pytest.approx(boxes[0].x0, abs=0.5)
        assert box.x1 == pytest.approx(boxes[0].x1, abs=0.5)


@pytest.mark.infra
@pytest.mark.merge
def test_the_tree_spans_the_genome_panels_between_its_barcodes(drawn: Drawn) -> None:
    """(a) spans (c): root `NAME_GAP` in, barcodes on the right edge, to 0.5 px."""
    from port.sim.truth_figure import truth_combined_figure

    figure = truth_combined_figure(read(drawn.path))
    renderer = figure.canvas.get_renderer()
    tree_panel, _, genomic = figure.subfigs
    (tree_ax,) = tree_panel.axes
    track = genomic.axes[0].get_window_extent(renderer)
    barcodes = [t for t in tree_ax.texts if t.get_gid() == "barcode"]
    names = [t for t in tree_ax.texts if t.get_gid() == "name"]
    boxes = [t.get_window_extent(renderer) for t in barcodes]
    tree = tree_ax.get_window_extent(renderer)

    assert len(barcodes) == len(names) == len(read(drawn.path).clones)
    assert tree.x0 == pytest.approx(track.x0, abs=0.5)
    assert tree_ax.transData.transform((0.0, 0.0))[0] == pytest.approx(
        track.x0 + NAME_GAP * figure.dpi / 72.0, abs=0.5
    )
    for leaf in boxes:
        assert leaf.x1 == pytest.approx(track.x1, abs=0.5)
    for name in names:
        box = name.get_window_extent(renderer)
        assert not any(box.overlaps(b) for b in boxes)

    headed = {
        t.get_text()
        for ax in genomic.axes
        for t in ax.texts
        if t.get_visible() and "(" in t.get_text()
    }
    # NB the normal clone's pair gives way to the phase track (#745)
    assert headed == {
        f"{n.get_text()} ({b.get_text()})"
        for n, b in zip(
            sorted(names, key=lambda t: t.xy[1]),
            sorted(barcodes, key=lambda t: t.get_position()[1]),
            strict=True,
        )
        if n.get_text() != "$m_N$"
    }


@pytest.mark.infra
@pytest.mark.merge
def test_the_truth_page_writes_byte_for_byte_at_its_size(
    drawn: Drawn, tmp_path: Path
) -> None:
    """Two writes are byte-equal (#452); MediaBox is the page, to 0.1 pt (T- #733, #740)."""
    import re

    from port.sim.truth_figure import write_truth_combined

    r = read(drawn.path)
    first = write_truth_combined(r, tmp_path / "a.pdf").read_bytes()
    second = write_truth_combined(r, tmp_path / "b.pdf").read_bytes()
    box = re.search(rb"/MediaBox\s*\[\s*[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)", first)

    assert first == second
    assert box is not None
    assert float(box.group(1)) == pytest.approx(468.31 / 72.27 * 72.0, abs=0.1)
    assert float(box.group(2)) == pytest.approx((590.99 / 72.27 - 1.0) * 72.0, abs=0.1)


@pytest.fixture(scope="module")
def dense(tmp_path_factory: pytest.TempPathFactory) -> Drawn:
    """`dev_tree_1s_dense`'s tree, r0 (`33e3471e`)'s 64 events, on a 20 x 20 array."""
    resources = references()
    if resources is None:
        pytest.skip("CalicoST's GRCh38_resources not found; set $PORT_GRCH38")
    return draw(
        draw_manifest("dev_tree_1s_dense"),
        tmp_path_factory.mktemp("dense"),
        resources=resources,
    )


@pytest.mark.infra
def test_a_barcode_over_10_bits_keeps_4_bits_at_each_end() -> None:
    """`shown` keeps up to 10 bits whole, else 4 bits each side of "..." (PR- #701)."""
    from port.sim.analysis import BARCODE_SHOWN, MANY_EVENTS, shown

    assert (MANY_EVENTS, BARCODE_SHOWN) == (10, 8)
    assert shown("10110") == "10110"
    assert shown("10110100") == "10110100"
    assert shown("101101001") == "101101001"
    assert shown("1011010010") == "1011010010"
    assert shown("1100" + "0" * 56 + "0011") == "1100\N{HORIZONTAL ELLIPSIS}0011"
    assert shown("10110100101") == "1011\N{HORIZONTAL ELLIPSIS}0101"


def _panels(r: Any) -> tuple[Any, Any, Any]:
    from port.sim.truth_figure import truth_combined_figure

    figure = truth_combined_figure(r)
    tree_panel, _, genomic = figure.subfigs
    (tree_ax,) = tree_panel.axes
    return figure, tree_ax, genomic


def _headed_in_order(genomic: Any) -> list[str]:
    """(c)'s headers, top to bottom on the page."""
    return [
        t.get_text()
        for ax in sorted(genomic.axes, key=lambda ax: -ax.get_position().y1)
        for t in ax.texts
        if t.get_visible() and "(" in t.get_text()
    ]


def _headed(genomic: Any) -> set[str]:
    return {
        t.get_text()
        for ax in genomic.axes
        for t in ax.texts
        if t.get_visible() and "(" in t.get_text()
    }


def _a_is_the_tree(r: Any) -> Any:
    """Assert (a) equals `draw_tree(edges=True)` and (c) heads each clone with its barcode."""
    import matplotlib.pyplot as plt
    from port.qa.combined_figure import FONT_SIZE
    from port.sim.analysis import draw_tree, shown
    from port.sim.truth_figure import _symbol

    t = tree(r)
    _, tree_ax, genomic = _panels(r)
    figure, ax = plt.subplots()
    draw_tree(ax, r, event_size=FONT_SIZE, node_size=FONT_SIZE, dot=18.0,
              name=_symbol(r), ancestors=False, edges=True)  # fmt: skip

    assert [x.get_text() for x in tree_ax.texts] == [x.get_text() for x in ax.texts]
    assert len(tree_ax.lines) == len(ax.lines) > 0
    assert len(tree_ax.collections) == len(ax.collections)
    assert {x.get_text() for x in tree_ax.texts if x.get_gid() == "barcode"} == {
        t.barcode[c] for c in r.clones
    }
    symbol = _symbol(r)
    # NB the normal clone's pair gives way to the phase track, unheaded (#745)
    assert _headed(genomic) == {
        f"{symbol(c)} ({shown(t.barcode[c])})" for c in r.clones if c != "normal"
    }
    plt.close(figure)
    return tree_ax


@pytest.mark.infra
@pytest.mark.merge
def test_at_10_events_or_fewer_a_is_the_tree(drawn: Drawn) -> None:
    """At <= `MANY_EVENTS`, (a) is `draw_tree`'s tree with events (PR- #701)."""
    from port.sim.analysis import MANY_EVENTS

    r = read(drawn.path)
    t = tree(r)
    tree_ax = _a_is_the_tree(r)

    assert len(t.events) <= MANY_EVENTS
    assert set(t.events["label"]) <= {x.get_text() for x in tree_ax.texts}


def _marks(ax: Any) -> list[Any]:
    """`ax`'s visible minor tick marks within its x limits."""
    lo, hi = ax.get_xlim()
    return [t for t in ax.xaxis.get_minor_ticks(len(ax.xaxis.get_minorticklocs()))
            if t.tick1line.get_visible() and lo <= t.get_loc() <= hi]  # fmt: skip


@pytest.mark.infra
@pytest.mark.merge
def test_only_the_last_track_marks_every_10_mb_at_paper_width(drawn: Drawn) -> None:
    """Only the last track marks every 10 Mb, 2 pt x 0.5 pt (PR- #701, PR- #715)."""
    import matplotlib.pyplot as plt
    from matplotlib.markers import TICKDOWN

    r = read(drawn.path)
    expected = int(np.sum(np.asarray(r.lengths) // 10_000_000))
    figure, _, genomic = _panels(r)
    figure.canvas.draw()
    *others, last = genomic.axes
    marks = _marks(last)

    assert expected > 0
    assert len(marks) == expected
    assert all(t.tick1line.get_markersize() == 2.0 for t in marks)
    assert all(t.tick1line.get_markeredgewidth() == 0.5 for t in marks)
    assert all(t.tick1line.get_marker() == TICKDOWN for t in marks)
    for ax in (*figure.subfigs[1].axes, *others):
        assert _marks(ax) == []
    plt.close(figure)


@pytest.mark.infra
@pytest.mark.merge
@pytest.mark.parametrize("which", ["drawn", "dense"])
def test_no_mb_label_is_drawn_and_every_contig_is_named_once_clear(
    which: str, request: pytest.FixtureRequest
) -> None:
    """No Mb labels; each contig named once under the last track (#743, #745, PR- #715)."""
    import re

    import matplotlib.pyplot as plt
    from port.sim.analysis import binned_axis

    r = read(request.getfixturevalue(which).path)
    figure, _, genomic = _panels(r)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axes = [ax for panel in figure.subfigs for ax in panel.axes]
    last = genomic.axes[-1]

    for ax in axes:
        ticks = ax.xaxis.get_minor_ticks(len(ax.xaxis.get_minorticklocs()))
        assert not [t for t in ticks if t.label1.get_visible() and t.label1.get_text()]
    # NB a `cnaster` contig name is `chr` plus its name
    assert not [t for t in _visible_texts(figure)
                if re.fullmatch(r"chr\w+", t.get_text())]  # fmt: skip
    names = sorted(
        (t for t in last.texts if t.get_gid() == "contig" and t.get_visible()),
        key=lambda t: float(t.get_position()[0]),
    )
    edges = binned_axis(r).edges
    spans = [(a, b) for a, b in itertools.pairwise(edges.tolist()) if b > a]

    assert len(names) == len(spans)
    for text, (a, _b) in zip(names, spans, strict=True):
        assert float(text.get_position()[0]) == pytest.approx(a)
        assert text.get_horizontalalignment() == "left"
        assert re.fullmatch(r"\w+", text.get_text())
    from port.extensions.genomic_axis import CONTIG_PAD, STAGGERED

    boxes = [t.get_window_extent(renderer) for t in names]
    # NB rows step half a line (#743): neighbours may meet within `CONTIG_PAD`
    pad = CONTIG_PAD * figure.dpi / 72.0
    for i, j in itertools.combinations(range(len(boxes)), 2):
        if boxes[i].y0 == pytest.approx(boxes[j].y0):
            assert not boxes[i].overlaps(boxes[j]), (
                names[i].get_text(),
                names[j].get_text(),
            )
        elif {names[i].get_text(), names[j].get_text()} <= set(STAGGERED):
            # NB 19-22 zigzag (#745): the lower clears the upper by half a height
            down = min(boxes[i].y1, boxes[j].y1) - max(boxes[i].y0, boxes[j].y0)
            assert down <= boxes[i].height / 2, (
                names[i].get_text(),
                names[j].get_text(),
            )
        else:
            across = min(boxes[i].x1, boxes[j].x1) - max(boxes[i].x0, boxes[j].x0)
            assert across <= pad, (names[i].get_text(), names[j].get_text())
    assert [t.get_text() for t in last.texts if t.get_gid() == "contig-axis"] == ["chr"]
    # NB barcodes and copy numbers are digits too; only contig names elsewhere
    assert all(
        t.get_gid() in ("contig", "contig-axis")
        for ax in [*figure.subfigs[1].axes[1:], *genomic.axes]
        for t in ax.texts
        if t.get_visible() and re.fullmatch(r"\d+|X|Y", t.get_text())
    )
    plt.close(figure)


@pytest.mark.infra
@pytest.mark.merge
def test_above_10_events_a_is_the_tree_without_events(dense: Drawn) -> None:
    """Above `MANY_EVENTS`, (a) has no events and (c) cuts barcodes by `shown` (PR- #701)."""
    from port.sim.analysis import MANY_EVENTS, shown

    r = read(dense.path)
    t = tree(r)
    tree_ax = _a_is_the_tree(r)

    assert len(t.events) > MANY_EVENTS
    assert {x.get_gid() for x in tree_ax.texts} == {"name", "barcode"}
    assert not set(t.events["label"]) & {x.get_text() for x in tree_ax.texts}
    assert all(
        len(x.get_text()) == len(t.events)
        for x in tree_ax.texts
        if x.get_gid() == "barcode"
    )
    assert all(
        len(shown(t.barcode[c])) == 9
        and shown(t.barcode[c])[4] == "\N{HORIZONTAL ELLIPSIS}"
        for c in r.clones
    )


def _leaves(ax: Any) -> list[tuple[str, tuple[float, ...]]]:
    """A tree axis's named nodes top to bottom, each with its dot's colour."""
    dots = {
        tuple(np.round(c.get_offsets()[0], 6)): tuple(c.get_facecolor()[0])
        for c in ax.collections
    }
    names = sorted(
        (x for x in ax.texts if x.get_gid() == "name"), key=lambda x: -x.xy[1]
    )
    return [(x.get_text(), dots[tuple(np.round(x.xy, 6))]) for x in names]


@pytest.mark.infra
@pytest.mark.merge
@pytest.mark.parametrize("fixture", ["drawn", "dense"])
def test_clones_read_n_1_2_down_the_tree_and_alike_in_every_truth_figure(
    fixture: str, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clones read $m_N$, $m_1$, ... top down and match by colour in every figure (PR- #701)."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from port.qa.combined_figure import clone_symbol
    from port.sim import analysis
    from port.sim.truth_figure import simulated_tree_figure

    r = read(request.getfixturevalue(fixture).path)
    symbols = [r"$m_N$", *(rf"$m_{k}$" for k in range(1, len(r.clones)))]
    figure, tree_ax, genomic = _panels(r)
    figure.canvas.draw()
    leaves = _leaves(tree_ax)
    colour = dict(leaves)
    rows = figure.subfigs[1].axes[-1].get_yticklabels()

    assert [name for name, _ in leaves] == symbols
    assert [x.get_text() for x in rows][::-1] == symbols
    # NB the phase track, unheaded, replaces the normal clone's pair (#745)
    assert [h.split(" (")[0] for h in _headed_in_order(genomic)] == symbols[1:]
    tree_figure = simulated_tree_figure(r)
    assert _leaves(tree_figure.axes[0]) == leaves
    plt.close(tree_figure)
    plt.close(figure)

    caught: dict[str, Any] = {}

    def keep(fig: Any, path: Path, *, tight: bool = True, dpi: int = 150) -> Path:
        caught[path.name] = fig
        return path

    monkeypatch.setattr(analysis, "_save", keep)
    for plotter in (analysis.plot_spatial, analysis.plot_clone_profiles,
                    analysis.plot_clones_genomic_truth):  # fmt: skip
        plotter(r, Path("unwritten"))

    (key,) = [ax.get_legend() for ax in caught["spatial.png"].axes if ax.get_legend()]
    assert [t.get_text() for t in key.get_texts()] == [
        symbols[r.clones.index(c)] for c in r.clones if (r.truth["labels"] == c).any()
    ]
    for text, handle in zip(key.get_texts(), key.legend_handles, strict=True):
        assert mcolors.to_rgba(handle.get_color()) == colour[text.get_text()]
    profile = caught["clone_profiles.png"].axes[0].get_yticklabels()
    assert [clone_symbol(x.get_text()) for x in profile][::-1] == symbols
    named = [x for ax in caught["clones_genomic.png"].axes for x in ax.texts
             if x.get_text().startswith("Clone")]  # fmt: skip
    spots = [x for ax in caught["clones_genomic.png"].axes for x in ax.texts
             if x.get_text().endswith("snp-umis")]  # fmt: skip
    assert [clone_symbol(x.get_text()) for x in named] == symbols
    for k, clone in enumerate(r.clones):
        assert colour[symbols[k]] == mcolors.to_rgba(
            analysis.clone_colour(clone, r.clones)
        )
        count = int((r.truth["labels"] == clone).sum())
        assert spots[k].get_text().startswith(f"{count:_} spots")
    for fig in caught.values():
        plt.close(fig)


@pytest.mark.infra
@pytest.mark.merge
def test_the_tree_s_edges_carry_events_up_to_10_and_none_above(
    drawn: Drawn, dense: Drawn
) -> None:
    """`draw_tree` puts events on edges at <= `MANY_EVENTS`, none above (PR- #701)."""
    import matplotlib.pyplot as plt
    from port.sim.analysis import MANY_EVENTS, draw_tree

    for fixture, many in ((drawn, False), (dense, True)):
        r = read(fixture.path)
        t = tree(r)
        figure, ax = plt.subplots()
        draw_tree(ax, r)
        labels = {x.get_text() for x in ax.texts} & set(t.events["label"])
        edges = len(ax.lines) - (0 if many else len(t.events))

        assert (len(t.events) > MANY_EVENTS) is many
        assert labels == (set() if many else set(t.events["label"]))
        assert edges == len(t.parent)
        plt.close(figure)


@pytest.mark.infra
@pytest.mark.merge
def test_the_mirror_key_starts_on_b_s_left_edge_and_is_labelled_on_its_right(
    drawn: Drawn,
) -> None:
    """(b)'s mirror key: on the left edge, labelled right, to 0.5 px (PR- #715)."""

    figure, _, _ = _panels(read(drawn.path))
    _, profile, _ = figure.subfigs
    legend_ax, profile_ax = profile.axes[:2]
    mirror_key_holds(legend_ax, profile_ax)


@pytest.mark.infra
@pytest.mark.merge
def test_truth_combined_reads_clones_profile_tracks(drawn: Drawn) -> None:
    """The truth page's panels follow `PANELS`, the run page's order (PR- #715)."""
    import matplotlib.pyplot as plt
    from port.qa.combined_figure import PANELS

    figure, tree_ax, genomic = _panels(read(drawn.path))
    _, profile, _ = figure.subfigs
    letters = [t for panel in figure.subfigs for t in panel.texts]

    assert (
        tuple(
            panels_in_order(
                letters,
                {
                    "clones": [tree_ax],
                    "profile": list(profile.axes),
                    "tracks": list(genomic.axes),
                },
            )
        )
        == PANELS
    )
    plt.close(figure)
