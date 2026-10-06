"""#452: the truth figures of a `port.sim.draw` realization.

The figures are read off the written realization; what they must say is
checked against the draw that wrote it: the profile rows merge segments
without moving a breakpoint or changing a state, and each leaf's barcode is
the events on its path from `normal`.
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

from tests.test_sim_draw import _manifest


@pytest.fixture(scope="module")
def drawn(tmp_path_factory: pytest.TempPathFactory) -> Drawn:
    resources = references()
    if resources is None:
        pytest.skip("CalicoST's GRCh38_resources not found; set $PORT_GRCH38")
    return draw(
        _manifest("dev_tree"), tmp_path_factory.mktemp("qa"), resources=resources
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
    """A node's barcode has a 1 exactly at the events on its path; the founder's is the lead.

    Events are ordered by their time since `normal`, so every tumour clone's
    barcode starts with the trunk's events and `normal`'s is all zeros.
    """
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
    """#454: the dashed box is the slices' overlap, at `(x, -y)` like the spots.

    `dev_tree`'s second slice sits half an array to the right, so the shared
    region is the first slice's right half: every spot of either slice inside
    it lies within the box, and no spot more than half a spacing outside it.
    """
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
    manifest = _manifest("dev_tree", {"sample": {"realizations": 2}})
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
    """One PNG per figure and the page, nonempty; what each shows is the two tests above."""
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
    """`llncs`'s 122 mm by 193 mm, lettered (a) to (c), no text over `FONT_SIZE`,
    and every text and legend on the page to half a pixel."""
    from port.extensions.combined_figure import FONT_SIZE, TEXT_HEIGHT
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

    assert figure.get_size_inches()[0] * 25.4 == pytest.approx(122.0)
    assert figure.get_size_inches()[1] == pytest.approx(TEXT_HEIGHT)
    assert [panel.texts[-1].get_text() for panel in figure.subfigs] == [
        f"({k})" for k in "abc"
    ]
    assert max(t.get_fontsize() for t in texts) <= FONT_SIZE

    for artist in [*texts, *legends]:
        extent = artist.get_window_extent(renderer)
        assert extent.x0 >= page.x0 - 0.5, artist
        assert extent.x1 <= page.x1 + 0.5, artist
        assert extent.y0 >= page.y0 - 0.5, artist
        assert extent.y1 <= page.y1 + 0.5, artist


@pytest.mark.infra
@pytest.mark.merge
def test_the_genome_panels_share_one_left_and_one_right_edge(drawn: Drawn) -> None:
    """(b)'s key and rows and (c)'s tracks start and end at one x, so a
    chromosome boundary is at one place in both."""
    from port.sim.truth_figure import truth_combined_figure

    figure = truth_combined_figure(read(drawn.path))
    renderer = figure.canvas.get_renderer()
    _, profile, genomic = figure.subfigs
    axes = [*profile.axes, *genomic.axes]
    boxes = [ax.get_window_extent(renderer) for ax in axes]

    assert len(genomic.axes) == 2 * len(read(drawn.path).clones)
    for box in boxes:
        assert box.x0 == pytest.approx(boxes[0].x0, abs=0.5)
        assert box.x1 == pytest.approx(boxes[0].x1, abs=0.5)


@pytest.mark.infra
@pytest.mark.merge
def test_the_tree_spans_the_genome_panels_between_its_barcodes(drawn: Drawn) -> None:
    """(a)'s root barcode starts on (c)'s left edge and each leaf's ends on its
    right; no name runs into a barcode or past its node's side."""
    from port.sim.truth_figure import truth_combined_figure

    figure = truth_combined_figure(read(drawn.path))
    renderer = figure.canvas.get_renderer()
    tree_panel, _, genomic = figure.subfigs
    (tree_ax,) = tree_panel.axes
    track = genomic.axes[0].get_window_extent(renderer)
    barcodes = [t for t in tree_ax.texts if t.get_gid() == "barcode"]
    names = [t for t in tree_ax.texts if t.get_gid() == "name"]
    boxes = sorted(
        (t.get_window_extent(renderer) for t in barcodes), key=lambda b: b.x0
    )
    root, leaves = boxes[0], boxes[1:]

    assert len(barcodes) == len(names) == len(read(drawn.path).clones)
    assert root.x0 == pytest.approx(track.x0, abs=0.5)
    for leaf in leaves:
        assert leaf.x1 == pytest.approx(track.x1, abs=0.5)
    for name in names:
        box = name.get_window_extent(renderer)
        assert not any(box.overlaps(b) for b in boxes)

    # NB (c) names each clone with the barcode (a) gives it.
    headed = {
        t.get_text()
        for ax in genomic.axes
        for t in ax.texts
        if t.get_visible() and "(" in t.get_text()
    }
    assert headed == {
        f"{n.get_text()} ({b.get_text()})"
        for n, b in zip(
            sorted(names, key=lambda t: t.xy[1]),
            sorted(barcodes, key=lambda t: t.get_position()[1]),
            strict=True,
        )
    }


@pytest.mark.infra
@pytest.mark.merge
def test_the_truth_page_writes_byte_for_byte_at_its_size(
    drawn: Drawn, tmp_path: Path
) -> None:
    """Two writes are one file: no creation date (#452); its MediaBox is the page to 0.1 pt."""
    import re

    from port.sim.truth_figure import write_truth_combined

    r = read(drawn.path)
    first = write_truth_combined(r, tmp_path / "a.pdf").read_bytes()
    second = write_truth_combined(r, tmp_path / "b.pdf").read_bytes()
    box = re.search(rb"/MediaBox\s*\[\s*[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)", first)

    assert first == second
    assert box is not None
    assert float(box.group(1)) == pytest.approx(122.0 / 25.4 * 72.0, abs=0.1)
    assert float(box.group(2)) == pytest.approx(193.0 / 25.4 * 72.0, abs=0.1)
