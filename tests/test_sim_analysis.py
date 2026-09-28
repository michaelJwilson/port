"""#452: the truth figures of a `port.sim.draw` realization.

The figures are read off the written realization; what they must say is
checked against the draw that wrote it: the profile rows merge segments
without moving a breakpoint or changing a state, and each leaf's barcode is
the events on its path from `normal`.
"""

from __future__ import annotations

import itertools
from pathlib import Path

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

from tests.sim_fixtures import references
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
    """One PNG per figure, nonempty; what each shows is the two tests above."""
    written = plot(drawn.path)

    assert len(written) == len(PLOTS)
    assert all(Path(p).stat().st_size > 10_000 for p in written)
    assert np.unique([Path(p).name for p in written]).size == len(PLOTS)
