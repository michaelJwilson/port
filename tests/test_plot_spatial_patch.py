"""`port.patch.plotting.spatial` against `cnaster.plotting.plot_clones_spatial` (#309).

The patch changes one thing, a spot's area. What upstream asserts -- which
clone and which tumour proportion each spot shows, at which position -- is
compared spot by spot: every colour bitwise, every tile centred on the point
upstream draws. The tiles are then held to their own claim: `TILE` of the
pitch on a side, so a gap of `1 - TILE` between neighbours.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest


def _instance() -> tuple[np.ndarray, Any, np.ndarray]:
    rng = np.random.default_rng(23)
    rows, columns = np.unravel_index(np.arange(12 * 9), (12, 9))
    coords = np.column_stack([rows, columns]).astype(float)
    assignment = pd.Series(
        [f"clone {clone}" for clone in rng.integers(0, 3, rows.size)]
    )
    proportion = rng.uniform(0.2, 1.0, rows.size)
    proportion[::17] = np.nan

    return coords, assignment, proportion


def _upstream_by_spot(figure: Any, coords: np.ndarray) -> np.ndarray:
    """RGBA per spot from upstream's scatters, keyed back by position."""
    colours = np.full((coords.shape[0], 4), np.nan)
    lookup = {(x, -y): spot for spot, (x, y) in enumerate(coords)}

    for collection in figure.axes[0].collections:
        offsets = np.asarray(collection.get_offsets())
        faces = np.asarray(collection.get_facecolors())

        for offset, face in zip(
            offsets, np.broadcast_to(faces, (len(offsets), 4)), strict=True
        ):
            colours[lookup[tuple(offset)]] = face

    return colours


@pytest.mark.patch
@pytest.mark.parametrize("with_proportion", [False, True])
def test_every_spot_has_upstreams_colour_at_upstreams_point(
    with_proportion: bool,
) -> None:
    """Colour and opacity bitwise, centre exact, on 108 spots and 3 clones."""
    import matplotlib as mpl

    mpl.use("Agg")

    from cnaster.plotting import plot_clones_spatial as upstream
    from port.patch.plotting.spatial import TILE, pitch, plot_clones_spatial

    coords, assignment, proportion = _instance()
    tumour = proportion if with_proportion else None

    theirs = _upstream_by_spot(upstream(coords, assignment, tumour), coords)
    ours = plot_clones_spatial(coords, assignment, tumour)

    (tiles,) = ours.axes[0].collections
    faces = np.asarray(tiles.get_facecolors())
    corners = np.array([path.vertices[:4] for path in tiles.get_paths()])

    np.testing.assert_array_equal(faces, theirs)

    centres = corners.mean(axis=1)
    np.testing.assert_array_equal(
        centres, np.column_stack([coords[:, 0], -coords[:, 1]])
    )

    side = np.ptp(corners[:, :, 0], axis=1)
    np.testing.assert_allclose(side, TILE * pitch(coords), rtol=1e-12)
    assert pitch(coords) == 1.0


@pytest.mark.patch
def test_two_samples_are_offset_and_titled_as_upstream_does() -> None:
    """Sample 1 shifted right by sample 0's width plus 10, colours bitwise.

    Upstream lays samples side by side along `x`, each after the previous
    one's largest `x` plus 10, and titles the page with the sample names.
    The same 108 spots split into two samples by row parity.
    """
    import matplotlib as mpl

    mpl.use("Agg")

    from cnaster.plotting import plot_clones_spatial as upstream
    from port.patch.plotting.spatial import plot_clones_spatial

    coords, assignment, _ = _instance()
    samples = (np.arange(coords.shape[0]) % 2).astype(int)
    names = ["S0", "S1"]

    shifted = coords.copy()
    shifted[samples == 1, 0] += coords[samples == 0, 0].max() + 10

    theirs_figure = upstream(coords.copy(), assignment, None, names, samples)
    theirs = _upstream_by_spot(theirs_figure, shifted)
    ours = plot_clones_spatial(coords.copy(), assignment, None, names, samples)

    (tiles,) = ours.axes[0].collections
    centres = np.array([path.vertices[:4] for path in tiles.get_paths()]).mean(axis=1)

    np.testing.assert_array_equal(np.asarray(tiles.get_facecolors()), theirs)
    # NB a centre is recovered as the mean of four float corners, which is
    #    exact only where the half-side sums without round-off; realized 4.4e-16.
    np.testing.assert_allclose(
        centres, np.column_stack([shifted[:, 0], -shifted[:, 1]]), rtol=0, atol=1e-12
    )
    assert [text.get_text() for text in ours.axes[0].texts] == ["S0, S1"]
