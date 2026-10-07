"""The mock H&E slide, read back the way `run_cnaster` reads one (#309).

`cnaster.he.get_he_image` joins each spot to its nearest pixel. The slide is
drawn as the transpose of the lattice, so a slide written the obvious way
round would put every spot on another spot's tissue and still load without a
warning; the referee is the planted labelling each pixel was stained from.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from port.sim.he_slide import mock_he, write_he_slide
from port.sim.truth import clone_bands

LATTICE = (20, 16)
N_CLONES = 4


def _read(tmp_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    from cnaster.he import get_he_image

    labels = clone_bands(*LATTICE, N_CLONES)
    write_he_slide(mock_he(labels, LATTICE, seed=5), tmp_path)

    rows, columns = np.unravel_index(np.arange(labels.size), LATTICE)
    # NB `tissue_positions.csv` as `python/port/sim/inputs.py` writes it.
    positions = pd.DataFrame(
        {
            "barcode": [f"BC{spot}" for spot in range(labels.size)],
            "x": rows,
            "y": columns,
        }
    )

    return get_he_image(str(tmp_path), pos=positions), labels


@pytest.mark.end2end
def test_each_spot_reads_its_own_clone_and_darker_away_from_normal(
    tmp_path: Path,
) -> None:
    """Every spot joins a pixel of its own clone; mean gray falls with clone.

    Exact: 320 of 320 spots. The slide's own per-pixel clone, looked up at
    the pixel `get_he_image` matched, is the planted label. And the per-clone
    mean of the `gray` it computes is strictly decreasing, normal first,
    which is what its percentile `label` -- and `run_cnaster`'s `he_label`
    refinement -- takes a slide to mean. Realized 0.69, 0.56, 0.45, 0.37.
    """
    from cnaster.he import get_he_image

    frame, labels = _read(tmp_path)
    slide = mock_he(labels, LATTICE, seed=5)
    pixels = get_he_image(str(tmp_path), pos=None)

    # NB `dist` is to the matched pixel; recover it from `x, y`.
    scale = slide.scalefactor
    row = np.rint(frame["y"].to_numpy() * scale).astype(int)
    column = np.rint(frame["x"].to_numpy() * scale).astype(int)

    assert pixels.shape[0] == slide.clone.size
    np.testing.assert_array_equal(slide.clone[row, column], labels)
    assert np.all(frame["dist"].to_numpy() == 0.0)

    gray = frame.assign(clone=labels).groupby("clone")["gray"].mean().to_numpy()

    assert np.all(np.diff(gray) < 0.0), f"mean gray by clone: {gray}"


@pytest.mark.bug
def test_the_brightest_pixel_takes_a_label_past_num_labels(tmp_path: Path) -> None:
    """`get_he_image(num_labels=4)` returns a fifth label, on one pixel.

    `he.py:112` bins with `np.digitize` against the 0th to 100th
    percentiles, whose last edge is the maximum; `digitize` puts a value
    equal to the last edge past it, so the brightest pixel is labelled
    `num_labels + 1`. On the spots it is a label nobody asked for, and
    `run_cnaster` factorizes it into an initial clone of its own.
    """
    from cnaster.he import get_he_image

    _read(tmp_path)
    pixels = get_he_image(str(tmp_path), pos=None, num_labels=4)
    labels = pixels["label"].to_numpy()

    assert labels.max() == 5
    assert np.sum(labels == 5) == np.sum(pixels["gray"] == pixels["gray"].max())


@pytest.mark.analytic
def test_ports_labels_are_the_num_labels_asked_for(tmp_path: Path) -> None:
    """`port.patch.he.he_image(num_labels=4)` labels every pixel `1..4` (#311).

    The brightest pixels, which `cnaster` labels 5, take label 4, and no
    other label changes.
    """
    from cnaster.he import get_he_image
    from port.patch.he import he_image

    _read(tmp_path)
    upstream = get_he_image(str(tmp_path), pos=None, num_labels=4)["label"].to_numpy()
    labels = he_image(str(tmp_path), pos=None, num_labels=4)["label"].to_numpy()

    assert set(np.unique(labels)) == {1, 2, 3, 4}
    np.testing.assert_array_equal(labels[upstream <= 4], upstream[upstream <= 4])
    assert (labels[upstream == 5] == 4).all()


@pytest.mark.infra
def test_patched_every_cnaster_caller_reads_labels_one_to_num_labels(
    tmp_path: Path,
) -> None:
    """Patched, `get_he_image` is `port.patch.he.he_image` wherever `cnaster`
    binds it -- `run_cnaster`'s figure frame among them -- and labels every
    pixel `1..4`; `cnaster`'s own is back, and labels a fifth, on exit
    (T- #771)."""
    import cnaster.scripts.run_cnaster as script
    from cnaster.he import get_he_image
    from port.pipeline import SWAPS, patched, swap_sites

    _read(tmp_path)
    sites = {site.module for site in swap_sites(SWAPS) if site.name == "get_he_image"}

    with patched():
        labels = script.get_he_image(str(tmp_path), pos=None, num_labels=4)

    assert {"cnaster.he", "cnaster.io", "cnaster.scripts.run_cnaster"} <= sites
    assert set(np.unique(labels["label"])) == {1, 2, 3, 4}
    assert get_he_image(str(tmp_path), pos=None, num_labels=4)["label"].max() == 5


@pytest.mark.end2end
def test_the_spots_h_and_e_class_darkens_away_from_normal(tmp_path: Path) -> None:
    """`he_classes` at the spots, as `run_cnaster` reads them: every class in
    `1..4`, and each planted clone's mean class strictly falls with the clone
    index, normal brightest -- the planted labelling is the referee (T- #771)."""
    from port.extensions.combined_figure import he_classes

    frame, labels = _read(tmp_path)
    coords = frame[["x", "y"]].to_numpy(dtype=np.float64)
    classes = he_classes(str(tmp_path), coords)
    mean = pd.Series(classes).groupby(labels).mean().to_numpy()

    assert set(np.unique(classes)) == {1, 2, 3, 4}
    assert np.all(np.diff(mean) < 0.0), f"mean class by clone: {mean}"


@pytest.mark.analytic
def test_each_contour_is_its_level_between_successive_classes(tmp_path: Path) -> None:
    """One contour per boundary, at `k + 1/2`, and every vertex at its level
    to 1e-9 under linear interpolation of the classes on the spots at
    `(x, -y)`, where `draw_clones_spatial` tiles them (T- #771)."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import LinearTriInterpolator, Triangulation
    from port.extensions.combined_figure import draw_he_contours, he_classes

    frame, _ = _read(tmp_path)
    coords = frame[["x", "y"]].to_numpy(dtype=np.float64)
    classes = he_classes(str(tmp_path), coords)
    _, ax = plt.subplots()
    contours = draw_he_contours(ax, coords, classes)
    surface = LinearTriInterpolator(
        Triangulation(coords[:, 0], -coords[:, 1]), classes.astype(np.float64)
    )

    np.testing.assert_array_equal(contours.levels, [1.5, 2.5, 3.5])
    for level, segments in zip(contours.levels, contours.allsegs, strict=True):
        vertices = np.concatenate(segments)
        assert vertices.size > 0, level
        np.testing.assert_allclose(
            surface(vertices[:, 0], vertices[:, 1]), level, atol=1e-9
        )
    plt.close("all")
