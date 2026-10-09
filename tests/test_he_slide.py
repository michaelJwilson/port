"""The mock H&E slide, read by `cnaster.he.get_he_image`, against the planted labelling
(#309).

The slide is the lattice's transpose, so a wrong orientation would load silently.
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
    """All 320 spots join a pixel of their own clone; mean gray strictly falls with
    clone.
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
    """`get_he_image(num_labels=4)` labels the brightest pixel 5 (`np.digitize` on the
    100th percentile).
    """
    from cnaster.he import get_he_image

    _read(tmp_path)
    pixels = get_he_image(str(tmp_path), pos=None, num_labels=4)
    labels = pixels["label"].to_numpy()

    assert labels.max() == 5
    assert np.sum(labels == 5) == np.sum(pixels["gray"] == pixels["gray"].max())


@pytest.mark.patch
def test_ports_labels_are_the_num_labels_asked_for(tmp_path: Path) -> None:
    """`port.patch.he.he_image(num_labels=4)` matches cnaster's labels bitwise on
    `1..4`; label 5 becomes 4 (#311, T- #771).
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
    """Patched, every `get_he_image` binding is port's and labels `1..4`; cnaster's
    returns on exit (T- #771).
    """
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
    """`he_classes` lie in `1..4` and each planted clone's mean class falls with clone
    index (T- #771).
    """
    from port.qa.combined_figure import he_classes

    frame, labels = _read(tmp_path)
    coords = frame[["x", "y"]].to_numpy(dtype=np.float64)
    classes = he_classes(str(tmp_path), coords)
    mean = pd.Series(classes).groupby(labels).mean().to_numpy()

    assert set(np.unique(classes)) == {1, 2, 3, 4}
    assert np.all(np.diff(mean) < 0.0), f"mean class by clone: {mean}"
