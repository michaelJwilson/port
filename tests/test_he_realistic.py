"""`port.sim.he_realistic`'s slide against the clone map it was painted from, and `he_draw`'s Visium HD positions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from port.sim.he_draw import visium_hd_positions
from port.sim.he_realistic import MICRONS_PER_PIXEL, realistic_he


def _labels(lattice: tuple[int, int]) -> np.ndarray:
    """Clone per spot, row-major over `lattice`: quadrants 0..3, normal and three tumour clones."""
    rows, cols = np.indices(lattice)
    return ((rows >= lattice[0] // 2) * 2 + (cols >= lattice[1] // 2)).ravel()


@pytest.mark.end2end
@pytest.mark.parametrize(
    ("lattice", "microns_per_unit", "microns_per_pixel"),
    [((24, 20), 100.0, MICRONS_PER_PIXEL), ((100, 80), 16.0, 3.80812)],
    ids=["visium", "visium_hd"],
)
def test_each_spot_sits_on_its_planted_clone(
    lattice: tuple[int, int], microns_per_unit: float, microns_per_pixel: float
) -> None:
    """At Visium's scale and at 16 um bins painted at 10x's hires resolution, spot (r, c) is drawn at image row
    c * scale, column r * scale, scale = microns_per_unit / microns_per_pixel, and the planted clone there is the spot's.

    The image is (columns, rows) x scale pixels, rounded to whole pixels. The clone map's boundaries are rounded and
    displaced by up to 50 um, so a spot on a quadrant edge may read its neighbour's: at least 90% of the spots read
    their own (24 x 20 Visium spots, 2.4 x 2 mm; 100 x 80 bins, 1.6 x 1.3 mm), against 25% for a map unrelated to the
    labels. Fails if the lattice is drawn untransposed,
    at the wrong scale, or if a length is held in pixels where the scale changes.
    """
    labels = _labels(lattice)
    image, clone = realistic_he(
        labels,
        lattice,
        seed=0,
        microns_per_unit=microns_per_unit,
        microns_per_pixel=microns_per_pixel,
    )
    scale = microns_per_unit / microns_per_pixel
    rows, cols = np.indices(lattice)

    assert image.shape == (round(lattice[1] * scale), round(lattice[0] * scale), 3)
    at = clone[
        np.clip((cols.ravel() * scale).astype(int), 0, clone.shape[0] - 1),
        np.clip((rows.ravel() * scale).astype(int), 0, clone.shape[1] - 1),
    ]
    assert np.mean(at == labels) >= 0.9


@pytest.mark.bug
def test_a_slide_under_600_um_is_refused() -> None:
    """24 x 20 bins of 16 um (384 x 320 um): `realistic_he` raises rather than paint two clones off the slide."""
    with pytest.raises(ValueError, match="MIN_SIDE_MICRONS"):
        realistic_he(
            _labels((24, 20)),
            (24, 20),
            microns_per_unit=16.0,
            microns_per_pixel=3.80812,
        )


@pytest.mark.infra
def test_visium_hd_positions_are_full_resolution_pixels(tmp_path: Path) -> None:
    """`visium_hd_positions` replaces the CSV by a parquet whose pixel columns are the array's, transposed.

    A spot at array (row 3, column 7) is at full-resolution (pxl_row 7, pxl_col 3): the image's row is the array's
    column, as the painter draws it. Fails if the CSV is left beside the parquet (a reader then takes the slice as
    Visium) or if the transpose is dropped.
    """
    spatial = tmp_path / "spatial"
    spatial.mkdir()
    pd.DataFrame([["a-1", 1, 3, 7, 0, 0], ["b-1", 1, 0, 2, 0, 0]]).to_csv(
        spatial / "tissue_positions_list.csv", header=False, index=False
    )

    visium_hd_positions(spatial)

    assert not (spatial / "tissue_positions_list.csv").exists()
    found = pd.read_parquet(spatial / "tissue_positions.parquet")
    assert found["barcode"].tolist() == ["a-1", "b-1"]
    np.testing.assert_array_equal(found["pxl_row_in_fullres"], [7.0, 2.0])
    np.testing.assert_array_equal(found["pxl_col_in_fullres"], [3.0, 0.0])
