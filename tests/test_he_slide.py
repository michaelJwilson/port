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

from tests.fixtures import clone_bands
from tests.he_slide import mock_he, write_he_slide

LATTICE = (20, 16)
N_CLONES = 4


def _read(tmp_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    from cnaster.he import get_he_image

    labels = clone_bands(*LATTICE, N_CLONES)
    write_he_slide(mock_he(labels, LATTICE, seed=5), tmp_path)

    rows, columns = np.unravel_index(np.arange(labels.size), LATTICE)
    # NB `tissue_positions.csv` as `tests/tmp_inputs.py` writes it.
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
