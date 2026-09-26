"""The genomic, spatial and combined figures, pixel for pixel, against a frozen copy (#342).

Frozen on #341's head, so a change to `port.extensions.combined_figure` that
is meant to leave the figures alone -- a refactor -- is shown to, rather than
argued to. Each figure is drawn on the 3 by 3 test instance at 100 dpi and
compared with `tests/data/figures/`, bitwise. A change that is meant to move
a figure re-freezes it in the same commit: `python -m tests.test_figure_snapshot`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

FROZEN = Path(__file__).parent / "data" / "figures"
DPI = 100


def _pixels(figure: Any) -> np.ndarray:
    """`figure` as drawn at `DPI`, RGBA bytes."""
    import io

    import matplotlib.image as mimage
    from port.extensions.combined_figure import page_style

    buffer = io.BytesIO()

    with page_style():
        figure.savefig(buffer, format="png", dpi=DPI, facecolor="white")
    buffer.seek(0)
    return np.asarray(np.round(mimage.imread(buffer) * 255.0), dtype=np.uint8)


def _drawn(tmp_path: Path) -> dict[str, np.ndarray]:
    import matplotlib as mpl

    mpl.use("Agg")
    # NB the state a run leaves: `cnaster.plotting` sets a serif face when
    #    imported, and `cnaster`'s plots set seaborn's context and style; the figures
    #    follow neither.
    import cnaster.plotting  # noqa: F401
    import seaborn as sns  # type: ignore[import-untyped]

    # NB as `cnaster.plot_validation_stats` sets it.
    sns.set_context("paper", font_scale=0.9)
    sns.set_style("ticks")
    from port.extensions.combined_figure import (
        combined_figure,
        genomic_figure,
        spatial_figure,
    )

    from tests.test_combined_figure import _recorded

    recorded, frame = _recorded(tmp_path)
    return {
        "genomic": _pixels(genomic_figure(recorded)),
        "spatial": _pixels(spatial_figure(recorded, frame)),
        "combined": _pixels(combined_figure(recorded, frame)),
    }


@pytest.mark.snapshot
@pytest.mark.merge
def test_the_figures_are_the_frozen_ones(cnaster_config: None, tmp_path: Path) -> None:
    """Both figures bitwise equal to `tests/data/figures/`: same size, every
    channel of every pixel."""
    import matplotlib.image as mimage

    for name, drawn in _drawn(tmp_path).items():
        frozen = np.asarray(
            np.round(mimage.imread(FROZEN / f"{name}.png") * 255.0), dtype=np.uint8
        )

        assert drawn.shape == frozen.shape, name
        differ = np.any(drawn != frozen, axis=-1)
        assert not differ.any(), f"{name}: {int(differ.sum())} pixels differ"


def main() -> None:
    """Re-freeze: draw the figures and write them over `tests/data/figures/`."""
    import tempfile

    import matplotlib.image as mimage

    from tests.conftest import SHIPPED_EM_FTOL, install_cnaster_config

    FROZEN.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as root:
        # NB the `cnaster_config` fixture's config, as the test draws under.
        install_cnaster_config(Path(root), em_ftol=SHIPPED_EM_FTOL, em_maxiter=100)

        for name, pixels in _drawn(Path(root)).items():
            mimage.imsave(FROZEN / f"{name}.png", pixels)
            print(f"froze {name}: {pixels.shape}", file=sys.stderr)


if __name__ == "__main__":
    main()
