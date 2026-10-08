"""The genomic, spatial and combined figures against frozen PNGs, pixel for pixel (#342).

Re-freeze with `python -m tests.test_figure_snapshot`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests import TESTS
from tests.figure_checks import recorded_combined_calls

FROZEN = TESTS / "data" / "figures"
DPI = 100


def _pixels(figure: Any) -> np.ndarray:
    """Return `figure` drawn at `DPI` as RGBA bytes."""
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
    # NB a run's state: `cnaster.plotting` sets a serif face, its plots seaborn's style
    import cnaster.plotting  # noqa: F401
    import seaborn as sns  # type: ignore[import-untyped]

    # NB as `cnaster.plot_validation_stats` sets it
    sns.set_context("paper", font_scale=0.9)
    sns.set_style("ticks")
    from port.extensions.combined_figure import (
        combined_figure,
        genomic_figure,
        spatial_figure,
    )

    recorded, frame = recorded_combined_calls(tmp_path)
    return {
        "genomic": _pixels(genomic_figure(recorded)),
        "spatial": _pixels(spatial_figure(recorded, frame)),
        "combined": _pixels(combined_figure(recorded, frame)),
    }


@pytest.mark.snapshot
# NB too specific for every change (#403): reruns where this module or the lock changes
@pytest.mark.deprecate
def test_the_figures_are_the_frozen_ones(cnaster_config: None, tmp_path: Path) -> None:
    """Every figure equals `tests/data/figures/`, every channel of every pixel."""
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
    from port.sim.inputs import written_config

    from tests.conftest import SHIPPED_EM_FTOL, cnaster_test_config

    FROZEN.mkdir(parents=True, exist_ok=True)

    with (
        tempfile.TemporaryDirectory() as root,
        # NB the `cnaster_config` fixture's config, as the test draws under
        written_config(cnaster_test_config(Path(root), SHIPPED_EM_FTOL, 100)),
    ):
        for name, pixels in _drawn(Path(root)).items():
            mimage.imsave(FROZEN / f"{name}.png", pixels)
            print(f"froze {name}: {pixels.shape}", file=sys.stderr)


if __name__ == "__main__":
    main()
