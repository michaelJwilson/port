"""`write_fig` leaves no written figure holding its PDF's raster buffers (T- #692 part
2).

Buffers are counted with the figure referenced and the collector off, as after
`run_cnaster`.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib as mpl
import pytest

from tests.figure_checks import CREATION_DATE, wide_rasterized_figure

mpl.use("Agg")


def _pinned(writer: Callable[..., None], path: Path, **keywords: Any) -> int:
    """Bytes of raster buffer held by the `PdfFile`s `writer` created."""
    from matplotlib.backends.backend_pdf import PdfFile

    gc.collect()
    gc.disable()
    try:
        before = {id(live) for live in gc.get_objects() if isinstance(live, PdfFile)}
        figure = wide_rasterized_figure()
        writer(str(path), figure, **keywords)
        buffers: dict[int, int] = {}
        for live in gc.get_objects():
            if isinstance(live, PdfFile) and id(live) not in before:
                for image, _, _ in live._images.values():
                    base = image
                    while getattr(base, "base", None) is not None:
                        base = base.base
                    buffers[id(base)] = base.nbytes
        del figure
    finally:
        gc.enable()
        gc.collect()

    return sum(buffers.values())


@pytest.mark.bug
def test_cnasters_write_fig_leaves_the_rasters_resident(tmp_path: Path) -> None:
    """cnaster pins one full-page RGBA buffer (22,602,720 bytes) for four rasterized
    scatters.
    """
    from cnaster.utils import write_fig

    assert _pinned(write_fig, tmp_path / "theirs.pdf") > 20_000_000


@pytest.mark.patch
@pytest.mark.parametrize(
    "keywords",
    [{}, {"dpi": 150, "group_rasters": True}],
    ids=["cnaster-defaults", "figure-swaps"],
)
def test_ports_write_fig_releases_them_and_writes_the_same_file(
    tmp_path: Path, keywords: dict[str, Any]
) -> None:
    """Zero bytes pinned, at cnaster's defaults and `FIGURE_SWAPS`' options; cnaster's
    file byte for byte.
    """
    from cnaster.utils import write_fig as upstream
    from port.patch.utils import write_fig

    assert _pinned(write_fig, tmp_path / "ours.pdf", **keywords) == 0

    if not keywords:
        _pinned(upstream, tmp_path / "theirs.pdf")
        ours, theirs = (
            CREATION_DATE.sub(b"", (tmp_path / name).read_bytes())
            for name in ("ours.pdf", "theirs.pdf")
        )
        assert ours == theirs
