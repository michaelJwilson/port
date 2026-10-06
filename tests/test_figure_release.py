"""`write_fig` leaves no written figure holding its PDF's rasters (T- #692 part 2).

Every `Text` caches the renderer that last drew it. After a PDF write that is
the `MixedModeRenderer`, which holds the `PdfFile`, which holds each
rasterizing group's image as a view of that group's full-page `RendererAgg`
buffer. `cnaster`'s `write_fig` leaves them for as long as the figure lives,
and `run_cnaster` keeps each figure in a local and then in a reference cycle.

The figure stays referenced and the collector stays off while the buffers are
counted: the state a figure is in after `run_cnaster` writes it.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib as mpl
import pytest

from tests.test_figure_dpi import CREATION_DATE, _figure

mpl.use("Agg")


def _pinned(writer: Callable[..., None], path: Path, **keywords: Any) -> int:
    """Bytes of raster buffer live `PdfFile`s hold after `writer` returns."""
    from matplotlib.backends.backend_pdf import PdfFile

    gc.collect()
    gc.disable()
    try:
        # NB only this write's files: another test's live figure, in the same
        #    worker, holds its own `PdfFile`, and is not this writer's to release
        before = {id(live) for live in gc.get_objects() if isinstance(live, PdfFile)}
        figure = _figure()
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
    """Four consecutive rasterized scatters, one group, on a 20 x 4 inch page.

    One full-page RGBA buffer at 300 dpi stays pinned: 22,602,720 bytes,
    under the untrimmed 20 * 4 * 300**2 * 4 = 28,800,000 because the tight
    box trims the page. Fails once `cnaster` or `matplotlib` releases it.
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
    """Zero bytes pinned on return, at `cnaster`'s defaults and at the options
    `FIGURE_SWAPS` binds; at `cnaster`'s defaults the file is `cnaster`'s,
    byte for byte once the creation date is removed."""
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
