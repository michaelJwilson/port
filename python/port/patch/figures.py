"""`cnaster.utils.write_fig`, at a resolution the figures are read at (#195).

**One constant, and it is 47 per cent of a whole `run_cnaster`.** `cProfile`
over a run on the dev instance puts `backend_agg.RendererAgg.__init__` at
28.13 s over 147 calls and `get_text_width_height_descent` at 8.70 s over
110 -- 36.8 s of 69 s, and neither is inference (#104).

**The mechanism is mixed-mode PDF.** The genomic and LOH plots set
`rasterized=True` on their collections, which puts the PDF backend into mixed
mode, and mixed mode allocates **a fresh full-figure `RendererAgg` per
rasterizing group**. A 20x10 inch figure at 300 dpi is 6,000 x 3,000 pixels,
72 MB of RGBA, and a run makes 136 rasterizing groups across 23 figures. The
second line is the same allocation reached another way: the PDF backend
builds an Agg renderer to measure mathtext.

`savefig(dpi=...)` sets the figure's dpi for the write, so the raster
resolution is decided here and not at the eight `plt.figure(dpi=300)`
construction sites. One default reaches every figure: no call site in
`cnaster` passes `dpi` -- checked, not assumed.

**This changes the output, and is the one patch in `port` that does.** Every
other replacement reproduces `cnaster` bitwise; a figure written at half the
dpi is a different file by design. That is why it is not in
`port.pipeline.SWAPS` and is installed only by `run_cnaster_port --figures`,
and why the test beside it compares the figure's **vector** content rather
than its bytes: the drawing is identical and the raster behind it is coarser.

A hypothesis died here and is recorded rather than carried forward:
`bbox_inches="tight"` renders the figure twice and looked like the cost. It
is **1.35x faster**, because the tight bbox shrinks the area that gets
rasterized. Dropping it is worth having only alongside a dpi change; on its
own it is a regression.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from cnaster.config import start_time
from cnaster.logger import get_logger

logger = get_logger(__name__, start_time=start_time)

FIGURE_DPI = 150
"""What a figure is written at, against `cnaster`'s 300.

Halving it quarters the raster: a 20x10 inch panel goes 6,000 x 3,000 pixels
to 3,000 x 1,500, 72 MB of RGBA to 18 MB. Measured on one figure with four
rasterized collections, written to PDF:

    dpi=300, tight bbox -- cnaster   2,033 ms   35.1 MB
    dpi=150, tight bbox                692 ms    9.9 MB   2.9x
    dpi=150, no tight bbox             489 ms    9.8 MB   4.2x
    dpi=300, tight, not rasterized   1,379 ms    2.1 MB

150 rather than lower because it is the floor at which a 20-inch panel still
carries 3,000 pixels across, which is more than any screen shows it at and
more than a page prints it at. Lower is available and is a judgement about
the figures rather than about the arithmetic, so it is left to whoever is
reading them.
"""


def write_fig(
    opath: str,
    fig: Any = None,
    transparent: bool = True,
    bbox_inches: str = "tight",
    dpi: int = FIGURE_DPI,
) -> None:
    """What `cnaster.utils.write_fig` does, at `FIGURE_DPI`.

    A drop-in: same name, same signature, same side effects -- the figure is
    written and closed. The only difference is the default `dpi`, which no
    caller in `cnaster` overrides.
    """
    if fig is None:
        fig = plt.figure()
        fig.add_subplot(111)

    logger.info(f"Writing figure to:\n{opath}")

    fig.savefig(
        opath,
        format="pdf",
        transparent=transparent,
        bbox_inches=bbox_inches,
        dpi=dpi,
    )

    plt.close(fig)
