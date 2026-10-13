"""`cnaster.utils.write_fig`, at a resolution and a group count a run can afford (#195).

Changes output (coarser raster, gridlines under the data), so it is a
`FIGURE_SWAPS` row. At `dpi=300, group_rasters=False` it is `cnaster`'s byte
for byte (`tests/test_figure_dpi.py`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from cnaster.config import start_time
from cnaster.logger import get_logger
from matplotlib.text import Text

logger = get_logger(__name__, start_time=start_time)


def collapse_rasterizing_groups(fig: Any, strategy: str = "sink") -> tuple[int, int]:
    """One rasterizing group per axes rather than two (#195 item 2).

    `cnaster`'s gridlines split each axes' rasterized run in two. `strategy`:
    ``sink`` moves interleaved vector artists below the run (default);
    ``sweep`` rasterizes them via `set_rasterization_zorder`; ``strict`` skips
    any axes with vector artists in the way. Returns (axes collapsed,
    rasterized artists in them); raises ValueError on an unknown strategy.
    """
    if strategy not in {"sink", "sweep", "strict"}:
        msg = f"unknown strategy {strategy!r}"
        raise ValueError(msg)

    collapsed, folded = 0, 0

    for axis in fig.axes:
        children = [child for child in axis.get_children() if child is not axis.patch]
        rasterized = [child for child in children if child.get_rasterized()]

        if len(rasterized) < 2:
            continue

        ceiling = max(child.get_zorder() for child in rasterized)
        floor = min(child.get_zorder() for child in rasterized)

        # NB only a vector artist between two rasterized ones splits the run.
        interleaved = [
            child
            for child in children
            if child.get_visible()
            and not child.get_rasterized()
            and floor <= child.get_zorder() <= ceiling
        ]

        if interleaved and strategy == "strict":
            continue

        if strategy == "sweep":
            for artist in rasterized:
                artist.set_rasterized(False)

            axis.set_rasterization_zorder(ceiling + 0.5)
        else:
            for artist in interleaved:
                artist.set_zorder(floor - 1.0)

        collapsed += 1
        folded += len(rasterized)

    return collapsed, folded


def write_fig(
    opath: str,
    fig: Any = None,
    transparent: bool = True,
    bbox_inches: str | None = "tight",
    dpi: int = 300,
    *,
    group_rasters: bool = False,
    group_strategy: str = "sink",
    png_copy: bool = False,
) -> None:
    """`cnaster.utils.write_fig`, with rasterizing groups collapsed on request.

    At the defaults, `cnaster`'s byte for byte; `FIGURE_SWAPS` binds
    `dpi=FIGURE_DPI` and `group_rasters=True` (#195, #517). `png_copy` also
    writes a metadata-free `<name>.png` for cross-run comparison (#452).
    Departure: each `Text`'s cached renderer is cleared, freeing the PDF's
    raster buffers (T- #692).
    """
    if fig is None:
        fig = plt.figure()
        fig.add_subplot(111)

    from port.extensions.figure_record import keep

    # NB the page's record into the run's `cnamaste.h5`, before the groups collapse (T- #817)
    keep(fig, opath, {"transparent": transparent, "bbox_inches": bbox_inches, "dpi": dpi,
                      "group_rasters": group_rasters, "group_strategy": group_strategy})  # fmt: skip

    if group_rasters:
        collapsed, folded = collapse_rasterizing_groups(fig, group_strategy)

        if collapsed:
            logger.info(
                f"Collapsed {folded} rasterized artists into {collapsed} groups."
            )

    logger.info(f"Writing figure to:\n{opath}")

    fig.savefig(
        opath,
        format="pdf",
        transparent=transparent,
        bbox_inches=bbox_inches,
        dpi=dpi,
    )

    if png_copy:
        fig.savefig(
            Path(opath).with_suffix(".png"),
            format="png",
            facecolor="white",
            bbox_inches=bbox_inches,
            dpi=dpi,
            metadata={"Software": None},
        )

    plt.close(fig)

    # NB the renderer each `Text` cached holds the PDF's rasters (T- #692).
    for text in fig.findobj(Text):
        text._renderer = None


def discard_fig(
    opath: str,
    fig: Any = None,
    transparent: bool = True,
    bbox_inches: str | None = "tight",
    dpi: int = 300,  # noqa: ARG001 -- cnaster's signature
) -> None:
    """`write_fig` under `run_cnaster_port --no-plots` (#403): close, write nothing."""
    import matplotlib.pyplot as plt

    if fig is not None:
        from port.extensions.figure_record import keep
        from port.pipeline import FIGURE_DPI

        # NB kept as `write_fig` keeps it, written as `FIGURE_SWAPS` writes: `run_plots` draws it later (T- #817)
        keep(fig, opath, {"transparent": transparent, "bbox_inches": bbox_inches, "dpi": FIGURE_DPI,
                          "group_rasters": True, "group_strategy": "sink"})  # fmt: skip
        plt.close(fig)
