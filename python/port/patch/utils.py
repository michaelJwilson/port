"""`cnaster.utils.write_fig`, at a resolution and a group count a run can afford (#195).

**This patch changes its output**, in two ways: a coarser raster, and
gridlines that paint under the data instead of over it. That is why it is in
`port.pipeline.FIGURE_SWAPS` rather than `SWAPS`, which `run_cnaster_port`
installs unless `--no-figure-swaps` or `--no-patch` is given. At `dpi=300, group_rasters=False` it is
`cnaster`'s function byte for byte, which `tests/test_figure_dpi.py` holds it
to.

It lowers the resolution (`FIGURE_DPI`), collapses rasterizing groups
(:func:`collapse_rasterizing_groups`), and keeps rasterizing and
`bbox_inches="tight"`. Measured: `docs/measurements.md`, `port.patch.utils`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from cnaster.config import start_time
from cnaster.logger import get_logger

logger = get_logger(__name__, start_time=start_time)


def collapse_rasterizing_groups(fig: Any, strategy: str = "sink") -> tuple[int, int]:
    """One rasterizing group per axes rather than two (#195 item 2).

    `matplotlib`'s `allow_rasterization` starts
    rasterizing at the first rasterized artist and stops at the first one
    that is **not**, so a run of consecutive rasterized artists shares one
    buffer.

    What splits `cnaster`'s runs is a gridline. `_format_track_axis` adds
    `ax.axhline(..., c="lightgray", linewidth=0.5, zorder=0)` per y tick
    (`plot_genomic.py:70`), and those land between the rasterized errorbar at
    zorder 0 and the rasterized scatter at zorder 1. Two groups per axes.
    Measured: `docs/measurements.md`,
    `port.patch.utils.collapse_rasterizing_groups`.

    So the floor is one group per axes, not one per figure, and reaching it
    costs a change to the drawing either way:

    ``sink``
        Move the interleaved vector artists **below** the rasterized run.
        Nothing that was vector becomes raster; the gridlines paint under the
        error bars instead of over them. This is the default, because the
        loss is a paint order that was arguably backwards and the other
        strategy's loss is resolution.
    ``sweep``
        Rasterize them with `Axes.set_rasterization_zorder`. The drawing
        order is untouched and the gridlines become raster at the figure's
        dpi -- a 0.5 pt line is one pixel at 150.
    ``strict``
        Refuse. Collapse only where nothing vector is in the way, which on
        `cnaster`'s own figures is **never**.

    Returns
    -------
    tuple[int, int]
        Axes collapsed, and rasterized artists in them.

    Raises
    ------
    ValueError
        On an unknown strategy.
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

        # NB only what is drawn *between* two rasterized artists splits the
        #    run. A vector artist above the ceiling never entered it.
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
    """What `cnaster.utils.write_fig` does, with rasterizing groups collapsed on request.

    A drop-in: `cnaster`'s signature and defaults, and at those defaults its
    function byte for byte, which `tests/test_figure_dpi.py` holds it to.
    `FIGURE_SWAPS` binds `dpi=FIGURE_DPI` and `group_rasters=True` at install
    (#195, #517): `group_rasters` collapses the groups by `group_strategy`.

    `png_copy` also writes `<name>.png` beside the PDF, without metadata, for
    figures compared across runs (#452): a matplotlib PDF
    carries its creation time, so two runs of the same code differ byte for
    byte and a PNG written without metadata does not.
    `run_cnaster_port --png-copies` binds it.
    """
    if fig is None:
        fig = plt.figure()
        fig.add_subplot(111)

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


def discard_fig(
    opath: str,
    fig: Any = None,
    transparent: bool = True,  # noqa: ARG001 -- cnaster's signature
    bbox_inches: str | None = "tight",  # noqa: ARG001
    dpi: int = 300,  # noqa: ARG001
) -> None:
    """`write_fig` under `run_cnaster_port --no-plots` (#403): close, write nothing.

    Every figure a run draws is still built -- the plotting code runs, and a
    coverage guard still reads it -- and only the rendering is skipped. For a
    run whose claim is not a figure. Measured: `docs/measurements.md`,
    `port.patch.utils.discard_fig`.
    """
    import matplotlib.pyplot as plt

    del opath
    if fig is not None:
        plt.close(fig)
