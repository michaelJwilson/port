"""`cnaster.utils.write_fig`, at a resolution and a group count a run can afford (#195).

**20.34 s of plotting to 3.84 s, and 8,287 MB of `RendererAgg` to 1,036 MB.**
Measured over a whole `run_cnaster` on the dev instance, 19 figures, with
every figure's groups and buffers counted as it was written:

| arm | plotting | groups | allocated | PDF |
| --- | ---: | ---: | ---: | ---: |
| `cnaster` | 20.34 s | 120 | 8,287 MB | 1,954 KB |
| `dpi=150` (#209) | 5.87 s | 120 | 2,072 MB | 1,048 KB |
| `dpi=150`, `sink` | **3.84 s** | **60** | **1,036 MB** | **839 KB** |
| `dpi=150`, `sweep` | 5.16 s | 60 | 1,036 MB | 1,021 KB |
| `dpi=150`, nothing rasterized | 5.08 s | 0 | 0 MB | 1,054 KB |

**Two of the ticket's three claims did not survive being measured**, and
both are recorded here rather than carried forward.

*The group count is not the artist count.* The ticket read mixed mode as
allocating a full-figure buffer per rasterized artist. `matplotlib`'s
`allow_rasterization` starts rasterizing at the first rasterized artist and
stops at the first one that is **not**, so a run of them shares one buffer.
What splits `cnaster`'s runs is a gridline: `_format_track_axis` adds
`ax.axhline(..., c="lightgray", linewidth=0.5, zorder=0)` per y tick
(`plot_genomic.py:70`), between the rasterized errorbar at zorder 0 and the
rasterized scatter at zorder 1. So the floor is one group per axes, not one
per figure, and a collapse that refuses to touch the drawing refuses
everywhere -- measured at 120 groups before and 120 after, byte for byte the
same files.

*Rasterizing is worth it, above about 500 bins.* The ticket asked whether
these panels need rasterizing at all. At the dev instance's 1,000 bins the
vector arm allocates nothing and is still slower, and it gets worse with the
bin count, which is `CLAUDE.md`'s rule that cost depends on the data rather
than only on its size. One panel, two clones, `dpi=150`:

| bins | `cnaster` | `sink` | vector |
| ---: | ---: | ---: | ---: |
| 250 | 0.406 s / 56 KB | 0.292 s / 41 KB | **0.275 s / 31 KB** |
| 1,000 | 0.436 s / 151 KB | **0.344 s / 115 KB** | 0.546 s / 107 KB |
| 4,000 | 0.729 s / 407 KB | **0.603 s / 312 KB** | 1.645 s / 387 KB |
| 16,000 | 1.509 s / 871 KB | **1.361 s / 635 KB** | 6.050 s / 1,506 KB |

The crossover is between 250 and 1,000 bins, and `expected_runtime.tex`'s own
derivation puts a genome at 2.9e5 segments (`docs/audit-paper-internal.md`
§3a), so every instance the method is aimed at is far above it. The decision
is **keep rasterizing**, and it is a decision rather than a default because
nothing had measured it.

**This is the one patch in `port` that changes its output**, and now in two
ways: a coarser raster, and gridlines that paint under the data instead of
over it. Every other replacement reproduces `cnaster` bitwise, which is why
this is not in `port.pipeline.SWAPS` and installs only under
`run_cnaster_port --figure-swaps`. At `dpi=300, group_rasters=False` it is
`cnaster`'s function byte for byte, which `tests/test_figure_dpi.py` holds it
to.

A third hypothesis died earlier and is kept for the same reason:
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


def collapse_rasterizing_groups(fig: Any, strategy: str = "sink") -> tuple[int, int]:
    """One rasterizing group per axes rather than two (#195 item 2).

    **The ticket's mechanism was wrong, and the measurement is why this
    function exists at all.** It read mixed mode as allocating a buffer per
    rasterized artist -- "four rasterized collections in one axes cost four
    full-figure buffers". `matplotlib` does not: `allow_rasterization` starts
    rasterizing at the first rasterized artist and stops at the first one
    that is **not**, so a run of consecutive rasterized artists shares one
    buffer.

    What splits `cnaster`'s runs is a gridline. `_format_track_axis` adds
    `ax.axhline(..., c="lightgray", linewidth=0.5, zorder=0)` per y tick
    (`plot_genomic.py:70`), and those land between the rasterized errorbar at
    zorder 0 and the rasterized scatter at zorder 1. Two groups per axes,
    measured: a whole run allocates **120 groups over 60 rasterized artists
    on 39 axes** -- exactly two per artist, one per artist per `bbox_inches`
    pass -- and 8,287 MB of `RendererAgg` at `cnaster`'s dpi.

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
        `cnaster`'s own figures is **never**: measured at 120 groups before
        and 120 after, byte for byte the same files.

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
    bbox_inches: str = "tight",
    dpi: int = FIGURE_DPI,
    group_rasters: bool = True,
    group_strategy: str = "sink",
) -> None:
    """What `cnaster.utils.write_fig` does, at `FIGURE_DPI` and one group per axes.

    A drop-in: same name, `cnaster`'s signature with one keyword appended,
    same side effects -- the figure is written and closed. Two differences,
    and each is a default a caller can put back:

    *   `dpi`, which no caller in `cnaster` overrides.
    *   `group_rasters`, which collapses the rasterizing groups by
        `group_strategy`. At `dpi=300, group_rasters=False` this is
        `cnaster`'s function byte for byte, which is what
        `tests/test_figure_dpi.py` holds it to.
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

    plt.close(fig)


def discard_fig(opath: str, fig: Any = None, *_: Any, **__: Any) -> None:
    """`write_fig` under `run_cnaster_port --no-plots` (#403): close, write nothing.

    Every figure a run draws is still built -- the plotting code runs, and a
    coverage guard still reads it -- and only the rendering is skipped, which
    is where a small run spends 31 to 45 per cent of its time, in PDF text
    layout. For a run whose claim is not a figure.
    """
    import matplotlib.pyplot as plt

    del opath
    if fig is not None:
        plt.close(fig)
