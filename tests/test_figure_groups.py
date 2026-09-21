"""What splits a rasterizing run, and what collapsing it costs (#195 item 2).

**The ticket's mechanism was wrong and this module is where that was
found.** It read mixed mode as allocating a full-figure `RendererAgg` per
rasterized artist -- "four rasterized collections in one axes cost four
full-figure buffers". `matplotlib` does not. `allow_rasterization` starts
rasterizing at the first rasterized artist and stops at the first one that is
**not**, so consecutive rasterized artists share a buffer.

What splits `cnaster`'s runs is a gridline. `_format_track_axis` adds
`ax.axhline(..., c="lightgray", linewidth=0.5, zorder=0)` per y tick
(`plot_genomic.py:70`) and those land between the rasterized errorbar at
zorder 0 and the rasterized scatter at zorder 1. Two groups per axes instead
of one, measured over a whole run at **120 groups for 60 rasterized artists
on 39 axes**.

So the floor is one group per axes, and reaching it costs a change to the
drawing. `sink` moves the gridlines under the rasterized run; `sweep`
rasterizes them; `strict` refuses, which on `cnaster`'s own figures means it
does nothing at all -- 120 groups before and 120 after, byte for byte the
same files.

**`sink` is the default because its cost is a paint order and `sweep`'s is
resolution**, and because the raster it produces is the per-artist raster to
one grey level: the gridlines were never in the buffers and still are not.
Over a whole run it is 3.84 s of plotting against `sweep`'s 5.16 s and
`cnaster`'s 20.34 s, and 839 KB of PDF against 1,021 and 1,954.
"""

from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pytest

mpl.use("Agg")

TICKS = (0.0, 1.0, 2.0, 3.0)
"""Where `_format_track_axis` puts a gridline, at the RDR track's limits."""

CHANNEL_TOLERANCE = 1
"""How far the one-buffer composite may sit from the two-buffer one, of 255.

Realized **1**, on **273 of 2,880,000 pixels** -- 9.5e-5 of the panel. Two
buffers composited in straight alpha and one buffer blended by Agg round
differently in an antialiased edge, and nowhere else: every differing pixel
is on the boundary of a mark. A realized value above this, or a fraction
above a part in ten thousand, would mean a mark had moved rather than an
edge having rounded.
"""


def _panel(n_clones: int = 2, n_obs: int = 400, *, gridlines: bool = True) -> Any:
    """`plot_genomic.plot_acns_genomic`'s shape, gridlines optional.

    Per clone an RDR axes and a BAF axes, each carrying a rasterized errorbar
    at zorder 0 and a rasterized scatter at zorder 1, and then the gridlines
    `_format_track_axis` adds at zorder 0 -- after the errorbar, so they sort
    between it and the scatter.

    `gridlines=False` is the control. Without it every claim below would be
    about a figure `cnaster` does not write.
    """
    import matplotlib.pyplot as plt

    generator = np.random.default_rng(11)
    figure, axes = plt.subplots(
        2 * n_clones, 1, figsize=(8, 2.0 * n_clones), dpi=300, facecolor="white"
    )

    x = np.arange(n_obs, dtype=float)

    for axis in np.atleast_1d(axes).ravel():
        y = generator.random(n_obs) * 3.0

        axis.errorbar(
            x,
            y,
            yerr=generator.random(n_obs) * 0.05,
            fmt="none",
            elinewidth=0.5,
            zorder=0,
            rasterized=True,
        )
        axis.scatter(x, y, s=2, edgecolors="none", zorder=1, rasterized=True)

        axis.set_ylim([-0.5, 3.5])

        if gridlines:
            for tick in TICKS:
                axis.axhline(y=tick, c="lightgray", linewidth=0.5, zorder=0)

    return figure


def _layers(figure: Any, path: Path) -> list[np.ndarray]:
    """Every rasterizing group's buffer, full figure size, in draw order.

    Captured from `MixedModeRenderer.stop_rasterizing` **before** it crops to
    the non-transparent extent, so every layer shares one frame and the
    composite below is an elementwise blend rather than a placement problem.

    `bbox_inches=None` for the same reason, and for a second: a tight
    bounding box renders the figure twice, so every group would be counted
    twice and the counts here would be about `savefig` rather than about the
    figure.
    """
    from matplotlib.backends.backend_mixed import MixedModeRenderer

    captured: list[np.ndarray] = []
    original = MixedModeRenderer.stop_rasterizing

    def capture(self: Any) -> Any:
        captured.append(np.asarray(self._raster_renderer.buffer_rgba()).copy())
        return original(self)  # type: ignore[no-untyped-call]

    MixedModeRenderer.stop_rasterizing = capture  # type: ignore[method-assign]

    try:
        figure.savefig(path, format="pdf", transparent=True, bbox_inches=None, dpi=300)
    finally:
        MixedModeRenderer.stop_rasterizing = original  # type: ignore[method-assign]

    return captured


def _composite(layers: list[np.ndarray]) -> np.ndarray:
    """`source over`, in straight alpha, in draw order.

    What a PDF viewer does with a stack of RGBA images at one position. The
    per-artist arm hands it two per axes and the collapsed arm hands it one,
    so this is the operation a collapse claims to be invariant under.
    """
    assert layers, "nothing was rasterized"

    out = np.zeros(layers[0].shape, dtype=np.float64)

    for layer in layers:
        source = layer.astype(np.float64) / 255.0
        source_alpha = source[..., 3:4]
        destination_alpha = out[..., 3:4]

        alpha = source_alpha + destination_alpha * (1.0 - source_alpha)
        colour = source[..., :3] * source_alpha + out[..., :3] * destination_alpha * (
            1.0 - source_alpha
        )

        out = np.concatenate(
            [
                np.divide(colour, alpha, out=np.zeros_like(colour), where=alpha > 0),
                alpha,
            ],
            axis=-1,
        )

    return np.rint(out * 255.0).astype(np.int16)


@pytest.mark.bug
def test_a_gridline_is_what_splits_the_rasterizing_run(tmp_path: Path) -> None:
    """The ticket's mechanism, corrected by measurement.

    Same two rasterized artists per axes either way. With the gridlines the
    run is cut in half and the figure allocates a buffer per artist; without
    them `matplotlib` coalesces and allocates one per axes. The artist count
    is not what decides it.

    A `bug` marker rather than a `smoke` one: a diagnostic gridline doubling
    the memory a figure allocates is a defect in `_format_track_axis`, and
    this fails when `cnaster` fixes it.
    """
    with_lines = _panel(gridlines=True)
    axes = len(with_lines.axes)
    split = len(_layers(with_lines, tmp_path / "split.pdf"))

    coalesced = len(_layers(_panel(gridlines=False), tmp_path / "whole.pdf"))

    assert coalesced == axes, f"{coalesced} groups for {axes} axes without gridlines"
    assert split == 2 * axes, f"{split} groups for {axes} axes with them"


@pytest.mark.patch
def test_sink_collapses_the_run_without_touching_the_raster(tmp_path: Path) -> None:
    """One group per axes, and the buffers hold what they held.

    The claim `sink` is the default on. The gridlines were never inside a
    rasterized group and are not afterwards -- only their place in the paint
    order moves -- so the composited raster is the per-artist raster to
    `CHANNEL_TOLERANCE`, which is one grey level on a ten-thousandth of the
    pixels and all of them on an antialiased edge.

    What does change is recorded by
    `test_sink_moves_the_gridlines_under_the_rasterized_run`, because a
    collapse that changed nothing at all would not be a collapse.
    """
    from port.patch.utils import collapse_rasterizing_groups

    reference = _panel()
    axes = len(reference.axes)
    before = _layers(reference, tmp_path / "before.pdf")

    figure = _panel()
    collapsed, folded = collapse_rasterizing_groups(figure, "sink")
    after = _layers(figure, tmp_path / "after.pdf")

    assert (collapsed, folded) == (axes, 2 * axes)
    assert len(before) == 2 * axes
    assert len(after) == axes

    difference = np.abs(_composite(after) - _composite(before))
    realized = int(difference.max())
    touched = int((difference.max(axis=-1) > 0).sum())

    assert realized <= CHANNEL_TOLERANCE, (
        f"the collapsed raster is {realized} of 255 from the original"
    )
    assert touched < difference[..., 0].size // 10_000, (
        f"{touched} pixels moved, which is more than an antialiased edge"
    )


@pytest.mark.snapshot
def test_sink_moves_the_gridlines_under_the_rasterized_run() -> None:
    """The cost, stated: the gridlines paint under the data instead of over it.

    `_format_track_axis` asks for `zorder=0`, the same as the errorbar, and
    they land on top of it only because they are added later. `sink` puts
    them below the rasterized run, which is where a gridline is usually meant
    to be -- but it is a change to the drawing and `CLAUDE.md` forbids making
    one silently, so it is pinned rather than described.
    """
    import matplotlib.pyplot as plt
    from port.patch.utils import collapse_rasterizing_groups

    figure = _panel(n_clones=1)
    axis = figure.axes[0]

    gridlines = [
        child
        for child in axis.get_children()
        if not child.get_rasterized() and child.get_zorder() == 0
    ]

    assert len(gridlines) == len(TICKS), "the panel no longer draws gridlines"

    collapse_rasterizing_groups(figure, "sink")

    rasterized = [child for child in axis.get_children() if child.get_rasterized()]

    assert all(
        line.get_zorder() < min(child.get_zorder() for child in rasterized)
        for line in gridlines
    )

    plt.close(figure)


@pytest.mark.patch
def test_sweep_collapses_the_run_by_rasterizing_the_gridlines(
    tmp_path: Path,
) -> None:
    """The other way to one group per axes, and why it is not the default.

    `set_rasterization_zorder` leaves the paint order alone and puts the
    gridlines inside the buffer, so the raster is **not** the per-artist
    raster -- the gridlines are in it now. Measured rather than asserted
    away, and it is why `sink` is the default: a paint order is recoverable
    from the figure and a resolution is not.

    Over a whole run the two allocate the same 1,036 MB, and `sweep` writes a
    1,021 KB of PDF against `sink`'s 839 KB in 5.16 s against 3.84 s.
    """
    from port.patch.utils import collapse_rasterizing_groups

    reference = _panel()
    axes = len(reference.axes)
    before = _composite(_layers(reference, tmp_path / "before.pdf"))

    figure = _panel()
    collapsed, folded = collapse_rasterizing_groups(figure, "sweep")
    after = _layers(figure, tmp_path / "after.pdf")

    assert (collapsed, folded) == (axes, 2 * axes)
    assert len(after) == axes
    assert all(not child.get_rasterized() for child in figure.axes[0].get_children())
    assert figure.axes[0].get_rasterization_zorder() == pytest.approx(1.5)

    assert int(np.abs(_composite(after) - before).max()) > 0, (
        "sweep left the raster unchanged, so it did not rasterize the gridlines"
    )


@pytest.mark.patch
def test_strict_refuses_the_figures_cnaster_writes(tmp_path: Path) -> None:
    """The refusal, and the measurement that made it the wrong default.

    `strict` collapses only where nothing vector is in the way. On a panel
    with gridlines that is nowhere, so it leaves 120 groups at 120 over a
    whole run and writes the same bytes. Kept as a strategy because it is the
    only one that changes no drawing at all, and pinned here so that reading
    for it is not necessary.
    """
    from port.patch.utils import collapse_rasterizing_groups

    figure = _panel()
    axes = len(figure.axes)

    assert collapse_rasterizing_groups(figure, "strict") == (0, 0)
    assert len(_layers(figure, tmp_path / "strict.pdf")) == 2 * axes

    without = _panel(gridlines=False)

    assert collapse_rasterizing_groups(without, "strict") == (axes, 2 * axes)


@pytest.mark.smoke
def test_an_axes_with_one_rasterized_artist_is_left_alone() -> None:
    """Nothing to collapse: a run of one is already one group."""
    import matplotlib.pyplot as plt
    from port.patch.utils import collapse_rasterizing_groups

    figure, axis = plt.subplots()
    axis.scatter([0.0, 1.0], [0.0, 1.0], rasterized=True)

    assert collapse_rasterizing_groups(figure) == (0, 0)
    assert axis.get_rasterization_zorder() is None

    plt.close(figure)


@pytest.mark.smoke
def test_an_unknown_strategy_is_refused() -> None:
    """Three strategies, named, and nothing else silently doing nothing."""
    import matplotlib.pyplot as plt
    from port.patch.utils import collapse_rasterizing_groups

    figure = plt.figure()

    with pytest.raises(ValueError, match="unknown strategy"):
        collapse_rasterizing_groups(figure, "collapse")

    plt.close(figure)
