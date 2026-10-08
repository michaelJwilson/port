"""What splits a rasterizing run and what collapsing it costs (#195 item 2).

A gridline between rasterized artists splits the run; `sink`, `sweep` and `strict` are
judged against the per-artist raster composited in straight alpha.
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
"""Max per-channel gap, of 255, between one-buffer and two-buffer composites (realized 1,
on edges).
"""


def _panel(n_clones: int = 2, n_obs: int = 400, *, gridlines: bool = True) -> Any:
    """`plot_acns_genomic`'s shape: per clone an RDR and a BAF axes, gridlines optional as
    the control.
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
    """Every rasterizing group's full-figure buffer in draw order, captured before
    cropping.
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
    """`source over` in straight alpha, in draw order, as a PDF viewer composites."""
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
    """With gridlines each rasterized artist gets its own group; without, one per axes
    (fails when `cnaster` fixes it).
    """
    with_lines = _panel(gridlines=True)
    axes = len(with_lines.axes)
    split = len(_layers(with_lines, tmp_path / "split.pdf"))

    coalesced = len(_layers(_panel(gridlines=False), tmp_path / "whole.pdf"))

    assert coalesced == axes, f"{coalesced} groups for {axes} axes without gridlines"
    assert split == 2 * axes, f"{split} groups for {axes} axes with them"


@pytest.mark.patch
# NB one figure's form (#403): reruns when this module or the lock changes, and at release.
@pytest.mark.deprecate
def test_sink_collapses_the_run_without_touching_the_raster(tmp_path: Path) -> None:
    """`sink` gives one group per axes; the composite matches the per-artist raster to
    `CHANNEL_TOLERANCE`.
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
    """`sink` paints the gridlines under the rasterized run rather than over it."""
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
# NB one figure's form (#403): reruns when this module or the lock changes, and at release.
@pytest.mark.deprecate
def test_sweep_collapses_the_run_by_rasterizing_the_gridlines(
    tmp_path: Path,
) -> None:
    """`sweep` gives one group per axes by rasterizing the gridlines, so the raster differs
    from per-artist.
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
# NB one figure's form (#403): reruns when this module or the lock changes, and at release.
@pytest.mark.deprecate
def test_strict_refuses_the_figures_cnaster_writes(tmp_path: Path) -> None:
    """`strict` leaves panels with gridlines unchanged."""
    from port.patch.utils import collapse_rasterizing_groups

    figure = _panel()
    axes = len(figure.axes)

    assert collapse_rasterizing_groups(figure, "strict") == (0, 0)
    assert len(_layers(figure, tmp_path / "strict.pdf")) == 2 * axes

    without = _panel(gridlines=False)

    assert collapse_rasterizing_groups(without, "strict") == (axes, 2 * axes)


@pytest.mark.smoke
# NB one figure's form (#403): reruns when this module or the lock changes, and at release.
@pytest.mark.deprecate
def test_an_axes_with_one_rasterized_artist_is_left_alone() -> None:
    """A run of one rasterized artist is left as one group."""
    import matplotlib.pyplot as plt
    from port.patch.utils import collapse_rasterizing_groups

    figure, axis = plt.subplots()
    axis.scatter([0.0, 1.0], [0.0, 1.0], rasterized=True)

    assert collapse_rasterizing_groups(figure) == (0, 0)
    assert axis.get_rasterization_zorder() is None

    plt.close(figure)


@pytest.mark.smoke
def test_an_unknown_strategy_is_refused() -> None:
    """An unknown strategy is refused."""
    import matplotlib.pyplot as plt
    from port.patch.utils import collapse_rasterizing_groups

    figure = plt.figure()

    with pytest.raises(ValueError, match="unknown strategy"):
        collapse_rasterizing_groups(figure, "collapse")

    plt.close(figure)
