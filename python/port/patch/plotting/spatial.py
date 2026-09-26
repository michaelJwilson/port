"""`cnaster.plotting.plot_clones_spatial`, each spot drawn as a tile (#309).

Upstream sizes its markers by count alone -- `s = clip(12000 / n, 0.1, 25)`
points squared -- so how much of the section a spot covers depends on the
figure size and the lattice. On the dev instance's 40 x 40 at 4 in a dot
is 0.53 of the pitch across, 2.74 pt against 5.20 pt, and the page reads as
white space with colour in it.

Here each spot is a square in **data units**, `TILE` of the lattice pitch
on a side, so the section is covered and a gap of `1 - TILE` of the pitch
stays between neighbours. The pitch is the median nearest-neighbour
distance, and a tile in data units is the same fraction of the pitch at any
figure size, which is what lets `port.extensions.combined_figure` draw it
into a 3.1 in panel.

Colours, opacity, clone order, legend and multi-sample offsets are
upstream's: `tests/test_plot_spatial_patch.py` pins each spot's colour
against `cnaster`'s.

**`sample_layout`, `(rows, columns)`, splits the samples into panels (#328).**
Upstream offsets each sample along `x` on one axis, which shifts twice a
sample already placed on a shared grid. With a layout each sample is drawn
in its own panel, in its own coordinates, with white space between, and
every panel colours the clones as the whole run does. Unset -- the default,
and `SAMPLE_LAYOUT`'s -- is upstream's single axis exactly.
`run_cnaster_port --sample-layout 3,1` sets `SAMPLE_LAYOUT`.
"""

from __future__ import annotations

import copy
from typing import Any

import matplotlib.colors as mcolors
import numpy as np
import scipy.spatial

TILE = 0.85
"""A tile's side, as a fraction of the lattice pitch; the rest is the gap."""

SAMPLE_LAYOUT: tuple[int, int] | None = None
"""The `sample_layout` a call that names none takes; `None` is upstream's axis."""

__all__ = [
    "SAMPLE_LAYOUT",
    "TILE",
    "clone_colours",
    "draw_clones_spatial",
    "pitch",
    "plot_clones_spatial",
]


def clone_colours(clone_ids: np.ndarray, palette: str = "rocket") -> list[str]:
    """Upstream's colour per clone id: `clone 0` light grey, the rest `palette`."""
    import seaborn as sns  # type: ignore[import-untyped]

    n_clones = len(clone_ids)

    if "clone 0" in clone_ids:
        return ["lightgrey", *sns.color_palette(palette, n_clones - 1).as_hex()]

    return list(sns.color_palette(palette, n_clones).as_hex())


def pitch(coords: np.ndarray) -> float:
    """The lattice spacing: the median distance to a spot's nearest neighbour."""
    if coords.shape[0] < 2:
        return 1.0

    distance, _ = scipy.spatial.cKDTree(coords).query(coords, k=2)

    return float(np.median(distance[:, 1]))


def spot_colours(
    assignment: Any,
    single_tumor_prop: np.ndarray | None = None,
    palette: str = "rocket",
    clone_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """RGBA per spot, the clone ids in upstream's order, and their colours.

    A spot with no clone is transparent, as upstream leaves it undrawn. Tumour
    proportion is the opacity, a NaN taken as 0.5, as upstream. `clone_ids`,
    when given, fixes the colours to a wider set than the spots carry, so a
    panel of one sample colours as the whole run does.
    """
    values = np.asarray(assignment.values)
    missing = assignment.isnull().to_numpy()
    if clone_ids is None:
        clone_ids = np.unique(values[~missing])
    colours = clone_colours(clone_ids, palette)

    rgba = np.zeros((values.size, 4))

    for colour, clone in zip(colours, clone_ids, strict=True):
        rgba[values == clone] = (*mcolors.to_rgb(colour), 1.0)

    if single_tumor_prop is not None:
        proportion = np.array(single_tumor_prop, dtype=float)
        proportion[np.isnan(proportion)] = 0.5
        rgba[:, 3] = np.clip(proportion, 0.0, 1.0)

    rgba[missing, 3] = 0.0

    return rgba, clone_ids, colours


def draw_clones_spatial(
    ax: Any,
    coords: np.ndarray,
    assignment: Any,
    single_tumor_prop: np.ndarray | None = None,
    palette: str = "rocket",
    tile: float = TILE,
    clone_ids: np.ndarray | None = None,
    legend: bool = True,
) -> None:
    """Tile each spot at `(x, -y)` into `ax`, with upstream's legend."""
    from cnaster.utils import cast_clone_label
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D

    rgba, clone_ids, colours = spot_colours(
        assignment, single_tumor_prop, palette, clone_ids
    )

    half = 0.5 * tile * pitch(coords)
    centres = np.column_stack([coords[:, 0], -coords[:, 1]])
    corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])

    ax.add_collection(
        PolyCollection(
            centres[:, None, :] + corners[None, :, :],
            facecolors=rgba,
            edgecolors="none",
            linewidths=0,
        )
    )
    ax.set_xlim(centres[:, 0].min() - half, centres[:, 0].max() + half)
    ax.set_ylim(centres[:, 1].min() - half, centres[:, 1].max() + half)
    ax.set_aspect("equal")
    ax.axis("off")

    if not legend:
        return

    ax.legend(
        [
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor=colour,
                label=clone,
                markersize=6,
            )
            for colour, clone in zip(colours, clone_ids, strict=True)
        ],
        [cast_clone_label(clone) for clone in clone_ids],
        handlelength=0.1,
        loc="upper left",
        bbox_to_anchor=(0.05, 0.02),
        ncol=len(clone_ids),
        frameon=False,
        fontsize=8,
        borderaxespad=0.0,
    )


def plot_clones_spatial(
    coords: np.ndarray,
    assignment: Any,
    single_tumor_prop: np.ndarray | None = None,
    sample_list: list[str] | None = None,
    sample_ids: np.ndarray | None = None,
    base_width: float = 4,
    base_height: float = 4,
    palette: str = "rocket",
    sample_layout: tuple[int, int] | None = None,
) -> Any:
    """Upstream's signature and page, each spot a tile of `TILE` the pitch.

    `sample_layout`, or unset `SAMPLE_LAYOUT`, draws one panel per sample;
    it needs `sample_ids`, and a module default is skipped for a run of one
    sample.
    """
    import matplotlib.pyplot as plt

    several = sample_ids is not None and np.unique(sample_ids).size > 1

    if sample_layout is None and several:
        sample_layout = SAMPLE_LAYOUT

    if sample_layout is not None:
        return _panels(
            coords,
            assignment,
            single_tumor_prop,
            sample_list,
            sample_ids,
            sample_layout,
            base_width,
            base_height,
            palette,
        )

    shifted = copy.copy(coords)

    if sample_ids is not None and sample_list is not None:
        offset = 0

        for sample, _ in enumerate(sample_list):
            index = np.where(sample_ids == sample)[0]
            shifted[index, 0] = shifted[index, 0] + offset
            offset += np.max(coords[index, 0]) + 10

    figure, ax = plt.subplots(
        1, 1, figsize=(base_width, base_height), dpi=300, facecolor="white"
    )
    draw_clones_spatial(ax, shifted, assignment, single_tumor_prop, palette)

    if sample_list is not None:
        ax.text(
            0.05,
            0.99,
            ", ".join(sample_list),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
        )

    return figure


def _panels(
    coords: np.ndarray,
    assignment: Any,
    single_tumor_prop: np.ndarray | None,
    sample_list: list[str] | None,
    sample_ids: np.ndarray | None,
    layout: tuple[int, int],
    base_width: float,
    base_height: float,
    palette: str,
) -> Any:
    """One panel per sample on `layout`, white space between, one legend."""
    import matplotlib.pyplot as plt

    from port.extensions.multisample import sample_panels

    if sample_ids is None:
        msg = "sample_layout needs sample_ids, the sample of each spot"
        raise ValueError(msg)

    rows, columns = layout
    values = np.asarray(assignment.values)
    clone_ids = np.unique(values[~assignment.isnull().to_numpy()])
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(base_width * columns, base_height * rows),
        dpi=300,
        facecolor="white",
        squeeze=False,
        gridspec_kw={"wspace": 0.15, "hspace": 0.15},
    )
    panels = sample_panels(np.asarray(sample_ids), layout)

    for i, (row, column, spots) in enumerate(panels):
        ax = axes[row, column]
        draw_clones_spatial(
            ax,
            np.asarray(coords)[spots],
            assignment.iloc[spots],
            None if single_tumor_prop is None else np.asarray(single_tumor_prop)[spots],
            palette,
            clone_ids=clone_ids,
            legend=i == 0,
        )

        if sample_list is not None:
            ax.set_title(sample_list[i], fontsize=9)

    for ax in axes.flat[len(panels) :]:
        ax.axis("off")

    return figure
