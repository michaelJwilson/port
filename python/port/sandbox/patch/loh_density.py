"""`cnaster.plot_loh_density.plot_loh_density`, with the clone loop unified (#278).

A drop-in replacement: same name, same signature, same figure. What changes
is the part that had three problems and no referee.

**The model half was nine lines carrying two defects and one duplication.**
`cnaster` writes:

    for c in range(n_clones):
        ...
        if is_concatenated:
            c_pred = pred_cnv[c * n_bins : (c + 1) * n_bins]
        else:
            c_pred = pred_cnv[:, c]

        c_p = p_binom[c_pred, c if p_binom.shape[1] > 1 else 0]

The layout test is re-derived here and at four sites in `plot_genomic`, and
they do not agree -- two take the modulus by `n_states` and three do not.
The column guard resolves to `0` on every run the fit can produce, so it
reads as a supported case that does not exist. Both go to
`port.patch.plot_genomic.clone_paths`, which is refereed against the expressions
it replaces rather than asserted.

**The rendering is copied unchanged.** It is 60 lines of `matplotlib` with
no branch worth unifying, and changing a pixel is out of scope: `#195`
established that a figure row does not reproduce bitwise, and this one adds
no such row. `nan_gaussian_filter1d` is imported from `cnaster` rather than
copied, so the smoothing stays upstream's.
"""

from __future__ import annotations

from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from cnaster.config import start_time
from cnaster.logger import get_logger
from cnaster.plot_loh_density import nan_gaussian_filter1d

from port.patch.plot_genomic.clone_paths import clone_path, parameter_by_path

logger = get_logger(__name__, start_time=start_time)

__all__ = ["loh_model", "plot_loh_density"]


def loh_model(res_combine: dict[str, Any], n_bins: int, n_spots: int) -> np.ndarray:
    """The inferred LOH signal per bin and spot.

    Extracted because it is the only part of the figure with a claim in it:
    everything downstream is rendering. `1 - 2 * min(p, 1 - p)` is zero at a
    balanced BAF and one at complete loss, per clone, broadcast over the
    spots that clone holds.

    A clone with no spots is skipped, as upstream does -- its column of the
    result stays zero rather than being written from an empty mask.
    """
    assignments = np.asarray(res_combine["new_assignment"])
    pred_cnv = np.asarray(res_combine["pred_cnv"])
    p_binom = np.asarray(res_combine["new_p_binom"])

    model = np.zeros((n_bins, n_spots))

    for clone in range(len(np.unique(assignments))):
        spots = assignments == clone

        if not np.any(spots):
            continue

        path = clone_path(pred_cnv, clone, n_bins)
        probability = parameter_by_path(p_binom, path)

        minor = np.minimum(probability, 1.0 - probability)
        model[:, spots] = (1.0 - 2.0 * minor)[:, None]

    return np.nan_to_num(model, nan=0.0)


def plot_loh_density(
    coords: np.ndarray,
    single_X: np.ndarray,
    single_total_bb_RD: np.ndarray,
    res_combine: dict[str, Any],
    lengths: np.ndarray | None = None,
    smooth_sigma: float = 0.1,
    max_alpha: float = 0.5,
    gamma: float = 1.0,
    color: str = "#0055ff",
    plot_type: str = "both",
) -> Any:
    """Upstream's, with the clone loop taken from `clone_paths`."""
    del lengths  # upstream's chromosome boundaries are commented out

    logger.info("Solving for loh density to render.")

    n_bins, _, n_spots = single_X.shape

    with np.errstate(divide="ignore", invalid="ignore"):
        baf_raw = single_X[:, 1, :] / single_total_bb_RD

    baf_smooth = nan_gaussian_filter1d(baf_raw, smooth_sigma, fill_value=0.5)

    minor_raw = np.minimum(baf_smooth, 1.0 - baf_smooth)
    loh_empirical = np.nan_to_num(np.clip(1.0 - 2.0 * minor_raw, 0.0, 1.0), nan=0.0)

    model = loh_model(res_combine, n_bins, n_spots)

    bins, spots = np.meshgrid(np.arange(n_bins), np.arange(n_spots), indexing="ij")

    x_flat = coords[spots.flatten(), 0]
    y_flat = -coords[spots.flatten(), 1]
    z_flat = bins.flatten()

    base_rgb = mcolors.to_rgb(color)

    def build_rgba(signal: np.ndarray) -> np.ndarray:
        rgba = np.zeros((n_bins * n_spots, 4))
        rgba[:, :3] = base_rgb
        rgba[:, 3] = np.clip(max_alpha * signal.flatten() ** gamma, 0.0, 1.0)

        return np.nan_to_num(rgba, nan=0.0)

    n_plots = 2 if plot_type == "both" else 1
    figure = plt.figure(figsize=(10 * n_plots, 10), dpi=300, facecolor="white")

    rgba_arrays = []

    if plot_type in ("empirical", "both"):
        rgba_arrays.append(build_rgba(loh_empirical))
    if plot_type in ("model", "both"):
        rgba_arrays.append(build_rgba(model))

    for index, rgba in enumerate(rgba_arrays):
        axis = figure.add_subplot(1, n_plots, index + 1, projection="3d")
        visible = rgba[:, 3] > 0.01

        # NB the axes are re-mapped: the genomic axis is plotted along x, and
        #    the two spatial axes along y and z.
        axis.scatter(
            z_flat[visible],
            x_flat[visible],
            y_flat[visible],
            c=rgba[visible],
            s=1.0,
            edgecolors="none",
            rasterized=True,
        )

        axis.set_box_aspect((2, 1, 1))
        axis.view_init(elev=20, azim=-70)

        axis.set_xlabel(r"$g$", labelpad=10)
        axis.set_ylabel(r"$x$", labelpad=10)
        axis.set_zlabel(r"$y$", labelpad=10)

        axis.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        axis.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        axis.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        axis.grid(False)

    figure.tight_layout()

    return figure
