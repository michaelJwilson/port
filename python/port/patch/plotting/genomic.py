"""`cnaster.plot_genomic.plot_clones_genomic`, with the clone loop unified (#278).

A drop-in replacement: same name, same signature, same figure. What changes
is that the four quantities the figure actually asserts are functions with
referees, instead of expressions buried in 338 lines of `matplotlib`.

**The decode appears twice inside this one function.** `pred_cnv` is sliced
at `plot_genomic.py:541` to colour the points and again at `:650` to draw the
Viterbi segments, and the two disagree: the first takes the modulus by
`new_p_binom.shape[0]`, the second by `new_log_mu.shape[0]`. They are the
same number, so nothing is wrong today -- but they are two derivations of one
thing, and #267 is the ticket about what happens when readings of an axis
drift apart. Both become `clone_path`.

**`clone_idx = 0 if new_log_mu.shape[1] == 1 else c`** is the spot-axis guard
(`:646`). It resolves to `0` on every run the fit can produce, so it is
replaced by `clone_column`, which refuses a second column rather than
silently indexing past one.

The rendering is copied unchanged: `#195` established that a figure row does
not reproduce bitwise, and this adds no such row and should change no pixel.
The four private helpers `_create_clone_gridspec`, `_format_track_axis`,
`_draw_chromosome_boundaries` and `_annotate_clone_stats` are imported from
`cnaster` rather than copied, so the layout stays upstream's.
"""

from __future__ import annotations

from typing import Any

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore[import-untyped]
from cnaster.config import start_time
from cnaster.logger import get_logger
from cnaster.palette import get_full_palette
from cnaster.plot_genomic import (
    NORMAL_OPACITY,
    _annotate_clone_stats,
    _create_clone_gridspec,
    _draw_chromosome_boundaries,
    _format_track_axis,
)
from cnaster.pseudobulk import merge_pseudobulk_by_index_mix
from cnaster.utils import get_intervals
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from port.patch.plotting.clone_paths import clone_column, clone_path

logger = get_logger(__name__, start_time=start_time)

__all__ = ["baf_track", "plot_clones_genomic", "rdr_track", "segment_levels"]


def rdr_track(
    X: np.ndarray, base_nb_mean: np.ndarray, clone: int
) -> tuple[np.ndarray, np.ndarray]:
    """The read-depth ratio and its Poisson error, for one clone.

    `X[:, 0, c] / base_nb_mean[:, c]`, with the error `sqrt(X) / base` --
    the Poisson standard deviation of the numerator, propagated through a
    division by a quantity treated as known.

    A bin with no baseline divides by zero. Upstream lets the *value* become
    `inf` or `nan` and only scrubs the *error*, which is what it does here
    too: the point is dropped by `matplotlib` while the error bar would
    otherwise raise.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        values = X[:, 0, clone] / base_nb_mean[:, clone]
        error = np.sqrt(X[:, 0, clone]) / base_nb_mean[:, clone]

    error[~np.isfinite(error)] = 0.0

    return values, error


def baf_track(
    X: np.ndarray, total_bb_RD: np.ndarray, clone: int
) -> tuple[np.ndarray, np.ndarray]:
    """The B-allele frequency and its Beta posterior error, for one clone.

    The error is the standard deviation of `Beta(k + 1, n - k + 1)` -- the
    posterior under a uniform prior, given `k` B-allele reads of `n`. Its
    closed form is `sqrt(ab / ((a + b)^2 (a + b + 1)))`, which is what
    upstream writes inline and what `tests/test_plot_genomic_tracks.py`
    checks against the distribution rather than against the expression.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        values = X[:, 1, clone] / total_bb_RD[:, clone]

    successes, trials = X[:, 1, clone], total_bb_RD[:, clone]
    alpha, beta = successes + 1, trials - successes + 1
    total = alpha + beta

    error = np.sqrt((alpha * beta) / (np.square(total) * (total + 1)))

    return values, error


def segment_levels(
    res_combine: dict[str, Any], clone: int, n_obs: int
) -> tuple[list[Any], np.ndarray, np.ndarray]:
    """The Viterbi segments and the rate and BAF each one sits at.

    Returns the segments, `exp(log_mu)` per segment and `p_binom` per
    segment -- the three things the drawn lines are made of. `clone_column`
    is what used to be `0 if new_log_mu.shape[1] == 1 else c`.
    """
    column = clone_column(np.asarray(res_combine["new_log_mu"]))
    n_states = np.asarray(res_combine["new_log_mu"]).shape[0]

    path = clone_path(res_combine["pred_cnv"], clone, n_obs, n_states)
    segments, labels = get_intervals(path)

    rates = np.exp(np.asarray(res_combine["new_log_mu"])[:, column])
    probabilities = np.asarray(res_combine["new_p_binom"])[:, column]

    return segments, rates[labels], probabilities[labels]


def plot_clones_genomic(
    lengths: np.ndarray,
    single_X: np.ndarray,
    single_base_nb_mean: np.ndarray,
    single_total_bb_RD: np.ndarray,
    df_cnv: pd.DataFrame | None = None,
    res_combine: dict[str, Any] | None = None,
    single_tumor_prop: np.ndarray | None = None,
    clone_index: list[Any] | None = None,
    sample_list: list[Any] | None = None,
    remove_xticks: bool = True,
    rdr_ylim: float = 6.0,
    chrtext_shift: float = -0.25,
    base_height: float = 3.2,
    pointsize: float = 3.0,
    linewidth: float = 1.0,
    palette_name: str = "chisel",
    plot_baf_errors: str = "beta",
    plot_rdr_errors: str = "poisson",
    phased_integer_copies: bool = False,
    known_nb_baseline: Any = None,
) -> Any:
    """Upstream's, with the two decodes and the column guard taken from above."""
    logger.info("Plotting aggregated rdr and baf for clones.")

    palette: list[Any] = []
    ordered_acn: list[Any] = []
    state_colors: list[Any] = []
    map_cn: dict[Any, int] = {}
    default_idx = 0
    n_states = 0

    if df_cnv is not None:
        assert res_combine is not None, "res_combine required if df_cnv is provided."

        unique_chrs = np.unique(df_cnv.CHR.to_numpy())
        final_clone_ids = np.sort(np.unique(res_combine["new_assignment"]))
        clone_index = [
            np.where(res_combine["new_assignment"] == c)[0] for c in final_clone_ids
        ]
        color_palette, ordered_acn = get_full_palette(palette_name)
        state_colors = [color_palette[c] for c in ordered_acn]
        map_cn = {x: i for i, x in enumerate(ordered_acn)}
        default_idx = map_cn.get((1, 1), 0)

    elif res_combine is not None:
        unique_chrs = 1 + np.arange(len(lengths))
        final_clone_ids = np.sort(np.unique(res_combine["new_assignment"]))
        clone_index = [
            np.where(res_combine["new_assignment"] == c)[0] for c in final_clone_ids
        ]
        n_states = res_combine["new_p_binom"].shape[0]
        palette = [
            mcolors.to_rgba(color, alpha=1.0)
            for color in sns.color_palette("deep", n_states)
        ]

    else:
        assert clone_index is not None, "clone_index must be provided."
        assert lengths is not None, "lengths must be provided."

        unique_chrs = 1 + np.arange(len(lengths))
        final_clone_ids = np.asarray([str(i) for i in range(len(clone_index))])

    assert single_X.shape[0] == np.sum(lengths), (
        "Mismatch in genomic segment defined X and lengths."
    )

    X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        clone_index,
        single_tumor_prop,
    )

    n_obs = X.shape[0]
    spots_per_clone = [len(xx) for xx in clone_index]
    nonempty_clones = np.where(np.sum(total_bb_RD, axis=0) > 0)[0]

    assert len(nonempty_clones) == total_bb_RD.shape[1]
    assert np.all(nonempty_clones == np.arange(len(final_clone_ids)))

    if known_nb_baseline is not None:
        base_nb_mean = known_nb_baseline.copy()

    has_rdr = base_nb_mean is not None and np.max(base_nb_mean) > 0
    axes_per_clone = 2 if has_rdr else 1

    fig, axes = _create_clone_gridspec(
        len(nonempty_clones), axes_per_clone, base_height, sample_list
    )
    x_vals = np.arange(n_obs)

    for s, raw_clone in enumerate(nonempty_clones):
        # NB `np.where` yields a numpy integer; the helpers take `int`.
        c = int(raw_clone)
        cid = final_clone_ids[c]
        ax_idx = s * axes_per_clone
        ax_rdr: Any = axes[ax_idx] if has_rdr else None
        ax_baf = axes[ax_idx + 1] if has_rdr else axes[ax_idx]

        hue_indices = np.zeros(0, dtype=int)
        this_pred = np.zeros(0, dtype=int)

        if df_cnv is not None:
            if phased_integer_copies:
                allele_1 = df_cnv[f"clone{cid} A"].to_numpy()
                allele_2 = df_cnv[f"clone{cid} B"].to_numpy()
            else:
                allele_1 = np.maximum(
                    df_cnv[f"clone{cid} A"].to_numpy(),
                    df_cnv[f"clone{cid} B"].to_numpy(),
                )
                allele_2 = np.minimum(
                    df_cnv[f"clone{cid} A"].to_numpy(),
                    df_cnv[f"clone{cid} B"].to_numpy(),
                )

            state_tuples = pd.Series(zip(allele_1, allele_2, strict=False))
            hue_indices = (
                state_tuples.map(map_cn).fillna(default_idx).astype(int).to_numpy()
            )

            if palette_name == "chisel":
                palette = [
                    mcolors.to_rgba(
                        color,
                        alpha=(NORMAL_OPACITY if ordered_acn[i] == (1, 1) else 1.0),
                    )
                    for i, color in enumerate(state_colors)
                ]
            else:
                palette = [mcolors.to_rgba(color, alpha=1.0) for color in state_colors]

            point_colors = np.array(palette)[hue_indices]
            unique_hues = np.unique(hue_indices)

        elif res_combine is not None:
            # NB one decode, from `clone_paths`. Upstream writes this here and
            #    again for the Viterbi segments below, taking the modulus by
            #    `new_p_binom.shape[0]` there and `new_log_mu.shape[0]` here.
            this_pred = clone_path(res_combine["pred_cnv"], c, n_obs, n_states)

            assert len(this_pred) == n_obs, (
                f"Clone {cid} copy states defined for {len(this_pred)}, "
                f"expected {n_obs}."
            )

            point_colors = np.array(palette)[this_pred]
            unique_hues = np.unique(this_pred)

        else:
            point_colors = "#4C72B0"
            unique_hues = np.zeros(0, dtype=int)

        if has_rdr:
            y_vals_rdr, std_err_rdr = rdr_track(X, base_nb_mean, c)

            if plot_rdr_errors == "poisson":
                ax_rdr.errorbar(
                    x_vals,
                    y_vals_rdr,
                    yerr=std_err_rdr,
                    fmt="none",
                    ecolor=point_colors,
                    elinewidth=0.5,
                    zorder=0,
                    rasterized=True,
                )

            ax_rdr.scatter(
                x_vals,
                y_vals_rdr,
                s=pointsize,
                c=point_colors,
                edgecolors="none",
                linewidth=linewidth,
                zorder=1,
                rasterized=True,
            )

            _format_track_axis(
                ax_rdr,
                "\nRDR",
                [-0.5, rdr_ylim],
                np.arange(0, rdr_ylim + 1.0, 1.0),
                remove_xticks,
                n_obs,
            )

        baf_vals, std_err_baf = baf_track(X, total_bb_RD, c)

        if plot_baf_errors == "beta":
            ax_baf.errorbar(
                x_vals,
                baf_vals,
                yerr=std_err_baf,
                fmt="none",
                ecolor=point_colors,
                elinewidth=0.5,
                zorder=0,
                rasterized=True,
            )

        ax_baf.scatter(
            x_vals,
            baf_vals,
            s=pointsize,
            c=point_colors,
            edgecolors="none",
            zorder=1,
            rasterized=True,
        )

        _format_track_axis(
            ax_baf,
            "\nBAF",
            [-0.05, 1.05],
            np.arange(0.0, 1.1, 0.2),
            remove_xticks,
            n_obs,
        )

        if res_combine is not None:
            segments, rdr_levels, baf_levels = segment_levels(res_combine, c, n_obs)

            rdr_lines, baf_major_lines, baf_minor_lines = [], [], []

            for i, seg in enumerate(segments):
                x_start, x_end = seg[0], seg[-1]

                if has_rdr:
                    rdr_lines.append([(x_start, rdr_levels[i]), (x_end, rdr_levels[i])])

                y_baf = baf_levels[i]
                baf_major_lines.append([(x_start, y_baf), (x_end, y_baf)])
                baf_minor_lines.append([(x_start, 1.0 - y_baf), (x_end, 1.0 - y_baf)])

            if has_rdr and rdr_lines:
                ax_rdr.add_collection(
                    LineCollection(rdr_lines, colors="k", linewidths=0.5, zorder=2)
                )

            if baf_major_lines:
                ax_baf.add_collection(
                    LineCollection(
                        baf_major_lines, colors="k", linewidths=0.5, zorder=2
                    )
                )
                ax_baf.add_collection(
                    LineCollection(
                        baf_minor_lines,
                        colors="k",
                        linewidths=0.5,
                        linestyles="--",
                        zorder=2,
                    )
                )

        if df_cnv is not None or res_combine is not None:
            legend_labels = ordered_acn if df_cnv is not None else [""] * n_states
            counted = hue_indices if df_cnv is not None else this_pred

            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=palette[i],
                    label=f"{100.0 * np.mean(counted == i):.1f}% {legend_labels[i]}",
                    markersize=10,
                    linestyle="None",
                )
                for i in unique_hues
            ]

            ax_legend = ax_rdr if has_rdr else ax_baf
            ax_legend.legend(
                handles=legend_elements,
                loc="upper right",
                bbox_to_anchor=(1, 1.25),
                ncol=len(legend_elements),
                frameon=False,
                bbox_transform=ax_legend.transAxes,
            )

        t_prop = (
            tumor_prop[c]
            if (single_tumor_prop is not None and tumor_prop is not None)
            else None
        )

        _annotate_clone_stats(
            ax_rdr if has_rdr else ax_baf,
            cid,
            spots_per_clone[c],
            np.sum(X[:, 0, c]),
            np.sum(total_bb_RD[:, c]),
            t_prop,
            paired_ax=ax_baf if has_rdr else None,
        )

    _draw_chromosome_boundaries(axes, lengths, unique_chrs, chrtext_shift)
    fig.tight_layout()

    return fig
