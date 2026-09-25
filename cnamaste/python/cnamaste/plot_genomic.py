from typing import Any, Dict, Optional

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

# from cnamaste.hmm_nophasing import hmm_nophasing
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from cnamaste.config import get_global_config, start_time
from cnamaste.hmm_phased import hmm_phased
from cnamaste.logger import get_logger
from cnamaste.palette import get_full_palette
from cnamaste.pseudobulk import merge_pseudobulk_by_index_mix
from cnamaste.utils import cast_clone_label, get_intervals, top_hat_sum, write_fig
from typing import NamedTuple
import scipy.special

logger = get_logger(__name__, start_time=start_time)

NORMAL_OPACITY = 0.75


def _create_clone_gridspec(
    n_pairs: int, axes_per_clone: int, base_height: float, sample_list: list = None
):
    """
    Creates a flexible GridSpec layout for plotting clones, automatically injecting
    vertical gaps between distinct clone tracks.
    """
    n_axes_total = axes_per_clone * n_pairs
    fig = plt.figure(figsize=(20, base_height * n_pairs), dpi=300, facecolor="white")

    height_ratios = []
    for i in range(n_pairs):
        height_ratios.extend([1] * axes_per_clone)
        if i < n_pairs - 1:
            height_ratios.append(0.25)  # Gap between clone tracks

    gs = gridspec.GridSpec(len(height_ratios), 1, height_ratios=height_ratios, hspace=0)

    axes, row = [], 0
    for i in range(n_axes_total):
        axes.append(fig.add_subplot(gs[row, 0]))
        row += 1
        if (i % axes_per_clone == axes_per_clone - 1) and (i < n_axes_total - 1):
            row += 1  # Skip the gap row

    if sample_list is not None:
        fig.suptitle(", ".join(sample_list), x=0.5, y=0.99, fontsize=16, ha="center")

    return fig, axes


def _format_track_axis(ax, ylabel, ylim, yticks, remove_xticks, n_obs):
    """Standardizes the styling for rdr and baf genomic tracks."""
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.1f}" for y in yticks])
    ax.set_xlim([0, n_obs])

    if remove_xticks:
        ax.set_xticks([])

    for y in yticks:
        ax.axhline(y=y, c="lightgray", linewidth=0.5, zorder=0)


def _draw_chromosome_boundaries(axes, lengths, unique_chrs, chrtext_shift):
    """Draws vertical contig boundaries and appends chromosome labels to the bottom axis."""
    for i in range(len(lengths)):
        start_len = np.sum(lengths[:i])

        # Label only on the bottom-most axis
        axes[-1].text(
            start_len,
            chrtext_shift,
            f"chr{unique_chrs[i]}",
            rotation=45,
            transform=axes[-1].get_xaxis_transform(),
            fontsize=10,
            ha="left",
        )
        # Draw boundaries across all axes
        for ax in axes:
            ax.axvline(x=start_len, c="black", linewidth=0.5)


def _annotate_clone_stats(
    ax,
    clone_label,
    n_spots,
    n_umis,
    n_snp_umis,
    tumor_prop=None,
    paired_ax=None,
):
    """Annotates the axis with clone identity and spot/UMI statistics."""
    x_offset = -0.04

    if paired_ax is None:
        ax.text(
            x_offset,
            0.5,
            cast_clone_label(str(clone_label)),
            ha="center",
            va="center",
            fontsize=12,
            rotation="vertical",
            transform=ax.transAxes,
            clip_on=False,
        )
    else:
        ax.text(
            x_offset,
            0.0,
            cast_clone_label(str(clone_label)),
            ha="center",
            va="center",
            fontsize=12,
            rotation="vertical",
            transform=ax.transAxes,
            clip_on=False,
        )

    theta_text = (
        f"$\\hat{{\\theta}}={tumor_prop:.2f}$" if tumor_prop is not None else ""
    )
    stats_text = f"{n_spots:_} spots; {int(n_umis):_} umis; {int(n_snp_umis):_} snp-umis; {theta_text}"

    ax.text(
        0.0,
        1.02,
        stats_text.strip("; "),
        ha="left",
        va="bottom",
        fontsize=12,
        transform=ax.transAxes,
    )


'''
def plot_clones_genomic(
    df_cnv,  # Can be None for plotting raw data, or __real__ states, rather than integer.
    lengths: np.ndarray,
    single_X: np.ndarray,
    single_base_nb_mean: np.ndarray,
    single_total_bb_RD: np.ndarray,
    res_combine: dict = None, # OWNS clone definition via new_assignment key.
    single_tumor_prop: np.ndarray = None,
    clone_ids: list = None, # DEPRECATE
    clone_index: list = None,
    sample_list: list = None,
    remove_xticks: bool = True,
    rdr_ylim: float = 6.0,
    chrtext_shift: float = -0.2,
    base_height: float = 3.2,
    pointsize: float = 3.0,
    linewidth: float = 1.0,
    palette_name: str = "chisel",
    plot_baf_errors: str = "beta",
    plot_rdr_errors: str = "poisson",
):
    """
    Plots aggregated rdr and baf (with error models) and best-fit continous copy states (mu, p).
    If df_cnv is None, functions as a raw data plotter without categorical integer states.
    """
    logger.info(f"Plotting aggregated rdr and baf for clones.")

    color_palette, ordered_acn = get_full_palette(palette_name)
    state_colors = [color_palette[c] for c in ordered_acn]

    assert clone_ids is None

    # NB defines final_clone_ids, unique_chrs and clone_index based on available data, in order of priority.
    if df_cnv is not None:
        map_cn = {x: i for i, x in enumerate(ordered_acn)}
        unique_chrs = np.unique(df_cnv.CHR.values)

        assert res_combine is not None

        # final_clone_ids = (
        #     df_cnv.columns.str.extract(r"^clone(.*) A$", expand=False).dropna().tolist()
        # )

        final_clone_ids = np.sort(np.unique(res_combine["new_assignment"]))
        clone_index = [
            np.where(res_combine["new_assignment"] == c)[0]
            for c, _ in enumerate(final_clone_ids)
        ]
    elif res_combine is not None:
        unique_chrs = 1 + np.arange(len(lengths))
        final_clone_ids = np.sort(np.unique(res_combine["new_assignment"]))

        clone_index = [
            np.where(res_combine["new_assignment"] == c)[0]
            for c, _ in enumerate(final_clone_ids)
        ]
    else:
        assert clone_index is not None, "clone_index must be provided."
        assert lengths is not None, "lengths must be provided."        

        unique_chrs = 1 + np.arange(len(lengths))
        final_clone_ids = [str(i) for i in range(len(clone_index))]

    # NB requires lengths to be provided.
    assert single_X.shape[0] == np.sum(
        lengths
    ), "Found mismatch for genomic segment defined X and lengths."

    # NB requires clone_index to be defined.
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

    has_rdr = base_nb_mean is not None and np.max(base_nb_mean) > 0

    axes_per_clone = 2 if has_rdr else 1
    fig, axes = _create_clone_gridspec(
        len(nonempty_clones), axes_per_clone, base_height, sample_list
    )

    # NB guards against trouble.
    # DEPRECATE
    # assert "0" in final_clone_ids
    assert np.all(nonempty_clones == np.arange(len(final_clone_ids))), f"Found nonempty clones={nonempty_clones}, expected={np.arange(len(final_clone_ids))}."

    # DEPRECATE s or c should be used, but not both.
    for s, c in enumerate(nonempty_clones):
        cid = final_clone_ids[c]
        ax_idx = s * axes_per_clone

        if has_rdr:
            ax_rdr = axes[ax_idx]
            ax_baf = axes[ax_idx + 1]
        else:
            ax_rdr = None
            ax_baf = axes[ax_idx]

        if df_cnv is not None:
            major = np.maximum(df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values)
            minor = np.minimum(df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values)

            default_idx = map_cn.get((1, 1), 0)
            hue_indices = [map_cn.get((major[i], minor[i]), default_idx) for i in range(len(major))]
            hue = pd.Categorical(hue_indices, categories=np.arange(len(ordered_acn)), ordered=True)

            if palette_name == "chisel":
                palette = [
                    mcolors.to_rgba(color, alpha=(NORMAL_OPACITY if ordered_acn[i] == (1, 1) else 1.0))
                    for i, color in enumerate(state_colors)
                ]
            else:
                base_pal = sns.color_palette(palette_name, len(ordered_acn))
                palette = [mcolors.to_rgba(color, alpha=1.0) for color in base_pal]

            point_colors = [palette[h] for h in hue.codes]
            scatter_kwargs = {"hue": hue, "palette": palette}
            
        elif res_combine is not None:
            n_states = res_combine["new_p_binom"].shape[0]

            if res_combine["pred_cnv"].ndim == 1 or res_combine["pred_cnv"].shape[1] == 1:
                this_pred = res_combine["pred_cnv"][(c * n_obs) : (c * n_obs + n_obs)].flatten() % n_states
            else:
                this_pred = res_combine["pred_cnv"][:, c] % n_states

            assert len(this_pred) == n_obs, (
                f"Clone {cid} copy states are defined for {len(this_pred)} segments, but data suggested {n_obs}."
            )

            hue = pd.Categorical(this_pred, categories=np.arange(n_states), ordered=True)

            base_pal = sns.color_palette("deep", n_states) 
            palette = [mcolors.to_rgba(color, alpha=1.0) for color in base_pal]
            point_colors = [palette[h] for h in hue.codes]
            scatter_kwargs = {"hue": hue, "palette": palette}
            
        else:
            point_colors = "#4C72B0"
            scatter_kwargs = {"color": point_colors}

        x_vals = np.arange(n_obs)

        if has_rdr:
            y_vals_rdr = X[:, 0, c] / base_nb_mean[:, c]

            if plot_rdr_errors == "poisson":
                with np.errstate(divide="ignore", invalid="ignore"):
                    std_err_rdr = np.sqrt(X[:, 0, c]) / base_nb_mean[:, c]
                    std_err_rdr[~np.isfinite(std_err_rdr)] = 0.0

                ax_rdr.errorbar(
                    x_vals, y_vals_rdr, yerr=std_err_rdr, fmt="none",
                    ecolor=point_colors, elinewidth=0.5, zorder=0,
                )

            sns.scatterplot(
                x=x_vals, y=y_vals_rdr, s=pointsize, edgecolor="none",
                linewidth=linewidth, legend=False, ax=ax_rdr, zorder=1,
                **scatter_kwargs,
            )

            _format_track_axis(
                ax_rdr, "\nRDR", [-0.5, rdr_ylim], np.arange(0, rdr_ylim + 1.0, 1.0),
                remove_xticks, n_obs,
            )


        baf_vals = X[:, 1, c] / total_bb_RD[:, c]

        if plot_baf_errors == "beta":
            k, n = X[:, 1, c], total_bb_RD[:, c]
            alpha_param, beta_param = k + 1, n - k + 1
            alpha_beta_sum = alpha_param + beta_param
            std_err_baf = np.sqrt(
                (alpha_param * beta_param) / (np.square(alpha_beta_sum) * (alpha_beta_sum + 1))
            )

            ax_baf.errorbar(
                x_vals, baf_vals, yerr=std_err_baf, fmt="none",
                ecolor=point_colors, elinewidth=0.5, zorder=0,
            )

        sns.scatterplot(
            x=x_vals, y=baf_vals, s=pointsize, edgecolor="none",
            legend=False, ax=ax_baf, zorder=1, **scatter_kwargs,
        )

        _format_track_axis(
            ax_baf, "\nBAF", [-0.05, 1.05], np.arange(0.0, 1.1, 0.2),
            remove_xticks, n_obs,
        )

        if res_combine is not None:
            # n_states = res_combine["n_states"]
            n_states = res_combine["new_log_mu"].shape[0]
            clone_idx = 0 if res_combine["new_log_mu"].shape[1] == 1 else c

            # NB clones concatenated along a single axis.
            if res_combine["pred_cnv"].ndim == 1 or res_combine["pred_cnv"].shape[1] == 1:
                this_pred = res_combine["pred_cnv"][(c * n_obs) : (c * n_obs + n_obs)] % n_states
            # NB one columne per clone.
            else:
                this_pred = res_combine["pred_cnv"][:, c] % n_states

            segments, labels = get_intervals(this_pred)
            
            for i, seg in enumerate(segments):
                if has_rdr:
                    ax_rdr.plot(
                        seg, [np.exp(res_combine["new_log_mu"][labels[i], clone_idx])] * 2,
                        c="k", linewidth=0.5, zorder=2,
                    )
                ax_baf.plot(
                    seg, [res_combine["new_p_binom"][labels[i], clone_idx]] * 2,
                    c="k", linewidth=0.5, zorder=2,
                )
                ax_baf.plot(
                    seg, [1.0 - res_combine["new_p_binom"][labels[i], clone_idx]] * 2,
                    c="k", linewidth=0.5, linestyle="--", zorder=2,
                )

        if df_cnv is not None or res_combine is not None:
            # NB we don't add a legend label for a state enumerator as confusing,
            #    just the percentage.
            legend_labels = ordered_acn if df_cnv is not None else n_states * [""]
            legend_elements = [
                Line2D(
                    [0], [0], marker="o", color="w", markerfacecolor=palette[i],
                    label=f"{100. * np.mean(hue == i):.1f}% {legend_labels[i]}",
                    markersize=10, linestyle="None",
                )
                for i in hue.unique()
            ]

            ax_legend = ax_rdr if has_rdr else ax_baf
            ax_legend.legend(
                handles=legend_elements, loc="upper right", bbox_to_anchor=(1, 1.25),
                ncol=len(legend_elements), frameon=False, bbox_transform=ax_legend.transAxes,
            )

        t_prop = tumor_prop[c] if (single_tumor_prop is not None and tumor_prop is not None) else None
        _annotate_clone_stats(
            ax_rdr if has_rdr else ax_baf, cid, spots_per_clone[c],
            np.sum(X[:, 0, c]), np.sum(total_bb_RD[:, c]), t_prop,
            paired_ax=ax_baf if has_rdr else None,
        )

    _draw_chromosome_boundaries(axes, lengths, unique_chrs, chrtext_shift)

    fig.tight_layout()

    return fig
'''


POINT_COLOUR = "#4C72B0"
"""Every bin's colour when there is neither a fit nor integer copies."""


def clone_groups(
    res_combine: Any, clone_index: list[np.ndarray] | None
) -> tuple[list[str], list[np.ndarray]]:
    """The clone labels, and the spots in each, in plotting order.

    From the fit's assignment when there is one, else from `clone_index`.
    """
    if res_combine is None:
        if clone_index is None:
            msg = "clone_index is required when there is no res_combine"
            raise ValueError(msg)

        return [str(i) for i in range(len(clone_index))], list(clone_index)

    assignment = np.asarray(res_combine["new_assignment"])
    labels = np.sort(np.unique(assignment))

    return [str(label) for label in labels], [
        np.flatnonzero(assignment == label) for label in labels
    ]


def clone_path(res_combine: Any, clone: int, n_obs: int) -> np.ndarray:
    """Clone `clone`'s decoded states, `(n_obs,)`, modulo the state count.

    `pred_cnv` comes either with one column per clone (`run_core_inference`
    deconcatenates) or with the clones concatenated along the genome.
    """
    pred = np.asarray(res_combine["pred_cnv"])
    n_states = np.asarray(res_combine["new_log_mu"]).shape[0]

    if pred.ndim == 2 and pred.shape[1] > 1:
        path = pred[:, clone]
    else:
        path = pred.reshape(-1)[clone * n_obs : (clone + 1) * n_obs]

    states: np.ndarray = np.asarray(path, dtype=np.int64) % n_states

    return states


def bin_colours(
    *,
    df_cnv: pd.DataFrame | None,
    res_combine: Any,
    label: str,
    clone: int,
    n_obs: int,
    palette_name: str,
    phased_integer_copies: bool,
) -> tuple[Any, list[tuple[Any, str]]]:
    """One colour per bin, and the legend entries `(colour, text)`.

    Integer copies when `df_cnv` is given, coloured by `cnamaste`'s palette
    with normal `(1, 1)` faded; else the decoded state, one seaborn colour
    each; else a single colour and no legend.
    """
    if df_cnv is not None:
        colour_of, ordered = get_full_palette(palette_name)
        first = df_cnv[f"clone{label} A"].to_numpy()
        second = df_cnv[f"clone{label} B"].to_numpy()

        if not phased_integer_copies:
            first, second = np.maximum(first, second), np.minimum(first, second)

        index = {pair: i for i, pair in enumerate(ordered)}
        normal = index.get((1, 1), 0)
        hue = np.array(
            [index.get(pair, normal) for pair in zip(first, second, strict=True)]
        )

        faded = palette_name == "chisel"
        palette = np.array(
            [
                mcolors.to_rgba(
                    colour_of[pair],
                    alpha=NORMAL_OPACITY if faded and pair == (1, 1) else 1.0,
                )
                for pair in ordered
            ]
        )
        names = [str(pair) for pair in ordered]

    elif res_combine is not None:
        hue = clone_path(res_combine, clone, n_obs)
        n_states = np.asarray(res_combine["new_log_mu"]).shape[0]
        palette = np.array(
            [mcolors.to_rgba(c) for c in sns.color_palette("deep", n_states)]
        )
        names = [""] * n_states

    else:
        return POINT_COLOUR, []

    legend = [
        (palette[i], f"{100.0 * np.mean(hue == i):.1f}% {names[i]}".strip())
        for i in np.unique(hue)
    ]

    return palette[hue], legend


class Levels(NamedTuple):
    """The runs of equal state along one clone, and where each is drawn."""

    starts: np.ndarray
    ends: np.ndarray
    rdr: np.ndarray
    baf: np.ndarray


def fitted_levels(
    res_combine: Any,
    clone: int,
    n_obs: int,
    single_base_nb_mean: np.ndarray,
    shifted: bool,
) -> Levels:
    """Each run of one state along clone `clone`, at its fitted RDR and BAF.

    RDR is `exp(log_mu)` unshifted and `exp(log_mu - log Z_c)` shifted, with
    `Z_c = sum_g lambda_g mu_{s_c(g)}` over this clone's path and `lambda`
    the baseline summed over spots and normalized, as `hmrf.py:476` builds
    it. BAF is the fitted `p`, drawn with its mirror `1 - p`.
    """
    from cnamaste.clone_paths import state_vector

    log_mu = state_vector(res_combine["new_log_mu"])
    p_binom = state_vector(res_combine["new_p_binom"])
    path = clone_path(res_combine, clone, n_obs)

    shift = 0.0

    if shifted:
        profile = np.asarray(single_base_nb_mean, dtype=np.float64).sum(axis=1)

        with np.errstate(divide="ignore"):
            log_lambda = np.log(profile / profile.sum())

        shift = float(scipy.special.logsumexp(log_mu[path] + log_lambda))

    segments, states = get_intervals(path)
    states = np.asarray(states, dtype=np.int64)

    return Levels(
        starts=np.array([segment[0] for segment in segments], dtype=np.float64),
        ends=np.array([segment[-1] for segment in segments], dtype=np.float64),
        rdr=np.exp(log_mu[states] - shift),
        baf=p_binom[states],
    )


def _horizontal(
    axis: Any, levels: Levels, heights: np.ndarray, style: str = "solid"
) -> None:
    lines = [
        [(start, height), (end, height)]
        for start, end, height in zip(levels.starts, levels.ends, heights, strict=True)
    ]
    axis.add_collection(
        LineCollection(lines, colors="k", linewidths=0.5, linestyles=style, zorder=2)
    )


def _points(
    axis: Any,
    x: np.ndarray,
    y: np.ndarray,
    error: np.ndarray | None,
    colours: Any,
    pointsize: float,
    linewidth: float,
) -> None:
    if error is not None:
        axis.errorbar(
            x,
            y,
            yerr=error,
            fmt="none",
            ecolor=colours,
            elinewidth=0.5,
            zorder=0,
            rasterized=True,
        )

    axis.scatter(
        x,
        y,
        s=pointsize,
        c=colours,
        edgecolors="none",
        linewidth=linewidth,
        zorder=1,
        rasterized=True,
    )


def clone_axes(figure: Any, n_pairs: int, per_clone: int) -> list[Any]:
    """`_create_clone_gridspec`'s axes, on a figure the caller owns.

    The same rows -- `per_clone` tracks per clone, a quarter-height gap
    between clones, no vertical space -- without the 20 in page or the
    title, which belong to whoever composes the figure.
    """
    ratios: list[float] = []

    for pair in range(n_pairs):
        ratios.extend([1.0] * per_clone)

        if pair < n_pairs - 1:
            ratios.append(0.25)

    grid = figure.add_gridspec(len(ratios), 1, height_ratios=ratios, hspace=0)
    rows = [row for row, ratio in enumerate(ratios) if ratio == 1.0]

    return [figure.add_subplot(grid[row, 0]) for row in rows]


def plot_clones_genomic(
    lengths: np.ndarray,
    single_X: np.ndarray,
    single_base_nb_mean: np.ndarray,
    single_total_bb_RD: np.ndarray,
    df_cnv: pd.DataFrame | None = None,
    res_combine: Any = None,
    single_tumor_prop: np.ndarray | None = None,
    clone_index: list[np.ndarray] | None = None,
    sample_list: list[str] | None = None,
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
    known_nb_baseline: np.ndarray | None = None,
    figure: Any = None,
) -> Any:
    """Per clone, RDR and BAF along the genome, with the fitted levels.

    `cnamaste`'s signature and page. The RDR level is shifted by the clone's
    `log Z_c` when the fit was (`port.patch.hmm_nophasing`'s flag), so the
    line sits on the bins it describes.

    `figure`, a `Figure` or `SubFigure`, is drawn into rather than a new
    20 in page, which is how `port.extensions.combined_figure` sets it in a
    column (#309). The layout is then the caller's, so no `tight_layout`.
    """
    from cnamaste.hmm_nophasing import hmm_nophasing

    if df_cnv is not None and res_combine is None:
        msg = "res_combine is required with df_cnv"
        raise ValueError(msg)

    if single_X.shape[0] != int(np.sum(lengths)):
        msg = f"{single_X.shape[0]} bins against lengths summing to {np.sum(lengths)}"
        raise ValueError(msg)

    labels, groups = clone_groups(res_combine, clone_index)

    X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(
        single_X, single_base_nb_mean, single_total_bb_RD, groups, single_tumor_prop
    )

    if not np.all(np.sum(total_bb_RD, axis=0) > 0):
        msg = "a clone holds no allele reads"
        raise ValueError(msg)

    if known_nb_baseline is not None:
        base_nb_mean = known_nb_baseline.copy()

    has_rdr = base_nb_mean is not None and np.max(base_nb_mean) > 0
    shifted = bool(hmm_nophasing.apply_logmu_shift) and has_rdr

    n_obs = X.shape[0]
    x = np.arange(n_obs)
    per_clone = 2 if has_rdr else 1

    if figure is None:
        figure, axes = _create_clone_gridspec(
            len(labels), per_clone, base_height, sample_list
        )
        owned = True
    else:
        axes = clone_axes(figure, len(labels), per_clone)
        owned = False

    for clone, label in enumerate(labels):
        ax_rdr = axes[per_clone * clone] if has_rdr else None
        ax_baf = axes[per_clone * clone + per_clone - 1]

        colours, legend = bin_colours(
            df_cnv=df_cnv,
            res_combine=res_combine,
            label=label,
            clone=clone,
            n_obs=n_obs,
            palette_name=palette_name,
            phased_integer_copies=phased_integer_copies,
        )

        counts, trials = X[:, 0, clone], total_bb_RD[:, clone]
        successes = X[:, 1, clone]

        if ax_rdr is not None:
            baseline = base_nb_mean[:, clone]

            with np.errstate(divide="ignore", invalid="ignore"):
                rdr = counts / baseline
                rdr_error = np.nan_to_num(np.sqrt(counts) / baseline, posinf=0.0)

            _points(
                ax_rdr,
                x,
                rdr,
                rdr_error if plot_rdr_errors == "poisson" else None,
                colours,
                pointsize,
                linewidth,
            )
            _format_track_axis(
                ax_rdr,
                "\nRDR",
                [-0.5, rdr_ylim],
                np.arange(0, rdr_ylim + 1.0, 1.0),
                remove_xticks,
                n_obs,
            )

        # NB the posterior sd of a beta(k + 1, n - k + 1), as upstream.
        a, b = successes + 1, trials - successes + 1
        baf_error = np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))

        _points(
            ax_baf,
            x,
            successes / trials,
            baf_error if plot_baf_errors == "beta" else None,
            colours,
            pointsize,
            linewidth,
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
            levels = fitted_levels(
                res_combine, clone, n_obs, single_base_nb_mean, shifted
            )

            if ax_rdr is not None:
                _horizontal(ax_rdr, levels, levels.rdr)

            _horizontal(ax_baf, levels, levels.baf)
            _horizontal(ax_baf, levels, 1.0 - levels.baf, "--")

        if legend:
            anchor = ax_rdr if ax_rdr is not None else ax_baf
            anchor.legend(
                handles=[
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=colour,
                        label=text,
                        markersize=10,
                        linestyle="None",
                    )
                    for colour, text in legend
                ],
                loc="upper right",
                bbox_to_anchor=(1, 1.25),
                ncol=len(legend),
                frameon=False,
                bbox_transform=anchor.transAxes,
            )

        _annotate_clone_stats(
            ax_rdr if ax_rdr is not None else ax_baf,
            label,
            len(groups[clone]),
            np.sum(counts),
            np.sum(trials),
            tumor_prop[clone] if single_tumor_prop is not None else None,
            paired_ax=ax_baf if ax_rdr is not None else None,
        )

    chromosomes = (
        np.unique(df_cnv.CHR.values)
        if df_cnv is not None
        else 1 + np.arange(len(lengths))
    )
    _draw_chromosome_boundaries(axes, lengths, chromosomes, chrtext_shift)

    if owned:
        figure.tight_layout()

    return figure


# TODO define width
def plot_cna_mixture(
    init_log_mu,
    init_alphas,
    init_p_binom,
    init_taus,
    X,
    base_nb_mean,
    total_bb_RD,
    width=1,
    prefix="initial",
    max_rdr=None,
):
    # NB base_nb_mean is zero until post-BAF normal identication; in which case,
    #    these will be NAN.
    X_gmm_rdr = np.vstack(
        [X[:, 0, s] / base_nb_mean[:, s] for s in range(X.shape[2])]
    ).T

    assert X_gmm_rdr.shape == (len(X[:, 0, 0]), X.shape[2])

    valid = ~np.isnan(X_gmm_rdr) & ~np.isinf(X_gmm_rdr)

    if np.all(~valid):
        X_gmm_rdr[~valid] = np.random.normal(
            loc=1.0, scale=0.01, size=np.count_nonzero(~valid)
        )

    # TODO clipping?
    X_gmm_baf = np.vstack(
        [
            top_hat_sum(X[:, 1, s], width) / top_hat_sum(total_bb_RD[:, s], width)
            for s in range(X.shape[2])
        ]
    ).T

    if init_p_binom is None:
        init_p_binom, _ = get_betabinom_start_params()
        init_p_binom = np.tile(np.array(init_p_binom).reshape(-1, 1), (1, X.shape[2]))

    if init_log_mu is None:
        init_log_mu, _ = get_nbinom_start_params()
        init_log_mu = np.array(init_log_mu).reshape(-1, 1)
        init_log_mu = np.tile(init_log_mu, (1, X.shape[2]))

    if init_alphas is None:
        config = get_global_config()
        init_alphas = config.nbinom.start_disp * np.ones_like(init_log_mu)
        init_alphas = np.tile(init_alphas, (1, X.shape[2]))

    if init_taus is None:
        config = get_global_config()
        init_taus = config.betabinom.start_disp * np.ones_like(init_p_binom)
        init_taus = np.tile(init_taus, (1, X.shape[2]))

    init_mu = np.exp(init_log_mu)

    num_clones, num_segments = X.shape[2], X.shape[0]

    # NB S,n = 3,4 ... [0 1 2 0 1 2 0 1 2 0 1 2], i.e. column major.
    clone_idx = np.tile(np.arange(num_clones), num_segments)

    palette = sns.color_palette(n_colors=num_clones)

    x = X_gmm_baf.ravel()
    y = X_gmm_rdr.ravel()

    g = sns.JointGrid(x=x, y=y, height=8, ratio=3, space=0.15)
    valid_mask = np.isfinite(x) & np.isfinite(y)

    lnlike_rdr, lnlike_baf = hmm_phased.compute_emission_probability_nb_betabinom(
        X, base_nb_mean, init_log_mu, init_alphas, total_bb_RD, init_p_binom, init_taus
    )

    lnlike = lnlike_baf + lnlike_rdr

    # TODO track finite.
    lnlike[~np.isfinite(lnlike)] = -np.inf

    # NB emission prob. under (0.0, 0.5)
    # best_lnlike = lnlike[0,:,:].ravel()

    # NB emission prob. under best state (includes phase flip complement).
    best_lnlike = np.max(lnlike, axis=0).ravel()

    like_ratio = np.exp(best_lnlike - best_lnlike.max())

    # alpha = 0.2 + (like_ratio - like_ratio.min()) * 0.8 / (1.0 - like_ratio.min())
    alpha = like_ratio

    for c in range(num_clones):
        clone_mask = (clone_idx == c) & valid_mask

        if np.any(clone_mask):
            g.ax_joint.scatter(
                x[clone_mask],
                y[clone_mask],
                s=1,
                marker=".",
                alpha=alpha,
                color=palette[c],
            )

    g.ax_joint.scatter(
        init_p_binom, init_mu, marker="*", facecolor="none", edgecolor="k", s=25
    )

    if max_rdr is not None:
        g.ax_joint.set_ylim(-1, max_rdr)

    xticks = np.arange(0.0, 1.05, 0.05)
    g.ax_joint.set_xticks(xticks)
    g.ax_joint.set_xticklabels(
        [f"{x:.2f}" if ((1 + ii) % 2) else "" for ii, x in enumerate(xticks)]
    )

    bins = 50

    validx = np.isfinite(x)
    validy = np.isfinite(y)

    bins_x = np.arange(-0.01, 1.0, 5.0e-3)
    bins_y = np.arange(-0.1, 10.0, 0.1)

    centers_x = 0.5 * (bins_x[:-1] + bins_x[1:])
    width_x = bins_x[1] - bins_x[0]

    centers_y = 0.5 * (bins_y[:-1] + bins_y[1:])
    height_y = bins_y[1] - bins_y[0]

    legend_patches = []

    for c in range(num_clones):
        clone_mask = clone_idx == c

        assert np.any(clone_mask)

        counts_x, _ = np.histogram(x[clone_mask], bins=bins_x)
        counts_y, _ = np.histogram(y[clone_mask], bins=bins_y)

        g.ax_marg_x.bar(
            centers_x,
            counts_x,
            width=width_x,
            align="center",
            facecolor="none",
            edgecolor=palette[c],
            linewidth=1.0,
            alpha=1.0,
        )

        g.ax_marg_y.barh(
            centers_y,
            counts_y,
            height=height_y,
            align="center",
            facecolor="none",
            edgecolor=palette[c],
            linewidth=1.0,
            alpha=0.5,
        )

        legend_patches.append(
            mpatches.Patch(
                facecolor="none",
                edgecolor=palette[c],
                label=cast_clone_label(f"clone {c}") if num_clones > 1 else "",
            )
        )

    g.set_axis_labels("ZHF", "RDR")
    g.ax_joint.legend(handles=legend_patches, loc="upper left", framealpha=0.0)

    fig = g.fig

    config = get_global_config()

    # {config.hmrf.n_clones_rdr}
    output_dir = f"{config.paths.output_dir}/clone{config.hmrf.n_clones}_rectangle{config.hmrf.random_state}_w{config.hmrf.spatial_weight:.1f}/"
    fig_path = f"{output_dir}/plots/{prefix}_rdr_baf.pdf"

    logger.info(f"Writing initial copy state mixture plot to:\n{fig_path}")

    write_fig(fig_path, fig, transparent=True, bbox_inches="tight")
