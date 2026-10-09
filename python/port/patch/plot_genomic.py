"""Replaces `cnaster.plot_genomic.plot_clones_genomic`, split into its parts (#299).

`clone_groups`, `bin_colours`, `fitted_levels` and the layout; helpers are
`cnaster`'s, imported. Departure: with the shift on (#293), the RDR line is at
`exp(log_mu - log Z_c)`, where the points are, not upstream's `exp(log_mu)`.
`colour_by` picks integer `(A, B)` or HMM-state colours.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore[import-untyped]
from cnaster.palette import get_full_palette
from cnaster.plot_genomic import (
    NORMAL_OPACITY,
    _annotate_clone_stats,
    _create_clone_gridspec,
    _draw_chromosome_boundaries,
    _format_track_axis,
)
from cnaster.plot_genomic import plot_clones_genomic as UPSTREAM
from cnaster.pseudobulk import merge_pseudobulk_by_index_mix
from cnaster.utils import get_intervals
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from port.extensions.genomic_axis import GenomicAxis, Ticks, resolve

__all__ = [
    "COLOUR_MODES",
    "UPSTREAM",
    "UPSTREAM_WIDTH",
    "Levels",
    "Pooled",
    "bin_colours",
    "clone_axes",
    "clone_groups",
    "draw_genomic",
    "fitted_clone_path",
    "fitted_levels",
    "plot_clones_genomic",
    "pool_genomic",
]

POINT_COLOUR = "#4C72B0"
"""Every bin's colour when there is neither a fit nor integer copies."""

COLOUR_MODES = ("integer", "states")
"""Deduplicated integer `(A, B)`, or one colour per continuous HMM state."""


UPSTREAM_WIDTH = 20.0
"""What `_create_clone_gridspec` hardcodes, in inches."""


def clone_groups(
    res_combine: Any, clone_index: list[np.ndarray] | None
) -> tuple[list[str], list[np.ndarray]]:
    """The clone labels and their spots, in plotting order: from the fit, else `clone_index`."""
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


def fitted_clone_path(res_combine: Any, clone: int, n_obs: int) -> np.ndarray:
    """Clone `clone`'s decoded states, `(n_obs,)`, modulo the state count; either `pred_cnv` layout."""
    from port.patch._clone_paths import clone_path

    pred = np.asarray(res_combine["pred_cnv"], dtype=np.int64)
    n_states = np.asarray(res_combine["new_log_mu"]).shape[0]

    if pred.ndim == 2 and pred.shape[1] > 1:
        pred, clone = pred[:, clone], 0

    return clone_path(pred, clone, n_obs, n_states)


def bin_colours(
    *,
    df_cnv: pd.DataFrame | None,
    res_combine: Any,
    label: str,
    clone: int,
    n_obs: int,
    palette_name: str,
    phased_integer_copies: bool,
    colour_by: str | None = None,
) -> tuple[Any, list[tuple[Any, str]]]:
    """One colour per bin, and the legend entries `(colour, text)`.

    Integer copies (`cnaster`'s palette) when `df_cnv` is given, else the
    decoded state, else one colour. `colour_by="states"` forces states
    (legend `2 mu`, `p`); `"integer"` raises ValueError without `df_cnv`.
    """
    if colour_by not in (None, *COLOUR_MODES):
        msg = f"colour_by is one of {COLOUR_MODES} or None, got {colour_by!r}"
        raise ValueError(msg)

    if colour_by == "integer" and df_cnv is None:
        msg = "colour_by='integer' needs df_cnv, the decoded integer copies"
        raise ValueError(msg)

    if colour_by == "states" and res_combine is None:
        msg = "colour_by='states' needs res_combine, the fitted states"
        raise ValueError(msg)

    if df_cnv is not None and colour_by != "states":
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
        hue = fitted_clone_path(res_combine, clone, n_obs)
        n_states = np.asarray(res_combine["new_log_mu"]).shape[0]
        palette = np.array(
            [mcolors.to_rgba(c) for c in sns.color_palette("deep", n_states)]
        )
        names = [""] * n_states

        if colour_by == "states":
            from port.patch._clone_paths import state_vector

            mu = np.exp(state_vector(res_combine["new_log_mu"]))
            p = state_vector(res_combine["new_p_binom"])
            names = [f"2mu={2.0 * m:.2f} p={q:.2f}" for m, q in zip(mu, p, strict=True)]

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

    RDR is `exp(log_mu)`, or `exp(log_mu - log Z_c)` when `shifted`, with
    `Z_c = sum_g lambda_g mu_{s_c(g)}` (`hmrf.py:476`). BAF is `p` (and `1 - p`).
    """
    from port.patch._clone_paths import state_vector

    log_mu = state_vector(res_combine["new_log_mu"])
    p_binom = state_vector(res_combine["new_p_binom"])
    path = fitted_clone_path(res_combine, clone, n_obs)

    shift = 0.0

    if shifted:
        from port.patch.hmm_nophasing.logmu_shift import clone_log_normalizers

        normalizers = clone_log_normalizers(log_mu, path[:, None], single_base_nb_mean)
        shift = 0.0 if normalizers is None else float(normalizers[0])

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


CLONE_GAP = 0.45
"""The gap row between clones, as a fraction of a track (#339)."""


def clone_axes(figure: Any, n_pairs: int, per_clone: int) -> list[Any]:
    """`_create_clone_gridspec`'s axes on a caller-owned figure, without page size or title."""
    ratios: list[float] = []

    for pair in range(n_pairs):
        ratios.extend([1.0] * per_clone)

        if pair < n_pairs - 1:
            ratios.append(CLONE_GAP)

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
    *,
    figure: Any = None,
    colour_by: str | None = None,
    preferred_colour_by: str | None = None,
    logmu_shift: bool = False,
    axis: GenomicAxis | Ticks | None = None,
) -> Any:
    """Per clone, RDR and BAF along the genome, with the fitted levels.

    `cnaster`'s signature and page, plus: `figure` (draw into a `Figure`/
    `SubFigure`, no `tight_layout`; #309); `colour_by`, defaulting to
    `preferred_colour_by` where possible; `logmu_shift` (RDR line at
    `mu / Z_c`; #299, #517); `axis`, a `GenomicAxis`/`Ticks` coordinate (T- #683).
    """

    if df_cnv is not None and res_combine is None:
        msg = "res_combine is required with df_cnv"
        raise ValueError(msg)

    if single_X.shape[0] != int(np.sum(lengths)):
        msg = f"{single_X.shape[0]} bins against lengths summing to {np.sum(lengths)}"
        raise ValueError(msg)

    if colour_by is None and preferred_colour_by is not None:
        possible = df_cnv is not None if preferred_colour_by == "integer" else True
        colour_by = (
            preferred_colour_by if possible and res_combine is not None else None
        )

    pooled = pool_genomic(
        single_X, single_base_nb_mean, single_total_bb_RD, res_combine, clone_index,
        single_tumor_prop, known_nb_baseline,
    )  # fmt: skip

    page = draw_genomic(
        lengths, pooled, df_cnv, res_combine, sample_list,
        remove_xticks=remove_xticks, rdr_ylim=rdr_ylim, chrtext_shift=chrtext_shift,
        base_height=base_height, pointsize=pointsize, linewidth=linewidth,
        palette_name=palette_name, plot_baf_errors=plot_baf_errors,
        plot_rdr_errors=plot_rdr_errors, phased_integer_copies=phased_integer_copies,
        figure=figure, colour_by=colour_by, logmu_shift=logmu_shift, axis=axis,
    )  # fmt: skip

    if figure is None:
        from port.extensions import figure_record

        figure_record.attach(
            page, "genomic", figure_record.genomic(lengths, pooled, df_cnv, res_combine, clone_index, single_tumor_prop,
                                  known_nb_baseline, sample_list,
                                  (single_X, single_base_nb_mean, single_total_bb_RD)),
            {"remove_xticks": remove_xticks, "rdr_ylim": rdr_ylim, "chrtext_shift": chrtext_shift,
             "base_height": base_height, "pointsize": pointsize, "linewidth": linewidth,
             "palette_name": palette_name, "plot_baf_errors": plot_baf_errors,
             "plot_rdr_errors": plot_rdr_errors, "phased_integer_copies": phased_integer_copies,
             "colour_by": colour_by, "logmu_shift": logmu_shift, "axis": figure_record.axis_option(axis)},
        )  # fmt: skip

    return page


class Pooled(NamedTuple):
    """Each clone's pooled counts, size and the baseline `profile` (summed over spots) (T- #817)."""

    labels: list[str]
    sizes: np.ndarray
    X: np.ndarray
    base_nb_mean: np.ndarray
    total_bb_RD: np.ndarray
    tumor_prop: np.ndarray | None
    profile: np.ndarray


def pool_genomic(
    single_X: np.ndarray,
    single_base_nb_mean: np.ndarray,
    single_total_bb_RD: np.ndarray,
    res_combine: Any,
    clone_index: list[np.ndarray] | None,
    single_tumor_prop: np.ndarray | None,
    known_nb_baseline: np.ndarray | None,
) -> Pooled:
    """The spots pooled per clone, as `plot_clones_genomic` pools them."""
    labels, groups = clone_groups(res_combine, clone_index)

    X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(
        single_X, single_base_nb_mean, single_total_bb_RD, groups, single_tumor_prop
    )

    if not np.all(np.sum(total_bb_RD, axis=0) > 0):
        msg = "a clone holds no allele reads"
        raise ValueError(msg)

    if known_nb_baseline is not None:
        base_nb_mean = known_nb_baseline.copy()

    sizes = np.array([len(g) for g in groups], dtype=np.int64)
    profile = np.asarray(single_base_nb_mean, dtype=np.float64).sum(axis=1)
    return Pooled(labels, sizes, X, base_nb_mean, total_bb_RD, tumor_prop, profile)


def draw_genomic(
    lengths: np.ndarray,
    pooled: Pooled,
    df_cnv: pd.DataFrame | None,
    res_combine: Any,
    sample_list: list[str] | None,
    *,
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
    figure: Any = None,
    colour_by: str | None = None,
    logmu_shift: bool = False,
    axis: GenomicAxis | Ticks | None = None,
) -> Any:
    """`plot_clones_genomic`'s page from `pooled`: what `run_plots` redraws from `cnamaste.h5`."""
    labels = pooled.labels
    X, base_nb_mean, total_bb_RD = pooled.X, pooled.base_nb_mean, pooled.total_bb_RD
    tumor_prop = pooled.tumor_prop
    single_base_nb_mean = pooled.profile[:, None]

    has_rdr = base_nb_mean is not None and np.max(base_nb_mean) > 0
    shifted = logmu_shift and has_rdr

    n_obs = X.shape[0]
    genome = resolve(axis, df_cnv, n_obs)
    x = np.arange(n_obs) if genome is None else genome.warp(np.arange(n_obs))
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
            colour_by=colour_by,
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

            if genome is not None:
                levels = levels._replace(
                    starts=genome.warp(levels.starts), ends=genome.warp(levels.ends)
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
            int(pooled.sizes[clone]),
            np.sum(counts),
            np.sum(trials),
            tumor_prop[clone] if tumor_prop is not None else None,
            paired_ax=ax_baf if ax_rdr is not None else None,
        )

    chromosomes = (
        np.unique(df_cnv.CHR.values)
        if df_cnv is not None
        else 1 + np.arange(len(lengths))
    )
    if genome is None:
        _draw_chromosome_boundaries(axes, lengths, chromosomes, chrtext_shift)
    else:
        _draw_chromosome_boundaries(
            axes, np.diff(genome.edges), chromosomes, chrtext_shift
        )

        # NB each clone's lowest track ticked, the page's last labelled.
        for ax in axes[per_clone - 1 :: per_clone]:
            genome.draw(ax, labels=ax is axes[-1])

    if owned:
        figure.tight_layout()

    return figure
