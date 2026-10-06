"""#540: `port.studies.copy_state_stream`'s runs against runtime, each start numbered as in the table beside it.

`run_study --copy-state-plot STREAM.record` writes `<stem>.png` beside
the record: each run's log-likelihood below the best any run reached on its
realization -- there is no bound for an HMM's likelihood -- on a log axis whose
bottom tick, "0", holds the runs at that best. A filled marker is the start's
own states, decoded by the HMM without fitting; the open marker after the
arrow is the same runs after `cnaster`'s Baum-Welch, drawn `DODGE` right of its
runtime. The solid line and dark grey band are Baum-Welch from the planted
states: its median and 10-90% range over realizations.

Each point is a start's median over realizations x seeds, its error bars the
10-90% range on both axes. The table's Missed column is the percentage of rows
(clone x bin) not in their planted state under the best 1-1 matching of
states, the median over the same runs: at the start / after Baum-Welch.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from port.qa.statistics import bars, ranks
from port.studies import records
from port.studies.figures import key_below, merged, stamp, tab20, tt

TABLE = (
    ("CalicoST, port", (
        ("calicost-gmm", f"CalicoST's {tt('initialization_by_gmm')}, clones stacked"),
        ("lattice", "Integer (A, B) lattice, chosen by the rows"),
    )),
    ("sal, one draw", (
        ("prior", "Drawn from a prior on the observed range"),
        ("kmeans++", f"{tt('k-means++')} on the raw count pair"),
        ("emission++", "Seeds by the NB x BB Bregman divergence"),
    )),
    ("sal, samplers on the HMM (#634)", (
        ("tempering-hmm", "Best point of sal's parallel tempering, 4 HMC replicas"),
        ("hmc-hmm", "Best draw of sal's HMC chain after dual-averaging warm-up"),
    )),
)  # fmt: skip
"""The starts drawn (T- #660). Set aside from the figure, still in the registry: `cnaster-gmm`,
`distinct`, `lattice-em`, `rdr-quantiles`, `data`, `quantile`, the emission++ variants and `anneal-hmm` (#716)."""


LABEL = {
    "calicost-gmm": "calicost-gmm", "lattice": "lattice", "prior": "prior", "kmeans++": "k-means++",
    "emission++": "emission++", "gaussian-em": "gaussian-em",
    "anneal-hmm": "anneal", "tempering-hmm": "parallel tempering", "hmc-hmm": "hmc",
}  # fmt: skip
"""A start's label."""
NUMBER = {name: k + 1 for k, name in enumerate(n for _, rows in TABLE for n, _ in rows)}
NUMBER_TEXT = {name: str(k) for name, k in NUMBER.items()}
SOURCE = {
    **{name: "sal" for _, rows in TABLE for name, _ in rows},
    "calicost-gmm": "CalicoST", "lattice": "port",
}  # fmt: skip
"""Each start's source: the package whose code it runs."""
KEY_NAMES = {
    "calicost-gmm": "CalicoST-GMM", "lattice": "Lattice", "prior": "Prior", "kmeans++": r"$k$-means++",
    "emission++": "Emission++",
    "tempering-hmm": "Parallel tempering", "hmc-hmm": "HMC",
}  # fmt: skip
"""The names `solver_combined`'s key prints (#716)."""

COLOUR = {name: tab20(k) for name, k in NUMBER.items()}
"""One colour per start, by its number."""

DODGE = 1.12
FLOOR = 1e-2

RUNTIME_FLOOR = 0.5
"""With `key`, runtimes below it are drawn at it, a left arrow marking the bound [s]."""
"""The "0" tick: runs within `FLOOR` nats of the best."""


def degenerate_counts(record: dict[str, Any]) -> dict[str, int]:
    """Each start's runs flagged degenerate (`known_copy.degenerate`): `cnaster`'s NB at probability 1."""
    rows = pd.DataFrame(record["rows"])
    rows = rows[rows.problem.isin(record["done"])]
    if "degenerate" not in rows:
        return {}
    flagged = rows.degenerate.fillna(False).astype(bool) | rows.start_degenerate.fillna(
        False
    ).astype(bool)
    return {str(k): int(v) for k, v in rows[flagged].groupby("start").size().items()}


def frame(record: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray]:
    """The runs with their gaps to the realization's best non-degenerate run, and the truth's gaps.

    A degenerate run scores rows at probability 1 through `cnaster`'s negative
    binomial (`known_copy.degenerate`); as the best it would set every gap by
    thousands of nats, so it is neither the reference nor a point.
    """
    rows = pd.DataFrame(record["rows"])
    rows = rows[rows.problem.isin(record["done"])]
    if "error" in rows:
        rows = rows[rows.error.isna()]
    if "degenerate" in rows:
        flagged = rows.degenerate.fillna(False).astype(
            bool
        ) | rows.start_degenerate.fillna(False).astype(bool)
        rows = rows[~flagged]
    problems = record["problems"]
    best = rows.groupby("problem").llf.max().to_dict()
    for i in best:
        best[i] = max(best[i], problems[i]["truth_llf"])
    top = rows.problem.map(best)
    n_rows = rows.problem.map({i: p["n_rows"] for i, p in problems.items()})
    rows = rows.assign(y=(top - rows.start_llf).clip(lower=FLOOR), by=(top - rows.llf).clip(lower=FLOOR),
                       bseconds=rows.seconds + rows.bw_seconds, start_missed_pct=100 * rows.start_missed / n_rows,
                       missed_pct=100 * rows.missed / n_rows)  # fmt: skip
    truth = np.array(
        [max(best[i] - problems[i]["truth_llf"], FLOOR) for i in record["done"]]
    )
    return rows, truth


def _table(
    tab: Any,
    missed: dict[str, tuple[float, float]],
    flagged: dict[str, int],
    rank_cost: dict[str, int],
    rank_missed: dict[str, int],
    ran: frozenset[str] = frozenset(),
) -> None:
    n_rows = (
        sum(1 + len(rows) for _, rows in TABLE) + len(TABLE) - 1
    )  # NB a blank row between groups
    head = 0.06
    step = (1 - head) / (n_rows + 0.5)

    def rule(y: float, lw: float) -> None:
        tab.plot(
            [0.0, 1.0], [y, y], color="k", lw=lw, transform=tab.transAxes, clip_on=False
        )

    rule(1.0, 1.2)
    for x, text in (
        (0.01, "#"),
        (0.07, "Algorithm"),
        (0.49, "$R_C$"),
        (0.55, "$R_M$"),
        (0.72, "Missed [%]"),
        (0.75, "Description"),
        (0.35, "Source"),
    ):
        tab.text(x, 1 - head / 2, text, fontsize=8.5, weight="bold", transform=tab.transAxes, va="center",
                 ha="right" if 0.4 < x < 0.74 else "left")  # fmt: skip
    rule(1 - head, 0.7)
    y = 1 - head + step * 0.25
    for g, (group, rows) in enumerate(TABLE):
        y -= step * (2 if g else 1)
        tab.text(
            0.07,
            y,
            group,
            fontsize=8,
            style="italic",
            transform=tab.transAxes,
            va="center",
        )
        for name, text in rows:
            y -= step
            tab.text(
                0.01,
                y,
                str(NUMBER[name]),
                fontsize=8,
                transform=tab.transAxes,
                va="center",
            )
            tab.text(
                0.09,
                y,
                LABEL.get(name, name),
                fontsize=8,
                family="monospace",
                transform=tab.transAxes,
                va="center",
            )
            tab.text(0.75, y, text, fontsize=8, transform=tab.transAxes, va="center")
            tab.text(
                0.35, y, SOURCE[name], fontsize=8, transform=tab.transAxes, va="center"
            )
            for x, rank in (
                (0.49, rank_cost.get(name)),
                (0.55, rank_missed.get(name)),
            ):
                if rank is not None:
                    tab.text(
                        x,
                        y,
                        str(rank),
                        fontsize=8,
                        transform=tab.transAxes,
                        va="center",
                        ha="right",
                    )
            a, b = missed.get(name, (np.nan, np.nan))
            k = flagged.get(name, 0)
            if np.isnan(a):
                cell = (
                    f"degenerate ({k})"
                    if k
                    else ("refused" if name in ran else "not run")
                )
            else:
                cell = f"{a:.1f} / {b:.1f}" + (f" ({k} degen.)" if k else "")
            tab.text(
                0.72,
                y,
                cell,
                fontsize=8,
                transform=tab.transAxes,
                va="center",
                ha="right",
            )
    rule(0.0, 1.2)
    tab.set_ylim(0, 1)


def draw(
    ax: Any,
    record: dict[str, Any],
    key: bool = False,
    key_style: dict[str, float] | None = None,
) -> pd.DataFrame:
    """The gap panel on `ax`: each start's runs against runtime, numbered as in `TABLE`; returns the runs drawn.

    `key` draws, for a figure with no table beside it (`solver_combined`, T- #660),
    a key below the axes instead of the legend: the stages' markers, then each
    start unnumbered with its missed % after Baum-Welch, `port`'s marked and
    named in a footnote.
    """
    from matplotlib.ticker import FixedLocator, FuncFormatter

    d, truth = frame(record)
    # NB every realization with rows counts, reused ones included; one still running is also named in progress
    n_problems = int(d.problem.nunique())
    n_partial = len(set(d.problem) - set(record.get("complete", record["done"])))
    ax.axhspan(
        float(np.quantile(truth, 0.1)),
        float(np.quantile(truth, 0.9)),
        color="0.55",
        alpha=0.35,
        lw=0,
        zorder=0,
    )
    ax.axhline(float(np.median(truth)), color="k", lw=0.9, zorder=0)

    points: list[tuple[float, float, str]] = []
    rightmost = 0.0
    top = 0.0
    for name, g in d.groupby("start"):
        if name not in NUMBER:
            continue
        colour = COLOUR[str(name)]
        limited = key and float(g.seconds.median()) < RUNTIME_FLOOR
        x, xe = bars(g.seconds.clip(lower=RUNTIME_FLOOR) if key else g.seconds)
        # NB each start displaced by its own factor, up to 0.1 decades either side, so equal runtimes do not overlap
        spread = 10 ** (0.2 * (NUMBER[str(name)] / max(NUMBER.values()) - 0.5))
        x *= spread
        xe = [[e * spread for e in side] for side in xe]
        y, ye = bars(g.y)
        top = max(top, y + ye[1][0])
        ax.errorbar(
            x, y, xerr=xe, yerr=ye, fmt="o", color=colour, ms=5, lw=0.8, capsize=2.5
        )
        bx, bxe = bars(g.bseconds)
        by, bye = bars(g.by)
        # NB the Baum-Welch points crowd near 10 s: spread three times as wide, up to 0.3 decades either side
        bspread = spread**3
        bx *= DODGE * bspread
        bxe = [[e * bspread for e in side] for side in bxe]
        ax.annotate(
            "",
            (bx, by),
            (x, y),
            arrowprops={"arrowstyle": "->", "color": colour, "lw": 0.8, "alpha": 0.7},
        )
        ax.errorbar(
            bx,
            by,
            xerr=bxe,
            yerr=bye,
            fmt="o",
            color=colour,
            ms=5,
            mfc="white",
            lw=0.8,
            capsize=2.5,
        )
        if limited:
            # NB an upper limit: the median start ran in under RUNTIME_FLOOR
            ax.annotate("", (x / 2.2, y), (x, y), arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 1.0,
                                                               "shrinkA": 4, "shrinkB": 0})  # fmt: skip
        points.append((x, y, str(name)))
        rightmost = max(rightmost, bx + bxe[1][0])

    # NB numbers placed left of their points, stacked upward in 0.25-decade steps where they would overlap
    placed: list[tuple[float, float]] = []
    for x, y, name in [] if key else sorted(points, key=lambda p: (p[0], p[1])):
        lx, ly = np.log10(x), np.log10(y)
        while any(abs(lx - px) < 0.3 and abs(ly - py) < 0.2 for px, py in placed):
            ly += 0.25
        placed.append((lx, ly))
        ax.annotate(NUMBER_TEXT[name], (x, y), xytext=(10**lx / 1.15, 10**ly), textcoords="data", fontsize=8, weight="bold", color=COLOUR[name],
                    ha="right", va="center", arrowprops={"arrowstyle": "-", "color": "0.6", "lw": 0.4, "shrinkA": 0, "shrinkB": 2}
                    if abs(ly - np.log10(y)) > 1e-9 else None)  # fmt: skip

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(
        RUNTIME_FLOOR / 3 if key else float(d.seconds.quantile(0.02)) * 0.5,
        rightmost * 1.3,
    )
    # NB one decade below the truth's median gap; three above, or above the highest start's error bar
    ax.set_ylim(
        float(np.median(truth)) / 10, max(float(np.median(truth)) * 1e3, top * 2)
    )
    ax.yaxis.set_major_locator(FixedLocator([FLOOR, *10.0 ** np.arange(-1, 8)]))
    ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda v, _: "0" if v == FLOOR else f"$10^{{{round(np.log10(v))}}}$"
        )
    )
    ax.set_xlabel("Runtime [s]")
    ax.set_ylabel("Gap [nats]" if key else "Gap [Nats]")
    if key:
        after = d.groupby("start").missed_pct.median()
        ordered = sorted(set(d.start) & set(NUMBER), key=lambda n: NUMBER[n])
        key_below(
            ax,
            f"{Path(record['manifest']).stem}: median of {n_problems} realization{'s' if n_problems != 1 else ''}"
            + (f" ({n_partial} in progress)" if n_partial else ""),
            [({"marker": "o", "color": "0.4", "markersize": 5}, "Initialized"),
             ({"marker": "o", "color": "0.4", "markerfacecolor": "white", "markersize": 5}, "Baum-Welch"),
             ({"line": True, "color": "k"}, "Truth")],
            [(KEY_NAMES.get(n, n), COLOUR[n], float(after[n]))
             for n in ordered],
            [],
            **(key_style or {}),
        )  # fmt: skip
        return d
    ax.plot([], [], "o", color="0.4", label="Start")
    ax.plot([], [], "o", color="0.4", mfc="white", label="Baum-Welch")
    ax.plot([], [], color="k", lw=0.9, label="Truth")
    # NB a start added to a running stream covers fewer realizations until it catches up: say which, and on how many
    counts = d[d.start.isin(NUMBER)].groupby("start").problem.nunique()
    short = counts[counts < d.problem.nunique()]
    behind = "".join(
        f"; #{', #'.join(str(NUMBER[s]) for s in sorted(g.index, key=lambda s: NUMBER[s]))} on {k}"
        for k, g in short.groupby(short)
    )
    if key:
        for name in sorted(set(d.start) & set(NUMBER), key=lambda n: NUMBER[n]):
            ax.plot(
                [],
                [],
                "o",
                color=COLOUR[name],
                label=f"{NUMBER[name]} {LABEL.get(name, name)}",
            )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=7.5, frameon=False,
              title=f"{Path(record['manifest']).stem}: Median for {n_problems} realization{'s' if n_problems != 1 else ''}"
                    + (f" ({n_partial} in progress)" if n_partial else "") + f" $\\times$ {record['seeds']} seeds{behind}",
              title_fontsize=7.5)  # fmt: skip
    return d


def figure(record: dict[str, Any], out: Path) -> Path:
    """The gap figure and its table, written to `out`."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9})
    fig = plt.figure(figsize=(15.5, 6.2))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.04)
    ax, tab = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    tab.axis("off")
    d = draw(ax, record)
    missed = {
        str(n): (float(g.start_missed_pct.median()), float(g.missed_pct.median()))
        for n, g in d.groupby("start")
    }
    # NB ranked after Baum-Welch, the polish the study measures: R_C by the median gap, R_M by the median Missed
    rank_cost = ranks({str(n): float(g.by.median()) for n, g in d.groupby("start")})
    rank_missed = ranks({n: m[1] for n, m in missed.items()})
    ran = frozenset(str(n) for n in pd.DataFrame(record["rows"]).start.unique())
    _table(tab, missed, degenerate_counts(record), rank_cost, rank_missed, ran)
    stamp(fig, record)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> None:
    """`STREAM.record [EARLIER.record ...]`: the figure beside the first, over all of them."""
    paths = [Path(p) for p in (argv if argv is not None else sys.argv[1:])]
    record = merged([records.read(p) for p in paths])
    print(figure(record, paths[0].with_suffix(".png")))
