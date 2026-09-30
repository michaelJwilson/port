"""#540: `tests.studies.copy_state_stream`'s runs against runtime, each start numbered as in the table beside it.

`python -m tests.studies.copy_state_plot STREAM.pkl` writes `<stem>.png` beside
the pickle: each run's log-likelihood below the best any run reached on its
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

import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TABLE = (
    ("cnaster, CalicoST, port", (
        ("cnaster-gmm", "cnaster's gmm_init: Gaussian mixtures"),
        ("calicost-gmm", "CalicoST's initialization_by_gmm, clones stacked"),
        ("distinct", "gmm_init among distinct components (#348)"),
        ("lattice", "integer (A, B) lattice, chosen by the rows"),
        ("lattice-em", "the lattice, by soft EM"),
        ("rdr-quantiles", "quantiles of log RDR, pooled BAF"),
    )),
    ("sal, one draw", (
        ("prior", "drawn from a prior on the observed range"),
        ("data", "on rows drawn uniformly"),
        ("kmeans++", "k-means++ on the raw count pair"),
        ("emission++", "D-squared sampling, the emission's divergence"),
        ("gaussian-em", "Gaussian mixture on read depth"),
        ("quantile", "each channel's quantiles, paired"),
        ("anneal", "best point of a falling temperature"),
        ("tempering", "best point on a temperature ladder"),
        ("hmc", "last draw of a Hamiltonian chain"),
    )),
    ("sal, best of 5 with EM", (
        ("datax5+em", "best of 5 data draws, each EM"),
        ("emission++x5+em", "best of 5 emission++, each EM"),
        ("kmeans++x5+em", "best of 5 kmeans++, each EM (--sal)"),
    )),
)  # fmt: skip
NUMBER = {name: k + 1 for k, name in enumerate(n for _, rows in TABLE for n, _ in rows)}
NUMBER_TEXT = {name: str(k) for name, k in NUMBER.items()}
SOURCE = {
    name: ("C1" if group == "cnaster, CalicoST, port" else "C0")
    for group, rows in TABLE
    for name, _ in rows
}

DODGE = 1.12
FLOOR = 1e-2
"""The "0" tick: runs within `FLOOR` nats of the best."""


def _bars(values: pd.Series) -> tuple[float, list[list[float]]]:
    m = float(values.median())
    return m, [[m - float(values.quantile(0.1))], [float(values.quantile(0.9)) - m]]


def frame(record: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray]:
    """The runs with their gaps to the realization's best, and the truth's gaps."""
    rows = pd.DataFrame(record["rows"])
    rows = rows[rows.problem.isin(record["done"])]
    if "error" in rows:
        rows = rows[rows.error.isna()]
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


def _table(tab: Any, missed: dict[str, tuple[float, float]]) -> None:
    n_rows = sum(1 + len(rows) for _, rows in TABLE)
    head = 0.06
    step = (1 - head) / (n_rows + 0.5)

    def rule(y: float, lw: float) -> None:
        tab.plot(
            [0.0, 1.0], [y, y], color="k", lw=lw, transform=tab.transAxes, clip_on=False
        )

    rule(1.0, 1.2)
    for x, text in (
        (0.01, "#"),
        (0.07, "Start"),
        (0.33, "Description"),
        (0.99, "Missed [%]"),
    ):
        tab.text(x, 1 - head / 2, text, fontsize=8.5, weight="bold", transform=tab.transAxes, va="center",
                 ha="right" if text == "Missed [%]" else "left")  # fmt: skip
    rule(1 - head, 0.7)
    y = 1 - head + step * 0.25
    for group, rows in TABLE:
        y -= step
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
            tab.text(0.09, y, name, fontsize=8, transform=tab.transAxes, va="center")
            tab.text(0.33, y, text, fontsize=8, transform=tab.transAxes, va="center")
            a, b = missed.get(name, (np.nan, np.nan))
            cell = "refused" if np.isnan(a) else f"{a:.1f} / {b:.1f}"
            tab.text(
                0.99,
                y,
                cell,
                fontsize=8,
                transform=tab.transAxes,
                va="center",
                ha="right",
            )
    rule(0.0, 1.2)
    tab.set_ylim(0, 1)


def figure(record: dict[str, Any], out: Path) -> Path:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter

    d, truth = frame(record)
    n_problems = len(record["done"])
    plt.rcParams.update({"font.size": 9})
    fig = plt.figure(figsize=(15.5, 6.2))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.04)
    ax, tab = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    tab.axis("off")
    ax.axhspan(FLOOR * 0.6, FLOOR * 1.4, color="0.92", zorder=0)
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
    for name, g in d.groupby("start"):
        if name not in NUMBER:
            continue
        colour = SOURCE[str(name)]
        x, xe = _bars(g.seconds)
        y, ye = _bars(g.y)
        ax.errorbar(
            x, y, xerr=xe, yerr=ye, fmt="o", color=colour, ms=5, lw=0.8, capsize=2.5
        )
        bx, bxe = _bars(g.bseconds)
        by, bye = _bars(g.by)
        bx *= DODGE
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
        points.append((x, y, str(name)))

    # NB numbers placed left of their points, stacked upward in 0.25-decade steps where they would overlap
    placed: list[tuple[float, float]] = []
    for x, y, name in sorted(points, key=lambda p: (p[0], p[1])):
        lx, ly = np.log10(x), np.log10(y)
        while any(abs(lx - px) < 0.3 and abs(ly - py) < 0.2 for px, py in placed):
            ly += 0.25
        placed.append((lx, ly))
        ax.annotate(NUMBER_TEXT[name], (x, y), xytext=(10**lx / 1.15, 10**ly), textcoords="data", fontsize=8, weight="bold",
                    ha="right", va="center", arrowprops={"arrowstyle": "-", "color": "0.6", "lw": 0.4, "shrinkA": 0, "shrinkB": 2}
                    if abs(ly - np.log10(y)) > 1e-9 else None)  # fmt: skip

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(
        float(d.seconds.quantile(0.02)) * 0.5,
        float(d.bseconds.quantile(0.98)) * DODGE * 1.6,
    )
    ax.set_ylim(FLOOR * 0.5, float(max(d.y.max(), d.by.max())) * 3)
    ax.yaxis.set_major_locator(FixedLocator([FLOOR, *10.0 ** np.arange(-1, 8)]))
    ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda v, _: "0" if v == FLOOR else f"$10^{{{round(np.log10(v))}}}$"
        )
    )
    ax.set_xlabel("Runtime [s]")
    ax.set_ylabel("Log-likelihood below the best [nats]")
    ax.plot([], [], "o", color="C1", label="cnaster, CalicoST, port")
    ax.plot([], [], "o", color="C0", label="sal")
    ax.plot([], [], "o", color="0.4", mfc="white", label="Baum-Welch")
    ax.fill_between([], [], [], color="0.55", alpha=0.35, lw=0, label="Truth, 10-90%")
    ax.plot([], [], color="k", lw=0.9, label="Truth")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=7.5, frameon=False,
              title=f"{Path(record['manifest']).stem}: median for {n_problems} realization{'s' if n_problems > 1 else ''} $\\times$ seeds",
              title_fontsize=7.5)  # fmt: skip
    missed = {
        str(n): (float(g.start_missed_pct.median()), float(g.missed_pct.median()))
        for n, g in d.groupby("start")
    }
    _table(tab, missed)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> None:
    (path,) = argv if argv is not None else sys.argv[1:]
    stream = Path(path)
    print(figure(pickle.loads(stream.read_bytes()), stream.with_suffix(".png")))


if __name__ == "__main__":
    main()
