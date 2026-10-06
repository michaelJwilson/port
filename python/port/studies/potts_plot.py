"""#556: `port.studies.potts_stream`'s runs against runtime, each solver numbered as in the table beside it.

`run_study --potts-plot STREAM.record` writes `<stem>.png` beside the
record: energy less TRW-S's lower bound, on a log axis whose bottom tick, "0",
holds every run at the bound, with a solid black line at the planted
labelling's gap (its median over realizations) in a dark grey band (its 10-90%
range).

Each point is a solver's median over realizations x random starts, its error
bars the 10-90% range on both axes; an open marker is the same runs after sal's
ICM, a diamond after the color merge that follows it. Polish stages are drawn
`DODGE` to the right of their runtime so stages do not overlap. The table's
last column, Missed, is the percentage of labels that differ from the planted
ones, the median over the same runs as the points: raw / after ICM and the color merge
(`missed`).
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

NAMES = {
    "field_argmax": "field-argmax", "anneal": "glauber", "tempering": "parallel tempering",
    "alpha-rust-fuse-merge": "alpha-rust-fuse", "trws": "trw-s",
}  # fmt: skip
"""The names the figure prints where they differ from the solver's own."""


TABLE = (
    ("Graph cuts", (
        ("sal:alpha-expansion", "Each clone in turn claims spots by a min cut"),
        ("sal:alpha-beta-swap", "Min-cut swaps between two clones at a time"),
        ("port:alpha-rust-fuse-merge", f"Rust {tt('alpha-expansion')} fused with argmax descent (--sal)"),
    )),
    ("Local descent", (
        ("sal:icm", "Each spot to its best clone given neighbours, in order"),
        ("sal:icm-random", f"{tt('icm')} in a random order: heat bath at T = 0"),
        ("sal:field_argmax", "Each spot's best clone, neighbours ignored"),
    )),
    ("Sampling, message passing", (
        ("sal:anneal", "Single-site heat bath, annealed"),
        ("sal:swendsen-wang-heat-bath", "Every bonded cluster relabelled by its field's heat bath, annealed"),
        ("sal:wolff-heat-bath", "One grown cluster relabelled by its field's heat bath, annealed"),
        ("sal:tempering", f"{tt('glauber')} replicas on a temperature ladder, swapped"),
        ("sal:max-product", "Loopy max-product belief propagation"),
        ("sal:trws", "Tree-reweighted message passing: its decode"),
    )),
)  # fmt: skip
"""The solvers drawn (T- #660). Set aside from the figure, still in `clone_label_arms` or `--only`:
`alpha-rust`, `alpha-rust-icm`, `icm-numba`, cnaster's `icm`, the four field-weighted cluster moves
and cluster tempering."""
NUMBER = {
    solver: k + 1 for k, solver in enumerate(s for _, rows in TABLE for s, _ in rows)
}
"""Each solver's number: its row in the table."""

DODGE = 1.12
"""Each polish stage drawn 12% right of its runtime."""

FLOOR = 1e-2
"""The gap figure's "0": runs within `FLOOR` nats of the bound."""


KEY_NAMES = {
    "sal:alpha-expansion": "Alpha-expansion", "sal:alpha-beta-swap": "Alpha-beta-swap",
    "sal:icm": "ICM-vector", "sal:icm-random": "ICM-random", "sal:field_argmax": "Field-argmax",
    "sal:anneal": "Glauber", "sal:swendsen-wang-heat-bath": "Swendsen-Wang", "sal:wolff-heat-bath": "Wolff",
    "sal:tempering": "Parallel tempering", "sal:max-product": "Max-product", "sal:trws": "TRW-S",
}  # fmt: skip
"""The names `solver_combined`'s key prints, and the solvers it draws: `alpha-rust-fuse` is not
among them (deprecated from the figure, #716)."""


def label(solver: str) -> str:
    """The name printed for `solver`, its `sal:`/`port:` source dropped."""
    name = solver.split(":", 1)[1]
    return NAMES.get(name, name)


def frame(record: dict[str, Any]) -> pd.DataFrame:
    """The record's successful runs on finished realizations, with their bound, truth and cumulative runtimes."""
    rows = pd.DataFrame(record["rows"])
    # NB a stream may hold solvers `TABLE` no longer draws (T- #660)
    rows = rows[rows.problem.isin(record["done"]) & rows.solver.isin(NUMBER)]
    if "error" in rows:
        rows = rows[rows.error.isna()]
    problems = record["problems"]
    return rows.assign(
        truth=rows.problem.map({i: p["truth_energy"] for i, p in problems.items()}),
        bound=rows.problem.map({i: p["bound"] for i, p in problems.items()}),
        pseconds=rows.seconds + rows.polish_seconds,
        bseconds=rows.seconds + rows.both_seconds,
    )


def missed(d: pd.DataFrame, n_labels: int) -> dict[str, tuple[float, float]]:
    """Per solver, the percentage of labels unlike the planted, median over all its runs: raw, after both polishes."""
    return {str(solver): (100 * float(g.wrong.median()) / n_labels, 100 * float(g.both_wrong.median()) / n_labels)
            for solver, g in d.groupby("solver")}  # fmt: skip


def n_spots(record: dict[str, Any]) -> int:
    """The spots each realization labels: the manifest's array."""
    from port.sim.draw import read_manifest

    array = read_manifest(Path(record["manifest"])).array
    return int(array["rows"]) * int(array["columns"])


def _table(
    tab: Any,
    wrong: dict[str, tuple[float, float]],
    rank_cost: dict[str, int],
    rank_missed: dict[str, int],
) -> None:
    """The solver table beside the figure: number, name, source, ranks, Missed, description."""
    n_rows = (
        sum(1 + len(rows) for _, rows in TABLE) + len(TABLE) - 1
    )  # NB a blank row between groups
    head = 0.06
    step = (1 - head) / (n_rows + 0.5)

    def rule(y: float, lw: float) -> None:
        tab.plot(
            [0.0, 1.0],
            [y, y],
            color="k",
            lw=lw,
            transform=tab.transAxes,
            clip_on=False,
        )

    # NB the table's top and bottom rules sit on the plot's y limits
    rule(1.0, 1.2)
    for x, text in (
        (0.01, "#"),
        (0.07, "Algorithm"),
        (0.79, "Description"),
        (0.38, "Source"),
        (0.53, "$R_C$"),
        (0.59, "$R_M$"),
        (0.76, "Missed [%]"),
    ):
        tab.text(
            x,
            1 - head / 2,
            text,
            fontsize=8.5,
            weight="bold",
            transform=tab.transAxes,
            va="center",
            ha="right" if 0.4 < x < 0.78 else "left",
        )
    rule(1 - head, 0.7)
    y = 1 - head + step * 0.25
    for k, (group, rows) in enumerate(TABLE):
        y -= step * (2 if k else 1)
        tab.text(
            0.07,
            y,
            group,
            fontsize=8,
            style="italic",
            transform=tab.transAxes,
            va="center",
        )
        for solver, text in rows:
            y -= step
            tab.text(
                0.01,
                y,
                str(NUMBER[solver]),
                fontsize=8,
                transform=tab.transAxes,
                va="center",
            )
            tab.text(
                0.09,
                y,
                label(solver),
                fontsize=8,
                family="monospace",
                transform=tab.transAxes,
                va="center",
            )
            tab.text(0.79, y, text, fontsize=8, transform=tab.transAxes, va="center")
            tab.text(
                0.38,
                y,
                solver.split(":", 1)[0],
                fontsize=8,
                transform=tab.transAxes,
                va="center",
            )
            for x, rank in (
                (0.53, rank_cost.get(solver)),
                (0.59, rank_missed.get(solver)),
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
            if solver in wrong:
                raw, polished = wrong[solver]
                tab.text(
                    0.76,
                    y,
                    f"{raw:.1f} / {polished:.1f}",
                    fontsize=8,
                    transform=tab.transAxes,
                    va="center",
                    ha="right",
                )
    rule(0.0, 1.2)
    tab.set_ylim(0, 1)


def draw(
    ax: Any, record: dict[str, Any], key: bool = False, centre: bool = False
) -> pd.DataFrame:
    """The gap panel on `ax`: each solver's runs against runtime, numbered as in `TABLE`; returns the runs drawn.

    `key` draws, for a figure with no table beside it (`solver_combined`, T- #660),
    a key below the axes instead of the legend: the stages' markers, then each
    solver unnumbered with its missed % after both polishes, `port`'s marked
    and named in a footnote; every solver a circle.
    `centre` widens the gap axis, in decades, until the truth's median is its
    midpoint; no limit narrows, so no point is clipped.
    """
    from matplotlib.ticker import FixedLocator, FuncFormatter

    d = frame(record)
    if key:
        d = d[d.solver.isin(KEY_NAMES)]
    d = d.assign(y=d.energy - d.bound, py=d.polished - d.bound, by=d.both - d.bound)
    d[["y", "py", "by"]] = d[["y", "py", "by"]].clip(lower=FLOOR)
    n_problems, n_starts = len(record["done"]), record["starts"]

    truths = np.array(
        [
            max(p["truth_energy"] - p["bound"], FLOOR)
            for i, p in record["problems"].items()
            if i in record["done"]
        ]
    )
    # NB the truth's own spread over realizations: its 10-90% range, as the points' bars
    ax.axhspan(
        float(np.quantile(truths, 0.1)),
        float(np.quantile(truths, 0.9)),
        color="0.55",
        alpha=0.35,
        lw=0,
        zorder=0,
    )
    ax.axhline(float(np.median(truths)), color="k", lw=0.9, zorder=0)

    lowest = float(d.groupby("solver").y.median().min())
    crowded: list[tuple[float, float, str]] = []
    for solver, g in d.groupby("solver"):
        colour = tab20(NUMBER[str(solver)])
        marker = "o" if key or not solver.startswith("port:") else "s"
        x, xe = bars(g.seconds)
        # NB each solver displaced by its own factor, up to 0.1 decades either side, so equal runtimes do not overlap
        spread = 10 ** (0.2 * (NUMBER[str(solver)] / max(NUMBER.values()) - 0.5))
        x *= spread
        xe = [[e * spread for e in side] for side in xe]
        y, ye = bars(g.y)
        ax.errorbar(
            x, y, xerr=xe, yerr=ye, fmt=marker, color=colour, ms=5, lw=0.8, capsize=2.5
        )
        px, pxe = bars(g.pseconds)
        py, pye = bars(g.py)
        px *= DODGE * spread
        pxe = [[e * spread for e in side] for side in pxe]
        if (g.y - g.py).abs().max() > 0.5:
            ax.annotate(
                "",
                (px, py),
                (x, y),
                arrowprops={
                    "arrowstyle": "->",
                    "color": colour,
                    "lw": 0.8,
                    "alpha": 0.7,
                },
            )
            ax.errorbar(
                px,
                py,
                xerr=pxe,
                yerr=pye,
                fmt=marker,
                color=colour,
                ms=5,
                mfc="white",
                lw=0.8,
                capsize=2.5,
            )
        bx, bxe = bars(g.bseconds)
        by, bye = bars(g.by)
        bx *= DODGE**2 * spread
        bxe = [[e * spread for e in side] for side in bxe]
        if (g.py - g.by).abs().max() > 0.5:
            ax.annotate(
                "",
                (bx, by),
                (px, py),
                arrowprops={
                    "arrowstyle": "->",
                    "color": colour,
                    "lw": 0.8,
                    "ls": "--",
                    "alpha": 0.7,
                },
            )
            ax.errorbar(
                bx,
                by,
                xerr=bxe,
                yerr=bye,
                fmt="D",
                color=colour,
                ms=4,
                mfc="white",
                lw=0.8,
                capsize=2.5,
            )
        if key:
            continue
        if abs(y - lowest) < FLOOR:
            crowded.append((x, y, solver))
        else:
            ax.annotate(
                str(NUMBER[solver]),
                (x, y),
                xytext=(6, 3),
                textcoords="offset points",
                fontsize=8,
                weight="bold",
                color=colour,
            )

    # NB solvers at the lowest energy: numbers in a row above them, each tied to its point
    crowded.sort()
    if crowded:
        # NB each number above its own point, pushed right only as far as keeps a 1.35x gap to the last
        spots: list[float] = []
        for x, _, _ in crowded:
            spots.append(max(x, spots[-1] * 1.35) if spots else x)
        top = FLOOR * 30
        for (x, y, solver), tx in zip(crowded, spots, strict=True):
            ax.annotate(str(NUMBER[solver]), (x, y), xytext=(tx, top), textcoords="data", fontsize=8, weight="bold", color=tab20(NUMBER[solver]),
                        ha="center", va="bottom", arrowprops={"arrowstyle": "-", "color": "0.6", "lw": 0.5, "shrinkA": 0, "shrinkB": 3})  # fmt: skip

    ax.set_xscale("log")
    # NB right edge: the slowest solver's 90% runtime, polish stages included, so no point is clipped
    slowest = (
        max(float(g.quantile(0.9)) for _, g in d.groupby("solver").bseconds) * DODGE**2
    )
    ax.set_xlim(
        float(d.groupby("solver").seconds.quantile(0.1).min()) * 0.6, slowest * 1.3
    )
    ax.set_yscale("log")
    low, high = FLOOR * 0.5, float(d.y.max()) * 3
    if centre:
        truth = float(np.median(truths))
        reach = max(truth / low, high / truth)
        low, high = truth / reach, truth * reach
    ax.set_ylim(low, high)
    ax.yaxis.set_major_locator(FixedLocator([FLOOR, *10.0 ** np.arange(-1, 7)]))
    ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda v, _: "0" if v == FLOOR else f"$10^{{{round(np.log10(v))}}}$"
        )
    )
    ax.set_ylabel("Gap [nats]" if key else "Gap [Nats]")
    ax.set_xlabel("Runtime [s]")

    if key:
        polished = missed(d, n_spots(record))
        ordered = sorted(set(d.solver), key=lambda s: NUMBER[s])
        key_below(
            ax,
            f"{Path(record['manifest']).stem}: median of {n_problems} realization{'s' if n_problems > 1 else ''}",
            [({"marker": "o", "color": "0.4", "markersize": 5}, "Initialized"),
             ({"marker": "o", "color": "0.4", "markerfacecolor": "white", "markersize": 5}, "ICM polish"),
             ({"marker": "D", "color": "0.4", "markerfacecolor": "white", "markersize": 4}, "Color merge"),
             ({"line": True, "color": "k"}, "Truth")],
            [(KEY_NAMES[s], tab20(NUMBER[s]), polished[s][1])
             for s in ordered],
            [],
        )  # fmt: skip
        return d
    ax.plot([], [], "o", color="0.4", label="sal")
    ax.plot([], [], "s", color="0.4", label="port")
    ax.plot([], [], "o", color="0.4", mfc="white", label="ICM polish")
    ax.plot([], [], "D", color="0.4", mfc="white", ms=4, label="Color merge")
    ax.plot([], [], color="k", lw=0.9, label="Truth")
    # NB a solver added to a finished stream runs on fewer realizations until it catches up: say which, and on how many
    counts = d.groupby("solver").problem.nunique()
    short = counts[counts < n_problems]
    behind = "".join(
        f"; #{', #'.join(str(NUMBER[s]) for s in sorted(g.index, key=lambda s: NUMBER[s]))} on {k}"
        for k, g in short.groupby(short)
    )
    if key:
        for solver in sorted(set(d.solver), key=lambda s: NUMBER[s]):
            ax.plot([], [], "s" if solver.startswith("port:") else "o", color=tab20(NUMBER[solver]),
                    label=f"{NUMBER[solver]} {label(solver)}")  # fmt: skip
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=7.5, frameon=False,
              title=f"{Path(record['manifest']).stem}: Median for {n_problems} realization{'s' if n_problems > 1 else ''} $\\times$ {n_starts} random starts{behind}",
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
    # NB ranked on each solver's own output: R_C by the median gap to the bound, R_M by the median Missed
    wrong = missed(d, n_spots(record))
    rank_cost = ranks({str(n): float(g.y.median()) for n, g in d.groupby("solver")})
    rank_missed = ranks({n: m[0] for n, m in wrong.items()})
    _table(tab, wrong, rank_cost, rank_missed)
    stamp(fig, record)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def table_tex() -> str:
    """The table as a booktabs `tabular`."""

    def escape(s: str) -> str:
        return s.replace("_", r"\_").replace("--sal", r"\texttt{--sal}")

    lines = [
        r"\begin{tabular}{rll}",
        r"\toprule",
        r"\# & Solver & Description \\",
        r"\midrule",
    ]
    for k, (group, rows) in enumerate(TABLE):
        if k:
            lines.append(r"\addlinespace")
        lines.append(rf"& \multicolumn{{2}}{{l}}{{\emph{{{group}}}}} \\")
        for solver, text in rows:
            mark = "s" if solver.startswith("sal:") else "p"
            lines.append(
                rf"{NUMBER[solver]} & {label(solver)}$^{{\mathrm{{{mark}}}}}$ & {escape(text)} \\"
            )
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}"]) + "\n"


def main(argv: list[str] | None = None) -> None:
    """`STREAM.record [EARLIER.record ...]`: the figure beside the first, over all of them."""
    paths = [Path(p) for p in (argv if argv is not None else sys.argv[1:])]
    stream = paths[0]
    record = merged([records.read(p) for p in paths])
    print(figure(record, stream.with_suffix(".png")))
    stream.with_name("potts_solvers_table.tex").write_text(table_tex())
