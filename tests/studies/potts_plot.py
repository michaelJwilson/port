"""#556: `tests.studies.potts_stream`'s runs against runtime, each solver numbered as in the table beside it.

`python -m tests.studies.potts_plot STREAM.pkl` writes `<stem>.png` beside the
pickle: energy less TRW-S's lower bound, on a log axis whose bottom tick, "0",
holds every run at the bound, with a solid black line at the planted
labelling's gap (its median over realizations).

Each point is a solver's median over realizations x random starts, its bars
the 10-90% range on both axes; an open marker is the same runs after sal's
ICM, a diamond after the color merge that follows it. Polish stages are drawn
`DODGE` to the right of their runtime so stages do not overlap. The table's
last column counts the labels that differ from the planted ones at each
solver's lowest-energy run, median over realizations: raw / after ICM and the
color merge (`wrong_at_best`).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

NAMES = {
    "field_argmax": "Field-argmax", "icm": "ICM", "icm-random": "ICM-random", "anneal": "Glauber",
    "swendsen-wang": "Swendsen-Wang", "wolff": "Wolff", "tempering": "Parallel tempering",
    "alpha-expansion": "Alpha-expansion", "alpha-beta-swap": "Alpha-beta-swap", "max-product": "Max-product",
    "alpha-rust": "Alpha-rust", "icm-numba": "ICM-numba", "alpha-rust-icm": "Alpha-rust-ICM",
    "alpha-rust-fuse-merge": "Alpha-rust-fuse", "trws": "TRW-S",
}  # fmt: skip
TABLE = (
    ("Graph cuts", (
        ("sal:alpha-expansion", "each clone in turn claims spots by a min cut"),
        ("sal:alpha-beta-swap", "min-cut swaps between two clones at a time"),
        ("port:alpha-rust", "alpha-expansion, Rust min cut"),
        ("port:alpha-rust-fuse-merge", "Alpha-rust fused with argmax descent (--sal)"),
        ("port:alpha-rust-icm", "Alpha-rust, then cnaster's ICM"),
    )),
    ("Local descent", (
        ("sal:icm", "each spot to its best clone given neighbours, in order"),
        ("port:icm-numba", "sal's ICM, compiled"),
        ("sal:icm-random", "ICM in a random order: heat bath at T = 0"),
        ("port:icm", "cnaster's ICM, random spot order"),
        ("sal:field_argmax", "each spot's best clone, neighbours ignored"),
    )),
    ("Sampling, message passing", (
        ("sal:anneal", "single-site heat bath (Glauber), annealed"),
        ("sal:swendsen-wang", "cluster moves over bonded spots, annealed"),
        ("sal:wolff", "one grown cluster flipped per move, annealed"),
        ("sal:tempering", "replicas on a temperature ladder, swapped"),
        ("sal:max-product", "loopy max-product belief propagation"),
        ("sal:trws", "tree-reweighted message passing: its decode"),
    )),
)  # fmt: skip
NUMBER = {
    solver: k + 1 for k, solver in enumerate(s for _, rows in TABLE for s, _ in rows)
}
"""Each solver's number: its row in the table."""

DODGE = 1.12
"""Each polish stage drawn 12% right of its runtime."""

FLOOR = 1e-2
"""The gap figure's "0": runs within `FLOOR` nats of the bound."""


def label(solver: str) -> str:
    kind, name = solver.split(":", 1)
    return f"{NAMES[name]}$^{{{'s' if kind == 'sal' else 'p'}}}$"


def _bars(values: pd.Series) -> tuple[float, list[list[float]]]:
    m = float(values.median())
    return m, [[m - float(values.quantile(0.1))], [float(values.quantile(0.9)) - m]]


def frame(record: dict[str, Any]) -> pd.DataFrame:
    rows = pd.DataFrame(record["rows"])
    rows = rows[rows.problem.isin(record["done"])]
    if "error" in rows:
        rows = rows[rows.error.isna()]
    problems = record["problems"]
    return rows.assign(
        truth=rows.problem.map({i: p["truth_energy"] for i, p in problems.items()}),
        bound=rows.problem.map({i: p["bound"] for i, p in problems.items()}),
        pseconds=rows.seconds + rows.polish_seconds,
        bseconds=rows.seconds + rows.both_seconds,
    )


def wrong_at_best(d: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Per solver, labels unlike the planted at each realization's lowest-energy run, median over realizations: raw, polished."""
    out: dict[str, tuple[float, float]] = {}
    for solver, g in d.groupby("solver"):
        raw = g.loc[g.groupby("problem").energy.idxmin(), "wrong"]
        polished = g.loc[g.groupby("problem").both.idxmin(), "both_wrong"]
        out[str(solver)] = (float(raw.median()), float(polished.median()))
    return out


def _table(
    tab: Any, wrong: dict[str, tuple[float, float]], tuned: dict[str, dict[str, float]]
) -> None:
    n_rows = sum(1 + len(rows) for _, rows in TABLE)
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
        (0.07, "Solver"),
        (0.33, "Description"),
        (0.99, "Wrong"),
    ):
        tab.text(
            x,
            1 - head / 2,
            text,
            fontsize=8.5,
            weight="bold",
            transform=tab.transAxes,
            va="center",
            ha="right" if text == "Wrong" else "left",
        )
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
                0.09, y, label(solver), fontsize=8, transform=tab.transAxes, va="center"
            )
            tab.text(0.33, y, text, fontsize=8, transform=tab.transAxes, va="center")
            if solver in wrong:
                raw, polished = wrong[solver]
                tab.text(
                    0.99,
                    y,
                    f"{raw:,.0f} / {polished:,.0f}",
                    fontsize=8,
                    transform=tab.transAxes,
                    va="center",
                    ha="right",
                )
    rule(0.0, 1.2)
    notes = ("$^s$ sal (snakes_and_ladders);  $^p$ port.  Wrong: labels unlike the planted at the best run, raw / ICM + color merge.",
             "Color merge: one clone relabelled into another, the best pair, while the energy drops (cnaster).",
             "Polish stages drawn 12% right of their runtime.",
             "Tuned on held-out realizations (T0, sweeps): "
             + ";  ".join(f"{NUMBER[k]} {v['t_start']:g}, {v['sweeps']:,.0f}" for k, v in sorted(tuned.items(), key=lambda kv: NUMBER[kv[0]]))
             if tuned else "")  # fmt: skip
    for k, note in enumerate(notes):
        tab.text(
            0.01,
            -0.045 - 0.04 * k,
            note,
            fontsize=7.5,
            color="0.35",
            transform=tab.transAxes,
            va="center",
        )
    tab.set_ylim(0, 1)


def figure(record: dict[str, Any], out: Path) -> Path:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    d = frame(record)
    d = d.assign(y=d.energy - d.bound, py=d.polished - d.bound, by=d.both - d.bound)
    d[["y", "py", "by"]] = d[["y", "py", "by"]].clip(lower=FLOOR)
    n_problems, n_starts = len(record["done"]), record["starts"]

    plt.rcParams.update({"font.size": 9})
    fig = plt.figure(figsize=(15.5, 6.2))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.04)
    ax, tab = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    tab.axis("off")
    ax.axhspan(FLOOR * 0.6, FLOOR * 1.4, color="0.92", zorder=0)
    truth = np.median(
        [
            max(p["truth_energy"] - p["bound"], FLOOR)
            for i, p in record["problems"].items()
            if i in record["done"]
        ]
    )
    ax.axhline(truth, color="k", lw=0.9, zorder=0)

    lowest = float(d.groupby("solver").y.median().min())
    crowded: list[tuple[float, float, str]] = []
    for solver, g in d.groupby("solver"):
        port = solver.startswith("port:")
        colour, marker = ("C1", "s") if port else ("C0", "o")
        x, xe = _bars(g.seconds)
        y, ye = _bars(g.y)
        ax.errorbar(
            x, y, xerr=xe, yerr=ye, fmt=marker, color=colour, ms=5, lw=0.8, capsize=2.5
        )
        px, pxe = _bars(g.pseconds)
        py, pye = _bars(g.py)
        px *= DODGE
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
        bx, bxe = _bars(g.bseconds)
        by, bye = _bars(g.by)
        bx *= DODGE**2
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
            )

    # NB solvers at the lowest energy: numbers in a row above them, each tied to its point
    crowded.sort()
    if crowded:
        xs = np.array([x for x, _, _ in crowded])
        centre = float(np.exp(np.log(xs).mean()))
        spots = centre * 1.45 ** (np.arange(xs.size) - (xs.size - 1) / 2)
        top = FLOOR * 30
        for (x, y, solver), tx in zip(crowded, spots, strict=True):
            ax.annotate(str(NUMBER[solver]), (x, y), xytext=(tx, top), textcoords="data", fontsize=8, weight="bold",
                        ha="center", va="bottom", arrowprops={"arrowstyle": "-", "color": "0.6", "lw": 0.5, "shrinkA": 0, "shrinkB": 3})  # fmt: skip

    ax.set_xscale("log")
    slowest = d[d.solver == "sal:max-product"].seconds
    ax.set_xlim(float(d.groupby("solver").seconds.quantile(0.1).min()) * 0.6,
                float(slowest.quantile(0.9) if len(slowest) else d.seconds.max()) * 1.6)  # fmt: skip
    from matplotlib.ticker import FixedLocator, FuncFormatter

    ax.set_yscale("log")
    ax.set_ylim(FLOOR * 0.5, float(d.y.max()) * 3)
    ax.yaxis.set_major_locator(FixedLocator([FLOOR, *10.0 ** np.arange(-1, 7)]))
    ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda v, _: "0" if v == FLOOR else f"$10^{{{round(np.log10(v))}}}$"
        )
    )
    ax.set_ylabel("TRW-S Gap [nats]")
    ax.set_xlabel("Runtime [s]")

    ax.plot([], [], "o", color="C0", label="sal")
    ax.plot([], [], "s", color="C1", label="port")
    ax.plot([], [], "o", color="0.4", mfc="white", label="ICM polish")
    ax.plot([], [], "D", color="0.4", mfc="white", ms=4, label="Color merge")
    ax.plot([], [], color="k", lw=0.9, label="Truth")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=7.5, frameon=False,
              title=f"median over {n_problems} realization{'s' if n_problems > 1 else ''} $\\times$ {n_starts} random starts; "
              "bars: 10-90%", title_fontsize=7.5)  # fmt: skip
    _table(tab, wrong_at_best(d), record.get("tuned", {}))
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
            kind, name = solver.split(":", 1)
            mark = "s" if kind == "sal" else "p"
            lines.append(
                rf"{NUMBER[solver]} & {NAMES[name]}$^{{\mathrm{{{mark}}}}}$ & {escape(text)} \\"
            )
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}"]) + "\n"


def main(argv: list[str] | None = None) -> None:
    (path,) = argv if argv is not None else sys.argv[1:]
    stream = Path(path)
    record = pickle.loads(stream.read_bytes())
    print(figure(record, stream.with_suffix(".png")))
    stream.with_name("potts_solvers_table.tex").write_text(table_tex())


if __name__ == "__main__":
    main()
