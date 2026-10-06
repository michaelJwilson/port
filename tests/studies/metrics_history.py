"""Clone and copy-state ARI across `main`'s merges, from the metrics ledger's history runs (`port.qa.ledger.read`, #620).

`python -m tests.studies.metrics_history [OUT.png [OUT_CLASSES.png]]`, by default
`.cache/plots/metrics_history{,_classes}.png` (`port.qa.provenance.PLOTS`), untracked.

A history row is a `tests/sim_audit.py::main` run whose note starts with
`HISTORY`: `--sal` measured at an earlier merge of `main`, newest to oldest,
recorded after the fact. The x axis is `main`'s first-parent order, read from
git, so a merge that touched no pipeline code and carries its predecessor's
figures is absent rather than drawn flat, and a run of three or more merges
whose plotted metrics did not move keeps its first and last merge, the last
with `>>>>` drawn between them (`ticks`, `axis`). One panel per `fixture_hash`,
labelled with its fixture name, so a panel holds one dataset (#588); one line
per metric (`SERIES`); a second figure plots integer copy recovery by
planted class (`CLASSES`). The figure carries its data hash and code commit.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from port.qa import provenance
from port.qa.ledger import ROOT, SIM_TEST, UNMEASURED, read
from port.qa.provenance import PLOTS

HISTORY = "HISTORY"
"""The note prefix that marks a row as a measurement at an earlier merge."""

OUT = PLOTS / "metrics_history.png"

SERIES = {
    "clone_ari": "clone ARI",
    "clone_ari_int": "integer clone ARI",
    "copy_ari": "copy ARI, integer (A, B)",
    "state_ari": "copy ARI, continuous state",
    "exact_altered": "exact altered",
    "exact_altered_pf": "exact altered, phase-free",
}
"""Each metric plotted, and its legend name: ARIs solid, the exact-altered shares dotted."""

CLASSES = {
    "exact_loh": "LOH",
    "exact_loh_pf": "LOH, phase-free",
    "exact_bgain": "balanced gain",
    "exact_bgain_pf": "balanced gain, phase-free",
    "exact_ugain": "unbalanced gain",
    "exact_ugain_pf": "unbalanced gain, phase-free",
}
"""Integer copy recovery by planted class, the second figure: phased solid, phase-free dotted."""

OUT_CLASSES = PLOTS / "metrics_history_classes.png"

SKIP = ">>>>"
"""Drawn horizontally in the gap between a folded run's first and last tick,
below the axis: the merges between them are skipped."""


def history() -> list[dict[str, str]]:
    """The history rows, in the table's order."""
    return [
        r for r in read() if r["test"] == SIM_TEST and r["note"].startswith(HISTORY)
    ]


def ticks(rows: list[dict[str, str]], order: dict[str, int]) -> list[list[str]]:
    """`rows`' merges in `order`, grouped into ticks: a merge joins the
    previous tick where no plotted metric (`SERIES` or `CLASSES`, so both
    figures share their ticks) on any fixture differs from its last recorded
    value, compared as recorded strings. A fixture compares only where it has
    values; a value that moves, or a fixture or metric first measured, starts
    a new tick."""
    by: dict[str, dict[tuple[str, str, str], str]] = {}
    for row in rows:
        values = by.setdefault(row["commit"].rstrip("+"), {})
        for metric in (*SERIES, *CLASSES):
            if row[metric] != UNMEASURED:
                values[(row["fixture"], row["fixture_hash"], metric)] = row[metric]
    groups: list[list[str]] = []
    last: dict[tuple[str, str, str], str] = {}
    for commit in sorted(by, key=order.__getitem__):
        if not groups or any(last.get(k) != v for k, v in by[commit].items()):
            groups.append([commit])
        else:
            groups[-1].append(commit)
        last |= by[commit]
    return groups


def axis(
    groups: list[list[str]], names: dict[str, str]
) -> tuple[list[tuple[str, str]], list[int]]:
    """The x ticks, as `(commit, "#NNN")`, and the folds: each of `ticks`'
    runs of three or more unchanged merges keeps its first and its last
    merge, and the fold is the index of the first, so `SKIP` goes between it
    and the next tick. Shorter runs keep every merge."""
    shown: list[tuple[str, str]] = []
    folds: list[int] = []
    for group in groups:
        if len(group) < 3:
            shown += [(c, names[c]) for c in group]
        else:
            folds.append(len(shown))
            shown += [(group[0], names[group[0]]), (group[-1], names[group[-1]])]
    return shown, folds


def first_parent() -> list[str]:
    """`main`'s merges, oldest first, as 7-character hashes."""
    log = subprocess.run(
        ["git", "log", "--first-parent", "--format=%h", "--abbrev=7", "origin/main"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return log[::-1]


def label(row: dict[str, str]) -> str:
    """The merge's PR number, read from the note: `HISTORY #NNN ...`."""
    words = row["note"].split()
    return words[1] if len(words) > 1 else row["commit"]


def figure(
    rows: list[dict[str, str]],
    out: Path,
    series: dict[str, str] | None = None,
) -> Path:
    """One panel per `fixture_hash`, one line per metric in `series` (default `SERIES`), x in `main`'s merge order."""
    series = SERIES if series is None else series
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    order = {c: k for k, c in enumerate(first_parent())}
    rows = [r for r in rows if r["commit"].rstrip("+") in order]
    names = {r["commit"].rstrip("+"): label(r) for r in rows}
    shown, folds = axis(ticks(rows, order), names)
    x = {c: k for k, (c, _) in enumerate(shown)}
    # NB the stamp hashes every history row on main, skipped merges included
    stamped = rows
    rows = [r for r in rows if r["commit"].rstrip("+") in x]
    datasets = sorted({(r["fixture"], r["fixture_hash"]) for r in rows})

    fig, axes = plt.subplots(
        len(datasets), 1, sharex=True, squeeze=False,
        figsize=(max(6.0, 0.42 * len(shown) + 2.5), 1.9 * len(datasets) + 1.2),
    )  # fmt: skip
    for ax, (fixture, digest) in zip(axes[:, 0], datasets, strict=True):
        mine = sorted(
            (r for r in rows if (r["fixture"], r["fixture_hash"]) == (fixture, digest)),
            key=lambda r: x[r["commit"].rstrip("+")],
        )
        for k, (column, name) in enumerate(series.items()):
            points = [
                (x[r["commit"].rstrip("+")], float(r[column]))
                for r in mine
                if r[column] != UNMEASURED
            ]
            if points:
                xs, ys = zip(*points, strict=True)
                ax.plot(xs, ys, ("-" if k < 4 else ":") if series is SERIES else ("-" if k % 2 == 0 else ":"), marker="o", ms=2.5, lw=1,
                        color=plt.get_cmap("tab10")(k if series is SERIES else k // 2), label=name)  # fmt: skip
        ax.set_ylabel(f"{fixture}\n{digest}", fontsize=7)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", lw=0.3)
    axes[0, 0].legend(
        fontsize=6.5, ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02)
    )
    ax = axes[-1, 0]
    ax.set_xticks(range(len(shown)))
    ax.set_xticklabels([name for _, name in shown], rotation=90, fontsize=7)
    for k in folds:
        ax.annotate(
            SKIP,
            xy=(k + 0.5, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -4),
            textcoords="offset points",
            ha="center",
            va="top",
            rotation=0,
            fontsize=7,
        )
    ax.set_xlabel("Merged PR (oldest left)")
    fig.text(
        0.99,
        0.01,
        stamp(stamped),
        ha="right",
        va="bottom",
        fontsize=6,
        family="monospace",
        color="0.4",
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def stamp(rows: list[dict[str, Any]]) -> str:
    """`data <hash> · code <commit>`: SHA-256 of the rows plotted, and the repository's commit."""
    return provenance.stamp(provenance.digest(repr(rows).encode()))


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    rows = history()
    print(figure(rows, Path(args[0]) if args else OUT))
    print(
        figure(
            rows,
            Path(args[1]) if len(args) > 1 else OUT_CLASSES,
            CLASSES,
        )
    )


if __name__ == "__main__":
    main()
