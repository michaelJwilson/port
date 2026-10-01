"""Clone and copy-state ARI across `main`'s merges, from `docs/metrics.md`'s history rows.

`python -m tests.studies.metrics_history [OUT.png]`

A history row is a `tests/sim_audit.py::main` row whose note starts with
`HISTORY`: `--sal` measured at an earlier merge of `main`, newest to oldest,
recorded after the fact. The x axis is `main`'s first-parent order, read from
git, so a merge that touched no pipeline code and carries its predecessor's
figures is absent rather than drawn flat. One panel per fixture, one line per
metric (`SERIES`). The figure carries its data hash and code commit.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any

from tests.metrics import ROOT, SIM_TEST, UNMEASURED, read

HISTORY = "HISTORY"
"""The note prefix that marks a row as a measurement at an earlier merge."""

OUT = ROOT / "docs" / "plots" / "metrics_history.png"

SERIES = {
    "clone_ari": "clone ARI",
    "clone_ari_int": "integer clone ARI",
    "copy_ari": "copy ARI, integer (A, B)",
    "state_ari": "copy ARI, continuous state",
    "exact_altered": "exact altered",
    "exact_altered_pf": "exact altered, phase-free",
}
"""Each metric plotted, and its legend name: ARIs solid, the exact-altered shares dotted."""


def history() -> list[dict[str, str]]:
    """The history rows, in the table's order."""
    return [
        r for r in read() if r["test"] == SIM_TEST and r["note"].startswith(HISTORY)
    ]


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


def figure(rows: list[dict[str, str]], out: Path) -> Path:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    order = {c: k for k, c in enumerate(first_parent())}
    rows = [r for r in rows if r["commit"].rstrip("+") in order]
    commits = sorted({r["commit"].rstrip("+") for r in rows}, key=order.__getitem__)
    x = {c: k for k, c in enumerate(commits)}
    names = {r["commit"].rstrip("+"): label(r) for r in rows}
    fixtures = sorted({r["fixture"] for r in rows})

    fig, axes = plt.subplots(
        len(fixtures), 1, sharex=True, squeeze=False,
        figsize=(max(6.0, 0.42 * len(commits) + 2.5), 1.9 * len(fixtures) + 1.2),
    )  # fmt: skip
    for ax, fixture in zip(axes[:, 0], fixtures, strict=True):
        mine = sorted(
            (r for r in rows if r["fixture"] == fixture),
            key=lambda r: x[r["commit"].rstrip("+")],
        )
        for k, (column, name) in enumerate(SERIES.items()):
            points = [
                (x[r["commit"].rstrip("+")], float(r[column]))
                for r in mine
                if r[column] != UNMEASURED
            ]
            if points:
                xs, ys = zip(*points, strict=True)
                ax.plot(xs, ys, "-" if k < 4 else ":", marker="o", ms=2.5, lw=1,
                        color=plt.get_cmap("tab10")(k), label=name)  # fmt: skip
        ax.set_ylabel(fixture, fontsize=7)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", lw=0.3)
    axes[0, 0].legend(
        fontsize=6.5, ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02)
    )
    ax = axes[-1, 0]
    ax.set_xticks(range(len(commits)))
    ax.set_xticklabels([names[c] for c in commits], rotation=90, fontsize=7)
    ax.set_xlabel("Merge to main (oldest left)")
    fig.text(
        0.99,
        0.01,
        stamp(rows),
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
    data = hashlib.sha256(repr(rows).encode()).hexdigest()[:8]
    commit = (
        subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        or "unknown"
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return f"data {data} · code {commit}{'+' if dirty else ''}"


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    print(figure(history(), Path(args[0]) if args else OUT))


if __name__ == "__main__":
    main()
