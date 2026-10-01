"""#541: `docs/nb/clone_label_study.ipynb`, built and executed from `clone_labels run`'s results.

    python -m tests.studies.clone_label_notebook RESULTS.pkl

writes `docs/nb/data/clone_label_study_r0.json` (the rows, without the
workers' tracebacks), then the notebook, executed by `nbclient`, which reads
only that file and draws `.cache/plots/studies/clone_label_study.png`
(untracked, `tests.plots_dir`) beside its own output.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "nb" / "data" / "clone_label_study_r0.json"
NOTEBOOK = ROOT / "docs" / "nb" / "clone_label_study.ipynb"
INTRO = ROOT / "docs" / "nb" / "clone_label_study.md"
"""The notebook's opening cell: the result and its reading, written once the rows are in."""


def summarize(results: dict[str, Any], out: Path) -> None:
    """The rows as JSON, floats rounded, tracebacks dropped."""

    def plain(value: Any) -> Any:
        if isinstance(value, np.floating | float):
            return round(float(value), 6)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items()}
        return value

    held = {
        "rows": [
            {k: plain(v) for k, v in row.items() if k != "trace"}
            for row in results["rows"]
        ],
        "oracle": plain(results["oracle"]),
        "grid2_seconds": plain(results["grid2_seconds"]),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(held, indent=0, sort_keys=True) + "\n")


CELLS = [
    (
        "code",
        """import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

held = json.loads(Path("data/clone_label_study_r0.json").read_text())
rows = pd.DataFrame(held["rows"])
if "error" not in rows:
    rows["error"] = None
oracle = held["oracle"]
print(f"{len(rows)} rows; {rows.error.notna().sum()} refused or crashed; "
      f"the planted labels' own field: ARI {oracle['ari']:.4f}, {oracle['clones']} clones")""",
    ),
    (
        "markdown",
        "## The starts\n\nEach start's labelling before any solver: its clone ARI against the planted labels, its clones and its smallest, and what it and its problem cost -- the copy states (`kmeans++x5+em`), profiles and field its labels give. Stochastic starts over seeds 0-9: median, with the range.",
    ),
    (
        "code",
        """starts = rows[(rows.arm == "starts") & (rows.solver == "start")]
table = starts.groupby("start", sort=False).agg(
    realizations=("seed", "size"),
    ari=("ari", "median"), ari_low=("ari", "min"), ari_high=("ari", "max"),
    clones=("clones", "median"), smallest=("smallest", "median"),
    start_seconds=("start_seconds", "median"), build_seconds=("build_seconds", "median"),
)
table.sort_values("ari", ascending=False).round(4)""",
    ),
    (
        "markdown",
        "## Every solver from every start, the field held\n\nOn each start's own field, from its labels (realizations 0-2), `sal`'s methods at #492's budget and port's rows, floorless. The gap is the energy above TRW-S's certified bound on that field, per spot; ARI is read raw and after `--sal`'s floor merge. `field_argmax`, `tempering` and `max-product` take no start (`startless`). Median over starts and seeds.",
    ),
    (
        "code",
        """solved = rows[(rows.arm == "starts") & (rows.solver != "start") & rows.error.isna()].copy()
solved["gap"] = (solved.energy - solved.bound) / 6000
solvers = solved.groupby("solver").agg(
    gap=("gap", "median"), gap_high=("gap", "max"),
    ari=("ari", "median"), floored_ari=("floored_ari", "median"),
    floored_ari_low=("floored_ari", "min"), clones=("floored_clones", "median"),
    seconds=("solve_seconds", "median"),
)
solvers.sort_values(["floored_ari", "gap"], ascending=[False, True]).round(4)""",
    ),
    (
        "markdown",
        "## The best solver from each start\n\nPer start, the solver whose floored ARI is highest (median over realizations 0-2), beside `--sal`'s `alpha-rust-fuse-merge`: what a start is worth once solved.",
    ),
    (
        "code",
        """per = solved.pivot_table(index="start", columns="solver", values="floored_ari", aggfunc="median")
best = pd.DataFrame({
    "start ARI": starts.groupby("start").ari.median(),
    "best solver": per.idxmax(axis=1),
    "its floored ARI": per.max(axis=1),
    "--sal's (alpha-rust-fuse-merge)": per["port:alpha-rust-fuse-merge"],
})
best.sort_values("its floored ARI", ascending=False).round(4)""",
    ),
    (
        "markdown",
        "## Runtime against gap, and what each start is worth\n\nLeft: each solver's median energy gap per spot above TRW-S's bound against its median solve time, over every start and seed, with the range. Right: each start's clone ARI as it stands (open), after one `--sal` solve and floor (bar), and after four alternating rounds (filled); in brackets, the start's and its states' and field's seconds. This figure replaces #492's `potts_solvers_sal.png` and `potts_solvers_port.png`.",
    ),
    (
        "code",
        """plt.rcParams.update({"font.size": 8})
figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.15]})
cuts = {"sal:alpha-expansion", "sal:alpha-beta-swap", "port:alpha", "port:alpha-rust",
        "port:alpha-rust-merge", "port:alpha-rust-fuse-merge", "port:alpha-rust-icm"}
for solver, group in solved.groupby("solver"):
    x, y = group.solve_seconds.median(), group.gap.median() + 1e-3
    port = solver.startswith("port:")
    left.errorbar(x, y, yerr=[[y - (group.gap.min() + 1e-3)], [group.gap.max() + 1e-3 - y]],
                  fmt="s" if port else "o", color="C1" if port else "C0", ms=4, lw=0.6, capsize=2)
    if solver not in cuts:
        left.annotate(solver.split(":")[1], (x, y), fontsize=6, xytext=(4, 3), textcoords="offset points")
left.annotate("graph cuts: sal alpha-expansion, alpha-beta-swap;\\nport alpha-rust, -merge, -fuse-merge, -icm, alpha",
              (0.1, 1.1e-3), fontsize=6, xytext=(8, 22), textcoords="offset points")
left.plot([], [], "o", color="C0", label="sal"); left.plot([], [], "s", color="C1", label="port")
left.legend(loc="upper left", fontsize=7, frameon=False)
left.set(xscale="log", yscale="log", xlabel="solve seconds (median over starts and seeds)",
         ylabel="energy above TRW-S bound per spot, + 1e-3", title="solvers from every start, field held")
alternating = rows[(rows.arm == "alternating") & rows.error.isna()]
fuse = solved[solved.solver == "port:alpha-rust-fuse-merge"].groupby("start").floored_ari.median()
cost = starts.groupby("start").start_seconds.median() + starts.groupby("start").build_seconds.median()
start_ari = starts.groupby("start").ari.median()
alt = alternating[(alternating.solver == "port:alpha-rust-fuse-merge") & (alternating["round"] == 4)].set_index("start").ari
order = alt.reindex(start_ari.index).fillna(fuse).sort_values().index
y = np.arange(len(order))
right.hlines(y, start_ari[order], alt.reindex(order), color="0.8", lw=1)
right.scatter(start_ari[order], y, facecolors="none", edgecolors="C0", s=22, label="start", zorder=3)
right.scatter(fuse.reindex(order), y, marker="|", color="C1", s=60, label="solved once, floored", zorder=3)
right.scatter(alt.reindex(order), y, color="C0", s=22, label="alternated, 4 rounds", zorder=3)
right.set_yticks(y, [f"{n}  ({cost[n]:.0f} s)" for n in order], fontsize=7)
right.axvline(oracle["ari"], color="C2", lw=0.6, ls="--")
right.legend(loc="lower right", fontsize=7, frameon=False)
right.set(xlabel="clone ARI", title="starts, with --sal's solver and floor (start + states and field, s)")
figure.tight_layout()
import hashlib, subprocess
data = hashlib.sha256(Path("data/clone_label_study_r0.json").read_bytes()).hexdigest()[:8]
commit = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"], capture_output=True, text=True).stdout.strip()
dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", "../../python", "../../tests"],
                       capture_output=True, text=True).stdout.strip()
figure.text(0.995, 0.005, f"data {data} · code {commit or 'unknown'}{'+' if dirty else ''}",
            ha="right", va="bottom", fontsize=6, color="0.4")
# NB untracked (`tests.plots_dir`): the figure is this notebook's output above.
Path("../../.cache/plots/studies").mkdir(parents=True, exist_ok=True)
figure.savefig("../../.cache/plots/studies/clone_label_study.png", dpi=150, metadata={"Software": None})""",
    ),
    (
        "markdown",
        "## Alternating: the loop `--sal` runs\n\nSolve, floor, refit the states warm, rebuild the field; four rounds from each start (seed 0) with three solvers. ARI by round, and the wall to the last round.",
    ),
    (
        "code",
        """alternating = rows[(rows.arm == "alternating") & rows.error.isna()]
rounds = alternating.pivot_table(index=["start", "solver"], columns="round", values="ari")
rounds["seconds"] = alternating.groupby(["start", "solver"]).seconds.max()
rounds["clones"] = alternating[alternating["round"] == 4].set_index(["start", "solver"]).clones
rounds.sort_values(4, ascending=False).round(4)""",
    ),
    (
        "markdown",
        "## Joint annealing and sampling of states and labels\n\nFrom four starts: heat-bath sweeps at T = 8, 4, 2, 1, 0.5 (annealing) or five times at T = 1 keeping the draw of highest data log-likelihood (sampling), the states refitted between, then `alpha-rust-fuse-merge` and the floor. Against alternating with `alpha-rust-fuse-merge` from the same start.",
    ),
    (
        "code",
        """joint = rows[rows.arm.isin(["joint-anneal", "joint-sample"]) & rows.error.isna()]
final = joint[joint.step == "solved"].set_index(["start", "arm"])[["ari", "clones", "seconds"]]
fused = alternating[(alternating.solver == "port:alpha-rust-fuse-merge") & (alternating["round"] == 4)]
fused = fused.assign(arm="alternating").set_index(["start", "arm"])[["ari", "clones", "seconds"]]
pd.concat([final, fused.loc[fused.index.get_level_values(0).isin(final.index.get_level_values(0))]]).sort_index().round(4)""",
    ),
    (
        "markdown",
        "## The coupling\n\nThe leading starts solved at the run's coupling x 0.1, 1 and 10 (#247, #420), floored.",
    ),
    (
        "code",
        """beta = rows[(rows.arm == "beta") & rows.error.isna()]
beta.pivot_table(index=["start", "solver"], columns="factor", values="ari").round(4)""",
    ),
    (
        "markdown",
        "## Refused and crashed\n\nEvery row with an error, by arm and solver.",
    ),
    (
        "code",
        """errors = rows[rows.error.notna()]
errors.groupby(["arm", "solver"]).error.agg(["size", "first"]) if len(errors) else "none" """,
    ),
]


def build(intro: str, out: Path) -> None:
    """The notebook: the intro's markdown, then `CELLS`, executed in `docs/nb`."""
    import nbformat
    from nbclient import NotebookClient

    v4: Any = nbformat.v4

    notebook = v4.new_notebook()
    # NB format 4.4: no cell ids, which are random per build and would make
    #    every rebuild a diff.
    notebook.nbformat_minor = 4
    notebook.cells = [v4.new_markdown_cell(intro)] + [
        v4.new_code_cell(source) if kind == "code" else v4.new_markdown_cell(source)
        for kind, source in CELLS
    ]
    notebook.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    NotebookClient(
        notebook, timeout=600, resources={"metadata": {"path": str(out.parent)}}
    ).execute()
    for cell in notebook.cells:
        cell.pop("id", None)
        if cell.cell_type == "code":
            cell.metadata = {}
            cell.execution_count = None
            for output in cell.get("outputs", []):
                output.pop("execution_count", None)
    writer: Any = nbformat.write
    writer(notebook, out)
    import shutil
    import subprocess

    ruff = shutil.which("ruff")
    if ruff is not None:
        subprocess.run([ruff, "format", "-q", str(out)], check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("results", type=Path, nargs="?")
    arguments = parser.parse_args(argv)
    if arguments.results is not None:
        with arguments.results.open("rb") as fh:
            summarize(pickle.load(fh), DATA)
    build(INTRO.read_text(), NOTEBOOK)


if __name__ == "__main__":
    main()
