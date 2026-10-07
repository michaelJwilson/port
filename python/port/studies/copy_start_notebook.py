"""#540: `docs/nb/copy_state_starts.ipynb`, built and executed from `copy_starts run`'s results.

    run_study --copy-start-notebook RESULTS.pkl

writes `docs/nb/data/copy_state_starts_r0.json` (the rows, without the
workers' tracebacks), then the notebook, executed by `nbclient`, which reads
only that file and draws `.cache/plots/studies/copy_state_starts.png`
(untracked, `port.qa.provenance.PLOTS`) beside its own output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from port.studies import notebook

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "docs" / "nb" / "data" / "copy_state_starts_r0.json"
NOTEBOOK = ROOT / "docs" / "nb" / "copy_state_starts.ipynb"
FIGURE = ROOT / ".cache" / "plots" / "studies" / "copy_state_starts.png"
INTRO = ROOT / "docs" / "nb" / "copy_state_starts.md"
"""The notebook's opening cell: the result and its reading, written once the rows are in."""


def summarize(results: dict[str, Any], out: Path) -> None:
    """The rows as JSON: arrays as lists, planted states keyed `"A,B"`, tracebacks dropped."""

    def plain(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return [round(float(v), 6) for v in value.ravel()]
        if isinstance(value, np.floating | float):
            return round(float(value), 6)
        if isinstance(value, dict):
            return {
                ",".join(map(str, k)) if isinstance(k, tuple) else k: plain(v)
                for k, v in value.items()
            }
        if isinstance(value, tuple | list):
            return [plain(v) for v in value]
        return value

    rows = [
        {k: plain(v) for k, v in row.items() if k != "trace"} for row in results["rows"]
    ]
    held = {
        "rows": rows,
        "planted": plain(results["planted"]),
        "settings": plain(results["settings"]),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(held, indent=0, sort_keys=True) + "\n")


CELLS = [
    (
        "code",
        """import json
from pathlib import Path

import numpy as np
import pandas as pd

held = json.loads(Path("data/copy_state_starts_r0.json").read_text())
rows = pd.DataFrame(held["rows"])
settings = held["settings"]
planted = held["planted"]
print(f"{len(rows)} trials; {rows.error.notna().sum()} refused; "
      f"{settings['seconds']:g} s a trial, seeds {settings['seeds']}, {settings['workers']} workers")""",
    ),
    (
        "markdown",
        "## The planted states at oracle clones\n\nEach planted state's pooled `(log mu, folded p, rows)` on the call: the truth a start is read against. `(0, 1)` is a one-copy loss, `(0, 2)` copy-neutral LOH, `(2, 2)` a balanced gain.",
    ),
    (
        "code",
        """pd.DataFrame(
    [(stage, key, *values) for stage, states in planted.items() for key, values in states.items()],
    columns=["stage", "(A, B)", "log mu", "p (folded)", "rows"],
)""",
    ),
    (
        "markdown",
        "## Every start\n\nThe gap is below the best log-likelihood any trial reached on the same call, so every gap is >= 0. `found` counts the planted states some fitted state lies within 0.1 in log mu and 0.05 in folded p of (BAF-only: p alone). Median over seeds.",
    ),
    (
        "code",
        """ok = rows[rows.error.isna()].copy()
clean = ok[ok.scored_on == "call"]
best = clean.groupby("stage").log_likelihood.max()
ok["gap"] = np.where(ok.scored_on == "call", ok.stage.map(best) - ok.log_likelihood, np.nan)
outliers = ok[ok.arm == "outlier"].copy()
outliers["call"] = outliers.variant.str.split(" masked").str[0]
ref = outliers.groupby(["stage", "call"]).log_likelihood.transform("max")
ok.loc[outliers.index, "gap"] = ref - outliers.log_likelihood
ok["found"] = ok.found.map(lambda f: sum(bool(v) for v in f.values()))


def table(frame):
    return (
        frame.groupby(["stage", "start"])
        .agg(gap=("gap", "median"), gap_max=("gap", "max"), seconds=("seconds", "median"),
             found=("found", "median"), trials=("seed", "size"))
        .round({"gap": 1, "gap_max": 1, "seconds": 1, "found": 1})
        .sort_values(["stage", "gap"])
    )


starts = ok[ok.arm == "starts"]
table(starts)""",
    ),
    (
        "code",
        """refused = rows[rows.error.notna()]
refused.assign(error=refused.error.str.slice(0, 90)).groupby(["arm", "stage", "start"]).error.first().to_frame()""",
    ),
    (
        "markdown",
        "## Runtime against gap\n\nEach start's median over seeds, with the seeds' range; one panel per stage. Named: the six nearest the best fit and the pipeline's own starts; the rest grey. Starred: the three best arms that seed on masked, smoothed or RDR-derived rows. The deprecated `docs/plots/studies/hmm_starts.png` (#489) drew the BAF + RDR stage alone.",
    ),
    (
        "code",
        """import matplotlib as mpl
import matplotlib.pyplot as plt
from port.extensions.figure_style import figure_rc

mpl.use("Agg")

with mpl.rc_context(figure_rc()):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    for ax, (stage, title) in zip(axes, [("baf", "BAF only"), ("rdrbaf", "BAF + RDR")], strict=True):
        frame = starts[starts.stage == stage]
        summary = frame.groupby("start").agg(
            gap=("gap", "median"), lo=("gap", "min"), hi=("gap", "max"),
            seconds=("seconds", "median"), found=("found", "median"))
        floor = 0.1
        # NB named: the six nearest the best fit, and the starts the pipeline
        #    installs (`distinct`, `cnaster-gmm`, `--sal`'s `kmeans++x5+em`);
        #    the rest grey, unnamed.
        named = set(summary.gap.nsmallest(6).index) | {"distinct", "cnaster-gmm", "kmeans++x5+em"}
        for name, row in summary.iterrows():
            y = max(row.gap, floor)
            shown = name in named
            ax.errorbar(row.seconds, y, yerr=[[y - max(row.lo, floor)], [max(row.hi, floor) - y]],
                        fmt="o", ms=4 if shown else 3, capsize=2, alpha=0.9 if shown else 0.35,
                        color=None if shown else "0.6", zorder=3 if shown else 1)
            if shown:
                ax.annotate(name, (row.seconds, y), fontsize=7, xytext=(4, 2), textcoords="offset points")
        # NB starred: the three best arms other than `starts` on this stage,
        #    each a start seeded on masked, smoothed or RDR-derived rows.
        armed = ok[(ok.stage == stage) & (ok.arm != "starts") & (ok.scored_on == "call")]
        tops = (armed.groupby(["arm", "variant", "start"])
                .agg(gap=("gap", "median"), seconds=("seconds", "median"))
                .nsmallest(3, "gap"))
        for (arm, variant, name), row in tops.iterrows():
            y = max(row.gap, floor)
            ax.plot(row.seconds, y, marker="*", ms=9, color="black", zorder=4)
            ax.annotate(f"{name}, {variant or arm}", (row.seconds, y), fontsize=7,
                        xytext=(4, -9), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("seconds, start and polish")
        ax.set_title(f"{title}: {len(summary)} starts", loc="left")
    axes[0].set_ylabel("gap below the best fit reached [nats] (0 drawn at 0.1)")
    figure.tight_layout()
    # NB untracked (`port.qa.provenance.PLOTS`): the figure is this notebook's output below.
    Path("../../.cache/plots/studies").mkdir(parents=True, exist_ok=True)
    figure.savefig("../../.cache/plots/studies/copy_state_starts.png", dpi=150, metadata={"Software": None})
    plt.show()""",
    ),
    (
        "markdown",
        "## Arms on the shortlist\n\nEach arm against the same start's own `starts` row: the change in median gap (negative is better) and in states found. `mask-seed` masks the rows a start seeds from; `mask-fit` also fits on them; `smooth` seeds on rows summed along the genome; every result is polished and scored on the whole call.",
    ),
    (
        "code",
        """base = table(starts)[["gap", "found"]]
arms = ok[~ok.arm.isin(["starts", "outlier"])]
by = (arms.groupby(["arm", "variant", "stage", "start"])
      .agg(gap=("gap", "median"), found=("found", "median"), seconds=("seconds", "median")))
joined = by.join(base, on=["stage", "start"], rsuffix="_start")
joined["d_gap"] = (joined.gap - joined.gap_start).round(1)
joined["d_found"] = joined.found - joined.found_start
joined[["gap", "d_gap", "found", "d_found", "seconds"]].round(1)""",
    ),
    (
        "markdown",
        "## Outliers\n\nThe shortlist on calls with 1% or 5% of rows replaced (read depth x8 or /8; B at 0 or at its trials), masked and not. The gap is on the corrupted call; `found` is against the clean call's states.",
    ),
    (
        "code",
        """(ok[ok.arm == "outlier"].groupby(["stage", "variant", "start"])
 .agg(gap=("gap", "median"), found=("found", "median"), seconds=("seconds", "median")).round(1))""",
    ),
]


def main(argv: list[str] | None = None) -> None:
    notebook.main(
        argv,
        description=(__doc__ or "").splitlines()[0],
        summarize=summarize,
        data=DATA,
        intro=INTRO,
        cells=CELLS,
        notebook=NOTEBOOK,
    )
