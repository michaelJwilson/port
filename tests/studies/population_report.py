"""The population study's curves, their error bars and whether they suffice (#544).

**Error bars are over realizations.** Clones and events of one member share
its draw and its run, so they are not independent: the interval on every
rate is a cluster bootstrap that resamples whole members (seeds) with
replacement, `BOOTSTRAP` times, and reads the 2.5th and 97.5th percentiles.
The same resampled seeds are used for every J, so a difference between two
J is paired, as the draws are.

**Each curve is summarized by where it crosses one half** -- the UMIs at
which half the clones are detected (`UMI50`), and the length at which half
the events of a class are recovered (`L50`) -- from a logistic fit on the
log of the covariate, bootstrapped the same way.

**Sufficiency is stated before the numbers are read** (`SUFFICIENT`): every
bin drawn holds at least `MIN_PER_BIN` items; every crossing's 95% interval
is at most `MAX_WIDTH_DEX` wide; and a J compared with Study 2's J either
differs (the paired interval on the difference excludes zero) or is reported
as unresolved. A study that fails the rule adds members until it passes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BOOTSTRAP = 2000
"""Resamples of the members, per curve."""
MIN_PER_BIN = 20
MAX_WIDTH_DEX = 0.3

UMI_EDGES = np.round(np.arange(5.5, 6.51, 0.2), 2)
"""log10 clone UMIs, 0.2 dex bins over the drawn range: 100 to 1,000 spots."""

LENGTH_EDGES = np.round(np.arange(6.0, 8.51, 0.25), 2)
"""log10 event length in bp, 1 Mb to about 300 Mb."""

J_RAMP = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
"""An ordinal blue ramp, light to dark: J is ordered (validated, `--ordinal`)."""


def j_colours(js: list[float]) -> dict[float, str]:
    """Each J its ramp step, the smallest lightest; at most four J."""
    ordered = sorted(js)
    steps = np.linspace(0, len(J_RAMP) - 1, len(ordered)).round().astype(int)
    return {j: J_RAMP[k] for j, k in zip(ordered, steps, strict=True)}


CLASS_COLOURS = {
    "LOH": "#2a78d6",
    "balanced gain": "#eb6834",
    "imbalanced gain": "#1baf7a",
    "all": "#52514e",
}
"""Categorical slots 1-3 in fixed order; `all` pools them, in secondary ink."""


EVENT_COLUMNS = ("seed", "J", "clone", "chr", "length", "a", "b", "class", "bins",
                 "correct", "recovered")  # fmt: skip


def failures(out: Path) -> dict[float, list[dict[str, Any]]]:
    """Per J, the runs that raised: their seed and error."""
    failed: dict[float, list[dict[str, Any]]] = {}
    for path in sorted((out / "records").glob("*.json")):
        record = json.loads(path.read_text())
        if "error" in record:
            failed.setdefault(float(record["J"]), []).append(
                {"seed": record["seed"], "error": record["error"]}
            )
    return failed


def load(out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(clones, events)`: one row per scored clone and per scored event.

    A run that raised has no rows; `failures` counts it.
    """
    clones, events = [], []
    for path in sorted((out / "records").glob("*.json")):
        record = json.loads(path.read_text())
        key = {"seed": record["seed"], "J": record["J"]}
        clones += [key | c for c in record["clones"]]
        events += [key | e for e in record["events"]]
    return pd.DataFrame(clones), pd.DataFrame(events, columns=list(EVENT_COLUMNS))


def _binned(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    binned: np.ndarray = np.digitize(values, edges) - 1
    return binned


def _crossing(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> float:
    """`x` at which a weighted logistic fit of `y` on `x` reads one half.

    Newton's method on the weighted log-likelihood (IRLS), 50 steps at most;
    NaN where the outcome does not vary, the fit separates, or the slope is
    not positive -- a curve that does not rise has no crossing to report.
    """
    kept = weight > 0
    if kept.sum() < 2 or y[kept].min() == y[kept].max():
        return float("nan")
    design = np.column_stack([np.ones(kept.sum()), x[kept]])
    target, w = y[kept], weight[kept]
    beta = np.zeros(2)
    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-(design @ beta)))
        gradient = design.T @ (w * (target - p))
        hessian = (design * (w * p * (1 - p))[:, None]).T @ design + 1e-9 * np.eye(2)
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if not np.all(np.isfinite(beta)) or abs(beta[1]) > 1e3:
            return float("nan")
        if np.max(np.abs(step)) < 1e-8:
            break
    intercept, slope = float(beta[0]), float(beta[1])
    return -intercept / slope if slope > 0 else float("nan")


def _weights(seeds: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """`(BOOTSTRAP, n_seeds)`: how often each member is drawn in each resample."""
    picks = rng.integers(0, seeds.size, (BOOTSTRAP, seeds.size))
    return np.stack([np.bincount(p, minlength=seeds.size) for p in picks])


def curve(
    frame: pd.DataFrame,
    x: str,
    y: str,
    edges: np.ndarray,
    seeds: np.ndarray,
    weights: np.ndarray,
    *,
    crossing: bool = True,
) -> dict[str, Any]:
    """The rate per bin, its bootstrap 95% interval, counts, and the crossing.

    `crossing` fits the half-point, for a 0/1 outcome only.

    A resample weights each member by how often it was drawn, so a member's
    clones enter together: the cluster bootstrap, computed without copying
    rows.
    """
    n_bins = edges.size - 1
    centres = ((edges[:-1] + edges[1:]) / 2).tolist()
    nan = [float("nan")] * n_bins
    if frame.empty:
        return {"centres": centres, "rate": nan, "low": nan, "high": nan,
                "n": [0] * n_bins, "crossing": float("nan"),
                "crossing_interval": [float("nan")] * 2,
                "crossing_draws": [float("nan")] * len(weights)}  # fmt: skip

    member = np.searchsorted(seeds, frame["seed"].to_numpy())
    bins = _binned(frame[x].to_numpy(), edges)
    inside = (bins >= 0) & (bins < n_bins)
    values = frame[y].to_numpy(dtype=float)
    sums = np.zeros((seeds.size, n_bins))
    counts = np.zeros((seeds.size, n_bins))
    np.add.at(sums, (member[inside], bins[inside]), values[inside])
    np.add.at(counts, (member[inside], bins[inside]), 1.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        rate = sums.sum(0) / counts.sum(0)
        rates = (weights @ sums) / (weights @ counts)
    low, high = np.nanpercentile(rates, [2.5, 97.5], axis=0)

    xs, ys = frame[x].to_numpy(dtype=float), frame[y].to_numpy(dtype=float)
    crossings = (
        np.array([_crossing(xs, ys, w[member]) for w in weights])
        if crossing
        else np.full(len(weights), np.nan)
    )
    finite = crossings[np.isfinite(crossings)]
    return {
        "centres": centres,
        "rate": rate.tolist(),
        "low": low.tolist(),
        "high": high.tolist(),
        "n": counts.sum(0).astype(int).tolist(),
        "crossing": _crossing(xs, ys, np.ones(xs.size)) if crossing else float("nan"),
        "crossing_interval": (
            np.percentile(finite, [2.5, 97.5]).tolist()
            if finite.size
            else [float("nan")] * 2
        ),
        "crossing_draws": crossings.tolist(),
    }


def summarize(out: Path, study2_j: float, seed: int = 544) -> dict[str, Any]:
    """Study 1 per J and Study 2 per class, with the sufficiency verdict."""
    clones, events = load(out)
    clones["log_umis"] = np.log10(clones["umis"])
    events["log_length"] = np.log10(events["length"])
    rng = np.random.default_rng(seed)
    # NB Study 1 is read on the members run at every J -- a member whose run
    #    failed at one J leaves every J -- with one set of resamples, so a
    #    J's curve and another's are read on the same members and their
    #    difference is paired.
    js = sorted(clones["J"].unique())
    at = clones.groupby("J")["seed"].apply(set)
    seeds = np.array(sorted(set.intersection(*(at[j] for j in js))))
    weights = _weights(seeds, rng)
    paired = clones[clones["seed"].isin(seeds)]

    study1: dict[float, dict[str, Any]] = {}
    for j, frame in paired.groupby("J"):
        study1[float(j)] = {
            "members": int(frame["seed"].nunique()),
            "clones": len(frame),
            "detected": curve(frame, "log_umis", "detected", UMI_EDGES, seeds, weights),
            "completeness": curve(
                frame,
                "log_umis",
                "completeness",
                UMI_EDGES,
                seeds,
                weights,
                crossing=False,
            ),
        }

    # NB Study 2 is read at one J, on every member run there.
    default = events[events["J"] == study2_j] if len(events) else events
    members2 = np.array(sorted(clones.loc[clones["J"] == study2_j, "seed"].unique()))
    weights2 = _weights(members2, rng)
    study2 = {}
    for name in ("LOH", "balanced gain", "imbalanced gain", "all"):
        frame = default if name == "all" else default[default["class"] == name]
        study2[name] = {
            "events": len(frame),
            "members": int(frame["seed"].nunique()) if len(frame) else 0,
            "recovered": curve(
                frame, "log_length", "recovered", LENGTH_EDGES, members2, weights2
            ),
        }

    differences = {}
    base = study1.get(study2_j, {}).get("detected", {}).get("crossing_draws")
    for j, entry in study1.items():
        if j == study2_j or base is None:
            continue
        change = np.array(entry["detected"]["crossing_draws"]) - np.array(base)
        change = change[np.isfinite(change)]
        interval = (
            np.percentile(change, [2.5, 97.5]).tolist()
            if change.size
            else [float("nan")] * 2
        )
        differences[j] = {
            "interval": interval,
            "resolved": bool(change.size and (interval[0] > 0 or interval[1] < 0)),
        }

    return {
        "study1": study1,
        "study2": study2,
        "differences": differences,
        "sufficient": SUFFICIENT(study1, study2),
        "members": int(seeds.size),
        "study2_members": int(members2.size),
        "study2_J": study2_j,
        "failures": {f"{j:g}": runs for j, runs in failures(out).items()},
    }


def verdict(entry: dict[str, Any], lo: float, hi: float) -> str:
    """Where a curve crosses one half, as the rule reads it.

    `crossed` if the crossing lies in `[lo, hi]`; `not reached` if every
    populated bin's upper bound is below one half; `exceeded` if every
    populated bin's lower bound is above it; `unresolved` otherwise.
    """
    populated = np.array(entry["n"]) >= MIN_PER_BIN
    if lo <= entry["crossing"] <= hi:
        return "crossed"
    if populated.any() and np.all(np.array(entry["high"])[populated] < 0.5):
        return "not reached"
    if populated.any() and np.all(np.array(entry["low"])[populated] > 0.5):
        return "exceeded"
    return "unresolved"


def SUFFICIENT(study1: dict[Any, Any], study2: dict[str, Any]) -> dict[str, Any]:
    """The rule stated before the numbers: per-bin counts and crossing widths.

    Study 1, per J: every bin holds `MIN_PER_BIN` clones, and UMI50's 95%
    interval is at most `MAX_WIDTH_DEX` wide. Study 2, per class: at least
    three bins hold `MIN_PER_BIN` events, and the curve either crosses one
    half with an L50 interval at most `MAX_WIDTH_DEX` wide, or is resolved
    as never reaching (or never falling below) one half in the drawn range.
    """
    failures = []
    for j, entry in study1.items():
        width = np.diff(entry["detected"]["crossing_interval"])[0]
        if not np.isfinite(width) or width > MAX_WIDTH_DEX:
            failures.append(f"J={j:g}: UMI50 interval {width:.2f} dex")
        thin = [c for c, n in zip(entry["detected"]["centres"], entry["detected"]["n"], strict=True)
                if n < MIN_PER_BIN]  # fmt: skip
        if thin:
            failures.append(f"J={j:g}: bins with < {MIN_PER_BIN} clones at {thin}")
    for name, entry in study2.items():
        curve_ = entry["recovered"]
        if sum(n >= MIN_PER_BIN for n in curve_["n"]) < 3:
            failures.append(f"{name}: fewer than 3 bins with {MIN_PER_BIN} events")
        read = verdict(curve_, LENGTH_EDGES[0], LENGTH_EDGES[-1])
        entry["verdict"] = read
        width = np.diff(curve_["crossing_interval"])[0]
        if read == "unresolved" or (
            read == "crossed" and (not np.isfinite(width) or width > MAX_WIDTH_DEX)
        ):
            failures.append(f"{name}: L50 {read}, interval {width:.2f} dex")
    return {"passes": not failures, "failures": failures}


def _panel(axis: Any, centres: list[float], entry: dict[str, Any], colour: str,
           label: str) -> None:  # fmt: skip
    rate = np.array(entry["rate"])
    low, high = np.array(entry["low"]), np.array(entry["high"])
    kept = np.array(entry["n"]) >= MIN_PER_BIN // 2
    x = np.array(centres)[kept]
    axis.errorbar(x, rate[kept], yerr=[rate[kept] - low[kept], high[kept] - rate[kept]],
                  color=colour, lw=2, marker="o", ms=5, capsize=3, label=label)  # fmt: skip
    if x.size:
        axis.annotate(label, (x[-1], rate[kept][-1]), xytext=(6, 0),
                      textcoords="offset points", fontsize=8, color="#52514e",
                      va="center")  # fmt: skip


def figures(summary: dict[str, Any], into: Path) -> list[Path]:
    """The two figures: clone detection and completeness by J; event recovery by class."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    style: Any = {"axes.spines.top": False, "axes.spines.right": False,
             "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b",
             "xtick.color": "#52514e", "ytick.color": "#52514e",
             "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
             "font.size": 9}  # fmt: skip
    into.mkdir(parents=True, exist_ok=True)
    paths = []

    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
        colours = j_colours(list(summary["study1"]))
        for j, entry in sorted(summary["study1"].items()):
            for axis, key in zip(axes, ("detected", "completeness"), strict=True):
                _panel(axis, entry[key]["centres"], entry[key], colours[j],
                       f"J = {j:g}")  # fmt: skip
        axes[0].set_ylabel("recovery rate (completeness ≥ 0.90)")
        axes[1].set_ylabel("mean completeness")
        for axis in axes:
            axis.set_xlabel("clone UMIs, log10")
            axis.set_ylim(-0.02, 1.02)
            axis.legend(frameon=False, fontsize=8, loc="lower right")
        fig.suptitle(
            f"Clone detection against clone UMIs, {summary['members']} realizations, "
            "95% intervals over realizations",
            fontsize=10,
        )
        path = into / "population_clone_umis.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

        fig, axis = plt.subplots(figsize=(5.5, 3.6), constrained_layout=True)
        for name, entry in summary["study2"].items():
            _panel(axis, entry["recovered"]["centres"], entry["recovered"],
                   CLASS_COLOURS[name], name)  # fmt: skip
        axis.set_xlabel("event length, log10 bp")
        axis.set_ylabel("recovery rate (≥ 0.90 of bins)")
        axis.set_ylim(-0.02, 1.02)
        axis.legend(frameon=False, fontsize=8, loc="lower right")
        axis.set_title(
            f"CNA recovery against length, detected clones, J = {summary['study2_J']:g}",
            fontsize=10,
        )
        path = into / "population_cna_length.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    return paths


def report(out: Path, study2_j: float) -> dict[str, Any]:
    """Summarize `out`'s records, write the summary and figures beside them."""
    summary = summarize(out, study2_j)
    figures(summary, out / "figures")
    slim: dict[str, Any] = json.loads(json.dumps(summary, default=float))
    for entry in slim["study1"].values():
        for key in ("detected", "completeness"):
            entry[key].pop("crossing_draws")
    for entry in slim["study2"].values():
        entry["recovered"].pop("crossing_draws")
    (out / "summary.json").write_text(json.dumps(slim, indent=1) + "\n")
    (out / "tables.md").write_text(tables(slim))
    print(json.dumps({"members": slim["members"], "sufficient": slim["sufficient"]},
                     indent=1))  # fmt: skip
    return slim


EMPTY = "-"
"""A table cell with no item in its bin."""


def _cell(rate: float, low: float, high: float, n: int) -> str:
    if n == 0 or not np.isfinite(rate):
        return EMPTY
    return f"{rate:.2f} [{low:.2f}, {high:.2f}] ({n})"


def tables(summary: dict[str, Any]) -> str:
    """Markdown tables of both studies: rate [95% over members] (count) per bin."""
    lines = []
    study1 = summary["study1"]
    centres = next(iter(study1.values()))["detected"]["centres"]
    lines.append(
        "| clone UMIs, log10 | "
        + " | ".join(f"J = {float(j):g}" for j in study1)
        + " |"
    )
    lines.append("| --- |" + " --- |" * len(study1))
    for k, centre in enumerate(centres):
        cells = []
        for entry in study1.values():
            e = entry["detected"]
            cells.append(_cell(e["rate"][k], e["low"][k], e["high"][k], e["n"][k]))
        if any(c != EMPTY for c in cells):
            lines.append(f"| {centre:.2f} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("| J | members | clones | UMI50, log10 [95%] |")
    lines.append("| --- | --- | --- | --- |")
    for j, entry in study1.items():
        e = entry["detected"]
        low, high = e["crossing_interval"]
        lines.append(f"| {float(j):g} | {entry['members']} | {entry['clones']} | "
                     f"{e['crossing']:.2f} [{low:.2f}, {high:.2f}] |")  # fmt: skip
    lines.append("")

    study2 = summary["study2"]
    centres = next(iter(study2.values()))["recovered"]["centres"]
    lines.append("| event length, log10 bp | " + " | ".join(study2) + " |")
    lines.append("| --- |" + " --- |" * len(study2))
    for k, centre in enumerate(centres):
        cells = []
        for entry in study2.values():
            e = entry["recovered"]
            cells.append(_cell(e["rate"][k], e["low"][k], e["high"][k], e["n"][k]))
        if any(c != EMPTY for c in cells):
            lines.append(f"| {centre:.2f} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("| class | events | L50, log10 bp [95%] | reading |")
    lines.append("| --- | --- | --- | --- |")
    for name, entry in study2.items():
        e = entry["recovered"]
        low, high = e["crossing_interval"]
        lines.append(f"| {name} | {entry['events']} | "
                     f"{e['crossing']:.2f} [{low:.2f}, {high:.2f}] | "
                     f"{entry.get('verdict', '')} |")  # fmt: skip
    return "\n".join(lines) + "\n"
