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
is at most `MAX_WIDTH_DEX` wide; and a J compared with the default either
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
MIN_PER_BIN = 20
MAX_WIDTH_DEX = 0.3

UMI_EDGES = np.round(np.arange(4.7, 6.71, 0.25), 2)
"""log10 clone UMIs, 0.25 dex bins."""

LENGTH_EDGES = np.round(np.arange(6.0, 8.51, 0.25), 2)
"""log10 event length in bp, 1 Mb to about 300 Mb."""

J_COLOURS = {0.0: "#86b6ef", 0.5: "#3987e5", 1.0: "#1c5cab", 2.0: "#0d366b"}
"""An ordinal blue ramp: J is ordered, so one hue light to dark (validated)."""

CLASS_COLOURS = {
    "LOH": "#2a78d6",
    "balanced gain": "#eb6834",
    "imbalanced gain": "#1baf7a",
    "all": "#52514e",
}
"""Categorical slots 1-3 in fixed order; `all` pools them, in secondary ink."""


EVENT_COLUMNS = ("seed", "J", "clone", "chr", "length", "a", "b", "class", "bins",
                 "correct", "recovered")  # fmt: skip


def load(out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(clones, events)`: one row per scored clone and per scored event."""
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
    """`x` at which a weighted logistic fit of `y` on `x` reads one half."""
    from sklearn.linear_model import LogisticRegression

    kept = weight > 0
    if kept.sum() < 2 or y[kept].min() == y[kept].max():
        return float("nan")
    fit = LogisticRegression(C=1e6, max_iter=1000)
    fit.fit(x[kept, None], y[kept], sample_weight=weight[kept])
    slope, intercept = float(fit.coef_[0, 0]), float(fit.intercept_[0])
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


def summarize(out: Path, seed: int = 544) -> dict[str, Any]:
    """Study 1 per J and Study 2 per class, with the sufficiency verdict."""
    clones, events = load(out)
    clones["log_umis"] = np.log10(clones["umis"])
    events["log_length"] = np.log10(events["length"])
    seeds = np.array(sorted(clones["seed"].unique()))
    # NB one set of resamples for every curve: a J's curve and the default's
    #    are read on the same members, so their difference is paired.
    weights = _weights(seeds, np.random.default_rng(seed))

    study1: dict[float, dict[str, Any]] = {}
    for j, frame in clones.groupby("J"):
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

    default = events[events["J"] == 1.0] if len(events) else events
    study2 = {}
    for name in ("LOH", "balanced gain", "imbalanced gain", "all"):
        frame = default if name == "all" else default[default["class"] == name]
        study2[name] = {
            "events": len(frame),
            "members": int(frame["seed"].nunique()) if len(frame) else 0,
            "recovered": curve(
                frame, "log_length", "recovered", LENGTH_EDGES, seeds, weights
            ),
        }

    differences = {}
    base = study1.get(1.0, {}).get("detected", {}).get("crossing_draws")
    for j, entry in study1.items():
        if j == 1.0 or base is None:
            continue
        paired = np.array(entry["detected"]["crossing_draws"]) - np.array(base)
        paired = paired[np.isfinite(paired)]
        interval = (
            np.percentile(paired, [2.5, 97.5]).tolist()
            if paired.size
            else [float("nan")] * 2
        )
        differences[j] = {
            "interval": interval,
            "resolved": bool(paired.size and (interval[0] > 0 or interval[1] < 0)),
        }

    return {
        "study1": study1,
        "study2": study2,
        "differences": differences,
        "sufficient": SUFFICIENT(study1, study2),
        "members": int(seeds.size),
    }


def SUFFICIENT(study1: dict[Any, Any], study2: dict[str, Any]) -> dict[str, Any]:
    """The rule stated before the numbers: per-bin counts and crossing widths."""
    failures = []
    for j, entry in study1.items():
        width = np.diff(entry["detected"]["crossing_interval"])[0]
        if not np.isfinite(width) or width > MAX_WIDTH_DEX:
            failures.append(f"J={j:g}: UMI50 interval {width:.2f} dex")
        thin = [c for c, n in zip(entry["detected"]["centres"], entry["detected"]["n"], strict=True)
                if 0 < n < MIN_PER_BIN]  # fmt: skip
        if thin:
            failures.append(f"J={j:g}: bins with < {MIN_PER_BIN} clones at {thin}")
    for name, entry in study2.items():
        width = np.diff(entry["recovered"]["crossing_interval"])[0]
        if np.isfinite(entry["recovered"]["crossing"]) and (
            not np.isfinite(width) or width > MAX_WIDTH_DEX
        ):
            failures.append(f"{name}: L50 interval {width:.2f} dex")
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
        for j, entry in sorted(summary["study1"].items()):
            for axis, key in zip(axes, ("detected", "completeness"), strict=True):
                _panel(axis, entry[key]["centres"], entry[key], J_COLOURS.get(j, "#52514e"),
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
        axis.set_title("CNA recovery against length, detected clones, J = 1",
                       fontsize=10)  # fmt: skip
        path = into / "population_cna_length.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    return paths


def report(out: Path) -> dict[str, Any]:
    """Summarize `out`'s records, write the summary and figures beside them."""
    summary = summarize(out)
    figures(summary, out / "figures")
    slim: dict[str, Any] = json.loads(json.dumps(summary, default=float))
    for entry in slim["study1"].values():
        for key in ("detected", "completeness"):
            entry[key].pop("crossing_draws")
    for entry in slim["study2"].values():
        entry["recovered"].pop("crossing_draws")
    (out / "summary.json").write_text(json.dumps(slim, indent=1) + "\n")
    print(json.dumps({"members": slim["members"], "sufficient": slim["sufficient"]},
                     indent=1))  # fmt: skip
    return slim
