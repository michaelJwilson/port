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

**Sufficiency is stated before the numbers are read** (`sufficiency`): every
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

from port.qa.statistics import bootstrap_interval, resample_weights

BOOTSTRAP = 2000
"""Resamples of the members, per curve."""
MIN_PER_BIN = 20
MAX_WIDTH_DEX = 0.3

UMI_EDGES = np.round(np.arange(5.5, 6.51, 0.2), 2)
"""log10 clone UMIs, 0.2 dex bins over the drawn range: 100 to 1,000 spots."""

UMI_DISPLAY = np.round(np.arange(5.5, 6.51, 0.1), 2)
"""The figure's finer bins; the rule reads `UMI_EDGES`."""

LENGTH_EDGES = np.round(np.arange(6.0, 8.51, 0.25), 2)
"""log10 event length in bp, 1 Mb to about 300 Mb."""

LENGTH_DISPLAY = np.round(np.arange(6.0, 8.51, 0.125), 3)

STAY_EDGES = np.arange(-9.5, -1.4, 1.0)
"""log10 `1 - t`: one bin per value of `population.STAY` (#729)."""
"""The figure's finer bins; the rule reads `LENGTH_EDGES`."""

SNP_EDGES = np.round(np.arange(0.5, 3.51, 0.5), 2)
"""log10 SNP-covering UMIs of a `(1, 1)` segment: Study 3's bins."""

SNP_DISPLAY = np.round(np.arange(0.5, 3.51, 0.25), 2)

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


EVENT_COLUMNS = ("seed", "J", "manifest", "clone", "chr", "length", "a", "b", "class", "bins",
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


def neutral(out: Path) -> pd.DataFrame:
    """Per member, J, log10 SNP UMIs (to 0.01) and outcome: how many `(1, 1)`
    segments. Records scored before `neutral_segments` have none.

    Segments number about 2,000 per clone, so they are counted rather than
    listed: a segment enters the rates and the fit through its row's `count`.
    """
    rows = []
    for path in sorted((out / "records").glob("*.json")):
        record = json.loads(path.read_text())
        for clone in record.get("neutral_segments", []):
            umis = np.asarray(clone["snp_umis"], dtype=float)
            kept = umis > 0
            rows.append(pd.DataFrame({
                "seed": record["seed"], "J": record["J"],
                "log_snp_umis": np.round(np.log10(umis[kept]), 2),
                "specific": np.asarray(clone["specific"])[kept],
            }))  # fmt: skip
    if not rows:
        return pd.DataFrame(columns=["seed", "J", "log_snp_umis", "specific", "count"])
    frame = pd.concat(rows, ignore_index=True)
    keys = ["seed", "J", "log_snp_umis", "specific"]
    return frame.groupby(keys).size().rename("count").reset_index()


def load(out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(clones, events)`: one row per scored clone and per scored event.

    A run that raised has no rows; `failures` counts it.
    """
    clones, events = [], []
    for path in sorted((out / "records").glob("*.json")):
        record = json.loads(path.read_text())
        key = {"seed": record["seed"], "J": record["J"],
               "manifest": record.get("manifest", "population")}  # fmt: skip
        clones += [key | c for c in record["clones"]]
        events += [key | e for e in record["events"]]
    return pd.DataFrame(clones), pd.DataFrame(events, columns=list(EVENT_COLUMNS))


def _binned(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    binned: np.ndarray = np.digitize(values, edges) - 1
    return binned


def _fit(
    x: np.ndarray, y: np.ndarray, weight: np.ndarray
) -> tuple[float, float] | None:
    """`(intercept, slope)` of a weighted logistic fit of `y` on `x`, or `None`.

    Newton's method on the weighted log-likelihood (IRLS), 50 steps at most;
    `None` where the outcome does not vary or the fit separates.
    """
    kept = weight > 0
    if kept.sum() < 2 or y[kept].min() == y[kept].max():
        return None
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
            return None
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(beta[0]), float(beta[1])


def _crossing(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> float:
    """`x` at which the logistic fit reads one half; NaN if it does not rise."""
    fit = _fit(x, y, weight)
    if fit is None or fit[1] <= 0:
        return float("nan")
    return -fit[0] / fit[1]


def _logistic(fit: tuple[float, float] | None, grid: np.ndarray) -> np.ndarray:
    if fit is None:
        return np.full(grid.size, np.nan)
    curve_: np.ndarray = 1.0 / (1.0 + np.exp(-(fit[0] + fit[1] * grid)))
    return curve_


def curve(
    frame: pd.DataFrame,
    x: str,
    y: str,
    edges: np.ndarray,
    seeds: np.ndarray,
    weights: np.ndarray,
    *,
    crossing: bool = True,
    display: np.ndarray | None = None,
    count: str | None = None,
) -> dict[str, Any]:
    """The rate per bin, its bootstrap 95% interval, counts, and the crossing.

    `crossing` fits the half-point, for a 0/1 outcome only, and with it the
    fitted curve on a grid of 101 points over `edges` and its 95% band over
    the same resamples. `display` bins the rate again, finer, for the figure
    alone; the rule reads `edges`.

    A resample weights each member by how often it was drawn, so a member's
    clones enter together: the cluster bootstrap, computed without copying
    rows. `count` names a column of how many items a row stands for.
    """
    n_bins = edges.size - 1
    centres = ((edges[:-1] + edges[1:]) / 2).tolist()
    nan = [float("nan")] * n_bins
    if frame.empty:
        return {"grid": [], "fitted": [], "band": [[], []], "display": {},
                "centres": centres, "rate": nan, "low": nan, "high": nan,
                "n": [0] * n_bins, "crossing": float("nan"),
                "crossing_interval": [float("nan")] * 2,
                "crossing_draws": [float("nan")] * len(weights)}  # fmt: skip

    member = np.searchsorted(seeds, frame["seed"].to_numpy())
    bins = _binned(frame[x].to_numpy(), edges)
    inside = (bins >= 0) & (bins < n_bins)
    values = frame[y].to_numpy(dtype=float)
    many = np.ones(len(frame)) if count is None else frame[count].to_numpy(dtype=float)
    sums = np.zeros((seeds.size, n_bins))
    counts = np.zeros((seeds.size, n_bins))
    np.add.at(sums, (member[inside], bins[inside]), (values * many)[inside])
    np.add.at(counts, (member[inside], bins[inside]), many[inside])

    with np.errstate(invalid="ignore", divide="ignore"):
        rate = sums.sum(0) / counts.sum(0)
    low, high = bootstrap_interval(sums, counts, weights)

    xs, ys = frame[x].to_numpy(dtype=float), frame[y].to_numpy(dtype=float)
    grid = np.linspace(edges[0], edges[-1], 101)
    fits = [_fit(xs, ys, w[member] * many) for w in weights] if crossing else []
    crossings = np.array(
        [np.nan if f is None or f[1] <= 0 else -f[0] / f[1] for f in fits]
        if crossing
        else np.full(len(weights), np.nan)
    )
    finite = crossings[np.isfinite(crossings)]
    fitted = _logistic(_fit(xs, ys, many), grid) if crossing else grid * np.nan
    with np.errstate(invalid="ignore"):
        band = (
            np.nanpercentile(
                np.array([_logistic(f, grid) for f in fits]), [2.5, 97.5], axis=0
            )
            if crossing
            else np.full((2, grid.size), np.nan)
        )
    shown = {}
    if display is not None:
        fine = _binned(frame[x].to_numpy(), display)
        within = (fine >= 0) & (fine < display.size - 1)
        fine_sums = np.zeros((seeds.size, display.size - 1))
        fine_counts = np.zeros((seeds.size, display.size - 1))
        np.add.at(fine_sums, (member[within], fine[within]), (values * many)[within])
        np.add.at(fine_counts, (member[within], fine[within]), many[within])
        with np.errstate(invalid="ignore", divide="ignore"):
            fine_rate = fine_sums.sum(0) / fine_counts.sum(0)
        fine_low, fine_high = bootstrap_interval(fine_sums, fine_counts, weights)
        shown = {
            "centres": ((display[:-1] + display[1:]) / 2).tolist(),
            "rate": fine_rate.tolist(),
            "low": fine_low.tolist(),
            "high": fine_high.tolist(),
            "n": fine_counts.sum(0).astype(int).tolist(),
        }
    return {
        "grid": grid.tolist(),
        "fitted": fitted.tolist(),
        "band": band.tolist(),
        "display": shown,
        "centres": centres,
        "rate": rate.tolist(),
        "low": low.tolist(),
        "high": high.tolist(),
        "n": counts.sum(0).astype(int).tolist(),
        "crossing": _crossing(xs, ys, many) if crossing else float("nan"),
        "crossing_interval": (
            np.percentile(finite, [2.5, 97.5]).tolist()
            if finite.size
            else [float("nan")] * 2
        ),
        "crossing_draws": crossings.tolist(),
    }


LONG_ARM = "population_long"
"""The long-event arm's manifest (#544): Study 1 reads every other manifest's members as the
base population, `population`'s records and `study`'s since T- #807."""


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
    # NB Study 1 reads the base population alone: the long-event arm alters
    #    more of each clone's genome, which bears on detection.
    base_clones = clones[clones["manifest"] != LONG_ARM]
    js = sorted(base_clones["J"].unique())
    at = base_clones.groupby("J")["seed"].apply(set)
    seeds = np.array(sorted(set.intersection(*(at[j] for j in js))))
    weights = resample_weights(seeds.size, BOOTSTRAP, rng)
    paired = base_clones[base_clones["seed"].isin(seeds)]

    study1: dict[float, dict[str, Any]] = {}
    for j, frame in paired.groupby("J"):
        study1[float(j)] = {
            "members": int(frame["seed"].nunique()),
            "clones": len(frame),
            "detected": curve(
                frame,
                "log_umis",
                "detected",
                UMI_EDGES,
                seeds,
                weights,
                display=UMI_DISPLAY,
            ),  # fmt: skip
            "completeness": curve(
                frame,
                "log_umis",
                "completeness",
                UMI_EDGES,
                seeds,
                weights,
                crossing=False,
                display=UMI_DISPLAY,
            ),
        }

    # NB Study 2 is read at one J, on every member run there.
    default = events[events["J"] == study2_j] if len(events) else events
    members2 = np.array(sorted(clones.loc[clones["J"] == study2_j, "seed"].unique()))
    weights2 = resample_weights(members2.size, BOOTSTRAP, rng)
    study2 = {}
    for name in ("LOH", "balanced gain", "imbalanced gain", "all"):
        frame = default if name == "all" else default[default["class"] == name]
        study2[name] = {
            "events": len(frame),
            "members": int(frame["seed"].nunique()) if len(frame) else 0,
            "recovered": curve(
                frame,
                "log_length",
                "recovered",
                LENGTH_EDGES,
                members2,
                weights2,
                display=LENGTH_DISPLAY,
            ),
        }

    # NB Study 3, the false-positive rate (1 - specificity), is read on
    #    Study 2's members and resamples.
    segments = neutral(out)
    segments = segments[segments["J"] == study2_j]
    segments = segments.assign(false_positive=1 - segments["specific"].astype(int))
    study3 = {
        "segments": int(segments["count"].sum()),
        "members": int(segments["seed"].nunique()) if len(segments) else 0,
        "false_positive": curve(
            segments,
            "log_snp_umis",
            "false_positive",
            SNP_EDGES,
            members2,
            weights2,
            display=SNP_DISPLAY,
            count="count",
        ),
        "pooled": _pooled(segments, "false_positive", members2, weights2),
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

    summary = {
        "study1": study1,
        "study2": study2,
        "study3": study3,
        "differences": differences,
        "sufficient": sufficiency(study1, study2),
        "members": int(seeds.size),
        "study2_members": int(members2.size),
        "study2_J": study2_j,
        "failures": {f"{j:g}": runs for j, runs in failures(out).items()},
    }
    arm = stay(out, rng)
    if arm:
        summary["t_arm"] = arm
    return summary


def stay(out: Path, rng: np.random.Generator) -> dict[str, Any]:
    """#729's arm per `1 - t`, from `out/stay/`: CNA sensitivity against log10
    event length, every class, as study 2's length curve is per J (#745)."""
    rows = []
    for path in sorted((out / "stay").glob("*.json")):
        record = json.loads(path.read_text())
        rows += [{"seed": record["seed"], **r} for r in record["rows"]]
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    frame["log_length"] = np.log10(frame["length"])
    members = np.array(sorted(frame["seed"].unique()))
    weights = resample_weights(members.size, BOOTSTRAP, rng)
    arm = {}
    for omt, part in frame.groupby("omt"):
        arm[f"{omt:g}"] = {
            "events": len(part),
            "members": int(part["seed"].nunique()),
            "recovered": curve(
                part, "log_length", "recovered", LENGTH_EDGES, members, weights
            ),  # fmt: skip
        }
    return arm


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


def sufficiency(study1: dict[Any, Any], study2: dict[str, Any]) -> dict[str, Any]:
    """The rule stated before the numbers: per-bin counts and crossing widths.

    Study 1, per J: every bin holds `MIN_PER_BIN` clones, and UMI50's 95%
    interval is at most `MAX_WIDTH_DEX` wide. Study 2, per class: at least
    three bins hold `MIN_PER_BIN` events, and the curve either crosses one
    half with an L50 interval at most `MAX_WIDTH_DEX` wide, or is resolved
    as never reaching (or never falling below) one half in the drawn range.
    """
    failed: list[str] = []
    for j, entry in study1.items():
        width = np.diff(entry["detected"]["crossing_interval"])[0]
        if not np.isfinite(width) or width > MAX_WIDTH_DEX:
            failed.append(f"J={j:g}: UMI50 interval {width:.2f} dex")
        thin = [c for c, n in zip(entry["detected"]["centres"], entry["detected"]["n"], strict=True)
                if n < MIN_PER_BIN]  # fmt: skip
        if thin:
            failed.append(f"J={j:g}: bins with < {MIN_PER_BIN} clones at {thin}")
    for name, entry in study2.items():
        curve_ = entry["recovered"]
        if sum(n >= MIN_PER_BIN for n in curve_["n"]) < 3:
            failed.append(f"{name}: fewer than 3 bins with {MIN_PER_BIN} events")
        read = verdict(curve_, LENGTH_EDGES[0], LENGTH_EDGES[-1])
        entry["verdict"] = read
        width = np.diff(curve_["crossing_interval"])[0]
        if read == "unresolved" or (
            read == "crossed" and (not np.isfinite(width) or width > MAX_WIDTH_DEX)
        ):
            failed.append(f"{name}: L50 {read}, interval {width:.2f} dex")
    return {"passes": not failed, "failures": failed}


CLASS_NAMES = {
    "LOH": "LOH",
    "balanced gain": r"$p = 0.5,\ \mu > 1$",
    "imbalanced gain": r"$p \neq 0.5,\ \mu > 1$",
    "all": "All",
}
"""The legend's names for the copy-state classes: a gain by its BAF `p` and
read-depth ratio `mu`, LOH and All by name (#743)."""


def _panel(axis: Any, entry: dict[str, Any], colour: str, label: str,
           dodge: float, unit: float | None) -> None:  # fmt: skip
    """The fitted curve and its 95% band, with the finer binned rates on it.

    Curves and bins are in log10 of the covariate; they are drawn at
    `10^x / unit` on a log axis, or at `x` itself where `unit` is None.
    Points are drawn for every bin holding an item, shifted `dodge` dex so
    bars of neighbouring series do not overlap.
    """

    def at(x: np.ndarray) -> np.ndarray:
        return x if unit is None else 10**x / unit

    grid = np.array(entry["grid"])
    fitted = np.array(entry["fitted"])
    curved = bool(grid.size and np.isfinite(fitted).any())
    if curved:
        low, high = np.array(entry["band"])
        axis.fill_between(at(grid), low, high, color=colour, alpha=0.15, lw=0)
        axis.plot(at(grid), fitted, color=colour, lw=FIT_WIDTH, label=label)
    shown = entry.get("display") or entry
    rate = np.array(shown["rate"])
    low, high = np.array(shown["low"]), np.array(shown["high"])
    kept = np.array(shown["n"]) > 0
    x = at(np.array(shown["centres"])[kept] + dodge)
    axis.errorbar(x, rate[kept], yerr=[rate[kept] - low[kept], high[kept] - rate[kept]],
                  color=colour, lw=BAR_WIDTH, ls="none", marker="o", ms=MARKER,
                  capsize=0.8,
                  label=None if curved else label)  # fmt: skip


def _false_positives(axis: Any, entry: dict[str, Any]) -> None:
    """The false-positive rate on a log axis: rare, so a linear one reads 0.

    The fit and its band where positive, and a bin's rate with its resampled
    95% interval where it saw a false positive; a bin with none has no point.
    """
    colour = CLASS_COLOURS["all"]
    grid, fitted = np.array(entry["grid"]), np.array(entry["fitted"])
    if grid.size and np.isfinite(fitted).any():
        low, high = np.array(entry["band"])
        axis.fill_between(
            grid, low, high, where=high > 0, color=colour, alpha=0.15, lw=0
        )
        axis.plot(grid, fitted, color=colour, lw=FIT_WIDTH)
    shown = entry.get("display") or entry
    x, n = np.array(shown["centres"]), np.array(shown["n"], dtype=float)
    rate = np.array(shown["rate"])
    low, high = np.array(shown["low"]), np.array(shown["high"])
    seen = (n > 0) & (rate > 0)
    axis.errorbar(x[seen], rate[seen],
                  yerr=[rate[seen] - np.maximum(low[seen], rate[seen] / 10),
                        high[seen] - rate[seen]],
                  color=colour, lw=BAR_WIDTH, ls="none", marker="o", ms=MARKER,
                  capsize=0.8)  # fmt: skip
    axis.set_yscale("log")


def _dodges(n: int, width: float) -> list[float]:
    return list(np.linspace(-width, width, n)) if n > 1 else [0.0]


FIT_WIDTH = 1.0
"""Points: a fitted curve, twice `PROFILE_LINEWIDTH`, the page's rule weight."""

BAR_WIDTH = 0.5
MARKER = 1.5
"""Points: an error bar at the page's rule weight, and a bin's marker."""

COMBINED = "pop_combined.png"
"""#729's 2 x 2, the population study's one figure in `docs/plots/paper/` (#743): (a) clones by UMIs per J, (b) CNAs by `1 - t` per class at
oracle clones, (c) false positives by SNP UMIs, (d) CNAs by length per class."""

COMBINED_HEIGHT = 6.6
"""Inches: the 2 x 2 at the paper's width with square panels, under the text
block less `CAPTION_ROOM` (6.68 in) (#743)."""

FPR_FLOOR = 3.0
"""The false-positive axis starts at the lowest binned rate over this: a bar
reaching toward 0 on a log axis otherwise sets the range (#743)."""

FPR_TICKS = (3e-4, 1e-3, 3e-3, 1e-2)


def _style() -> None:
    import matplotlib as mpl

    from port.extensions.combined_figure import FONT_SIZE
    from port.patch.plot_copy_number_profile import LINEWIDTH

    mpl.rcParams.update({"font.size": FONT_SIZE, "axes.linewidth": LINEWIDTH,
                         "xtick.major.width": LINEWIDTH,
                         "ytick.major.width": LINEWIDTH,
                         "xtick.minor.width": LINEWIDTH,
                         "ytick.minor.width": LINEWIDTH,
                         "xtick.major.size": 2, "ytick.major.size": 2,
                         "xtick.minor.size": 1, "ytick.minor.size": 1,
                         "legend.fontsize": FONT_SIZE,
                         "legend.borderaxespad": 0.2})  # fmt: skip


def _clones(axis: Any, summary: dict[str, Any]) -> None:
    """Clone sensitivity by log10 clone UMIs, one curve per J."""
    colours = j_colours(list(summary["study1"]))
    series = sorted(summary["study1"].items())
    for (j, entry), dodge in zip(series, _dodges(len(series), 0.015), strict=True):
        _panel(axis, entry["detected"], colours[j], f"J = {j:g}", dodge, None)
    axis.set_xlabel(r"$\log_{10} |{\rm Clone\ UMIs}|$")
    axis.set_ylabel("Clone sensitivity")


def _by_class(axis: Any, entries: dict[str, Any], unit: float | None) -> None:
    # NB the gains first in the legend: they are the classes the length
    #    curve separates.
    order = ("imbalanced gain", "balanced gain", "LOH", "all")
    classes = [(k, entries[k]) for k in order if k in entries]
    for (name, entry), dodge in zip(classes, _dodges(len(classes), 0.02), strict=True):
        _panel(axis, entry["recovered"], CLASS_COLOURS[name], CLASS_NAMES[name],
               dodge, unit)  # fmt: skip
    axis.set_xscale("log")
    axis.set_ylabel("CNA sensitivity")


def _lengths(axis: Any, summary: dict[str, Any]) -> None:
    """CNA sensitivity by length, per class."""
    _by_class(axis, summary["study2"], 1e6)
    axis.set_xlabel("CNA length [Mb]")


STAY_SHOWN = (1e-8, 1e-6, 1e-4, 1e-2)
"""The `1 - t` drawn on (b), every other one run: four curves, as (a) has four J."""


def _stay(axis: Any, summary: dict[str, Any]) -> None:
    """CNA sensitivity by length at oracle clones, one curve per `1 - t` in
    `STAY_SHOWN`, coloured as (a)'s J are: #729's arm (`summary["t_arm"]`)."""
    shown = [omt for omt in STAY_SHOWN if f"{omt:g}" in summary["t_arm"]]
    colours = j_colours(shown)
    for omt, dodge in zip(shown, _dodges(len(shown), 0.015), strict=True):
        label = rf"$1 - t = 10^{{{round(np.log10(omt))}}}$"
        _panel(axis, summary["t_arm"][f"{omt:g}"]["recovered"], colours[omt], label,
               dodge, 1e6)  # fmt: skip
    axis.set_xscale("log")
    axis.set_xlabel("CNA length [Mb]")
    axis.set_ylabel("CNA sensitivity")


def _fpr(axis: Any, summary: dict[str, Any]) -> None:
    """The false-positive rate by SNP UMIs, floored at the lowest binned rate over `FPR_FLOOR`."""
    import matplotlib as mpl

    entry = summary["study3"]["false_positive"]
    _false_positives(axis, entry)
    shown = entry.get("display") or entry
    rates = np.array(shown["rate"], dtype=float)
    rates = rates[np.isfinite(rates) & (rates > 0)]
    if rates.size:
        floor = float(rates.min()) / FPR_FLOOR
        axis.set_ylim(floor, None)
        ticks = [t for t in FPR_TICKS if t >= floor]
        axis.set_yticks(ticks, [_power(t) for t in ticks])
    axis.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    axis.set_xlabel(r"$\log_{10} |{\rm SNP\ UMIs\ in\ segment}|$")
    axis.set_ylabel("False positive rate")


def _power(value: float) -> str:
    exponent = int(np.floor(np.log10(value)))
    mantissa = round(value / 10**exponent)
    return (
        f"$10^{{{exponent}}}$"
        if mantissa == 1
        else f"${mantissa}\\times10^{{{exponent}}}$"
    )


def _finish(fig: Any, axes: list[Any], legends: list[Any]) -> None:
    """Square panels, sensitivities on [0, 1], keys inside, a letter over each panel."""
    from port.extensions.combined_figure import FONT_SIZE, LABEL_SIZE

    for axis in axes:
        axis.set_box_aspect(1)
    for axis in legends:
        axis.set_ylim(-0.02, 1.02)
        axis.set_yticks(np.linspace(0.0, 1.0, 6))
        axis.legend(loc="upper left", frameon=True, framealpha=0.85,
                    edgecolor="none", fancybox=False, borderpad=0.15,
                    labelspacing=0.1, handlelength=0.8, handletextpad=0.3,
                    fontsize=FONT_SIZE)  # fmt: skip
    # NB constrained layout settles over draws; two, then frozen.
    fig.canvas.draw()
    fig.canvas.draw()
    fig.set_layout_engine("none")
    to_figure = fig.transFigure.inverted()
    for axis, letter in zip(axes, "abcd", strict=False):
        box = axis.get_tightbbox()
        if box is None:
            msg = "an axis with no extent: nothing drawn to place its letter beside"
            raise ValueError(msg)
        x0 = box.transformed(to_figure).x0
        top = axis.get_position().y1
        fig.text(x0, top + 0.01, f"({letter})", fontsize=LABEL_SIZE,
                 ha="left", va="bottom")  # fmt: skip


def combined(summary: dict[str, Any], into: Path) -> Path:
    """`COMBINED` into `into`: (a) clones by UMIs per J, (b) CNAs by length at
    oracle clones per `1 - t` (`summary["t_arm"]`, #729), (c) false positives by
    SNP UMIs, (d) CNAs by length per class; square panels, no stamp (#743).

    Where the summary has no `t_arm`, (b) is drawn empty and says so: the
    other three panels are current, and (b) is #729's arm.
    """
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.layout_engine import ConstrainedLayoutEngine

    from port.extensions.combined_figure import page_style
    from port.extensions.figure_style import PAPER_WIDTH

    into.mkdir(parents=True, exist_ok=True)
    with page_style():
        _style()
        fig, ((a, b), (c, d)) = plt.subplots(
            2, 2, figsize=(PAPER_WIDTH, COMBINED_HEIGHT), dpi=300, facecolor="white"
        )
        fig.set_layout_engine(
            ConstrainedLayoutEngine(rect=(0, 0, 1, 0.97), w_pad=0.02, h_pad=0.04)
        )
        _clones(a, summary)
        if "t_arm" in summary:
            _stay(b, summary)
        else:
            b.set_xscale("log")
            b.set_xlim(1e-9, 1e-2)
            b.set_xlabel(r"$1 - t$")
            b.set_ylabel("CNA sensitivity")
            b.text(0.5, 0.5, "not yet run (#729)", ha="center", va="center",
                   transform=b.transAxes, color="0.5")  # fmt: skip
        _fpr(c, summary)
        _lengths(d, summary)
        _finish(fig, [a, b, c, d], [a, b, d] if "t_arm" in summary else [a, d])
        b.set_ylim(-0.02, 1.02)
        # NB a PNG at 300 dpi, as the paper's other figures are (#745)
        fig.savefig(
            into / COMBINED, dpi=300, facecolor="white", metadata={"Software": None}
        )
        plt.close(fig)
    return into / COMBINED


def report(out: Path, study2_j: float) -> dict[str, Any]:
    """Summarize `out`'s records, write the summary and figures beside them."""
    summary = summarize(out, study2_j)
    combined(summary, out / "figures")
    slim: dict[str, Any] = json.loads(json.dumps(summary, default=float))
    for entry in slim["study1"].values():
        for key in ("detected", "completeness"):
            entry[key].pop("crossing_draws")
    for entry in slim["study2"].values():
        entry["recovered"].pop("crossing_draws")
    slim["study3"]["false_positive"].pop("crossing_draws")
    (out / "summary.json").write_text(json.dumps(slim, indent=1) + "\n")
    (out / "tables.md").write_text(tables(slim))
    print(json.dumps({"members": slim["members"], "sufficient": slim["sufficient"]},
                     indent=1))  # fmt: skip
    return slim


EMPTY = "-"
"""A table cell with no item in its bin."""


def _pooled(
    frame: pd.DataFrame, y: str, seeds: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    """One rate over every row, counted, with its 95% interval over members."""
    if frame.empty:
        return {"rate": float("nan"), "low": float("nan"), "high": float("nan"),
                "false": 0}  # fmt: skip
    member = np.searchsorted(seeds, frame["seed"].to_numpy())
    many = frame["count"].to_numpy(dtype=float)
    hits = np.bincount(member, frame[y].to_numpy() * many, seeds.size)
    total = np.bincount(member, many, seeds.size)
    low, high = bootstrap_interval(hits, total, weights)
    return {"rate": float(hits.sum() / total.sum()), "low": float(low),
            "high": float(high), "false": int(hits.sum())}  # fmt: skip


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
    lines.append("")

    study3 = summary["study3"]
    e = study3["false_positive"]
    lines.append(
        "| (1, 1) segment SNP UMIs, log10 | false positive rate [95%] (segments) |"
    )
    lines.append("| --- | --- |")
    for k, centre in enumerate(e["centres"]):
        if e["n"][k]:
            lines.append(f"| {centre:.2f} | {e['rate'][k]:.2e} [{e['low'][k]:.2e}, "
                         f"{e['high'][k]:.2e}] ({e['n'][k]}) |")  # fmt: skip
    pooled = study3["pooled"]
    lines.append(f"| all | {pooled['rate']:.2e} [{pooled['low']:.2e}, "
                 f"{pooled['high']:.2e}] ({study3['segments']}; "
                 f"{pooled['false']} false) |")  # fmt: skip
    return "\n".join(lines) + "\n"
