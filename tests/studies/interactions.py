"""#570: five settings that act on state allocation and clone labelling, by a sampled design.

`python -m tests.studies.interactions design OUT.json`
`python -m tests.studies.interactions run DESIGN.json OUT.jsonl SAMPLE...`
`python -m tests.studies.interactions analyse OUT.jsonl`

The full factorial is 2 x 3 x 2 x 4 x 4 = 192 configurations per sample.
`design` draws `RUNS` of them by coordinate exchange, D-optimal for `MODEL`:
every main effect, and the two-way interactions among the four factors that
act on state allocation (`ALLOCATION`). 32 configurations cannot carry all
38 two-way terms beside the 11 main ones, and the solver acts after state
allocation, on the clone labels, so it enters as a main effect alone.

`run` scores each configuration on each sample with `tests.sim_audit`, one
subprocess per run, appending a JSON line per run, so an interrupted stream
resumes where it stopped. `analyse` fits, per metric, a least-squares model
of `MODEL` plus a sample offset, with standard errors from the residuals.

Levels, and what selects each:

- floor: the read-depth segment floor in normal UMI (#551). `300` is
  `--sal`'s; `0` sets `quality.min_segment_normal_umi: false`.
- start: `--hmm-start`. `hmcx5+em` is `sal`'s HMC on its Gaussian surrogate,
  not #557's HMC on the HMM objective (`port.sandbox.known_copy`), which no
  pipeline option reaches (#563).
- split: `--split-state` (#481).
- dispersion: #566's arms, which are levels of one factor and refused
  together.
- solver: `PORT_LABEL_SOLVER`. `icm` is `cnaster`'s.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

FACTORS: dict[str, tuple[str, ...]] = {
    "floor": ("300", "0"),
    "start": ("kmeans++x5+em", "lattice", "hmcx5+em"),
    "split": ("off", "on"),
    "dispersion": ("shared", "per-state", "rescale", "two-component"),
    "solver": (
        "alpha-rust-fuse-merge",
        "sw-field-glauber-merge",
        "glauber-merge",
        "icm",
    ),
}
"""Each factor's levels, the first being `--sal`'s on `main` (the reference level)."""

ALLOCATION = ("floor", "start", "split", "dispersion")
"""The factors that act on state allocation; their pairwise interactions are in `MODEL`."""

MODEL: tuple[tuple[str, ...], ...] = (
    *((factor,) for factor in FACTORS),
    *itertools.combinations(ALLOCATION, 2),
)
"""Main effects, then the allocation factors' two-way interactions: 28 parameters with the intercept."""

RUNS = 32
SEED = 570

METRICS = (
    "exact_bgain",
    "exact_altered_pf",
    "exact_altered",
    "ari",
    "copy_ari",
    "exact_loh",
    "exact_loh_pf",
    "exact_ugain",
    "exact_ugain_pf",
)
"""`tests.sim_audit.SimRecovery` fields, #570's first two first."""

THRESHOLD = 0.05
"""An effect counts at this size or more and beyond two standard errors (#565's threshold)."""


def _coded(factor: str, level: str) -> np.ndarray:
    """Effect coding: `k - 1` columns, the reference level at -1 throughout."""
    levels = FACTORS[factor]
    column = np.zeros(len(levels) - 1)
    index = levels.index(level)
    if index == 0:
        column[:] = -1.0
    else:
        column[index - 1] = 1.0
    return column


def columns(config: dict[str, str]) -> np.ndarray:
    """`config`'s row of the `MODEL` matrix, intercept first."""
    parts: list[np.ndarray] = [np.ones(1)]
    for term in MODEL:
        coded = [_coded(factor, config[factor]) for factor in term]
        parts.append(
            coded[0] if len(coded) == 1 else np.outer(coded[0], coded[1]).ravel()
        )
    return np.concatenate(parts)


def names() -> list[str]:
    """One name per `MODEL` column, `term[level]` or `a[level]:b[level]`."""
    out = ["intercept"]
    for term in MODEL:
        if len(term) == 1:
            out += [f"{term[0]}[{level}]" for level in FACTORS[term[0]][1:]]
        else:
            a, b = term
            out += [
                f"{a}[{x}]:{b}[{y}]" for x in FACTORS[a][1:] for y in FACTORS[b][1:]
            ]
    return out


def _log_det(rows: list[dict[str, str]]) -> float:
    matrix = np.array([columns(r) for r in rows])
    sign, value = np.linalg.slogdet(matrix.T @ matrix)
    return float(value) if sign > 0 else -np.inf


def design(
    runs: int = RUNS, seed: int = SEED, restarts: int = 20
) -> list[dict[str, str]]:
    """`runs` configurations, coordinate exchange on `log det(X'X)`, best of `restarts` seeded starts.

    Each start balances every factor's levels (a 3-level factor to within one
    run), then exchanges one cell at a time while the determinant rises.
    """
    rng = np.random.default_rng(seed)
    best: list[dict[str, str]] = []
    best_value = -np.inf
    for _ in range(restarts):
        rows: list[dict[str, str]] = [{} for _ in range(runs)]
        for factor, levels in FACTORS.items():
            column = [levels[k % len(levels)] for k in range(runs)]
            for row, level in zip(rows, rng.permutation(column), strict=True):
                row[factor] = str(level)
        value = _log_det(rows)
        improved = True
        while improved:
            improved = False
            for row in rows:
                for factor, levels in FACTORS.items():
                    kept = row[factor]
                    for level in levels:
                        if level == kept:
                            continue
                        row[factor] = level
                        trial = _log_det(rows)
                        if trial > value + 1e-9:
                            value, kept, improved = trial, level, True
                    row[factor] = kept
        if value > best_value:
            best, best_value = [dict(r) for r in rows], value
    return best


def arm(config: dict[str, str]) -> tuple[list[str], list[str], dict[str, str]]:
    """`tests.sim_audit`'s flags, `--set` entries and environment for `config`."""
    flags = ["--sal", "--no-plots", "--hmm-start", config["start"]]
    sets: list[str] = []
    if config["floor"] == "0":
        sets.append("quality.min_segment_normal_umi=false")
    if config["split"] == "on":
        flags.append("--split-state")
    flags += {
        "shared": [],
        "per-state": ["--per-state-dispersion"],
        "rescale": ["--dispersion-rescale"],
        "two-component": ["--dispersion-rescale", "--dispersion-two-component"],
    }[config["dispersion"]]
    return flags, sets, {"PORT_LABEL_SOLVER": config["solver"]}


def run(design_path: Path, out: Path, samples: list[str], root: Path) -> None:
    """Every configuration on every sample, sample-major, skipping what `out` already holds."""
    configs = json.loads(design_path.read_text())["configs"]
    done = set()
    if out.exists():
        done = {
            (r["sample"], r["index"])
            for r in map(json.loads, out.read_text().splitlines())
        }
    for sample in samples:
        for index, config in enumerate(configs):
            if (sample, index) in done:
                continue
            flags, sets, env = arm(config)
            where = root / f"{Path(sample).name}_{index:02d}"
            command = [sys.executable, "-m", "tests.sim_audit", "--sample", sample,
                       "--root", str(where), *(f"--set={s}" for s in sets), "--", *flags]  # fmt: skip
            started = time.perf_counter()
            done_run = subprocess.run(command, capture_output=True, text=True, timeout=1800,
                                      env={**os.environ, **env}, check=False)  # fmt: skip
            line = next(
                (x for x in done_run.stdout.splitlines() if x.startswith("SIM ")), None
            )
            record: dict[str, Any] = {"sample": sample, "index": index, **config,
                                      "rc": done_run.returncode,
                                      "wall": round(time.perf_counter() - started, 1)}  # fmt: skip
            if line is not None:
                record["score"] = json.loads(line[4:])
            else:
                record["error"] = done_run.stderr[-2000:]
            with out.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                json.dumps({k: record[k] for k in ("sample", "index", "rc", "wall")}),
                flush=True,
            )


def analyse(out: Path) -> dict[str, list[dict[str, Any]]]:
    """Per metric, each `MODEL` coefficient with its standard error, a sample offset fitted beside them.

    Effects are reported as the coefficient: under effect coding, a level's
    departure from the mean over levels. A run that failed or whose metric
    is undefined (a class not planted) is left out of that metric's fit.
    """
    records = [r for r in map(json.loads, out.read_text().splitlines()) if "score" in r]
    samples = sorted({r["sample"] for r in records})
    table: dict[str, list[dict[str, Any]]] = {}
    for metric in METRICS:
        rows = [r for r in records if isinstance(r["score"].get(metric), int | float)
                and np.isfinite(r["score"][metric])]  # fmt: skip
        if len(rows) <= len(names()) + len(samples):
            continue
        x = np.array([
            np.concatenate([columns(r), [float(r["sample"] == s) for s in samples[1:]]])
            for r in rows
        ])  # fmt: skip
        y = np.array([float(r["score"][metric]) for r in rows])
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        residual = y - x @ coef
        dof = len(y) - np.linalg.matrix_rank(x)
        sigma2 = float(residual @ residual) / max(dof, 1)
        se = np.sqrt(np.clip(np.diag(np.linalg.pinv(x.T @ x)) * sigma2, 0.0, None))
        table[metric] = [
            {
                "term": n,
                "effect": round(float(c), 4),
                "se": round(float(s), 4),
                "counts": bool(abs(c) >= THRESHOLD and abs(c) > 2 * s),
            }  # fmt: skip
            for n, c, s in zip(names(), coef, se, strict=False)
        ]
    return table


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if args[0] == "design":
        configs = design()
        payload = {"seed": SEED, "runs": RUNS, "model": [list(t) for t in MODEL],
                   "log_det": _log_det(configs), "configs": configs}  # fmt: skip
        Path(args[1]).write_text(json.dumps(payload, indent=1) + "\n")
        for factor, levels in FACTORS.items():
            print(factor, {lv: sum(c[factor] == lv for c in configs) for lv in levels})
    elif args[0] == "run":
        root = Path(os.environ.get("INTERACTIONS_ROOT", Path(args[2]).parent / "runs"))
        run(Path(args[1]), Path(args[2]), args[3:], root)
    elif args[0] == "analyse":
        print(json.dumps(analyse(Path(args[1])), indent=1))
    else:
        msg = f"design, run or analyse, not {args[0]!r}"
        raise SystemExit(msg)


if __name__ == "__main__":
    main()
