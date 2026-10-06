"""#540: copy-state starts at known clones on a stream of drawn realizations, polished by the HMM's Baum-Welch.

`run_study --copy-state-stream MANIFEST OUT_DIR [--problems 5] [--seeds 3] [--held-out 3] [--first 0] [--settings PATH] [--workers 4] [--all | --starts NAME ...]`

`... --tune` tunes the samplers on the HMM (`anneal-hmm`, `tempering-hmm`,
`hmc-hmm`, `sal`'s since #634) on the `--held-out` realizations and writes `SETTINGS`, as
`potts_stream` tunes its samplers: a grid per sampler (`GRID`), `TUNING_SEEDS` seeds per setting, the
cheapest setting whose median gap in log-likelihood at the start's own states
is within `TOLERANCE` of the best setting's. The held-out realizations are
never evaluated: the stream starts after them.

The main process draws each realization of `MANIFEST` and builds its problem at
the planted clones (`port.sandbox.known_copy.problems`: 1 Mb bins under #551's
300 normal-UMI floor, phased allele reads); the pool runs the starts while the
next realization draws.

- **Starts.** `STARTS`, one per family of `port.sandbox.extensions.copy_starts`'
  registry (`--all`: every start), each seeded as `run_start` seeds it but without
  its `sal` mixture polish: the start is the algorithm's own output. A
  stochastic start runs `--seeds` seeds, a deterministic one seed 0.
- **Polish.** `cnaster`'s Baum-Welch on the clones stacked along the genome
  (`known_copy.baum_welch`), from the start's states.
- **Scored.** The log-likelihood at the start's states (`known_copy.decode`)
  and after Baum-Welch; the rows whose state is not the planted one under the
  best 1-1 matching of states (`known_copy.missed`), before and after.
- **Truth.** Per realization, the planted states decoded and polished by the
  same Baum-Welch.

There is no bound: the gap is to the best log-likelihood any run reached on
that realization. When a realization's runs are all in, it pickles
`OUT_DIR/<stem>.pkl` and redraws `OUT_DIR/<stem>.png`
(`port.studies.copy_state_plot`). Each worker warms up on a small drawn call
first; seconds are per job with `--workers` jobs sharing the host.
"""

from __future__ import annotations

import argparse
import pickle
import time
import traceback
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from port.studies import stream as harness

STARTS = (
    "calicost-gmm", "lattice", "prior", "kmeans++", "emission++", "gaussian-em",
    "anneal-hmm", "tempering-hmm", "hmc-hmm",
)  # fmt: skip
"""The starts the paper's initialization figure draws (T- #660). Out of the study, still in the
registry (`--all` runs them): `cnaster-gmm`, `distinct`, `lattice-em`, `rdr-quantiles`, `data`,
`quantile`, the emission++ variants (`EMISSION_VARIANTS`), `sal`'s surrogate `anneal`,
`tempering`, `hmc` (snapped to observed rows, #563) and its best-of-5-with-EM starts,
`--sal`'s `kmeans++x5+em` among them."""

SECONDS = 60.0
"""A best-of-n start's budget for its own polishes, as `run_start` gives it."""

GRID: dict[str, tuple[dict[str, float], ...]] = {
    "anneal-hmm": tuple({"t_start": t, "step": e} for t in (1e2, 1e3, 1e4) for e in (1e-3, 3e-3, 1e-2)),
    "tempering-hmm": tuple({"t_top": t, "step": e} for t in (1e2, 1e3, 1e4) for e in (1e-3, 3e-3, 1e-2)),
    "hmc-hmm": tuple({"temperature": t} for t in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)),
    "emission++trim": tuple({"trim": t} for t in (0.005, 0.02, 0.05, 0.1)),
    "emission++trimx20hmm": tuple({"trim": t, "draws": n} for t in (0.005, 0.02, 0.05) for n in (10, 20)),
    "emission++lloydx5hmm": tuple({"lloyd": r} for r in (1, 3, 10)),
    "emission++anchor": tuple({"lloyd": r} for r in (1, 3, 10)),
    "emission++knn": tuple({"knn": k} for k in (0.003, 0.01, 0.03)),
}  # fmt: skip
"""Each tuned start's settings: the samplers' knobs (`port.sandbox.known_copy.hmm_objective`) and
the emission++ variants' knobs (`port.sandbox.extensions.copy_starts.EMISSION_VARIANTS`); each untuned default is in its grid.

The samplers are `sal`'s since #634, on port's deleted samplers' grid shapes, 9 / 9 / 7: the
budget is held at what port's tuned samplers spent (`hmm_objective.DEFAULTS`) and the fixed step
searched in its place, around the 3.3-3.8e-3 port's step adaptation settled at on realization 0;
`hmc-hmm` adapts its step and mass by `sal`'s dual averaging, so only the temperature is searched.
Port's grid varied the budget; these hold it, so `TOLERANCE`'s "cheapest" is a tie broken by
seconds."""

UNTUNED = {"anneal-hmm": 2, "tempering-hmm": 2, "hmc-hmm": 2, "emission++trim": 1, "emission++trimx20hmm": 3,
           "emission++lloydx5hmm": 1, "emission++anchor": 1, "emission++knn": 1}  # fmt: skip
"""Each grid's index of the schedule the samplers were written with, reported beside the tuned one."""

TUNING_SEEDS = 5
TOLERANCE = 1.0
"""Nats: a setting within this of the best median gap is as good, and the cheapest of those is kept."""

REDRAW = 0.0
"""Seconds between redraws from the runs finished so far, a realization in progress included: 0, after every job."""

SETTINGS = Path(__file__).with_name("copy_sampler_settings.json")
"""The settings tuned once on `dev_tree_1s_hard`'s first `--held-out` realizations (r0 `d2938975`), reused by `--settings`."""


def _raw(problem: Any) -> dict[str, Any]:
    """`cnaster`'s initializer arguments for `problem`: the clones stacked along the genome, one column.

    What `cnaster.hmm.pipeline_baum_welch` hands `gmm_init` when it seeds
    itself, and the arrays `known_copy.baum_welch` fits. Clones as columns
    instead made `gmm_init` return a state per clone per state: 28 for 7
    states on 4 clones.
    """
    from port.sandbox.known_copy.hmm import CONFIG

    def column(values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64)[:, None]

    config = (Path(__file__).resolve().parents[3] / CONFIG).read_text()
    return {"X": np.stack([problem.total, problem.b], axis=1)[:, :, None].astype(np.float64),
            "base_nb_mean": column(problem.exposure), "total_bb_RD": column(problem.trials),
            "lengths": np.asarray(problem.lengths), "log_sitewise_transmat": np.zeros(problem.total.size),
            "params": "smp", "config": config}  # fmt: skip


def _call(problem: Any) -> Any:
    from port.extensions.copy_starts import CopyCall

    return CopyCall("rdrbaf", problem.n_states, problem.total, problem.b, problem.exposure, problem.trials,
                    problem.clone, problem.contig, problem.start, problem.length, problem.planted, _raw(problem))  # fmt: skip


def seed_states(
    name: str,
    problem: Any,
    rng: np.random.Generator,
    setting: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """`name`'s states on `problem`, seeded as `copy_starts.run_start` seeds them, before its `sal` polish."""
    from port.sandbox.extensions.copy_starts import seed_states as seeded
    from port.sandbox.extensions.copy_starts import starts

    return seeded(
        name,
        _call(problem),
        rng,
        covariate=starts()[name].covariate,
        seconds=SECONDS,
        setting=setting,
    )


def solve(
    problem: Any,
    name: str,
    seed: int,
    setting: dict[str, float] | None = None,
    polish: bool = True,
) -> dict[str, Any]:
    """One start, scored at its states and, with `polish`, after Baum-Welch; a failure is a row."""
    import port.sandbox.known_copy as kc

    try:
        opened = time.perf_counter()
        log_mu, p = seed_states(
            name, problem, np.random.default_rng([seed, 540]), setting
        )
        seconds = time.perf_counter() - opened
        truth = problem.truth_label
        at_start = kc.decode(problem, log_mu, p)
        if not polish:
            return {"problem": problem.realization, "start": name, "seed": seed, "setting": setting,
                    "seconds": seconds, "start_llf": at_start.log_likelihood}  # fmt: skip
        fitted = kc.baum_welch(problem, log_mu, p)  # NB `--sal`'s Baum-Welch (#540)
        return {
            "problem": problem.realization, "start": name, "seed": seed, "setting": setting, "seconds": seconds,
            "start_llf": at_start.log_likelihood, "start_missed": kc.missed(at_start.label, truth),
            "bw_seconds": fitted.seconds, "llf": fitted.log_likelihood, "missed": kc.missed(fitted.label, truth),
            "start_degenerate": at_start.degenerate, "degenerate": fitted.degenerate,
            "log_mu": fitted.log_mu, "p_binom": fitted.p_binom,
        }  # fmt: skip
    except Exception as error:  # noqa: BLE001 -- a failed job is a result
        return {"problem": problem.realization, "start": name, "seed": seed,
                "error": f"{type(error).__name__}: {error}", "trace": traceback.format_exc(limit=4)}  # fmt: skip


def _warm() -> None:
    """Every start and the Baum-Welch once on a small call, so no compilation lands in a timing."""
    from types import SimpleNamespace

    rng = np.random.default_rng(0)
    n = 120
    clone = np.repeat([0, 1], n)
    state = np.where(np.arange(2 * n) % n < n // 2, 0, np.where(clone == 1, 1, 0))
    exposure = np.full(2 * n, 300.0)
    trials = rng.integers(20, 40, 2 * n).astype(float)
    tiny = SimpleNamespace(
        realization=-1, total=rng.poisson(exposure * np.where(state == 1, 0.5, 1.0)).astype(float),
        b=rng.binomial(trials.astype(int), np.where(state == 1, 0.1, 0.5)).astype(float), exposure=exposure,
        trials=trials, clone=clone, contig=np.full(2 * n, "1"), start=np.tile(np.arange(n) * 1e6, 2),
        length=np.full(2 * n, 1e6), lengths=np.array([n, n]), planted=np.column_stack([1 - state, np.ones(2 * n, int)]),
        n_states=2, truth_label=state,
    )  # fmt: skip
    for name in STARTS:
        solve(tiny, name, 0)
    for name, grid in GRID.items():
        if name in STARTS:
            solve(tiny, name, 0, grid[0], polish=False)


def _init() -> None:
    import logging

    logging.disable(logging.INFO)
    _warm()


def _describe(problem: Any) -> dict[str, Any]:
    import port.sandbox.known_copy as kc

    truth = problem.truth_label
    at = kc.decode(problem, problem.truth_log_mu, problem.truth_p_binom)
    fitted = kc.baum_welch(problem, problem.truth_log_mu, problem.truth_p_binom)
    return {"truth_start_llf": at.log_likelihood, "truth_start_missed": kc.missed(at.label, truth),
            "truth_llf": fitted.log_likelihood, "truth_missed": kc.missed(fitted.label, truth),
            "n_rows": int(problem.total.size), "n_states": problem.n_states, "states": problem.states.tolist(),
            "draw_seconds": problem.draw_seconds, "build_seconds": problem.build_seconds}  # fmt: skip


def tune(
    pool: ProcessPoolExecutor, held_out: list[Any], names: tuple[str, ...] = tuple(GRID)
) -> dict[str, dict[str, float]]:
    """Each of `GRID`'s samplers' setting: the cheapest within `TOLERANCE` of the best median gap on `held_out`."""
    import pandas as pd

    futures = [pool.submit(solve, p, name, seed, setting, False)
               for p in held_out for name in names for setting in GRID[name] for seed in range(TUNING_SEEDS)]  # fmt: skip
    rows = [f.result() for f in futures]
    failed = [r for r in rows if "error" in r]
    if failed:
        print(
            f"{len(failed)} tuning runs failed, e.g. {failed[0]['error'][:120]}",
            flush=True,
        )
    frame = pd.DataFrame([r for r in rows if "error" not in r])
    top = frame.groupby("problem").start_llf.max()
    frame["gap"] = frame.problem.map(top) - frame.start_llf
    frame["key"] = frame.setting.map(lambda s: tuple(sorted(s.items())))
    chosen: dict[str, dict[str, float]] = {}
    for name, g in frame.groupby("start"):
        key, best, by = harness.cheapest(g, TOLERANCE)
        default = GRID[str(name)][UNTUNED[str(name)]]
        chosen[str(name)] = {**dict(key), "median_gap": round(float(best.gap), 3),
                             "seconds": round(float(best.seconds), 3),
                             "default_median_gap": round(float(by.gap.get(tuple(sorted(default.items())), np.nan)), 3)}  # fmt: skip
        print(f"tuned {name}: {dict(key)}, median gap {best.gap:.1f} nats ({best.seconds:.2f} s); "
              f"untuned {chosen[str(name)]['default_median_gap']:.1f}", flush=True)  # fmt: skip
    return chosen


def run(
    manifest: Path,
    out_dir: Path,
    n_problems: int,
    seeds: int,
    workers: int,
    everything: bool,
    first: int = 0,
    merge: tuple[Path, ...] = (),
    held_out: int = 3,
    settings: Path | None = None,
    reuse: tuple[Path, ...] = (),
    only: tuple[str, ...] = (),
    drop: tuple[str, ...] = (),
) -> Path:
    """The stream after the `held_out` realizations; returns the pickle it keeps current.

    A start in `drop` gets no new job; its `reuse` rows are still kept, so a start can leave mid-stream
    and its realizations so far stay in the record.
    """
    import json
    import logging

    import port.sandbox.known_copy as kc
    from port.sandbox.extensions.copy_starts import starts

    logging.disable(logging.INFO)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (
        f"copy_{manifest.stem}_r{first}.pkl"
        if first or merge
        else f"copy_{manifest.stem}.pkl"
    )
    names = list(only) if only else list(starts()) if everything else list(STARTS)
    tuned: dict[str, dict[str, float]] = {}
    if settings is not None:
        loaded = json.loads(settings.read_text())
        tuned = {
            k: {q: v[q] for q in GRID[k][0]} for k, v in loaded.items() if k in GRID
        }
    registry = starts()
    reused_rows: dict[tuple[int, str, int], dict[str, Any]] = {}
    reused_truth: dict[int, dict[str, Any]] = {}
    for path in reuse:
        earlier = pickle.loads(path.read_bytes())
        reused_truth |= earlier["problems"]
        for row in earlier["rows"]:
            if row["start"] in names and row["problem"] in earlier["complete"]:
                reused_rows[(row["problem"], row["start"], row["seed"])] = row
    held: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    pending: dict[int, int] = {}
    total_jobs: dict[int, int] = {}
    done: list[int] = []
    futures: dict[Future[dict[str, Any]], int] = {}
    opened = time.perf_counter()

    drawn = [time.perf_counter()]

    def draw(partial: bool) -> None:
        shown = (
            sorted({*done, *(i for i in pending if pending[i] < total_jobs[i])})
            if partial
            else done
        )
        record = {"manifest": str(manifest), "problems": held, "rows": rows, "done": shown, "complete": list(done),
                  "starts": names, "seeds": seeds, "tuned": tuned, "held_out": held_out}  # fmt: skip
        out.write_bytes(pickle.dumps(record))
        harness.redraw("copy-state-plot", out, merge)
        drawn[0] = time.perf_counter()

    def drain(block: bool) -> None:
        for index, row in harness.finished(futures, block):
            rows.append(row)
            pending[index] -= 1
            if pending[index]:
                if time.perf_counter() - drawn[0] >= REDRAW:
                    draw(partial=True)
                    print(
                        f"[{time.perf_counter() - opened:6.0f}s] partial: {len(rows)} runs in; plot redrawn",
                        flush=True,
                    )
                continue
            done.append(index)
            draw(partial=False)
            record = {"rows": rows}
            errors = sum("error" in r for r in record["rows"])
            print(f"[{time.perf_counter() - opened:6.0f}s] problem {index} solved; {len(done)}/{n_problems} done, "
                  f"{errors} errors; plot redrawn", flush=True)  # fmt: skip

    with harness.pool(workers, _init) as pool:
        total = held_out + first + n_problems
        for problem in kc.problems(manifest, total, realizations=total):
            if problem.realization < held_out + first:
                continue
            held[problem.realization] = reused_truth.get(
                problem.realization
            ) or _describe(problem)
            print(f"[{time.perf_counter() - opened:6.0f}s] drew {problem.realization}: {problem.total.size} rows, "
                  f"{problem.n_states} states; truth after Baum-Welch missed {held[problem.realization]['truth_missed']}",
                  flush=True)  # fmt: skip
            jobs = [
                (name, seed)
                for name in names
                for seed in (range(seeds) if registry[name].stochastic else [0])
            ]
            kept = [reused_rows[(problem.realization, *job)] for job in jobs
                    if (problem.realization, *job) in reused_rows]  # fmt: skip
            rows.extend(kept)
            jobs = [
                j
                for j in jobs
                if (problem.realization, *j) not in reused_rows and j[0] not in drop
            ]
            print(f"  reused {len(kept)} runs, {len(jobs)} to run", flush=True)
            if not jobs:
                done.append(problem.realization)
                continue
            pending[problem.realization] = len(jobs)
            total_jobs[problem.realization] = len(jobs)
            for name, seed in jobs:
                futures[pool.submit(solve, problem, name, seed, tuned.get(name))] = (
                    problem.realization
                )
            drain(block=False)
        while futures:
            drain(block=True)
    return out


def retune(
    manifest: Path, held_out: int, workers: int, names: tuple[str, ...] = tuple(GRID)
) -> None:
    """`tune` of `names` on `manifest`'s first `held_out` realizations, merged into `SETTINGS` with its provenance."""
    import json
    import logging

    import port.sandbox.known_copy as kc

    logging.disable(logging.INFO)
    with harness.pool(workers, _init) as pool:
        chosen = tune(
            pool, list(kc.problems(manifest, held_out, realizations=held_out)), names
        )
    provenance = (f"run_study --copy-state-stream --tune on {manifest.name} realizations 0-{held_out - 1}, "
                  f"{TUNING_SEEDS} seeds per setting; the cheapest setting within {TOLERANCE} nats of the best "
                  "median gap in log-likelihood at the start's states (#540)")  # fmt: skip
    earlier = json.loads(SETTINGS.read_text()) if SETTINGS.exists() else {}
    SETTINGS.write_text(
        json.dumps({**earlier, "_provenance": provenance, **chosen}, indent=2) + "\n"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--problems", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--first",
        type=int,
        default=0,
        help="skip this many realizations: a window of the stream",
    )
    parser.add_argument(
        "--merge",
        nargs="+",
        type=Path,
        default=(),
        metavar="PKL",
        help="earlier windows to draw with this one",
    )
    parser.add_argument(
        "--reuse",
        nargs="+",
        type=Path,
        default=(),
        metavar="PKL",
        help="earlier records of this manifest: a (realization, start, seed) run of a complete "
        "realization there, and its truth, are taken rather than rerun",
    )
    parser.add_argument(
        "--drop",
        nargs="+",
        default=(),
        metavar="START",
        help="starts given no new job; their --reuse rows are kept",
    )
    parser.add_argument(
        "--held-out",
        type=int,
        default=3,
        help="realizations the tuning reads, never evaluated",
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=None,
        help=f"tuned sampler settings, e.g. {SETTINGS}",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help=f"tune on the held-out realizations, write {SETTINGS.name}, stop",
    )
    parser.add_argument(
        "--tune-starts",
        nargs="+",
        default=list(GRID),
        choices=list(GRID),
        help="with --tune, only these starts; the others' settings are kept",
    )
    parser.add_argument(
        "--starts",
        nargs="+",
        default=(),
        metavar="NAME",
        help="only these starts of the registry, in place of STARTS",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="every start of the registry, not one per family",
    )
    arguments = parser.parse_args(argv)
    if arguments.tune:
        retune(
            arguments.manifest,
            arguments.held_out,
            arguments.workers,
            tuple(arguments.tune_starts),
        )
        return
    run(
        arguments.manifest,
        arguments.out_dir,
        arguments.problems,
        arguments.seeds,
        arguments.workers,
        arguments.all,
        arguments.first,
        tuple(arguments.merge),
        arguments.held_out,
        arguments.settings,
        tuple(arguments.reuse),
        tuple(arguments.starts),
        tuple(arguments.drop),
    )
