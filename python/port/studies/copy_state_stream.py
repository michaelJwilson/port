"""#540: copy-state starts at the planted clones, each scored and polished by `run_cnaster_port --sal`'s own Baum-Welch (#730).

`run_study --copy-state-stream MANIFEST OUT_DIR [--problems 5] [--seeds 3] [--held-out 3] [--first 0] [--settings PATH] [--workers 4] [--all | --starts NAME ...]`

`... --tune` tunes the samplers on the HMM (`anneal-hmm`, `tempering-hmm`,
`hmc-hmm`, `sal`'s since #634) on the `--held-out` realizations and writes `SETTINGS`, as
`potts_stream` tunes its samplers: a grid per sampler (`GRID`), `TUNING_SEEDS` seeds per setting, the
cheapest setting whose median gap in log-likelihood at the start's own states
is within `TOLERANCE` of the best setting's. The held-out realizations are
never evaluated: the stream starts after them.

Each realization is drawn to disk as the run reads a sample
(`port.studies.stage.members`), and one pool job per realization runs
`run_cnaster_port --sal` on it at its planted clones up to the RDR + BAF
stage's Baum-Welch (`port.studies.stage.at_oracle_clones`, #730). There the
run's own call, with every argument as the run built it, scores every start:

- **Problem.** The run's: its segments, phasing, pseudobulk and exposure at
  the planted clones, clones stacked along the genome. Nothing is rebuilt by
  the study; the planted clones are the one oracle input.
- **Starts.** `STARTS`, one per family of `port.sandbox.extensions.copy_starts`'
  registry (`--all`: every start), each seeded as `run_start` seeds it but without
  its `sal` mixture polish: the start is the algorithm's own output. A
  stochastic start runs `--seeds` seeds, a deterministic one seed 0. A start's
  first call in a worker runs once untimed: its compilation.
- **Polish.** The run's `pipeline_baum_welch`, with the start as
  `init_log_mu` and `init_p_binom`.
- **Scored.** The same call with `max_iter = 0`, at the start's states, and
  after Baum-Welch; the rows whose state is not the planted `(A, B)` under the
  best 1-1 matching of states (`port.studies.stage.missed`), before and after.
- **References.** Per realization, the planted states (`oracle_states`: each
  planted class's pooled depth ratio and B share) and the run's own
  initializer, each scored and fitted by the same call.

There is no bound: the gap is to the best log-likelihood any run reached on
that realization. When a realization's runs are all in, it writes
`OUT_DIR/<stem>.record` (`port.studies.records`) and redraws `OUT_DIR/<stem>.png`
(`port.studies.copy_state_plot`). Seconds are per job with `--workers`
realizations sharing the host.
"""

from __future__ import annotations

import argparse
import time
import traceback
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from port.studies import records
from port.studies import stream as harness

STARTS = (
    "calicost-gmm", "lattice", "prior", "kmeans++", "emission++",
    "tempering-hmm", "hmc-hmm",
)  # fmt: skip
"""The starts the paper's initialization figure draws (T- #660). Out of the study, still in the
registry (`--all` runs them): `cnaster-gmm`, `distinct`, `lattice-em`, `rdr-quantiles`, `data`,
`quantile`, the emission++ variants (`EMISSION_VARIANTS`), `anneal-hmm` (#716), `sal`'s surrogate `anneal`,
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
"""Each tuned start's settings: the samplers' knobs (`port.sandbox.extensions.hmm_objective`) and
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


def _call(stage: Any) -> Any:
    """The run's arrays at `stage` as a `CopyCall`: one row per (clone, bin), clones stacked genome after genome."""
    from port.extensions.copy_starts import CopyCall

    def tiled(values: Any) -> np.ndarray:
        return np.tile(np.asarray(values), stage.n_clones)

    X = stage.X
    return CopyCall(
        "rdrbaf", stage.n_states, X[:, 0, 0], X[:, 1, 0], stage.base_nb_mean.reshape(-1),
        stage.total_bb_RD.reshape(-1), stage.clone, tiled(stage.contig).astype(str), tiled(stage.start),
        tiled(stage.length), stage.planted,
        {"X": X, "base_nb_mean": stage.base_nb_mean, "total_bb_RD": stage.total_bb_RD, "lengths": stage.lengths,
         "log_sitewise_transmat": np.asarray(stage.args[6]), "params": str(stage.arguments["params"]),
         "config": stage.config},
    )  # fmt: skip


def truth_label(stage: Any) -> np.ndarray:
    """Each stacked row's planted state: its planted `(A, B)` as a class."""
    _, inverse = np.unique(stage.planted, axis=0, return_inverse=True)
    return np.asarray(inverse, dtype=np.int64).ravel()


def seed_states(
    name: str,
    call: Any,
    rng: np.random.Generator,
    setting: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """`name`'s states on `call`, seeded as `copy_starts.run_start` seeds them, before its `sal` polish."""
    from port.sandbox.extensions.copy_starts import seed_states as seeded
    from port.sandbox.extensions.copy_starts import starts

    return seeded(
        name,
        call,
        rng,
        covariate=starts()[name].covariate,
        seconds=SECONDS,
        setting=setting,
    )


def scored(
    stage: Any, log_mu: Any, p_binom: Any, truth: np.ndarray, *, fit: bool
) -> dict[str, Any]:
    """The run's Baum-Welch call at `stage` from `(log_mu, p_binom)`: fitted, or with `fit` false scored at them (`max_iter = 0`)."""
    shape = np.shape(stage.arguments["init_log_mu"])
    log_mu = np.asarray(log_mu, dtype=np.float64).reshape(shape)
    p_binom = np.clip(np.asarray(p_binom, dtype=np.float64), 1e-4, 1 - 1e-4).reshape(
        shape
    )
    opened = time.perf_counter()
    result = stage.run(
        init_log_mu=log_mu, init_p_binom=p_binom, **({} if fit else {"max_iter": 0})
    )
    label = np.asarray(result.profile.pred_cnv, dtype=np.int64).ravel()
    return {"seconds": time.perf_counter() - opened, "llf": float(result.llf), "missed": stage_module().missed(label, truth),
            "log_mu": np.ravel(result.params.new_log_mu), "p_binom": np.ravel(result.params.new_p_binom)}  # fmt: skip


def stage_module() -> Any:
    from port.studies import stage

    return stage


def solve(
    stage: Any, call: Any, realization: int, name: str, seed: int,
    setting: dict[str, float] | None = None, polish: bool = True,
) -> dict[str, Any]:  # fmt: skip
    """One start, scored by the run's Baum-Welch at its states and, with `polish`, fitted from them; a failure is a row."""
    try:
        opened = time.perf_counter()
        log_mu, p = seed_states(name, call, np.random.default_rng([seed, 540]), setting)
        seconds = time.perf_counter() - opened
        truth = truth_label(stage)
        at_start = scored(stage, log_mu, p, truth, fit=False)
        if not polish:
            return {"problem": realization, "start": name, "seed": seed, "setting": setting,
                    "seconds": seconds, "start_llf": at_start["llf"]}  # fmt: skip
        fitted = scored(stage, log_mu, p, truth, fit=True)
        return {
            "problem": realization, "start": name, "seed": seed, "setting": setting, "seconds": seconds,
            "start_llf": at_start["llf"], "start_missed": at_start["missed"],
            "bw_seconds": fitted["seconds"], "llf": fitted["llf"], "missed": fitted["missed"],
            "log_mu": fitted["log_mu"], "p_binom": fitted["p_binom"],
        }  # fmt: skip
    except Exception as error:  # noqa: BLE001 -- a failed job is a result
        return {"problem": realization, "start": name, "seed": seed,
                "error": f"{type(error).__name__}: {error}", "trace": traceback.format_exc(limit=4)}  # fmt: skip


def oracle_states(stage: Any) -> tuple[np.ndarray, np.ndarray]:
    """The planted states as a start: each planted `(A, B)` class's pooled depth ratio and B share, the largest first.

    Read off the run's own arrays at the planted labels. Classes beyond the
    run's `n_states` are dropped, the fewest rows first; a run asking for
    more states than were planted repeats the largest class, so the start
    has the run's shape.
    """
    call = _call(stage)
    truth = truth_label(stage)
    order = np.argsort(-np.bincount(truth))[: stage.n_states]
    order = np.concatenate([order, np.repeat(order[:1], stage.n_states - order.size)])

    def pooled(values: np.ndarray, k: int) -> float:
        return float(values[truth == k].sum())

    log_mu = np.array(
        [np.log(pooled(call.total, k) / pooled(call.exposure, k)) for k in order]
    )
    p_binom = np.array(
        [pooled(call.b, k) / max(pooled(call.trials, k), 1.0) for k in order]
    )
    return log_mu, p_binom


def describe(stage: Any, realization: int) -> dict[str, Any]:
    """The realization's references: the planted states, and the run's own initializer, each scored and fitted."""
    truth = truth_label(stage)
    planted = oracle_states(stage)
    at, fitted = (scored(stage, *planted, truth, fit=f) for f in (False, True))
    run_at = scored(
        stage,
        stage.arguments["init_log_mu"],
        stage.arguments["init_p_binom"],
        truth,
        fit=False,
    )
    run_fit = scored(
        stage,
        stage.arguments["init_log_mu"],
        stage.arguments["init_p_binom"],
        truth,
        fit=True,
    )
    return {"truth_start_llf": at["llf"], "truth_start_missed": at["missed"],
            "truth_llf": fitted["llf"], "truth_missed": fitted["missed"],
            "run_start_llf": run_at["llf"], "run_llf": run_fit["llf"], "run_missed": run_fit["missed"],
            "n_rows": int(truth.size), "n_states": stage.n_states, "n_planted": int(truth.max()) + 1,
            "realization": realization}  # fmt: skip


_WARM: list[str] = []
"""The starts this process has run once, untimed: numba and sal compile on the first call."""


def member(
    path: str, realization: int, jobs: list[tuple[str, int, dict[str, float] | None]], polish: bool, root: str,
) -> dict[str, Any]:  # fmt: skip
    """`jobs` on one realization, each against the run's Baum-Welch at its planted clones (`port.studies.stage`)."""
    import logging
    import shutil

    from port.sim.fixtures import load_simulated
    from port.studies import stage as at

    logging.disable(logging.INFO)
    sample = load_simulated(Path(path).name, Path(path).parent)

    def study(found: Any) -> dict[str, Any]:
        call = _call(found)
        for name in sorted({job[0] for job in jobs} - set(_WARM)):
            # NB untimed: a start's first call pays its compilation
            solve(found, call, realization, name, 0, None, polish=False)
            _WARM.append(name)
        rows = [solve(found, call, realization, *job, polish=polish) for job in jobs]
        return {
            "rows": rows,
            "problem": describe(found, realization) if polish else None,
        }

    try:
        return at.at_oracle_clones(sample, study, root=Path(root))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def tune(
    pool: ProcessPoolExecutor, held_out: list[Any], names: tuple[str, ...], root: Path
) -> dict[str, dict[str, float]]:
    """Each of `GRID`'s samplers' setting: the cheapest within `TOLERANCE` of the best median gap on `held_out`.

    Two rounds (#716): every setting from one seed per held-out realization,
    then the settings `harness.halve` keeps get the other `TUNING_SEEDS - 1`
    seeds, and the choice is among those alone. The gap is to the best
    log-likelihood any tuning run reached on that realization. Each round
    is one job per realization, reaching its stage once (#730).
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []

    def run(jobs: list[tuple[str, int, dict[str, float] | None]]) -> pd.DataFrame:
        futures = [pool.submit(member, str(m.sample.path), m.realization, jobs, False,
                               str(root / f"tune_r{m.realization}")) for m in held_out]  # fmt: skip
        for f in futures:
            rows.extend(f.result()["rows"])
        frame = pd.DataFrame([r for r in rows if "error" not in r])
        top = frame.groupby("problem").start_llf.max()
        frame["gap"] = frame.problem.map(top) - frame.start_llf
        frame["key"] = frame.setting.map(lambda s: tuple(sorted(s.items())))
        return frame

    frame = run([(name, 0, setting) for name in names for setting in GRID[name]])
    kept = {
        str(name): harness.halve(g, TOLERANCE) for name, g in frame.groupby("start")
    }
    frame = run([(name, seed, dict(key)) for name, keys in kept.items() for key in keys
                 for seed in range(1, TUNING_SEEDS)])  # fmt: skip
    failed = [r for r in rows if "error" in r]
    if failed:
        print(
            f"{len(failed)} tuning runs failed, e.g. {failed[0]['error'][:120]}",
            flush=True,
        )
    print(f"tuning: {len(rows)} runs against the full grid's "
          f"{sum(len(GRID[n]) for n in names) * TUNING_SEEDS * len(held_out)}", flush=True)  # fmt: skip
    chosen: dict[str, dict[str, float]] = {}
    for name, g in frame.groupby("start"):
        full = g.groupby("key").gap.size() == TUNING_SEEDS * len(held_out)
        key, best, _ = harness.cheapest(g[g.key.isin(full[full].index)], TOLERANCE)
        by = g.groupby("key").agg(gap=("gap", "median"))
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
    """The stream after the `held_out` realizations; returns the record it keeps current.

    A start in `drop` gets no new job; its `reuse` rows are still kept, so a start can leave mid-stream
    and its realizations so far stay in the record.
    """
    import json
    import logging

    from port.sandbox.extensions.copy_starts import starts
    from port.studies import stage as at

    logging.disable(logging.INFO)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (
        f"copy_{manifest.stem}_r{first}{records.SUFFIX}"
        if first or merge
        else f"copy_{manifest.stem}{records.SUFFIX}"
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
        earlier = records.read(path)
        reused_truth |= earlier["problems"]
        for row in earlier["rows"]:
            if row["start"] in names and row["problem"] in earlier["complete"]:
                reused_rows[(row["problem"], row["start"], row["seed"])] = row
    held: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    done: list[int] = []
    futures: dict[Future[dict[str, Any]], int] = {}
    opened = time.perf_counter()

    def draw() -> None:
        record = {"manifest": str(manifest), "problems": held, "rows": rows, "done": done, "complete": list(done),
                  "starts": names, "seeds": seeds, "tuned": tuned, "held_out": held_out}  # fmt: skip
        records.write(out, record)
        harness.redraw("copy-state-plot", out, merge)

    def drain(block: bool) -> None:
        for index, result in harness.finished(futures, block):
            rows.extend(result["rows"])
            held[index] = reused_truth.get(index) or result["problem"]
            done.append(index)
            draw()
            errors = sum("error" in r for r in rows)
            print(f"[{time.perf_counter() - opened:6.0f}s] problem {index} solved; {len(done)}/{n_problems} done, "
                  f"{errors} errors; truth after Baum-Welch missed {held[index]['truth_missed']} of "
                  f"{held[index]['n_rows']}; plot redrawn", flush=True)  # fmt: skip

    with harness.pool(workers, _init) as pool:
        for m in at.members(
            manifest, out_dir / ".sim", n=n_problems, first=held_out + first
        ):
            jobs = [
                (name, seed)
                for name in names
                for seed in (range(seeds) if registry[name].stochastic else [0])
            ]
            kept = [
                reused_rows[(m.realization, *job)]
                for job in jobs
                if (m.realization, *job) in reused_rows
            ]
            rows.extend(kept)
            todo = [(name, seed, tuned.get(name)) for name, seed in jobs
                    if (m.realization, name, seed) not in reused_rows and name not in drop]  # fmt: skip
            print(f"[{time.perf_counter() - opened:6.0f}s] drew {m.realization} ({m.hash}): reused {len(kept)} runs, "
                  f"{len(todo)} to run", flush=True)  # fmt: skip
            futures[pool.submit(member, str(m.sample.path), m.realization, todo, True,
                                str(out_dir / ".runs" / f"r{m.realization}"))] = m.realization  # fmt: skip
            drain(block=False)
        while futures:
            drain(block=True)
    return out


def _init() -> None:
    import logging

    logging.disable(logging.INFO)


def retune(
    manifest: Path,
    held_out: int,
    workers: int,
    names: tuple[str, ...] = tuple(GRID),
    root: Path | None = None,
) -> None:
    """`tune` of `names` on `manifest`'s first `held_out` realizations, merged into `SETTINGS` with its provenance."""
    import json
    import logging
    import tempfile

    from port.studies import stage as at

    logging.disable(logging.INFO)
    root = Path(tempfile.mkdtemp()) if root is None else root
    with harness.pool(workers, _init) as pool:
        chosen = tune(
            pool, list(at.members(manifest, root / ".sim", n=held_out)), names, root
        )
    provenance = (f"run_study --copy-state-stream --tune on {manifest.name} realizations 0-{held_out - 1}, "
                  f"{TUNING_SEEDS} seeds per setting, at the run's Baum-Welch at oracle clones (#730); the "
                  "cheapest setting within {TOLERANCE} nats of the best median gap in log-likelihood at the "
                  "start's states (#540)").replace("{TOLERANCE}", str(TOLERANCE))  # fmt: skip
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
