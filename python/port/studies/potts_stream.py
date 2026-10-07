"""#556: Potts solvers from random labels on a stream of the run's clone-assignment problems, the plot redrawn per problem.

`run_study --potts-stream MANIFEST OUT_DIR [--problems 25] [--starts 50] [--held-out 3] [--workers 4] [--states run]`
`run_calibrate --potts MANIFEST OUT_DIR --samplers SAMPLER ...` tunes the
named samplers on the held-out realizations and merges them into `SETTINGS`
(`tune`, `retune`); the stream itself never tunes (#749 WP1).
`--only SOLVER ...` runs a subset, `--merge PKL ...` draws it with earlier streams.

The main process draws each realization of `MANIFEST` and runs
`run_cnaster_port --sal` on it to the RDR + BAF stage's clone assignment, at
the planted clones (`port.studies.stage.at_clone_assignment`, #735): the
field, graph and coupling are the ones the run's solver is handed there. The
Baum-Welch before it starts from the run's own initial states, or with
`--states planted` from the planted ones, a second oracle input. The pool
solves each problem while the next one is built. Until #735 the field was the
planted law's (`port.sandbox.known_field`, deleted): a different problem, so
its numbers do not compare with these.

- **Warmup.** Each worker runs every solver once on a 10 x 10 patch before
  any timed job, so no compilation lands in a timing.
- **Tuning.** `run_calibrate --potts` tunes the samplers (`TUNED`) on the
  first `--held-out` realizations, which the stream never evaluates; the
  stream reads the settings it wrote (`SETTINGS`, or `--settings`). Each
  sampler searches `GRID` -- the start temperature, the end fixed
  at `T_END` so the last sweeps are a descent, the sweep budget and the
  warm-up; for a tempering ladder, its hottest replica and its replica sweeps
  -- in three rounds (`tune`): cheap budgets first, the dearest only where
  more sweeps still help, then `TUNING_STARTS` starts for the settings that
  survive a one-start round. It keeps the cheapest of those whose median gap
  to TRW-S's bound is within `TOLERANCE` nats of the best's.
- **Evaluation.** The next `--problems` realizations run every solver of
  #541's harness (`port.studies.clone_label_arms`) but bifurcation, port's
  pure-Python `alpha` and the floor-merge row, plus TRW-S's own decoded
  labelling, from `--starts` random labellings; the samplers at their tuned
  settings. Each run is polished twice: sal's ICM, then the color merge
  (`port.sandbox.extensions.color_merge`, cnaster's `merge_assignment` rule).

Per problem it records the planted labelling's energy, TRW-S's lower bound and its spots,
per run the energy and the labels unlike the planted ones, raw and after each
polish. When a problem's runs are all in it writes `OUT_DIR/<stem>.record`
(`port.studies.records`: Parquet and JSON) and redraws `OUT_DIR/<stem>.png` (`port.studies.potts_plot`).

Timing: the graph is built once per worker per problem, outside the timed
solve; seconds are per job with `--workers` jobs sharing the host.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
import traceback
from collections.abc import Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np

from port.qa.provenance import CONFIGS
from port.studies import records
from port.studies import stage as at
from port.studies import stream as harness

DROPPED = frozenset({
    "sal:bifurcation", "port:alpha", "port:alpha-rust-merge",
    "port:alpha-rust", "port:alpha-rust-icm", "port:icm-numba", "port:icm",
    "sal:swendsen-wang", "sal:wolff",
})  # fmt: skip
"""Out of the stream: bifurcation (#541), `alpha` (Alpha-rust's pure-Python twin), the deprecated floor merge,
and, for the paper's figure (T- #660), `alpha-rust` and `alpha-rust-icm` (`alpha-rust-fuse-merge` stays),
`icm-numba` and cnaster's `icm`; and the uniform-proposal cluster moves, their heat-bath variants in their
place (#716). `clone_label_arms` still runs them."""

EXTRA = ("sal:trws",)
"""Entries beyond the harness's: TRW-S's decoded labelling. `CLUSTER_TEMPERING` left the stream with
T- #660; `--only` still runs it."""

SAMPLERS = {
    "sal:anneal": "single-site",
    "sal:swendsen-wang-heat-bath": "swendsen-wang-heat-bath",
    "sal:wolff-heat-bath": "wolff-heat-bath",
}
"""sal's annealed chains (`run_annealed`), by `sal`'s move set. The cluster moves are sal's heat-bath
variants (its #1142): each cluster's label drawn from its summed field, where the uniform proposal
of `swendsen-wang` and `wolff` is accepted on that field and freezes in a field of this size."""


TEMPERING = "sal:tempering"
"""sal's `parallel_tempering`: a ladder of single-site heat-bath replicas, swapped."""

CLUSTER_TEMPERING = "sal:cluster-tempering"
"""sal's `cluster_tempering` (its #1090): `TEMPERING`'s ladder, one Swendsen-Wang pass per replica per step and
Houdayer moves between replicas."""

TUNED = (*SAMPLERS, TEMPERING, CLUSTER_TEMPERING)
"""Every entry that runs at a tuned annealing setting."""

T_END = 0.05
"""sal's `ANNEAL_END`: cold enough that the last sweeps are a descent."""

GRID = tuple(
    {"t_start": t, "sweeps": s, "warm": w}
    for t, s, w in itertools.product(
        (0.25, 0.5, 1.0, 2.0, 8.0, 32.0), (250, 1000, 4000), (0.0, 0.1, 0.25)
    )
)
"""Start temperature x sweep budget x warm-up. sal's default is T = 2, 1,000 sweeps, no warm-up; the
field's margins run to 18 nats. `warm` is the fraction of the steps held at the start temperature
before the exponential ramp (:class:`WarmSchedule`); a tempering ladder has no ramp and takes `warm` 0."""


@dataclass(frozen=True)
class _Warmed:
    """A schedule held at `t_start` for its first `held` steps, then `ramp`."""

    n_steps: int
    t_start: float
    held: int
    ramp: Any

    def __call__(self, step: int) -> float:
        return self.t_start if step < self.held else float(self.ramp(step - self.held))


@dataclass(frozen=True)
class WarmSchedule:
    """sal's exponential `ScheduleParams` with a warm-up: `warm` of the steps at `t_start` first.

    The chain equilibrates at its hottest temperature before it cools, so the
    ramp starts from a sample of that temperature rather than from the random
    labelling. sal's `ScheduleParams` holds only at the end; `run_annealed`
    calls `build(n_steps)` on whatever it is given.
    """

    t_start: float
    t_end: float
    warm: float = 0.0

    def build(self, n_steps: int) -> _Warmed:
        from sal.sample.schedule import ScheduleParams, ScheduleShape

        held = min(int(self.warm * n_steps), n_steps - 1)
        ramp = ScheduleParams(
            ScheduleShape.EXPONENTIAL, self.t_start, self.t_end
        ).build(n_steps - held)
        return _Warmed(n_steps, self.t_start, held, ramp)


def grid(solver: str) -> tuple[dict[str, float], ...]:
    """`GRID` for `solver`: a tempering ladder's without the warm-up it has no use for."""
    if solver in (TEMPERING, CLUSTER_TEMPERING):
        return tuple(g for g in GRID if g["warm"] == 0.0)
    return GRID


TUNING_STARTS = 5
TOLERANCE = 0.1
"""Nats: a setting within this of the best median gap is as good, and the cheapest of those is kept."""

REPLICAS = 6
"""sal's `N_REPLICAS`: both tempering ladders, geometric between `T_END` and the start temperature."""

SETTINGS = CONFIGS / "potts_sampler_settings.json"
"""The samplers' settings, tuned by `run_calibrate --potts` on `dev_tree_1s_hard`'s first 3
realizations at the run's clone-assignment field (#723); the stream's default `--settings`."""

_GRAPHS: dict[tuple[int, float], Any] = {}


class Problem(NamedTuple):
    """One realization's clone-assignment problem, the run's (`port.studies.stage.Field`)."""

    realization: int
    hash: str
    field: np.ndarray
    planted: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    spatial_weight: float
    states: str
    draw_seconds: float
    field_seconds: float

    @property
    def n_spots(self) -> int:
        return int(self.field.shape[0])


def problems(
    manifest: Path, root: Path, n: int, first: int = 0, states: str = "run"
) -> Iterator[Problem]:
    """Realizations `first`, ..., `first + n - 1` of `manifest`, each drawn and run to its field only when asked for."""
    import shutil

    drawn = at.members(manifest, root / ".sim", n=n, first=first)
    while True:
        opened = time.perf_counter()
        member = next(drawn, None)
        if member is None:
            return
        draw_seconds = time.perf_counter() - opened
        scratch = root / f".run_r{member.realization}"
        try:
            found = at.at_clone_assignment(
                member.sample, lambda f: f, states=states, root=scratch
            )
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        yield Problem(member.realization, member.hash, found.field, found.planted, found.indptr, found.indices,
                      found.weights, found.spatial_weight, states, draw_seconds, found.seconds)  # fmt: skip


def hex_graph(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CSR `(indptr, indices, weights)` of each point's nearest neighbours at weight 1: the warm-up's patch."""
    import scipy.sparse as sp
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    spacing = float(np.min(tree.query(points, k=2)[0][:, 1]))
    pairs = tree.query_pairs(spacing * 1.01, output_type="ndarray")
    n = len(points)
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    matrix = sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
    matrix.sort_indices()
    return matrix.indptr, matrix.indices, matrix.data


def _init() -> None:
    """Build each problem's graph once per worker, then warm every solver up on a small patch."""
    import port.studies.clone_label_arms as arms

    build = arms._graph

    def memo(beta: float) -> Any:
        key = (arms._HELD["problem"], beta)
        if key not in _GRAPHS:
            _GRAPHS[key] = build(beta)
        return _GRAPHS[key]

    arms._graph = memo
    _warm()


def _warm() -> None:
    """Every solver once on a 10 x 10 patch, q = 3: numba compiles here, not in a timed job."""
    from types import SimpleNamespace

    import port.studies.clone_label_arms as arms

    rows, cols = np.divmod(np.arange(100), 10)
    points = np.column_stack([cols + 0.5 * (rows % 2), rows * np.sqrt(3) / 2])
    indptr, indices, weights = hex_graph(points)
    field = np.random.default_rng(0).normal(size=(100, 3))
    patch = SimpleNamespace(realization=-1, field=field, planted=field.argmax(1), indptr=indptr, indices=indices,
                            weights=weights, spatial_weight=1.0, n_spots=100)  # fmt: skip
    # NB every solver `--only` can name, the ones T- #660 dropped from the stream included
    retired = {"sal:bifurcation", "port:alpha", "port:alpha-rust-merge"}
    runnable = [s for s in arms._solvers() if s not in retired]
    for solver in [*runnable, *EXTRA, CLUSTER_TEMPERING]:
        solve(
            patch,
            solver,
            0,
            {"t_start": 2.0, "sweeps": 10} if solver in TUNED else None,
        )


def _hold(index: int, problem: Any) -> None:
    import port.studies.clone_label_arms as arms

    arms._HELD["capture"], arms._HELD["problem"] = problem, index


def _sample(solver: str, field: np.ndarray, start: np.ndarray, rng: np.random.Generator,
            graph: Any, setting: dict[str, float]) -> np.ndarray:  # fmt: skip
    """A sampler at `setting`: an annealed chain on its schedule, or a tempering ladder topped at `t_start`.

    A ladder runs ``sweeps // REPLICAS`` steps of `REPLICAS` replicas, so
    `sweeps` counts replica sweeps for every entry.
    """
    from sal.cost import Cost
    from sal.opt.budget import Budget
    from sal.sample.potts_mcmc.chains import cluster_tempering, parallel_tempering
    from sal.sample.potts_mcmc.moves import PottsMove
    from sal.search.ground_state import Problem as SalProblem
    from sal.search.ground_state import run_annealed

    t_start, sweeps = float(setting["t_start"]), int(setting["sweeps"])
    schedule = WarmSchedule(t_start, T_END, float(setting.get("warm", 0.0)))
    steps = max(1, sweeps // REPLICAS)
    best: Any
    if solver == CLUSTER_TEMPERING:
        # NB coldest first, as sal's cluster_tempering requires
        ladder = tuple(float(t) for t in np.geomspace(T_END, t_start, REPLICAS))
        best = cluster_tempering(graph, field, ladder, rng, steps).best
    elif solver == TEMPERING:
        ladder = tuple(float(t) for t in np.geomspace(t_start, T_END, REPLICAS))
        best = parallel_tempering(graph, field, ladder, rng, steps).best
    else:
        problem = SalProblem(graph, field, field.shape[1])
        budget = Budget(Cost.SITE_VISITS, sweeps * problem.visits_per_sweep)
        move = PottsMove(SAMPLERS[solver])
        calibrated: int | None = None
        if solver == "sal:wolff-heat-bath":
            # NB sal budgets a Wolff step at a sweep's visits and flips one cluster, so it spends a
            #    fraction of the budget; a pilot measures the visits a step costs, and the run takes
            #    as many steps as spend the budget the other samplers get. The pilot is timed too.
            #    A shorter pilot is not: on dev_tree_1s_hard r3 from a random start, a tenth-length
            #    one calibrates 267,525 steps against the full one's 15,104 (#716), since a cluster
            #    grows with the cooling a short schedule compresses.
            pilot = run_annealed(problem, budget, np.random.default_rng(rng.integers(2**63)), move,
                                 schedule=cast(Any, schedule), start=start)  # fmt: skip
            calibrated = max(1, round(sweeps * budget.size / max(pilot.spent, 1)))
        best = run_annealed(problem, budget, rng, move, schedule=cast(Any, schedule), steps=calibrated,
                            start=start).labelling  # fmt: skip
    return np.asarray(best, dtype=np.int64)


def solve(
    problem: Any, solver: str, seed: int, setting: dict[str, float] | None = None
) -> dict[str, Any]:
    """One run from random labels, then its two polishes; a failure is a row."""
    from sal.sim.potts import energy

    import port.studies.clone_label_arms as arms
    from port.sandbox.extensions.color_merge import color_merge
    from port.studies.stage import missed

    try:
        _hold(problem.realization, problem)
        field, beta = problem.field, problem.spatial_weight
        _, graph = arms._graph(beta)
        start = np.random.default_rng([seed, 7]).integers(
            0, field.shape[1], field.shape[0]
        )
        rng = np.random.default_rng([seed, 492])
        opened = time.perf_counter()
        if solver == "sal:trws":
            from sal.search.trws import trws

            out = np.asarray(trws(graph, field).labelling, dtype=np.int64)
        elif setting is not None:
            out = _sample(solver, field, start, rng, graph, setting)
        else:
            out = arms._solve(solver, field, start, rng, beta)
        seconds = time.perf_counter() - opened
        opened = time.perf_counter()
        polished = arms._solve("sal:icm", field, out, np.random.default_rng(0), beta)
        polish_seconds = time.perf_counter() - opened
        opened = time.perf_counter()
        both, merges = color_merge(
            field, polished, problem.indptr, problem.indices, problem.weights, beta
        )
        merge_seconds = time.perf_counter() - opened
        planted = problem.planted
        return {
            "problem": problem.realization, "solver": solver, "seed": seed, "setting": setting,
            "seconds": seconds, "energy": energy(graph, field, out),
            "start_energy": energy(graph, field, start),
            "polish_seconds": polish_seconds, "polished": energy(graph, field, polished),
            "both_seconds": polish_seconds + merge_seconds, "both": energy(graph, field, both),
            "merges": merges, "clones": int(np.unique(out).size), "both_clones": int(np.unique(both).size),
            "wrong": missed(out, planted), "polished_wrong": missed(polished, planted),
            "both_wrong": missed(both, planted),
        }  # fmt: skip
    except Exception as error:  # noqa: BLE001 -- a failed job is a result
        return {"problem": problem.realization, "solver": solver, "seed": seed, "setting": setting,
                "error": f"{type(error).__name__}: {error}", "trace": traceback.format_exc(limit=4)}  # fmt: skip


def _describe(problem: Any) -> dict[str, Any]:
    """The problem's bound, its planted labelling's energy, and what drawing it cost."""
    from sal.sim.potts import energy
    from sklearn.metrics import adjusted_rand_score

    import port.studies.clone_label_arms as arms

    _hold(problem.realization, problem)
    _, graph = arms._graph(problem.spatial_weight)
    bound, trws_energy, trws_seconds = arms._bound(
        problem.field, problem.spatial_weight
    )
    return {"bound": bound, "trws_energy": trws_energy, "trws_seconds": trws_seconds,
            "truth_energy": energy(graph, problem.field, problem.planted), "q": int(problem.field.shape[1]),
            "draw_seconds": problem.draw_seconds, "field_seconds": problem.field_seconds,
            "spots": problem.n_spots, "hash": problem.hash, "states": problem.states,
            "argmax_ari": adjusted_rand_score(problem.planted, problem.field.argmax(1))}  # fmt: skip


def tune(
    pool: ProcessPoolExecutor, held_out: list[Any], samplers: tuple[str, ...] = TUNED
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    """Each of `samplers`' setting: the cheapest in `GRID` within `TOLERANCE` of the best median gap on `held_out`.

    Three rounds rather than the whole grid at every start (#716):

    1. every start temperature and warm-up at the two cheaper sweep budgets,
       one start per held-out realization;
    2. the dearest budget only where the middle one still beat the cheapest by
       more than `TOLERANCE` -- the gap falls with sweeps, so where it has
       stopped falling the dearest budget is not run;
    3. the settings `harness.halve` keeps from rounds 1-2 get the remaining
       `TUNING_STARTS - 1` starts, and the choice is among those alone.

    Each round submits its longest jobs first. `default_gap` is sal's default
    setting's median over the runs it got: one start per realization unless it
    went on to round 3.
    """
    import pandas as pd

    bounds = {p.realization: _describe(p)["bound"] for p in held_out}
    rows: list[dict[str, Any]] = []

    def run(jobs: list[tuple[str, int, dict[str, float]]]) -> pd.DataFrame:
        # NB the longest first, so no worker idles behind one long job at a round's end
        jobs = sorted(jobs, key=lambda job: -job[2]["sweeps"])
        futures = [
            pool.submit(solve, p, solver, seed, setting)
            for solver, seed, setting in jobs
            for p in held_out
        ]
        rows.extend(f.result() for f in futures)
        frame = pd.DataFrame([r for r in rows if "error" not in r])
        frame["gap"] = frame.energy - frame.problem.map(bounds)
        frame["key"] = frame.setting.map(
            lambda s: (s["t_start"], s["sweeps"], s.get("warm", 0.0))
        )
        return frame

    cheap, middle, dear = sorted({g["sweeps"] for g in GRID})
    frame = run(
        [
            (solver, 0, g)
            for solver in samplers
            for g in grid(solver)
            if g["sweeps"] != dear
        ]
    )
    median = frame.groupby(["solver", "key"]).gap.median()
    falling = [(solver, 0, g) for solver in samplers for g in grid(solver) if g["sweeps"] == dear
               and median.get((solver, (g["t_start"], middle, g["warm"])), np.inf)
               < median.get((solver, (g["t_start"], cheap, g["warm"])), np.inf) - TOLERANCE]  # fmt: skip
    frame = run(falling)
    kept = {
        str(solver): harness.halve(g, TOLERANCE)
        for solver, g in frame.groupby("solver")
    }
    frame = run([(solver, seed, {"t_start": t, "sweeps": n, "warm": w})
                 for solver, keys in kept.items() for t, n, w in keys for seed in range(1, TUNING_STARTS)])  # fmt: skip
    print(f"tuning: {len(rows)} runs against the full grid's "
          f"{sum(len(grid(s)) for s in samplers) * TUNING_STARTS * len(held_out)}", flush=True)  # fmt: skip
    chosen: dict[str, dict[str, float]] = {}
    for solver, g in frame.groupby("solver"):
        full = g.groupby("key").gap.size() == TUNING_STARTS * len(held_out)
        key, best, _ = harness.cheapest(g[g.key.isin(full[full].index)], TOLERANCE)
        by = g.groupby("key").gap.median()
        t_start, sweeps, warm = key
        chosen[str(solver)] = {"t_start": float(t_start), "sweeps": int(sweeps), "warm": float(warm),
                               "gap": float(best.gap), "default_gap": float(by.get((2.0, 1000, 0.0), np.nan))}  # fmt: skip
        print(f"tuned {solver}: T0 {t_start}, {sweeps} sweeps, warm {warm}, median gap {best.gap:.2f} nats "
              f"({best.seconds:.2f} s); sal's default {chosen[str(solver)]['default_gap']:.2f}", flush=True)  # fmt: skip
    return chosen, rows


def run(
    manifest: Path,
    out_dir: Path,
    n_problems: int,
    starts: int,
    held_out: int,
    workers: int,
    settings: Path = SETTINGS,
    only: tuple[str, ...] | None = None,
    first: int = 0,
    merge: tuple[Path, ...] = (),
    states: str = "run",
) -> Path:
    """Tune on the first `held_out` realizations, then stream the next `n_problems`; returns the record it keeps current."""
    import port.studies.clone_label_arms as arms

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (
        f"{manifest.stem}_r{first}{records.SUFFIX}"
        if only or first
        else f"{manifest.stem}{records.SUFFIX}"
    )
    solvers = (
        list(only)
        if only
        else [s for s in arms._solvers() if s not in DROPPED] + list(EXTRA)
    )
    held: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    pending: dict[int, int] = {}
    done: list[int] = []
    futures: dict[Future[dict[str, Any]], int] = {}
    opened = time.perf_counter()
    with harness.pool(workers, _init) as pool:
        loaded = json.loads(settings.read_text())
        tuned = {k: v for k, v in loaded.items() if not k.startswith("_")}
        tuning_rows: list[dict[str, Any]] = []

        def drain(block: bool) -> None:
            for index, row in harness.finished(futures, block):
                rows.append(row)
                pending[index] -= 1
                if pending[index]:
                    continue
                done.append(index)
                record = {"manifest": str(manifest), "problems": held, "rows": rows, "done": done, "solvers": solvers,
                          "starts": starts, "tuned": tuned, "tuning": tuning_rows, "held_out": held_out,
                          "states": states}  # fmt: skip
                records.write(out, record)
                harness.redraw("potts-plot", out, merge)
                errors = sum("error" in r for r in rows)
                print(f"[{time.perf_counter() - opened:6.0f}s] problem {index} solved; "
                      f"{len(done)}/{n_problems} done, {errors} errors; plot redrawn", flush=True)  # fmt: skip

        # NB the next realization draws here while the pool solves the last; the held-out
        #    realizations are never built for evaluation
        for problem in problems(
            manifest, out_dir, n_problems, held_out + first, states
        ):
            held[problem.realization] = _describe(problem)
            print(f"[{time.perf_counter() - opened:6.0f}s] drew {problem.realization}: truth - bound "
                  f"{held[problem.realization]['truth_energy'] - held[problem.realization]['bound']:.2f} nats", flush=True)  # fmt: skip
            pending[problem.realization] = len(solvers) * starts
            for solver in solvers:
                setting = (
                    {
                        k: tuned[solver].get(k, 0.0)
                        for k in ("t_start", "sweeps", "warm")
                    }
                    if solver in tuned
                    else None
                )
                for seed in range(starts):
                    futures[pool.submit(solve, problem, solver, seed, setting)] = (
                        problem.realization
                    )
            drain(block=False)
        while futures:
            drain(block=True)
    return out


def retune(
    manifest: Path,
    samplers: tuple[str, ...],
    held_out: int,
    workers: int,
    root: Path,
    states: str = "run",
) -> None:
    """`tune` for `samplers` alone on `manifest`'s first `held_out` realizations, merged into `SETTINGS`."""
    unknown = set(samplers) - set(TUNED)
    if unknown:
        msg = f"not tunable: {sorted(unknown)}; tunable: {TUNED}"
        raise ValueError(msg)
    with harness.pool(workers, _init) as pool:
        chosen, _ = tune(
            pool, list(problems(manifest, root, held_out, states=states)), samplers
        )
    settings = json.loads(SETTINGS.read_text())
    settings["_provenance"] = (f"run_calibrate --potts on {manifest.name} realizations 0-{held_out - 1}, states {states}, "
                               f"{TUNING_STARTS} random starts per setting; the cheapest setting within {TOLERANCE} nats "
                               "of the best median gap to TRW-S's bound (#556, #723)")  # fmt: skip
    for solver, setting in chosen.items():
        settings[solver] = {"t_start": setting["t_start"], "sweeps": setting["sweeps"], "warm": setting["warm"],
                            "median_gap": round(setting["gap"], 3), "default_median_gap": round(setting["default_gap"], 3)}  # fmt: skip
    SETTINGS.write_text(json.dumps(settings, indent=2) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--problems", type=int, default=25)
    parser.add_argument("--starts", type=int, default=50)
    parser.add_argument("--held-out", type=int, default=3)
    parser.add_argument(
        "--settings",
        type=Path,
        default=SETTINGS,
        help=f"sampler settings, by default {SETTINGS}, which run_calibrate --potts writes",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="SOLVER",
        help="run only these solvers",
    )
    parser.add_argument(
        "--first",
        type=int,
        default=0,
        help="skip this many evaluated realizations: a window of the stream",
    )
    parser.add_argument(
        "--merge",
        nargs="+",
        type=Path,
        default=(),
        metavar="PKL",
        help="earlier streams drawn with this one",
    )
    parser.add_argument(
        "--states",
        choices=("run", "planted"),
        default="run",
        help="the Baum-Welch before the field starts from the run's states or the planted ones",
    )
    arguments = parser.parse_args(argv)
    run(arguments.manifest, arguments.out_dir, arguments.problems, arguments.starts, arguments.held_out,
        arguments.workers, arguments.settings, tuple(arguments.only) if arguments.only else None, arguments.first,
        tuple(arguments.merge), arguments.states)  # fmt: skip
