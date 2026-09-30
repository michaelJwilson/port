"""#556: Potts solvers from random labels on a stream of known-law problems, the plot redrawn per problem.

`python -m tests.studies.potts_stream MANIFEST OUT_DIR [--problems 25] [--starts 50] [--held-out 3] [--workers 4]`

The main process draws each realization of `MANIFEST` in memory and builds its
field at the planted copy states and profiles (`port.sandbox.known_field`);
the pool solves it while the next one draws.

- **Warmup.** Each worker runs every solver once on a 10 x 10 patch before
  any timed job, so no compilation lands in a timing.
- **Tuning.** The first `--held-out` realizations tune the samplers and are
  not evaluated; with `--settings`, the samplers take that file's settings
  (`SETTINGS`, tuned once on `dev_tree_1s_hard`) and nothing is tuned, the
  held-out realizations still skipped. Each sampler runs a grid of annealing schedules (`GRID`:
  the start temperature, the end fixed at `T_END` so the last sweeps are a
  descent, and the sweep budget) from `TUNING_STARTS` random labellings each.
  It keeps the cheapest setting whose median gap to TRW-S's bound is within
  `TOLERANCE` nats of the best setting's.
- **Evaluation.** The next `--problems` realizations run every solver of
  #541's harness (`tests.studies.clone_label_arms`) but bifurcation, port's
  pure-Python `alpha` and the floor-merge row, plus TRW-S's own decoded
  labelling, from `--starts` random labellings; the samplers at their tuned
  settings. Each run is polished twice: sal's ICM, then the color merge
  (`known_field.color_merge`, cnaster's `merge_assignment` rule).

Per problem it records the planted labelling's energy and TRW-S's lower bound,
per run the energy and the labels unlike the planted ones, raw and after each
polish. When a problem's runs are all in it pickles `OUT_DIR/<stem>.pkl` and
redraws `OUT_DIR/<stem>.png` (`tests.studies.potts_plot`).

Timing: the graph is built once per worker per problem, outside the timed
solve; seconds are per job with `--workers` jobs sharing the host.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import pickle
import subprocess
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np

DROPPED = frozenset({"sal:bifurcation", "port:alpha", "port:alpha-rust-merge"})
"""Out of the stream: bifurcation (#541), `alpha` (Alpha-rust's pure-Python twin), and the deprecated floor merge."""

EXTRA = ("sal:trws",)
"""Entries beyond the harness's: TRW-S's decoded labelling."""

SAMPLERS = {
    "sal:anneal": "single-site",
    "sal:swendsen-wang": "swendsen-wang",
    "sal:wolff": "wolff",
    "sal:tempering": None,
}
"""The tuned entries, by `sal`'s move set; tempering is a ladder of single-site replicas."""

T_END = 0.05
"""sal's `ANNEAL_END`: cold enough that the last sweeps are a descent."""

GRID = tuple(
    {"t_start": t, "sweeps": s}
    for t, s in itertools.product((0.25, 0.5, 1.0, 2.0, 8.0, 32.0), (250, 1000, 4000))
)
"""Start temperature x sweep budget. sal's default is T = 2, 1,000 sweeps; the field's margins run to 18 nats."""

TUNING_STARTS = 5
TOLERANCE = 0.1
"""Nats: a setting within this of the best median gap is as good, and the cheapest of those is kept."""

REPLICAS = 6
"""sal's `N_REPLICAS`: tempering's ladder, geometric from the start temperature to `T_END`."""

SETTINGS = Path(__file__).with_name("potts_sampler_settings.json")
"""The samplers' settings tuned once on `dev_tree_1s_hard`'s first 3 realizations, reused by `--settings`."""

_GRAPHS: dict[tuple[int, float], Any] = {}


def _init() -> None:
    """Build each problem's graph once per worker, then warm every solver up on a small patch."""
    import tests.studies.clone_label_arms as arms

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

    from port.sandbox.known_field import hex_graph

    import tests.studies.clone_label_arms as arms

    rows, cols = np.divmod(np.arange(100), 10)
    points = np.column_stack([cols + 0.5 * (rows % 2), rows * np.sqrt(3) / 2])
    indptr, indices, weights = hex_graph(points)
    field = np.random.default_rng(0).normal(size=(100, 3))
    patch = SimpleNamespace(realization=-1, field=field, planted=field.argmax(1), indptr=indptr, indices=indices,
                            weights=weights, spatial_weight=1.0, n_spots=100)  # fmt: skip
    for solver in [s for s in arms._solvers() if s not in DROPPED] + list(EXTRA):
        solve(
            patch,
            solver,
            0,
            {"t_start": 2.0, "sweeps": 10} if solver in SAMPLERS else None,
        )


def _hold(index: int, problem: Any) -> None:
    import tests.studies.clone_label_arms as arms

    arms._HELD["capture"], arms._HELD["problem"] = problem, index


def _sample(solver: str, field: np.ndarray, start: np.ndarray, rng: np.random.Generator,
            graph: Any, setting: dict[str, float]) -> np.ndarray:  # fmt: skip
    """A sampler at `setting`: sal's annealed run or tempering ladder, its schedule and budget replaced."""
    from sal.cost import Cost
    from sal.opt.budget import Budget
    from sal.sample.potts_mcmc.chains import parallel_tempering
    from sal.sample.potts_mcmc.moves import PottsMove
    from sal.sample.schedule import ScheduleParams, ScheduleShape
    from sal.search.ground_state import Problem, run_annealed

    sweeps = int(setting["sweeps"])
    if SAMPLERS[solver] is None:
        ladder = tuple(
            float(t) for t in np.geomspace(setting["t_start"], T_END, REPLICAS)
        )
        run = parallel_tempering(graph, field, ladder, rng, max(1, sweeps // REPLICAS))
        return np.asarray(run.best, dtype=np.int64)
    problem = Problem(graph, field, field.shape[1])
    budget = Budget(Cost.SITE_VISITS, sweeps * problem.visits_per_sweep)
    schedule = ScheduleParams(
        ScheduleShape.EXPONENTIAL, float(setting["t_start"]), T_END
    )
    labelling = run_annealed(
        problem,
        budget,
        rng,
        PottsMove(str(SAMPLERS[solver])),
        schedule=schedule,
        start=start,
    ).labelling
    return np.asarray(labelling, dtype=np.int64)


def solve(
    problem: Any, solver: str, seed: int, setting: dict[str, float] | None = None
) -> dict[str, Any]:
    """One run from random labels, then its two polishes; a failure is a row."""
    from port.sandbox.known_field import color_merge, missed
    from sal.sim.potts import energy

    import tests.studies.clone_label_arms as arms

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
        graph_csr = (problem.indptr, problem.indices, problem.weights)
        opened = time.perf_counter()
        both, merges = color_merge(field, polished, *graph_csr, beta)
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

    import tests.studies.clone_label_arms as arms

    _hold(problem.realization, problem)
    _, graph = arms._graph(problem.spatial_weight)
    bound, trws_energy, trws_seconds = arms._bound(
        problem.field, problem.spatial_weight
    )
    return {"bound": bound, "trws_energy": trws_energy, "trws_seconds": trws_seconds,
            "truth_energy": energy(graph, problem.field, problem.planted), "q": int(problem.field.shape[1]),
            "draw_seconds": problem.draw_seconds, "field_seconds": problem.field_seconds,
            "argmax_ari": adjusted_rand_score(problem.planted, problem.field.argmax(1))}  # fmt: skip


def tune(
    pool: ProcessPoolExecutor, held_out: list[Any]
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    """Each sampler's setting: the cheapest in `GRID` within `TOLERANCE` of the best median gap on `held_out`."""
    import pandas as pd

    bounds = {p.realization: _describe(p)["bound"] for p in held_out}
    futures = [pool.submit(solve, p, solver, seed, setting)
               for p in held_out for solver in SAMPLERS for setting in GRID for seed in range(TUNING_STARTS)]  # fmt: skip
    rows = [f.result() for f in futures]
    frame = pd.DataFrame([r for r in rows if "error" not in r])
    frame["gap"] = frame.energy - frame.problem.map(bounds)
    frame["key"] = frame.setting.map(lambda s: (s["t_start"], s["sweeps"]))
    chosen: dict[str, dict[str, float]] = {}
    for solver, g in frame.groupby("solver"):
        by = g.groupby("key").agg(gap=("gap", "median"), seconds=("seconds", "median"))
        good = by[by.gap <= by.gap.min() + TOLERANCE].sort_values("seconds")
        t_start, sweeps = good.index[0]
        chosen[str(solver)] = {"t_start": float(t_start), "sweeps": int(sweeps),
                               "gap": float(good.gap.iloc[0]), "default_gap": float(by.gap.get((2.0, 1000), np.nan))}  # fmt: skip
        print(f"tuned {solver}: T0 {t_start}, {sweeps} sweeps, median gap {good.gap.iloc[0]:.2f} nats "
              f"({good.seconds.iloc[0]:.2f} s); sal's default {chosen[str(solver)]['default_gap']:.2f}", flush=True)  # fmt: skip
    return chosen, rows


def run(
    manifest: Path,
    out_dir: Path,
    n_problems: int,
    starts: int,
    held_out: int,
    workers: int,
    settings: Path | None = None,
) -> Path:
    """Tune on the first `held_out` realizations, then stream the next `n_problems`; returns the pickle it keeps current."""
    from port.sandbox.known_field import problems

    import tests.studies.clone_label_arms as arms

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{manifest.stem}.pkl"
    solvers = [s for s in arms._solvers() if s not in DROPPED] + list(EXTRA)
    held: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    pending: dict[int, int] = {}
    done: list[int] = []
    futures: dict[Future[dict[str, Any]], int] = {}
    opened = time.perf_counter()
    stream = problems(
        manifest, held_out + n_problems, realizations=held_out + n_problems
    )
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=context, initializer=_init) as pool:
        skipped = list(itertools.islice(stream, held_out))
        if settings is None:
            tuned, tuning_rows = tune(pool, skipped)
        else:
            loaded = json.loads(settings.read_text())
            tuned = {k: v for k, v in loaded.items() if not k.startswith("_")}
            tuning_rows = []

        def drain(block: bool) -> None:
            finished, _ = wait(
                futures, timeout=None if block else 0, return_when=FIRST_COMPLETED
            )
            for future in finished:
                index = futures.pop(future)
                rows.append(future.result())
                pending[index] -= 1
                if pending[index]:
                    continue
                done.append(index)
                record = {"manifest": str(manifest), "problems": held, "rows": rows, "done": done, "solvers": solvers,
                          "starts": starts, "tuned": tuned, "tuning": tuning_rows, "held_out": held_out}  # fmt: skip
                out.write_bytes(pickle.dumps(record))
                subprocess.run(
                    [sys.executable, "-m", "tests.studies.potts_plot", str(out)],
                    check=False,
                )
                errors = sum("error" in r for r in rows)
                print(f"[{time.perf_counter() - opened:6.0f}s] problem {index} solved; "
                      f"{len(done)}/{n_problems} done, {errors} errors; plot redrawn", flush=True)  # fmt: skip

        # NB the next realization draws here while the pool solves the last
        for problem in stream:
            held[problem.realization] = _describe(problem)
            print(f"[{time.perf_counter() - opened:6.0f}s] drew {problem.realization}: truth - bound "
                  f"{held[problem.realization]['truth_energy'] - held[problem.realization]['bound']:.2f} nats", flush=True)  # fmt: skip
            pending[problem.realization] = len(solvers) * starts
            for solver in solvers:
                setting = (
                    {k: tuned[solver][k] for k in ("t_start", "sweeps")}
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
        default=None,
        help=f"sampler settings to reuse, e.g. {SETTINGS}",
    )
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args(argv)
    run(arguments.manifest, arguments.out_dir, arguments.problems, arguments.starts, arguments.held_out,
        arguments.workers, arguments.settings)  # fmt: skip


if __name__ == "__main__":
    main()
