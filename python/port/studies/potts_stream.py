"""#556: Potts solvers from random labels on a stream of the run's clone-assignment problems, the plot redrawn per problem.

`run_study --potts-stream MANIFEST OUT_DIR [--problems 25] [--starts 50] [--held-out 5] [--workers 4] [--states run]`
`MANIFEST` is `sim/manifests/study15.toml`, or `study10.toml`, the studies' draws (T- #807).
`run_calibrate --potts MANIFEST OUT_DIR --samplers SAMPLER ...` tunes the
named samplers on the held-out realizations and merges them into `SETTINGS`
(`tune`, `retune`); the stream itself never tunes (#749 WP1).
`--only SOLVER ...` runs a subset, `--merge PKL ...` draws it with earlier streams.

The main process draws each realization of `MANIFEST` and runs
`run_cnaster_port --sal` on it to the RDR + BAF stage's clone assignment, at
the planted clones (`port.qa.stage.at_clone_assignment`, #735): the
field, graph and coupling are the ones the run's solver is handed there. The
Baum-Welch before it starts from the run's own initial states, or with
`--states planted` from the planted ones, a second oracle input. The pool
solves each problem while the next one is built. Until #735 the field was the
planted law's (`port.sandbox.known_field`, deleted): a different problem, so
its numbers do not compare with these.

- **Warmup.** Each worker runs every solver once on a 10 x 10 patch before
  any timed job, so no compilation lands in a timing.
- **Tuning.** `run_calibrate --potts` tunes the annealed samplers (`TUNED`)
  on the first `--held-out` realizations, which the stream never evaluates;
  the stream reads the settings it wrote (`SETTINGS`, or `--settings`). The
  schedule is `sal`'s to choose (T- #777): `sal.sample.tune.tune_schedule`
  on one problem, the disjoint union of the held-out realizations, each
  `PILOT_SEEDS` times (`union`), whose energy is the sum of theirs, so sal
  ranks across problems itself. Its candidates are `SCHEDULES` -- each of
  sal's declared shapes, the start temperature, the warm-up, the end fixed
  at `T_END` so the last sweeps are a descent -- raced (sal #1337: a quarter
  of `SWEEPS` first, the better half kept and the steps doubled), from
  common random numbers, and ranked by the energy after ICM and the merge
  from each pilot's best (`Criterion.POLISHED_GAP`), the figure's polish
  (T- #829). Every annealed
  sampler, Wolff included, spends the same `SWEEPS` of site visits: sal's
  `anneal_potts(budget=...)` charges each step what it visited (sal #1344).
- **Evaluation.** The next `--problems` realizations run every solver of
  #541's harness (`port.studies.clone_label_arms`) but bifurcation, port's
  pure-Python `alpha` and the floor-merge row, plus TRW-S's own decoded
  labelling, from `--starts` random labellings; the samplers at their tuned
  settings. Each run is polished twice: sal's ICM, then sal's merge
  (`merge_labels`). The merge counts a boundary coupling once, the energy's
  change; **a stated departure from `cnaster`**, whose `merge_assignment`
  counts half of it (`cnaster/icm.py:169`, sal #1415, T- #854). A sampler's
  polish runs inside the anneal (`Polish.ICM_MERGE`, sal #1373, #1375), each
  stage recorded; every other solver's is `sal:icm`, then the merge, from its
  output, but alpha-expansion's (`EXPANSIONS`), which is the merge alone.
- **Backends.** sal's defaults: the anneal loop, Wolff and heat-bath
  Swendsen-Wang in Rust (sal #1362, #1364, #1368). Their streams differ from
  the Python loop's, so a figure before the bump does not replay at its seeds.

Per problem it records the planted labelling's energy, TRW-S's lower bound and its spots,
per run the energy and the labels unlike the planted ones, raw and after each
polish. When a problem's runs are all in it writes `OUT_DIR/<stem>.record`
(`port.qa.records`: Parquet and JSON) and redraws `OUT_DIR/<stem>.png` (`port.studies.potts_plot`).

Timing: the graph is built once per worker per problem, outside the timed
solve; seconds are per job with `--workers` jobs sharing the host.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from collections.abc import Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from port.qa import records
from port.qa import stage as at
from port.qa import stream as harness
from port.qa.provenance import CONFIGS

DROPPED = frozenset({
    "sal:bifurcation", "port:alpha", "port:alpha-rust-merge",
    "port:alpha-rust", "port:alpha-rust-icm", "port:icm-numba", "port:icm",
    "sal:swendsen-wang-heat-bath", "sal:wolff-heat-bath",
})  # fmt: skip
"""Out of the stream: bifurcation (#541), `alpha` (Alpha-rust's pure-Python twin), the deprecated floor merge,
and, for the paper's figure (T- #660), `alpha-rust` and `alpha-rust-icm` (`alpha-rust-fuse-merge` stays),
`icm-numba` and cnaster's `icm`; and sal's deprecated `*-heat-bath` cluster moves, which run without the
Gibbs sweep the bare moves compose (sal #1317, #1323, T- #854). `clone_label_arms` still runs them, and `--only` names any. Parallel and cluster tempering
are out of port: sal moved `parallel_tempering` to its sandbox (sal #1352)."""

EXTRA = ("sal:trws",)
"""Entries beyond the harness's: TRW-S's decoded labelling."""

SAMPLERS = {
    "sal:anneal": "single-site",
    "sal:swendsen-wang": "swendsen-wang",
    "sal:wolff": "wolff",
}
"""sal's annealed chains (`anneal_potts`), each under sal's name for its method and move (T- #854). A bare
cluster move is the move and a Gibbs sweep per step (sal #1323), each cluster's label drawn from its summed
field (`Recolour.PER_MOVE`): the move alone relabels only whole same-label regions (T- #829)."""

EXPANSIONS = ("sal:alpha-expansion",)
"""The entries polished by the merge alone: a converged expansion is a fixed point of every single-site
move, so ICM after it moved 0 nats in sal's 16 of 16 runs (T- #854)."""

TUNED = tuple(SAMPLERS)
"""The entries whose schedule `sal` tunes (`tune`), and every entry that runs at a setting from `SETTINGS`."""

T_END = 0.05
"""sal's `ANNEAL_END`: cold enough that the last sweeps are a descent."""

SWEEPS = 4000
"""Every annealed sampler's budget, in sweeps of site visits: what #723 chose for three of four."""

SHAPES = ("exponential", "linear", "cosine", "inverse_linear", "power", "logarithmic")
"""sal's declared ramps (#1333). `thermodynamic` is not among them: it is placed from a measured
`sigma_E(T)`, which sal builds from no pilot of its own yet; Huang's `adaptive` schedule runs online and is
no grid candidate."""

SCHEDULES = tuple(
    {"shape": shape, "t_start": t, "warm": w}
    for shape in SHAPES
    for t in (0.25, 0.5, 1.0, 2.0, 8.0, 32.0)
    for w in (0.0, 0.1, 0.25)
)
"""`tune_schedule`'s candidates: shape x start temperature x warm-up, each ending at `T_END`. `warm` is the
fraction of the steps held at the start temperature before the ramp (`sal`'s `ScheduleParams.warm`,
#1324). sal's default, exponential 2 -> 0.05 unheld, is among them."""


def schedule(setting: dict[str, Any]) -> Any:
    """`setting`'s `sal.sample.schedule.ScheduleParams`: exponential and unheld where it names neither."""
    from sal.sample.schedule import ScheduleParams, ScheduleShape

    return ScheduleParams(ScheduleShape(setting.get("shape", "exponential")),
                          float(setting["t_start"]), T_END, warm=float(setting.get("warm", 0.0)))  # fmt: skip


PILOT_SEEDS = 2
"""Copies of each held-out realization (`harness.HELD_OUT` of them) in the tuned union, each from its own
start."""

ICM_CAP = 100_000
"""The tuning polish's sweep cap, never reached: ICM stops at its fixed point, as `Polish.ICM` does."""

SETTINGS = CONFIGS / "potts_sampler_settings.json"
"""The samplers' settings, tuned by `run_calibrate --potts` at the run's clone-assignment field
(#723) on the realizations `HELD_OUT` names, of the manifest the file's `_provenance` states; the
stream's default `--settings`."""

_GRAPHS: dict[tuple[int, float], Any] = {}


class Problem(NamedTuple):
    """One realization's clone-assignment problem, the run's (`port.qa.stage.Field`)."""

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
    """Realizations `first`, ..., `first + n - 1` of `manifest`, each drawn and run to its field only when asked for.

    A field already under `root/.stage` (`stage.field_path`), which
    `copy_state_stream` writes from its own run, is read rather than rerun
    (T- #814); one this stream runs is kept there.
    """
    import shutil

    drawn = at.members(manifest, root / ".sim", n=n, first=first)
    while True:
        opened = time.perf_counter()
        member = next(drawn, None)
        if member is None:
            return
        draw_seconds = time.perf_counter() - opened
        kept = at.field_path(root / ".stage", member, states=states)
        if kept.is_file():
            found = at.load_field(kept)
        else:
            scratch = root / f".run_r{member.realization}"
            try:
                found = at.at_clone_assignment(
                    member.sample, lambda f: f, states=states, root=scratch
                )
            finally:
                shutil.rmtree(scratch, ignore_errors=True)
            at.save_field(kept, found)
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

    build = arms.potts_graph

    def memo(spatial_weight: float) -> Any:
        key = (arms.HELD["problem"], spatial_weight)
        if key not in _GRAPHS:
            _GRAPHS[key] = build(spatial_weight)
        return _GRAPHS[key]

    arms.potts_graph = memo
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
    runnable = [s for s in arms.solver_names() if s not in retired]
    for solver in [*runnable, *EXTRA]:
        solve_labelling(
            patch,
            solver,
            0,
            {"t_start": 2.0, "sweeps": 10} if solver in TUNED else None,
        )


def _hold(index: int, problem: Any) -> None:
    import port.studies.clone_label_arms as arms

    arms.HELD["capture"], arms.HELD["problem"] = problem, index


def _sample(solver: str, field: np.ndarray, start: np.ndarray, rng: np.random.Generator,
            graph: Any, setting: dict[str, float]) -> Any:  # fmt: skip
    """An annealed chain at `setting`, then ICM and the merge to their fixed points, in one call.

    `SWEEPS` sweeps of site visits for every move set: sal's step loop
    charges each step what it visited and stops at the budget (sal #1344),
    so a Wolff run, whose cluster flips cost less than a sweep, takes the
    steps that spend it, where a pilot used to count them.
    """
    from sal.cost import Cost
    from sal.opt.budget import Budget
    from sal.sample.potts_mcmc import PottsMove
    from sal.sample.potts_mcmc.chains import anneal_potts
    from sal.sample.schedule import Polish
    from sal.search.ground_state import Problem as SalProblem

    sweeps = int(setting["sweeps"])
    problem = SalProblem(graph, field, field.shape[1])
    budget = Budget(Cost.SITE_VISITS, sweeps * problem.visits_per_sweep)
    return anneal_potts(graph, field, schedule(setting).build(sweeps), rng, move=PottsMove(SAMPLERS[solver]),
                        start=start, budget=budget, polish=Polish.ICM_MERGE)  # fmt: skip


STAGES = ("init", "polish", "merge")
"""`Polish.ICM_MERGE`'s stages of an annealed run: the anneal's best, after ICM, after the merge."""


def merged(graph: Any, field: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """`labels` after sal's merge (`merge_labels`) to its fixed point: each boundary coupling once, where `cnaster`'s counts half (T- #854)."""
    from sal.search.icm import merge_labels

    return np.asarray(merge_labels(graph, field, labels).labelling, dtype=np.int64)


def solve_labelling(
    problem: Any, solver: str, seed: int, setting: dict[str, float] | None = None
) -> dict[str, Any]:
    """One run from random labels, then ICM and the merge; a failure is a row."""
    from sal.sim.potts import energy

    import port.studies.clone_label_arms as arms
    from port.qa.stage import missed

    try:
        _hold(problem.realization, problem)
        field, beta = problem.field, problem.spatial_weight
        _, graph = arms.potts_graph(beta)
        start = np.random.default_rng([seed, 7]).integers(
            0, field.shape[1], field.shape[0]
        )
        rng = np.random.default_rng([seed, 492])
        opened = time.perf_counter()
        if setting is not None:
            run = _sample(solver, field, start, rng, graph, setting)
            if tuple(stage.name for stage in run.stages) != STAGES:
                msg = f"anneal_potts stages {[stage.name for stage in run.stages]}, expected {STAGES}"
                raise RuntimeError(msg)
            out, polished, both = (
                np.asarray(stage.best, dtype=np.int64) for stage in run.stages
            )
            polish_seconds, merge_seconds = run.stages[1].seconds, run.stages[2].seconds
            seconds = time.perf_counter() - opened - polish_seconds - merge_seconds
        else:
            if solver == "sal:trws":
                from sal.search.trws import trws

                out = np.asarray(trws(graph, field).labelling, dtype=np.int64)
            else:
                out = arms.solve_from(solver, field, start, rng, beta)
            seconds = time.perf_counter() - opened
            opened = time.perf_counter()
            polished = (
                out
                if solver in EXPANSIONS
                else arms.solve_from(
                    "sal:icm", field, out, np.random.default_rng(0), beta
                )
            )
            polish_seconds = time.perf_counter() - opened
            opened = time.perf_counter()
            both = merged(graph, field, polished)
            merge_seconds = time.perf_counter() - opened
        merges = int(np.unique(polished).size - np.unique(both).size)
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
    _, graph = arms.potts_graph(problem.spatial_weight)
    bound, trws_energy, trws_seconds = arms._bound(
        problem.field, problem.spatial_weight
    )
    return {"bound": bound, "trws_energy": trws_energy, "trws_seconds": trws_seconds,
            "truth_energy": energy(graph, problem.field, problem.planted), "q": int(problem.field.shape[1]),
            "draw_seconds": problem.draw_seconds, "field_seconds": problem.field_seconds,
            "spots": problem.n_spots, "hash": problem.hash, "states": problem.states,
            "argmax_ari": adjusted_rand_score(problem.planted, problem.field.argmax(1))}  # fmt: skip


def union(held_out: list[Any], copies: int) -> tuple[Any, np.ndarray]:
    """The held-out problems, each `copies` times, as one Potts problem: block-diagonal couplings and stacked
    fields, padded with `-inf` to the most clones. Its energy is the sum of the problems'."""
    from sal.sim.graph import PottsGraph

    from port.patch.icm.alpha_expansion import potts_graph_from
    from port.patch.icm.interface import CsrGraph

    q = max(int(p.field.shape[1]) for p in held_out)
    edges: list[tuple[int, int]] = []
    coupling: list[float] = []
    fields, offset = [], 0
    for p in [p for p in held_out for _ in range(copies)]:
        graph = potts_graph_from(
            CsrGraph(p.indptr, p.indices, p.weights), p.spatial_weight
        )
        edges += [(i + offset, j + offset) for i, j in graph.edges]
        coupling += list(graph.coupling)
        fields.append(
            np.pad(
                p.field, ((0, 0), (0, q - p.field.shape[1])), constant_values=-np.inf
            )
        )
        offset += graph.n_nodes
    return PottsGraph(offset, tuple(edges), tuple(coupling)), np.vstack(fields)


def pilots(
    held_out: list[Any], solver: str, rng: np.random.Generator
) -> dict[str, Any]:
    """`sal`'s `tune_schedule` for `solver` on the held-out `union`: raced, common random numbers, ranked by
    the energy after ICM and the merge, as `_sample` polishes. Returns the chosen setting and every
    candidate's last pilot."""
    from sal.cost import Cost
    from sal.opt.budget import Budget
    from sal.sample.potts_mcmc import PottsMove, Recolour
    from sal.sample.tune import Criterion, tune_schedule
    from sal.search.icm import iterated_conditional_modes

    graph, field = union(held_out, PILOT_SEEDS)
    grid = tuple(schedule(c) for c in SCHEDULES)

    def polish(labels: np.ndarray) -> np.ndarray:
        # NB `Polish.ICM_MERGE`'s pair: ICM in index order to its fixed point, then the merge. A callable
        #    until sal #1398 lets `tune_schedule` take the member itself.
        settled = iterated_conditional_modes(graph, field, np.random.default_rng(0), start=labels,
                                             max_iterations=ICM_CAP)  # fmt: skip
        if not settled.termination.converged:
            msg = f"ICM did not reach a fixed point in {ICM_CAP} sweeps"
            raise RuntimeError(msg)
        return merged(graph, field, np.asarray(settled.labelling, dtype=np.int64))

    opened = time.perf_counter()
    tuned = tune_schedule(graph, field, move=PottsMove(SAMPLERS[solver]), recolour=Recolour.PER_MOVE,
                          budget=Budget(Cost.SITE_VISITS, len(grid) * SWEEPS * graph.n_nodes),
                          criterion=Criterion.POLISHED_GAP, rng=rng, grid=grid, racing=True, common=True,
                          polish=polish)  # fmt: skip
    index = grid.index(tuned.params)
    return {"solver": solver, **SCHEDULES[index], "sweeps": SWEEPS, "seconds": time.perf_counter() - opened,
            "rounds": [list(r) for r in tuned.rounds],
            "candidates": [{**c, "lowest": run.lowest_energy, "polished": run.polished_energy, "spent": run.spent}
                           for c, run in zip(SCHEDULES, tuned.candidates, strict=True)]}  # fmt: skip


def tune(
    pool: ProcessPoolExecutor, held_out: list[Any], samplers: tuple[str, ...] = TUNED
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Each of `samplers`' schedule, `sal`'s choice on the held-out union (`pilots`); one job per sampler."""
    futures = [pool.submit(pilots, held_out, solver, np.random.default_rng([777, k]))
               for k, solver in enumerate(samplers)]  # fmt: skip
    rows = [f.result() for f in futures]
    chosen = {
        row["solver"]: {k: row[k] for k in ("shape", "t_start", "warm", "sweeps")}
        for row in rows
    }
    for row in rows:
        print(f"tuned {row['solver']}: {row['shape']} T0 {row['t_start']}, warm {row['warm']}, {SWEEPS} sweeps, "
              f"rounds {row['rounds']}, {row['seconds']:.0f} s", flush=True)  # fmt: skip
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
        else [s for s in arms.solver_names() if s not in DROPPED] + list(EXTRA)
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
                        k: tuned[solver][k]
                        for k in ("shape", "t_start", "sweeps", "warm")
                        if k in tuned[solver]
                    }
                    if solver in tuned
                    else None
                )
                for seed in range(starts):
                    futures[
                        pool.submit(solve_labelling, problem, solver, seed, setting)
                    ] = problem.realization
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
    provenance = (f"run_calibrate --potts on {manifest.name} realizations 0-{held_out - 1}, states {states}: "
                  f"sal's tune_schedule on their union, each {PILOT_SEEDS} times, {len(SCHEDULES)} schedules "
                  f"(shape x t_start x warm, t_end {T_END}) at {SWEEPS} sweeps, raced, common random numbers, "
                  f"ranked by the energy after ICM and the merge (#556, #723, T- #777, T- #829)")  # fmt: skip
    harness.merge_settings(SETTINGS, provenance, chosen)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--problems", type=int, default=25)
    parser.add_argument("--starts", type=int, default=50)
    parser.add_argument("--held-out", type=int, default=harness.HELD_OUT)
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
