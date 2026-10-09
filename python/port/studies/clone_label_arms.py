"""#541's arms: every clone-label start, every Potts solver from it, and the joint arms.

Each job builds its problems from the capture and a labelling
(`port.sandbox.clone_starts.problem.build`) and returns rows; jobs run in
forked workers, which inherit the capture.

- `starts`: each start (stochastic ones at seeds 0-9), its problem -- copy
  states by `kmeans++x5+em`, profiles, field -- and TRW-S's bound on that
  field; then every solver from the start (realizations 0-2), 3 seeds for a
  stochastic solver, each scored raw and after `--sal`'s floor merge.
- `alternating`: the HMRF loop as `--sal` runs it: solve, floor, refit the
  states from the last (warm), rebuild the field; 4 rounds.
- `joint-anneal`: heat-bath sweeps at a falling temperature with the states
  refitted between, then a zero-temperature solve.
- `joint-sample`: heat-bath draws at T = 1 with the states refitted between,
  the draw of highest data log-likelihood kept, then solved.
- `beta`: the leading starts solved at the run's coupling x 0.1, 1 and 10.

Field-reading starts read `grid2`'s field: the profiles `grid2`'s labels give.
"""

from __future__ import annotations

import pickle
import time
import traceback
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from port.qa.statistics import measured

SWEEPS = 1000
"""#492's budget: 1,000 sweeps of site visits for each `sal` method."""

PORT_ROWS = (
    "icm",
    "icm-numba",
    "alpha",
    "alpha-rust",
    "alpha-rust-icm",
    "alpha-rust-merge",
    "alpha-rust-fuse-merge",
)
STARTLESS = (
    "field_argmax",
    "max-product",
    "bifurcation",
)
"""`sal` methods that take no start: each labelling is built from the field alone. The tempering arms left
sal's `METHODS` for its sandbox (sal #1352)."""
STOCHASTIC_SOLVERS = (
    "icm-random",
    "anneal",
    "swendsen-wang",
    "wolff",
)
REALIZATIONS = 10
SOLVED = 3
"""Realizations of a stochastic start every solver runs from."""
ALTERNATING = ("port:alpha-rust-fuse-merge", "sal:alpha-expansion", "port:icm")
JOINT_STARTS = ("grid2", "normal-first", "mean field", "posterior draw")
BETA_STARTS = ("grid2", "normal-first", "mean field", "posterior draw")
TEMPERATURES = (8.0, 4.0, 2.0, 1.0, 0.5)

HELD: dict[str, Any] = {}


class Job(NamedTuple):
    arm: str
    start: str
    seed: int
    solver: str = ""
    factor: float = 1.0


def solver_names() -> list[str]:
    from sal.search.ground_state import METHODS

    return [f"sal:{m}" for m in METHODS] + [f"port:{r}" for r in PORT_ROWS]


def potts_graph(spatial_weight: float) -> Any:
    from port.patch.icm.alpha_expansion import potts_graph_from
    from port.patch.icm.interface import CsrGraph

    capture = HELD["capture"]
    csr = CsrGraph(capture.indptr, capture.indices, capture.weights)
    return csr, potts_graph_from(csr, spatial_weight)


def solve_from(
    solver: str,
    field: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    spatial_weight: float,
) -> np.ndarray:
    """`solver` from `labels` on `field` at coupling `beta`, floorless."""
    csr, graph = potts_graph(spatial_weight)
    labels = np.asarray(labels, dtype=np.int64).copy()
    kind, name = solver.split(":", 1)

    if kind == "port":
        from port.extensions.label_solver import sweep_for
        from port.patch.icm.interface import icm_sweep
        from port.sandbox.extensions.label_solvers import SWEEPS as SET_ASIDE

        sweep = (
            icm_sweep
            if name == "icm"
            else SET_ASIDE[name]
            if name in SET_ASIDE
            else sweep_for(name)  # type: ignore[arg-type]
        )
        np.random.seed(int(rng.integers(2**31)))  # noqa: NPY002 -- cnaster's ICM reads it
        sweep(field, csr, labels, spatial_weight, min_clone_spots=0)
        return labels

    from sal.cost import Cost
    from sal.opt.budget import Budget
    from sal.search.ground_state import METHODS, Problem

    problem = Problem(graph, field, field.shape[1])
    budget = Budget(Cost.SITE_VISITS, SWEEPS * problem.visits_per_sweep)
    # NB `field_argmax`, `max-product` and `bifurcation` take no start by
    #    design: each runs from its own, and its rows say so (`startless`).
    start = None if name in STARTLESS else labels
    return np.asarray(
        METHODS[name](problem, budget, rng, start=start).labelling, dtype=np.int64
    )


def _floored(field: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """`--sal`'s floor merge: clones under the floor emptied, smallest first."""
    from port.patch.icm.floor import floor_clones

    labels = np.asarray(labels, dtype=np.int64).copy()
    floor_clones(field, labels, HELD["capture"].floor)
    return labels


def _labelling_scores(
    labels: np.ndarray, field: np.ndarray, beta: float
) -> dict[str, Any]:
    from sal.sim.potts import energy
    from sklearn.metrics import adjusted_rand_score

    _, graph = potts_graph(beta)
    labels = np.asarray(labels, dtype=np.int64)
    sizes = np.bincount(labels, minlength=field.shape[1])
    used = sizes[sizes > 0]
    return {
        "ari": float(adjusted_rand_score(HELD["capture"].planted, labels)),
        "clones": int(used.size),
        "smallest": int(used.min()),
        "energy": float(energy(graph, field, labels)),
        "loglik": float(field[np.arange(labels.size), labels].sum()),
    }


def _bound(field: np.ndarray, beta: float) -> tuple[float, float, float]:
    from sal.search.trws import trws

    _, graph = potts_graph(beta)
    with measured() as cost:
        found = trws(graph, field)
    return float(found.bound), float(found.energy), cost.wall_s


def _start(job: Job) -> tuple[np.ndarray, float]:
    from port.sandbox.clone_starts.starts import STARTS

    with measured() as cost:
        labels = STARTS[job.start].run(
            HELD["capture"],
            np.random.default_rng([job.seed, 541]),
            field=HELD["grid2"].field,
        )
    return np.asarray(labels, dtype=np.int64), cost.wall_s


def _build(labels: np.ndarray, seed: int, states: Any = None) -> Any:
    from port.sandbox.clone_starts.problem import build

    return build(
        HELD["capture"],
        labels,
        np.random.default_rng([seed, 540]),
        states=states,
        seconds=60.0 if states is None else 30.0,
    )


def _starts_arm(job: Job) -> list[dict[str, Any]]:
    labels, start_seconds = _start(job)
    problem = _build(labels, job.seed)
    beta = HELD["capture"].spatial_weight
    bound, trws_energy, trws_seconds = _bound(problem.field, beta)
    base = {**job._asdict(), "bound": bound, "trws_energy": trws_energy,
            "trws_seconds": trws_seconds, "start_seconds": start_seconds,
            "build_seconds": problem.seconds, "states_by": problem.states_by}  # fmt: skip
    rows = [
        {
            **base,
            "solver": "start",
            **_labelling_scores(problem.labels, problem.field, beta),
        }
    ]

    if job.seed >= SOLVED:
        return rows

    for solver in solver_names():
        seeds = range(3) if solver.split(":")[1] in STOCHASTIC_SOLVERS else range(1)
        for seed in seeds:
            with measured() as cost:
                try:
                    solved = solve_from(solver, problem.field, problem.labels,
                                    np.random.default_rng([seed, 492]), beta)  # fmt: skip
                except Exception as error:  # noqa: BLE001 -- a refusal is a result
                    rows.append({**base, "solver": solver, "solver_seed": seed,
                                 "error": f"{type(error).__name__}: {error}"})  # fmt: skip
                    continue
            seconds = cost.wall_s
            floored = _floored(problem.field, solved)
            rows.append({
                **base, "solver": solver, "solver_seed": seed, "solve_seconds": seconds,
                **_labelling_scores(solved, problem.field, beta),
                **{f"floored_{k}": v for k, v in _labelling_scores(floored, problem.field, beta).items()},
            })  # fmt: skip
    return rows


def _alternating_arm(job: Job) -> list[dict[str, Any]]:
    labels, start_seconds = _start(job)
    beta = HELD["capture"].spatial_weight
    problem = _build(labels, job.seed)
    elapsed = start_seconds + problem.seconds
    rows = [{**job._asdict(), "round": 0, "seconds": elapsed,
             **_labelling_scores(problem.labels, problem.field, beta)}]  # fmt: skip
    for round_ in range(1, 5):
        with measured() as cost:
            solved = solve_from(job.solver, problem.field, problem.labels,
                            np.random.default_rng([round_, 492]), beta)  # fmt: skip
            floored = _floored(problem.field, solved)
            problem = _build(
                floored, job.seed, states=(problem.log_mu, problem.p_binom)
            )
        elapsed += cost.wall_s
        rows.append({**job._asdict(), "round": round_, "seconds": elapsed,
                     **_labelling_scores(problem.labels, problem.field, beta)})  # fmt: skip
    return rows


def _heat_bath(
    field: np.ndarray, labels: np.ndarray, temperature: float, seed: int
) -> np.ndarray:
    from sal.sample.potts_mcmc import anneal_potts
    from sal.sample.schedule import ConstantTempSchedule

    _, graph = potts_graph(HELD["capture"].spatial_weight)
    run = anneal_potts(graph, field, ConstantTempSchedule(temperature, 20),
                       np.random.default_rng([seed, 7]), start=labels)  # fmt: skip
    return np.asarray(run.best, dtype=np.int64)


def _joint_arm(job: Job) -> list[dict[str, Any]]:
    labels, start_seconds = _start(job)
    beta = HELD["capture"].spatial_weight
    problem = _build(labels, job.seed)
    elapsed = start_seconds + problem.seconds
    rows = [{**job._asdict(), "step": "start", "seconds": elapsed,
             **_labelling_scores(problem.labels, problem.field, beta)}]  # fmt: skip
    best = problem
    temperatures = TEMPERATURES if job.arm == "joint-anneal" else (1.0,) * 5

    for step, temperature in enumerate(temperatures):
        with measured() as cost:
            drawn = _floored(
                problem.field,
                _heat_bath(problem.field, problem.labels, temperature, step),
            )
            problem = _build(drawn, job.seed, states=(problem.log_mu, problem.p_binom))
        elapsed += cost.wall_s
        scored = _labelling_scores(problem.labels, problem.field, beta)
        if (
            job.arm == "joint-sample"
            and scored["loglik"]
            > _labelling_scores(best.labels, best.field, beta)["loglik"]
        ):
            best = problem
        rows.append(
            {
                **job._asdict(),
                "step": f"T={temperature:g}",
                "seconds": elapsed,
                **scored,
            }
        )

    final = best if job.arm == "joint-sample" else problem
    with measured() as cost:
        solved = _floored(final.field, solve_from("port:alpha-rust-fuse-merge", final.field,
                                              final.labels, np.random.default_rng(0), beta))  # fmt: skip
        problem = _build(solved, job.seed, states=(final.log_mu, final.p_binom))
    elapsed += cost.wall_s
    rows.append({**job._asdict(), "step": "solved", "seconds": elapsed,
                 **_labelling_scores(problem.labels, problem.field, beta)})  # fmt: skip
    return rows


def _beta_arm(job: Job) -> list[dict[str, Any]]:
    labels, _ = _start(job)
    problem = _build(labels, job.seed)
    beta = HELD["capture"].spatial_weight * job.factor
    rows = []
    for solver in ("port:alpha-rust-fuse-merge", "sal:alpha-expansion"):
        solved = _floored(problem.field, solve_from(solver, problem.field, problem.labels,
                                                np.random.default_rng(0), beta))  # fmt: skip
        rows.append(
            {
                **job._asdict(),
                "solver": solver,
                **_labelling_scores(solved, problem.field, beta),
            }
        )
    return rows


ARMS = {
    "starts": _starts_arm,
    "alternating": _alternating_arm,
    "joint-anneal": _joint_arm,
    "joint-sample": _joint_arm,
    "beta": _beta_arm,
}


def _jobs(arms: list[str]) -> list[Job]:
    from port.sandbox.clone_starts.starts import STARTS

    jobs: list[Job] = []
    for arm in arms:
        if arm == "starts":
            for name, start in STARTS.items():
                for seed in range(REALIZATIONS if start.stochastic else 1):
                    jobs.append(Job(arm, name, seed))
        elif arm == "alternating":
            jobs += [Job(arm, n, 0, s) for n in STARTS for s in ALTERNATING]
        elif arm in ("joint-anneal", "joint-sample"):
            jobs += [Job(arm, n, 0) for n in JOINT_STARTS]
        elif arm == "beta":
            jobs += [
                Job(arm, n, 0, factor=f) for n in BETA_STARTS for f in (0.1, 1.0, 10.0)
            ]
    # NB the longest first, so the pool's tail is short jobs.
    order = {
        "starts": 0,
        "alternating": 1,
        "joint-anneal": 2,
        "joint-sample": 2,
        "beta": 3,
    }
    return sorted(jobs, key=lambda j: (order[j.arm], j.seed >= SOLVED))


def _run(job: Job) -> list[dict[str, Any]]:
    try:
        return ARMS[job.arm](job)
    except Exception as error:  # noqa: BLE001 -- a failed job is a result
        return [{**job._asdict(), "error": f"{type(error).__name__}: {error}",
                 "trace": traceback.format_exc(limit=4)}]  # fmt: skip


def _hold(capture_path: Path) -> None:
    from port.sandbox.clone_starts.problem import build, load
    from port.sandbox.clone_starts.starts import STARTS

    capture = load(capture_path)
    HELD["capture"] = capture
    grid2 = STARTS["grid2"].run(capture, np.random.default_rng(0))
    HELD["grid2"] = build(capture, grid2, np.random.default_rng([0, 540]))
    HELD["oracle"] = build(capture, capture.planted, np.random.default_rng([0, 540]))


def run_arms(
    capture_path: Path,
    out: Path,
    *,
    arms: list[str] | None = None,
    workers: int = 4,
    retry: Path | None = None,
) -> None:
    """Every job of `arms` (all by default) in `workers` spawned processes; rows pickled to `out` as they arrive.

    Spawned, each loading the capture once (`_hold`), not forked: a child
    forked after the parent has built a problem inherits compiled thread
    pools that do not survive the fork, and died on its first job.
    """
    from concurrent.futures import as_completed

    from port.studies.stream import pool as harness_pool

    _hold(capture_path)
    beta = HELD["capture"].spatial_weight
    oracle = HELD["oracle"]
    jobs = _jobs(arms or list(ARMS))
    rows: list[dict[str, Any]] = []
    if retry is not None:
        # NB every job with an errored row runs again; its other rows go.
        with retry.open("rb") as fh:
            earlier = pickle.load(fh)["rows"]
        fields = Job._fields
        failed = {tuple(r[f] for f in fields) for r in earlier if r.get("error")}
        rows = [r for r in earlier if tuple(r[f] for f in fields) not in failed]
        jobs = [j for j in jobs if tuple(j) in failed]
    print(f"{len(jobs)} jobs", flush=True)
    opened = time.perf_counter()
    with harness_pool(workers, _hold, capture_path) as pool:
        futures = {pool.submit(_run, job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            try:
                rows += future.result()
            except Exception as error:  # noqa: BLE001 -- a lost job is a result
                rows.append(
                    {**job._asdict(), "error": f"{type(error).__name__}: {error}"}
                )
            print(f"{done}/{len(jobs)} {job.arm} {job.start} {job.seed} {job.solver} "
                  f"{time.perf_counter() - opened:.0f} s", flush=True)  # fmt: skip
            with out.open("wb") as fh:
                pickle.dump({"rows": rows, "oracle": _labelling_scores(oracle.labels, oracle.field, beta),
                             "grid2_seconds": HELD["grid2"].seconds}, fh)  # fmt: skip
