"""#556: Potts solvers from random labels on a stream of known-law problems, the plot redrawn per problem.

`python -m tests.studies.potts_stream MANIFEST OUT_DIR [--problems 5] [--starts 25] [--workers 4]`

The main process draws each realization of `MANIFEST` in memory and builds its
field at the planted copy states and profiles (`port.sandbox.known_field`);
the pool solves it while the next one draws. Each problem runs every solver
of #541's harness (`tests.studies.clone_label_arms`) but bifurcation, port's
pure-Python `alpha` and the floor-merge row, plus TRW-S's own decoded
labelling, from `--starts` random labellings. Each run is polished twice:
sal's ICM, then the color merge (`known_field.color_merge`, cnaster's
`merge_assignment` rule). Per problem it records the planted labelling's
energy and TRW-S's lower bound, and when a problem's runs are all in,
pickles `OUT_DIR/<stem>.pkl` and redraws `OUT_DIR/<stem>.png` and
`OUT_DIR/<stem>_gap.png` (`tests.studies.potts_plot`).

Timing: the graph is built once per worker per problem, outside the timed
solve; seconds are per job with `--workers` jobs sharing the host.
"""

from __future__ import annotations

import argparse
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

_GRAPHS: dict[tuple[int, float], Any] = {}


def _init() -> None:
    """Build each problem's graph once per worker, not inside the timed solve."""
    import tests.studies.clone_label_arms as arms

    build = arms._graph

    def memo(beta: float) -> Any:
        key = (arms._HELD["problem"], beta)
        if key not in _GRAPHS:
            _GRAPHS[key] = build(beta)
        return _GRAPHS[key]

    arms._graph = memo


def _hold(index: int, problem: Any) -> None:
    import tests.studies.clone_label_arms as arms

    arms._HELD["capture"], arms._HELD["problem"] = problem, index


def solve(problem: Any, solver: str, seed: int) -> dict[str, Any]:
    """One run from random labels, then its two polishes; a failure is a row."""
    from port.sandbox.known_field import color_merge
    from sal.sim.potts import energy

    import tests.studies.clone_label_arms as arms

    try:
        _hold(problem.realization, problem)
        field, beta = problem.field, problem.spatial_weight
        _, graph = arms._graph(beta)
        start = np.random.default_rng([seed, 7]).integers(
            0, field.shape[1], field.shape[0]
        )
        opened = time.perf_counter()
        if solver == "sal:trws":
            from sal.search.trws import trws

            out = np.asarray(trws(graph, field).labelling, dtype=np.int64)
        else:
            out = arms._solve(
                solver, field, start, np.random.default_rng([seed, 492]), beta
            )
        seconds = time.perf_counter() - opened
        opened = time.perf_counter()
        polished = arms._solve("sal:icm", field, out, np.random.default_rng(0), beta)
        polish_seconds = time.perf_counter() - opened
        graph_csr = (problem.indptr, problem.indices, problem.weights)
        opened = time.perf_counter()
        both, merges = color_merge(field, polished, *graph_csr, beta)
        merge_seconds = time.perf_counter() - opened
        return {
            "problem": problem.realization, "solver": solver, "seed": seed,
            "seconds": seconds, "energy": energy(graph, field, out),
            "start_energy": energy(graph, field, start),
            "polish_seconds": polish_seconds, "polished": energy(graph, field, polished),
            "both_seconds": polish_seconds + merge_seconds, "both": energy(graph, field, both),
            "merges": merges, "clones": int(np.unique(out).size), "both_clones": int(np.unique(both).size),
        }  # fmt: skip
    except Exception as error:  # noqa: BLE001 -- a failed job is a result
        return {"problem": problem.realization, "solver": solver, "seed": seed,
                "error": f"{type(error).__name__}: {error}", "trace": traceback.format_exc(limit=4)}  # fmt: skip


def run(
    manifest: Path, out_dir: Path, n_problems: int, starts: int, workers: int
) -> Path:
    """The stream; returns the pickle it keeps current."""
    from port.sandbox.known_field import problems
    from sal.sim.potts import energy
    from sklearn.metrics import adjusted_rand_score

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
            record = {"manifest": str(manifest), "problems": held, "rows": rows, "done": done,
                      "solvers": solvers, "starts": starts}  # fmt: skip
            out.write_bytes(pickle.dumps(record))
            subprocess.run(
                [sys.executable, "-m", "tests.studies.potts_plot", str(out)],
                check=False,
            )
            errors = sum("error" in r for r in rows)
            print(f"[{time.perf_counter() - opened:6.0f}s] problem {index} solved; "
                  f"{len(done)}/{n_problems} done, {errors} errors; plots redrawn", flush=True)  # fmt: skip

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=context, initializer=_init) as pool:
        # NB the next realization draws here while the pool solves the last
        for problem in problems(manifest, n_problems):
            _hold(problem.realization, problem)
            _, graph = arms._graph(problem.spatial_weight)
            bound, trws_energy, trws_seconds = arms._bound(
                problem.field, problem.spatial_weight
            )
            held[problem.realization] = {
                "bound": bound, "trws_energy": trws_energy, "trws_seconds": trws_seconds,
                "truth_energy": energy(graph, problem.field, problem.planted),
                "q": int(problem.field.shape[1]), "draw_seconds": problem.draw_seconds,
                "field_seconds": problem.field_seconds,
                "argmax_ari": adjusted_rand_score(problem.planted, problem.field.argmax(1)),
            }  # fmt: skip
            print(f"[{time.perf_counter() - opened:6.0f}s] drew {problem.realization}: draw "
                  f"{problem.draw_seconds:.1f}s, field {problem.field_seconds:.1f}s; truth - bound "
                  f"{held[problem.realization]['truth_energy'] - bound:.2f} nats", flush=True)  # fmt: skip
            pending[problem.realization] = len(solvers) * starts
            for solver in solvers:
                for seed in range(starts):
                    futures[pool.submit(solve, problem, solver, seed)] = (
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
    parser.add_argument("--problems", type=int, default=5)
    parser.add_argument("--starts", type=int, default=25)
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args(argv)
    run(
        arguments.manifest,
        arguments.out_dir,
        arguments.problems,
        arguments.starts,
        arguments.workers,
    )


if __name__ == "__main__":
    main()
