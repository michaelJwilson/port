"""#540: copy-state starts at known clones on a stream of drawn realizations, polished by the HMM's Baum-Welch.

`python -m tests.studies.copy_state_stream MANIFEST OUT_DIR [--problems 5] [--seeds 3] [--workers 4] [--all]`

The main process draws each realization of `MANIFEST` and builds its problem at
the planted clones (`port.sandbox.known_copy.problems`: 1 Mb bins under #551's
300 normal-UMI floor, phased allele reads); the pool runs the starts while the
next realization draws.

- **Starts.** `STARTS`, one per family of `port.extensions.copy_starts`'
  registry (`--all`: all 42), each seeded as `run_start` seeds it but without
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
(`tests.studies.copy_state_plot`). Each worker warms up on a small drawn call
first; seconds are per job with `--workers` jobs sharing the host.
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

STARTS = (
    "cnaster-gmm", "calicost-gmm", "distinct", "lattice", "lattice-em", "rdr-quantiles",
    "prior", "data", "kmeans++", "emission++", "gaussian-em", "quantile", "anneal", "tempering", "hmc",
    "datax5+em", "emission++x5+em", "kmeans++x5+em",
)  # fmt: skip
"""One start per family of the registry; `kmeans++x5+em` is `--sal`'s."""

SECONDS = 60.0
"""A best-of-n start's budget for its own polishes, as `run_start` gives it."""


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

    config = (Path(__file__).resolve().parents[2] / CONFIG).read_text()
    return {"X": np.stack([problem.total, problem.b], axis=1)[:, :, None].astype(np.float64),
            "base_nb_mean": column(problem.exposure), "total_bb_RD": column(problem.trials),
            "lengths": np.asarray(problem.lengths), "log_sitewise_transmat": np.zeros(problem.total.size),
            "params": "smp", "config": config}  # fmt: skip


def _call(problem: Any) -> Any:
    from port.extensions.copy_starts import CopyCall

    return CopyCall("rdrbaf", problem.n_states, problem.total, problem.b, problem.exposure, problem.trials,
                    problem.clone, problem.contig, problem.start, problem.length, problem.planted, _raw(problem))  # fmt: skip


def seed_states(
    name: str, problem: Any, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """`name`'s states on `problem`, seeded as `copy_starts.run_start` seeds them, before its `sal` polish."""
    from port.extensions.copy_starts import seed_states as seeded
    from port.extensions.copy_starts import starts

    return seeded(
        name, _call(problem), rng, covariate=starts()[name].covariate, seconds=SECONDS
    )


def solve(problem: Any, name: str, seed: int) -> dict[str, Any]:
    """One start, scored at its states and after Baum-Welch; a failure is a row."""
    import port.sandbox.known_copy as kc

    try:
        opened = time.perf_counter()
        log_mu, p = seed_states(name, problem, np.random.default_rng([seed, 540]))
        seconds = time.perf_counter() - opened
        truth = problem.truth_label
        at_start = kc.decode(problem, log_mu, p)
        fitted = kc.baum_welch(problem, log_mu, p)
        return {
            "problem": problem.realization, "start": name, "seed": seed, "seconds": seconds,
            "start_llf": at_start.log_likelihood, "start_missed": kc.missed(at_start.label, truth),
            "bw_seconds": fitted.seconds, "llf": fitted.log_likelihood, "missed": kc.missed(fitted.label, truth),
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


def run(
    manifest: Path,
    out_dir: Path,
    n_problems: int,
    seeds: int,
    workers: int,
    everything: bool,
) -> Path:
    """The stream; returns the pickle it keeps current."""
    import logging

    import port.sandbox.known_copy as kc
    from port.extensions.copy_starts import starts

    logging.disable(logging.INFO)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"copy_{manifest.stem}.pkl"
    names = list(starts()) if everything else list(STARTS)
    registry = starts()
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
            record = {"manifest": str(manifest), "problems": held, "rows": rows, "done": done, "starts": names,
                      "seeds": seeds}  # fmt: skip
            out.write_bytes(pickle.dumps(record))
            subprocess.run(
                [sys.executable, "-m", "tests.studies.copy_state_plot", str(out)],
                check=False,
            )
            errors = sum("error" in r for r in rows)
            print(f"[{time.perf_counter() - opened:6.0f}s] problem {index} solved; {len(done)}/{n_problems} done, "
                  f"{errors} errors; plot redrawn", flush=True)  # fmt: skip

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=context, initializer=_init) as pool:
        for problem in kc.problems(manifest, n_problems, realizations=n_problems):
            held[problem.realization] = _describe(problem)
            print(f"[{time.perf_counter() - opened:6.0f}s] drew {problem.realization}: {problem.total.size} rows, "
                  f"{problem.n_states} states; truth after Baum-Welch missed {held[problem.realization]['truth_missed']}",
                  flush=True)  # fmt: skip
            jobs = [
                (name, seed)
                for name in names
                for seed in (range(seeds) if registry[name].stochastic else [0])
            ]
            pending[problem.realization] = len(jobs)
            for name, seed in jobs:
                futures[pool.submit(solve, problem, name, seed)] = problem.realization
            drain(block=False)
        while futures:
            drain(block=True)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--problems", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--all",
        action="store_true",
        help="every start of the registry, not one per family",
    )
    arguments = parser.parse_args(argv)
    run(
        arguments.manifest,
        arguments.out_dir,
        arguments.problems,
        arguments.seeds,
        arguments.workers,
        arguments.all,
    )


if __name__ == "__main__":
    main()
