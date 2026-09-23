"""Trial the BAF-stage clone initializers against the planted truth (#358).

Run as `python -m tests.clone_init_trial [--candidates a,b] [--fixtures x,y]
[--seeds N] [--jobs J]`. Each `(fixture, candidate, seed)` is one fresh
process running `run_cnaster_port` at the figures' configuration with the
BAF stage started from `port.sandbox.clone_init.CANDIDATES[candidate]`, and
prints one `TRIAL` line of JSON: the initial labelling's ARI, the final
clone ARI (continuous and integer), the copy-state ARIs and the wall time.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

FIXTURES = ("dev", "calicost", "mandelbrot", "lattice")


def _truth(fixture: str) -> Any:
    from tests import fixtures

    if fixture == "dev":
        return fixtures.dev_instance()
    if fixture == "calicost":
        return fixtures.calicost_instance()
    if fixture == "mandelbrot":
        return fixtures.dev_instance(labelling="mandelbrot")
    if fixture == "lattice":
        return fixtures.dev_instance(
            n_states=len(fixtures.COPY_LATTICE), copy_lattice=True
        )
    msg = f"unknown fixture {fixture!r}"
    raise ValueError(msg)


def one(fixture: str, candidate: str, seed: int) -> dict[str, Any]:
    """One run, scored."""
    import warnings

    import matplotlib as mpl
    from port.sandbox.clone_init import trial
    from sklearn.metrics import adjusted_rand_score

    from tests.recovery_audit import run_arm

    mpl.use("Agg")
    warnings.simplefilter("ignore")
    truth = _truth(fixture)
    n_states = len(truth.log_mu) if fixture == "lattice" else 8
    started = time.perf_counter()

    with trial(candidate, truth=truth.labels, seed=seed) as record:
        recovery, _ = run_arm(truth, ["--no-figures"], n_states=n_states)

    return {
        "fixture": fixture,
        "candidate": candidate,
        "seed": seed,
        "initial_ari": round(
            float(adjusted_rand_score(truth.labels, record["labels"])), 4
        ),
        "ari": recovery.ari,
        "ari_integer": recovery.ari_integer,
        "n_clones": recovery.n_clones,
        "copy_ari": recovery.copy_ari,
        "state_ari": recovery.state_ari,
        "wall": round(time.perf_counter() - started, 2),
    }


def main() -> None:
    from port.sandbox.clone_init import CANDIDATES

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidates", default=",".join(CANDIDATES))
    parser.add_argument("--fixtures", default=",".join(FIXTURES))
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=1)
    arguments = parser.parse_args()

    runs = [
        (fixture, candidate, seed)
        for fixture in arguments.fixtures.split(",")
        for candidate in arguments.candidates.split(",")
        for seed in range(arguments.seeds)
    ]
    context = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=arguments.jobs, mp_context=context, max_tasks_per_child=1
    ) as pool:
        futures = {pool.submit(one, *run): run for run in runs}
        for future, run in futures.items():
            try:
                print("TRIAL " + json.dumps(future.result()), flush=True)
            except Exception as error:
                print(
                    "TRIAL " + json.dumps({"run": run, "error": repr(error)}),
                    flush=True,
                )


if __name__ == "__main__":
    main()
