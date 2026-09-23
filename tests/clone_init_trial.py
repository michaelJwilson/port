"""Trial the BAF-stage clone initializers against the planted truth (#358).

Run as `python -m tests.clone_init_trial [--candidates a,b] [--fixtures x,y]
[--seeds N] [--jobs J]`. Each `(fixture, candidate, seed)` is one fresh
process running `run_cnaster_port` at the figures' configuration with the
BAF stage started from `port.sandbox.clone_init.CANDIDATES[candidate]`, and
prints one `TRIAL` line of JSON: the initial labelling's ARI, the final
clone ARI (continuous and integer), the copy-state ARIs and the wall time.

Every run is at copy cap 6 (`int_copy_num.max_total_copy`), as #348's
comparison is: at 12 one dev-instance run peaks at 7.4 GB. A run that dies
(out of memory, a crash) is reported as an error line and the rest go on.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

FIXTURES = ("dev", "calicost", "mandelbrot", "lattice")

CAP = {"int_copy_num.max_total_copy": 6}


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
        recovery, _ = run_arm(truth, ["--no-figures"], n_states=n_states, overrides=CAP)

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
    parser.add_argument("--one", nargs=3, metavar=("FIXTURE", "CANDIDATE", "SEED"))
    arguments = parser.parse_args()

    if arguments.one:
        fixture, candidate, seed = arguments.one
        print("TRIAL " + json.dumps(one(fixture, candidate, int(seed))), flush=True)
        return

    runs = [
        (fixture, candidate, seed)
        for fixture in arguments.fixtures.split(",")
        for candidate in arguments.candidates.split(",")
        for seed in range(arguments.seeds)
    ]

    def isolated(run: tuple[str, str, int]) -> str:
        command = [sys.executable, "-m", "tests.clone_init_trial", "--one"]
        done = subprocess.run(
            [*command, *map(str, run)], capture_output=True, text=True, check=False
        )
        lines = [x for x in done.stdout.splitlines() if x.startswith("TRIAL ")]
        if lines:
            return lines[-1]
        error = {"run": run, "error": f"exit {done.returncode}: {done.stderr[-300:]}"}
        return "TRIAL " + json.dumps(error)

    with ThreadPoolExecutor(max_workers=arguments.jobs) as pool:
        for line in pool.map(isolated, runs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
