"""#494: `run_cnaster_port --sal` and CalicoST under its own settings, on easy, hard and `dev_tree`.

`python -m tests.final_benchmark SAMPLE {port,calicost} [--repeats N]` runs one
tool on one sample in a child process and prints one `BENCH` JSON line: clone
ARI (clones), copy ARI, exact altered and its phase-free form, wall and the
child's peak RSS.

**port** is `tests.sim_audit --sal`, repeated `--repeats` times on a warm numba
cache; the wall is the median.

**CalicoST** is `run_calicost --shipped configuration_cna --no-align
--no-figures` on the same staged inputs: CalicoST's own configuration file as
its tutorial runs the simulated example, every value but the paths as
shipped, and its own hard-coded constants. It runs once under `timeout
TIMEOUT`; a case that reaches it is recorded as not finished.

Both are scored by `tests.sim_audit.score` on what they wrote.
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

TIMEOUT = 1800
"""CalicoST's budget per case, in seconds (#494)."""


def _child_peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024**2


def _shipped() -> Path:
    from tests.sim_fixtures import references

    resources = references()

    if resources is None:
        msg = "CalicoST's checkout is not installed; its configuration_cna is needed"
        raise FileNotFoundError(msg)

    return resources.parent / "configuration_cna"


def _sample(name: str) -> Any:
    from tests.sim_fixtures import load_simulated

    path = Path(name)
    return (
        load_simulated(path.name, path.parent)
        if path.is_absolute()
        else load_simulated(name)
    )


def port(name: str, repeats: int) -> dict[str, Any]:
    """`tests.sim_audit --sal` in a child, `repeats` times; the last run's scores, the median wall."""
    walls, row = [], {}

    for _ in range(repeats):
        started = time.perf_counter()
        done = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.sim_audit",
                "--sample",
                name,
                "--",
                "--sal",
                "--no-plots",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        walls.append(time.perf_counter() - started)
        line = next(x for x in done.stdout.splitlines() if x.startswith("SIM "))
        row = json.loads(line[4:])

    return {
        "tool": "port --sal",
        "ari": row["ari"],
        "clones": row["n_clones"],
        "copy_ari": row["copy_ari"],
        "exact_altered": row["exact_altered"],
        "exact_altered_minor": row["exact_altered_minor"],
        "wall": round(statistics.median(walls), 1),
        "walls": [round(w, 1) for w in walls],
        "peak_gb": round(_child_peak_gb(), 2),
    }


def calicost(name: str) -> dict[str, Any]:
    """CalicoST on its shipped configuration, under `TIMEOUT`, scored as port is."""
    from tests.sim_audit import _drawn_config, score
    from tests.sim_fixtures import write_sim_inputs

    sample = _sample(name)
    root = Path(tempfile.mkdtemp())
    config = (
        _drawn_config(sample, root, {})
        if (sample.path / "snp").is_dir()
        else write_sim_inputs(sample, root, {})
    )
    started = time.perf_counter()

    try:
        subprocess.run(
            [
                "timeout",
                str(TIMEOUT),
                sys.executable,
                "-c",
                "from port.scripts.run_calicost import main; raise SystemExit(main())",
                str(config),
                "--shipped",
                str(_shipped()),
                "--no-align",
                "--no-figures",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        wall = time.perf_counter() - started
        reached = [x for x in error.stderr.splitlines() if " - " in x][-1:]
        return {
            "tool": "CalicoST (shipped)",
            "finished": False,
            "returncode": error.returncode,
            "wall": round(wall, 1),
            "peak_gb": round(_child_peak_gb(), 2),
            "reached": reached[0][:200] if reached else error.stderr[-400:],
        }

    wall = time.perf_counter() - started
    print(f"calicost output: {root / 'output_calicost'}", file=sys.stderr, flush=True)

    try:
        recovery = score(sample, root / "output_calicost", "calicost", wall)
    except Exception as error:  # noqa: BLE001 -- the run is kept; the scoring is reported
        return {
            "tool": "CalicoST (shipped)",
            "finished": True,
            "scored": f"{type(error).__name__}: {error}",
            "wall": round(wall, 1),
            "peak_gb": round(_child_peak_gb(), 2),
            "output": str(root / "output_calicost"),
        }

    return {
        "tool": "CalicoST (shipped)",
        "finished": True,
        "ari": recovery.ari,
        "clones": recovery.n_clones,
        "copy_ari": recovery.copy_ari,
        "exact_altered": recovery.exact_altered,
        "exact_altered_minor": recovery.exact_altered_minor,
        "wall": round(wall, 1),
        "peak_gb": round(_child_peak_gb(), 2),
        "output": str(root / "output_calicost"),
    }


def main(argv: list[str] | None = None) -> None:
    """One tool on one sample; one `BENCH` line."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("sample")
    parser.add_argument("tool", choices=["port", "calicost"])
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args(argv)
    row = (
        port(arguments.sample, arguments.repeats)
        if arguments.tool == "port"
        else calicost(arguments.sample)
    )
    print("BENCH " + json.dumps({"sample": arguments.sample, **row}), flush=True)


if __name__ == "__main__":
    main()
