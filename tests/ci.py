"""The one way to run CI, locally or in a workflow (#403).

`python -m tests.ci` runs the gate. Each step prints its wall time, and the
first failure stops the run with that step's exit code.

| flag | runs | budget, 4 cores |
| --- | --- | --- |
| `--gate` (default) | ruff, mypy, every test in no tier or `critical` | 60 s |
| `--badges` | the judged and drop-in coverage guards, `check_badges` | pre-merge |
| `--full` | the gate, the badges, then `merge` tests neither guard ran | pre-merge |
| `--release` | `release`, `oracle` and `deprecate` | release |
| `--figures` | `docs/plots`, drawn locally | on a figure change |

The tiers partition the suite -- `critical`, none, `merge`, `release`,
`deprecate`, at most one per test (`tests/test_marker_discipline.py`) -- so
no step runs a test another step ran. A `deprecate` test is too specific to
run on every change: it has passed where it was merged, and runs again only
where its module differs from `--base` (`origin/main`), and at a release. `--record` writes the guards' measured figures into
`.badges/measurements.json` and regenerates the badges, so the change that
moves a figure carries it (`CLAUDE.md`: badges are local).

A guard whose recorded input hash (`tests.badges.inputs_hash`: `python/`,
`tests/`, `src/`, the locks, `pyproject.toml`, the coverage configs) is the
tree's is not re-measured: its figure is a function of those inputs, so it
still holds. `--force` measures it anyway.

`--install` sets the `badges` merge driver `.gitattributes` names: a merge
keeps this branch's badge files and figures rather than stopping on them,
and `--badges` then measures what the merged tree actually reads.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

WORKERS = 4
"""xdist workers: the host's cores, and 15 GB holds four gate workers."""

GATE = "not release and not oracle and not merge and not benchmark and not deprecate"
"""Every test in no tier or `critical`. Benchmarks measure, so they run
serially under `--full`, where pytest-benchmark is not disabled by xdist."""

JUDGED = "not release and end2end"
DROPIN = "not release and (patch or cnaster)"
"""The guards measure rather than gate, so they read `deprecate` tests too: a
correspondence test is what makes a replacement compared, however seldom the
gate runs it, and a guard that dropped it would read the replacement as
uncompared (#403: the drop-in figure fell 90.57 -> 89.53 without this)."""
MERGE_REST = "merge and not end2end and not patch and not cnaster"
"""The `merge` tests neither coverage guard runs, so `--full` runs each once."""

RELEASE = "release or oracle or deprecate"

DEPRECATE = "deprecate"
"""Run only in the test modules a change touches (`_changed_modules`)."""


def _changed_modules(base: str) -> list[str]:
    """Test modules that differ from `base`, committed or not.

    A moved `uv.lock` counts as every module changing: `bug` and `warning`
    tests pin what `cnaster` does at its locked version, so a pin bump is
    exactly when a `deprecate` one of them has something new to say.
    """
    merge_base = subprocess.run(
        ["git", "merge-base", "HEAD", base], capture_output=True, text=True, check=False
    ).stdout.strip()
    if not merge_base:
        return []
    changed = subprocess.run(
        ["git", "diff", "--name-only", merge_base, "--", "tests/", "uv.lock"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    if "uv.lock" in changed:
        changed = [str(path) for path in Path("tests").glob("test_*.py")]
    return sorted(
        path
        for path in changed
        if path.startswith("tests/test_")
        and path.endswith(".py")
        and Path(path).exists()
    )


def _pytest(
    marks: str, *, workers: int, cov: Sequence[str] = ("--no-cov",)
) -> list[str]:
    distribute = ["-n", str(workers), "--dist", "loadgroup"] if workers > 1 else []
    return [sys.executable, "-m", "pytest", "-q", "-m", marks, *distribute, *cov]


def _steps(
    arguments: argparse.Namespace,
) -> list[tuple[str, list[str], dict[str, str]]]:
    n = arguments.workers
    sysmon = {"COVERAGE_CORE": "sysmon"}
    steps: list[tuple[str, list[str], dict[str, str]]] = []

    if arguments.gate or arguments.full:
        steps += [
            ("ruff check", [sys.executable, "-m", "ruff", "check", "."], {}),
            (
                "ruff format",
                [sys.executable, "-m", "ruff", "format", "--check", "."],
                {},
            ),
            ("mypy", [sys.executable, "-m", "mypy"], {}),
            ("gate", _pytest(GATE, workers=n), {}),
        ]
        touched = _changed_modules(arguments.base)
        if touched:
            # NB exit code 5 is "no tests collected": no touched module holds
            #    a `deprecate` test, which is the common case and not a failure.
            steps.append(
                (
                    "deprecate, touched modules",
                    [*_pytest(DEPRECATE, workers=1), *touched],
                    {"PORT_CI_EMPTY_OK": "1"},
                )
            )

    if arguments.badges or arguments.full:
        unchanged = _unchanged() if not arguments.force else set()
        judged = (
            "judged coverage",
            _pytest(JUDGED, workers=n, cov=("--cov", "--cov-report=")),
            {**sysmon, "COVERAGE_FILE": ".coverage-e2e"},
        )
        dropin = (
            "drop-in coverage",
            _pytest(
                DROPIN,
                workers=n,
                cov=(
                    "--cov",
                    "--cov-config=.coveragerc-dropin",
                    "--cov-report=",
                    "--cov-fail-under=0",
                ),
            ),
            # NB `numba` reports nothing for a compiled body, so the guard
            #    runs its kernels as Python (#281).
            {**sysmon, "COVERAGE_FILE": ".coverage-dropin", "NUMBA_DISABLE_JIT": "1"},
        )
        steps += [
            step
            for name, step in (("judged", judged), ("dropin", dropin))
            if name not in unchanged
        ]
        check = [sys.executable, "-m", "tests.check_badges"]
        if arguments.record:
            check.append("--record")
        if unchanged:
            check.append("--skip=" + ",".join(sorted(unchanged)))
            print(
                f"[tests.ci] inputs unchanged, not re-measured: {', '.join(sorted(unchanged))}"
            )
        steps.append(("check badges", check, {}))

    if arguments.full:
        steps += [
            ("merge, serially", _pytest(MERGE_REST, workers=1), {}),
            (
                "benchmarks, serially",
                _pytest("benchmark and not release", workers=1),
                {},
            ),
        ]

    if arguments.release:
        steps.append(("release", _pytest(RELEASE, workers=min(n, 2)), {}))

    if arguments.figures:
        steps.append(("figures", [sys.executable, "-m", "tests.generate_plots"], {}))

    return steps


def _unchanged() -> set[str]:
    """Guards whose recorded input hash is the tree's: their figure still holds."""
    from tests.badges import inputs_hash, load

    digest = inputs_hash()
    return {
        name
        for name, guard in load()["coverage"].items()
        if name in {"judged", "dropin"} and guard.get("inputs") == digest
    }


def install() -> int:
    """Name the merge driver `.gitattributes` routes badges and figures to."""
    for key, value in (
        ("merge.badges.name", "keep this branch's; tests.ci --badges re-measures"),
        ("merge.badges.driver", "true"),
    ):
        subprocess.run(["git", "config", key, value], check=True)
    print("installed the `badges` merge driver")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.ci", description=__doc__)
    parser.add_argument("--gate", action="store_true", help="lint, types, the gate")
    parser.add_argument("--badges", action="store_true", help="coverage guards")
    parser.add_argument("--full", action="store_true", help="gate, badges, merge")
    parser.add_argument("--release", action="store_true", help="release and oracle")
    parser.add_argument("--figures", action="store_true", help="draw docs/plots")
    parser.add_argument(
        "--record", action="store_true", help="write the measured figures"
    )
    parser.add_argument(
        "--force", action="store_true", help="re-measure unchanged guards"
    )
    parser.add_argument(
        "--base", default="origin/main", help="what `deprecate` tests diff against"
    )
    parser.add_argument("--install", action="store_true", help="set the merge driver")
    parser.add_argument("-n", "--workers", type=int, default=WORKERS)
    arguments = parser.parse_args(argv)

    if arguments.install:
        return install()

    if not any(
        (
            arguments.gate,
            arguments.badges,
            arguments.full,
            arguments.release,
            arguments.figures,
        )
    ):
        arguments.gate = True

    started = time.perf_counter()

    for name, command, environment in _steps(arguments):
        began = time.perf_counter()
        code = subprocess.run(
            command, env={**os.environ, **environment}, check=False
        ).returncode
        if code == 5 and environment.get("PORT_CI_EMPTY_OK"):
            code = 0
        print(f"[tests.ci] {name}: {time.perf_counter() - began:.1f} s", flush=True)

        if code:
            print(f"[tests.ci] {name} failed with exit code {code}")
            return code

    print(f"[tests.ci] total: {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
