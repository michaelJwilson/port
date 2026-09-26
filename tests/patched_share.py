"""The share of `cnaster` lines a run executes that `port` has replaced (#302).

Run as `python -m tests.patched_share` to measure it and record it in
`.badges/measurements.json`; `python -m tests.badges` then renders the badge.

**Executed** is what an **unpatched** `run_cnaster` runs, on the dev
instance, under coverage of `cnaster` with `numba` disabled -- so an `@njit`
kernel's body counts as the lines it is, rather than as one call coverage
cannot see into. **Patched** is every executed line inside a function that a
row `run_cnaster_port` installs by default replaces: `SWAPS`,
`FIGURE_SWAPS` and `SHIFT_SWAPS`. For a class row, only the methods the
replacement class overrides count; what it inherits is still `cnaster`'s.

So the figure is the fraction of what a run exercises that no longer runs as
`cnaster` wrote it. It is not a quality claim and it asserts nothing, so the
badge is blue.
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

from tests.badges import MEASUREMENTS

INSTANCE = "dev"
"""What the run is measured on: `tests.fixtures.dev_instance`, five states."""


def _lines(function: Any) -> tuple[str, set[int]]:
    """The file and line numbers `function`'s source occupies."""
    function = inspect.unwrap(function)
    source, start = inspect.getsourcelines(function)

    return os.path.realpath(inspect.getsourcefile(function) or ""), set(
        range(start, start + len(source))
    )


def patched_lines() -> dict[str, set[int]]:
    """`{file: lines}` of the `cnaster` code the default rows replace."""
    import importlib

    from port.pipeline import FIGURE_SWAPS, SHIFT_SWAPS, SWAPS

    spans: dict[str, set[int]] = {}

    for swap in SWAPS + FIGURE_SWAPS + SHIFT_SWAPS:
        original = getattr(importlib.import_module(swap.module), swap.name)
        module_name, _, attribute = swap.replacement.partition(":")
        replacement = getattr(importlib.import_module(module_name), attribute)

        if inspect.isclass(original):
            targets = [
                getattr(original, name)
                for name, value in vars(replacement).items()
                if callable(value) or isinstance(value, staticmethod | classmethod)
                if hasattr(original, name)
            ]
        else:
            targets = [original]

        for target in targets:
            try:
                path, lines = _lines(target)
            except (OSError, TypeError):
                continue

            spans.setdefault(path, set()).update(lines)

    return spans


def executed_lines() -> dict[str, set[int]]:
    """`{file: lines}` an unpatched `run_cnaster` executes on the dev instance.

    In a subprocess, because `numba` reads `NUMBA_DISABLE_JIT` at import and
    this process has imported it already.
    """
    import json

    with tempfile.TemporaryDirectory() as scratch:
        report = Path(scratch) / "executed.json"
        subprocess.run(
            [sys.executable, "-m", "tests.patched_share", "--run", str(report)],
            check=True,
            env={**os.environ, "NUMBA_DISABLE_JIT": "1"},
        )
        recorded: dict[str, list[int]] = json.loads(report.read_text())

    return {path: set(lines) for path, lines in recorded.items()}


def _run(report: Path) -> None:
    """The subprocess: one unpatched run under coverage, written as JSON."""
    import json

    import cnaster
    import coverage
    import matplotlib as mpl

    mpl.use("Agg")

    from tests.fixtures import dev_instance
    from tests.generate_plots import STATES
    from tests.run_config import run_written

    # NB `__path__`, not `__file__`: this pin ships `cnaster` as a namespace
    #    package, with no `__init__.py` to name.
    root = str(Path(next(iter(cnaster.__path__))).resolve())
    tracer = coverage.Coverage(source=[root], data_file=None)

    with tempfile.TemporaryDirectory() as scratch, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tracer.start()
        try:
            run_written(
                dev_instance(),
                Path(scratch),
                port=False,
                max_iter_outer=1,
                max_iter=3,
                n_states=STATES,
            )
        finally:
            tracer.stop()

    data = tracer.get_data()
    report.write_text(
        json.dumps(
            {
                os.path.realpath(path): sorted(data.lines(path) or ())
                for path in data.measured_files()
            }
        )
    )


def share(
    executed: dict[str, set[int]], patched: dict[str, set[int]]
) -> tuple[int, int]:
    """`(patched, executed)` line counts: executed lines inside patched spans."""
    total = sum(len(lines) for lines in executed.values())
    hit = sum(len(lines & patched.get(path, set())) for path, lines in executed.items())

    return hit, total


def main() -> None:
    import json

    if len(sys.argv) == 3 and sys.argv[1] == "--run":
        _run(Path(sys.argv[2]))
        return

    hit, total = share(executed_lines(), patched_lines())
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    recorded = json.loads(MEASUREMENTS.read_text())
    recorded["patched"] = {
        "label": "patched",
        "percent": round(100.0 * hit / total, 2),
        "patched_lines": hit,
        "executed_lines": total,
        "instance": INSTANCE,
        "commit": commit,
        "note": (
            "Executed cnaster lines inside functions run_cnaster_port replaces "
            "by default (SWAPS, FIGURE_SWAPS, SHIFT_SWAPS; for a class, only "
            "overridden methods), over every cnaster line an unpatched "
            "run_cnaster executes on the dev instance with numba disabled. "
            "Measured by python -m tests.patched_share (#302); not checked per "
            "pull request, because it needs a whole run."
        ),
    }
    MEASUREMENTS.write_text(json.dumps(recorded, indent=1, ensure_ascii=False) + "\n")

    print(
        f"patched {hit} of {total} executed cnaster lines: {100.0 * hit / total:.2f}%"
    )


if __name__ == "__main__":
    main()
