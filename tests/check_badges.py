"""Refuse a recorded coverage figure that is not the one just measured.

`tests/test_badges_agree.py` refuses a **badge** that disagrees with
`.badges/measurements.json`. What it cannot catch is the other direction: a
guard that moved and whose new figure was never written down. The badge and
the record agree perfectly, and both are wrong.

So this reads the coverage data the CI job just wrote and compares it with
what is recorded. Run as `python -m tests.check_badges` after the coverage
steps.

**It refuses rather than rewrites.** A job that recomputed the numbers and
pushed them would make the badges always correct and never reviewed, which
is the opposite of what a floor is for: a coverage change should arrive in a
diff someone reads, with the work that caused it beside it.

Guard 3 is not checked, because CI does not run it -- #159 asks for exactly
that and is unmet. Nor are the two ratio badges: they are measured on the
`release` tier, because a whole `run_cnaster` at a stress size is minutes
per arm, so a per-pull-request job has nothing to compare against.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from tests.badges import load

TOLERANCE = 0.005
"""How far a recorded figure may sit from the measured one, in points.

Not zero: coverage reports two decimals, so a round trip through the file
can differ in the last digit without anything having changed. Wider than
that would let a real movement hide.
"""

GUARDS = (
    ("judged", Path(".coverage-e2e"), None),
    ("oracle", Path(".coverage-oracle"), Path(".coveragerc-oracle")),
)
"""Each checkable guard, with the data file and config its CI step used.

Each guard has its own data file, and neither is `.coverage`. Every run
defaulted to that name until this module needed to read them, so whichever
ran last silently owned it -- a check reading `.coverage` got the referee's
percentage under the subject's name. The config matters for the same
reason: the oracle's denominator is its `include` list, and reporting its
data under `pyproject.toml`'s rules would measure `cnaster` against a run
that never imported it.

Both names are hyphenated rather than dotted, which is load-bearing:
`.coverage.oracle` reads as a parallel-mode shard of `.coverage`, and the
`coverage erase` that pytest-cov runs at the start of every `--cov` session
deletes those with the main file. That is measured, not theoretical -- a
`--cov` run removes a file of that name outright.

`judged` reads `.coverage-e2e` rather than the gate's data because the two
are no longer one selection. CI still gates on `end2end or oracle` at the
floor #157 decided; the badge says `e2e`, so it measures `end2end` alone --
a strict subset, worth 62 statements and 0.88 points less.
"""


def measured(data_file: Path, config: Path | None) -> float | None:
    """One guard's percentage, from the coverage data CI just wrote."""
    try:
        import coverage
    except ImportError:  # pragma: no cover - coverage is a dev dependency
        return None

    if not data_file.exists():
        return None

    measurement = coverage.Coverage(
        data_file=str(data_file),
        config_file=str(config) if config is not None else True,
    )
    measurement.load()

    # NB `report` insists on writing its table somewhere and returns the
    #    total; the table is not what is wanted here, only the number.
    with Path(os.devnull).open("w") as sink:
        return float(measurement.report(file=sink))


def main() -> int:
    recorded = load()["coverage"]
    failed = 0

    for name, data_file, config in GUARDS:
        guard = recorded[name]

        if guard["percent"] is None:
            print(f"{name} is recorded as unmeasured; nothing to check")
            continue

        current = measured(data_file, config)

        if current is None:
            # NB a failure, not a skip. In CI a missing data file means the
            #    step that writes it did not run, or something erased it --
            #    which is exactly how `.coverage.oracle` used to disappear.
            #    Passing quietly there would leave the guard unchecked and
            #    look identical to a guard that agreed.
            print(
                f"{name}: no coverage data at {data_file}. Its step did not run, "
                "or a later --cov run erased it."
            )
            failed += 1
            continue

        if abs(current - guard["percent"]) > TOLERANCE:
            print(
                f"{name} coverage is {current:.2f}% and .badges/measurements.json "
                f"records {guard['percent']:.2f}%.\n"
                "Update it and run `python -m tests.badges`, so the README's badge "
                "arrives in the same diff as the change that moved it."
            )
            failed += 1
        else:
            print(f"{name} coverage {current:.2f}% matches the record")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
