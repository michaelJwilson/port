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
    ("judged", Path(".coverage"), None),
    ("oracle", Path(".coverage.oracle"), Path(".coveragerc-oracle")),
)
"""Each checkable guard, with the data file and config its CI step used.

The two data files are distinct on purpose. Both runs defaulted to
`.coverage` until this module needed to read them, and because the oracle
step runs second it overwrote the subject's data -- so a check reading
`.coverage` got the referee's percentage under the subject's name. The
config matters for the same reason: the oracle's denominator is its
`include` list, and reporting its data under `pyproject.toml`'s rules would
measure `cnaster` against a run that never imported it.
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
            print(f"{name}: no data at {data_file}; run this after its coverage step")
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
