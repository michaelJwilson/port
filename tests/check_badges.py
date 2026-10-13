"""Refuse a recorded coverage figure that is not the one just measured.

Run as `python -m tests.check_badges` after CI's coverage steps; refuses rather than
rewrites. Guard 3 (#159) and the release-tier ratio badges are not checked.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from port.qa.provenance import inputs_hash

from scripts.badges import MEASUREMENTS, load, write

TOLERANCE = 0.005
"""Allowed gap between recorded and measured figures, in points (rounding only)."""

GUARDS = (
    ("judged", Path(".coverage-e2e"), None),
    ("oracle", Path(".coverage-oracle"), Path(".coveragerc-oracle")),
    ("dropin", Path(".coverage-dropin"), Path(".coveragerc-dropin")),
)
"""Each checkable guard, with its data file and config.

Hyphenated names: pytest-cov's `coverage erase` deletes `.coverage.*` shards. `judged`
reads `.coverage-e2e`, the `end2end`-only selection.
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

    # NB only the total is wanted from `report`.
    with Path(os.devnull).open("w") as sink:
        return float(measurement.report(file=sink))


def main(argv: list[str] | None = None) -> int:
    """Check every guard; with `--record`, write what was measured instead (#403)."""
    arguments = sys.argv[1:] if argv is None else argv
    record = "--record" in arguments
    # NB `--skip` names guards whose recorded input hash matches the tree (#403).
    skipped = {
        name
        for flag in arguments
        if flag.startswith("--skip=")
        for name in flag.removeprefix("--skip=").split(",")
    }
    digest = inputs_hash()
    document = load()
    recorded = document["coverage"]
    failed = 0
    moved = False

    for name, data_file, config in GUARDS:
        guard = recorded[name]

        if guard["percent"] is None:
            print(f"{name} is recorded as unmeasured; nothing to check")
            continue

        if name in skipped:
            if guard.get("inputs") != digest:
                print(f"{name}: skipped, but its inputs changed; re-measure it")
                failed += 1
            else:
                print(
                    f"{name}: inputs unchanged since {guard['percent']:.2f}% was recorded"
                )
            continue

        current = measured(data_file, config)

        if current is None:
            # NB a missing data file fails: the step that writes it did not run.
            print(
                f"{name}: no coverage data at {data_file}. Its step did not run, "
                "or a later --cov run erased it."
            )
            failed += 1
            continue

        if record:
            if abs(current - guard["percent"]) > TOLERANCE:
                print(f"{name}: recorded {guard['percent']:.2f}% -> {current:.2f}%")
                guard["percent"] = round(current, 2)
            else:
                print(f"{name} coverage {current:.2f}% matches the record")
            guard["inputs"] = digest
            moved = True
        elif abs(current - guard["percent"]) > TOLERANCE:
            print(
                f"{name} coverage is {current:.2f}% and .badges/measurements.json "
                f"records {guard['percent']:.2f}%.\n"
                "Update it and run `python -m scripts.badges`, so the README's badge "
                "arrives in the same diff as the change that moved it."
            )
            failed += 1
        else:
            print(f"{name} coverage {current:.2f}% matches the record")

    if moved:
        MEASUREMENTS.write_text(
            json.dumps(document, indent=1, ensure_ascii=False) + "\n"
        )
        write()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
