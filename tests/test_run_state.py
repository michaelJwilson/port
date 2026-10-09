"""Module-level state `port` writes after import, held to a run (#517 E2).

`STATE` and `OUTLIVES` are declared, and checked against the source, in
`tests/test_rules.py` (`state-writes`, `state-outlives`).
"""

from __future__ import annotations

import copy
import importlib
import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import port.scripts.run_cnaster as entry_point
import pytest
from port.sim.run_config import write_for_run

from tests import ROOT
from tests.test_rules import OUTLIVES, STATE


def _value(name: str) -> Any:
    parts = name.split(".")

    for split in range(len(parts), 0, -1):
        try:
            value: Any = importlib.import_module(".".join(parts[:split]))
        except ImportError:
            continue

        for part in parts[split:]:
            value = getattr(value, part)

        return value

    raise LookupError(name)


def _same(before: Any, after: Any) -> bool:
    if before is after:
        return True

    try:
        return type(before) is type(after) and pickle.dumps(before) == pickle.dumps(
            after
        )
    except (pickle.PicklingError, TypeError, AttributeError):
        return False


def changed_by_a_run(argv: list[str]) -> list[str]:
    """Run the entry point on `argv` in this process; the declared names it changed."""

    before = {name: copy.deepcopy(_value(name)) for name in STATE}

    if entry_point.main(argv) != 0:
        msg = f"run_cnaster_port {argv} failed"
        raise RuntimeError(msg)

    return sorted(name for name in STATE if not _same(before[name], _value(name)))


@pytest.mark.smoke
@pytest.mark.merge
@pytest.mark.parametrize("flags", [(), ("--sal",)], ids=["default", "sal"])
def test_a_run_leaves_only_the_declared_state_behind(
    planted_instance: Any, tmp_path: Path, flags: tuple[str, ...]
) -> None:
    """After one run, every declared name but `OUTLIVES` is back at its import-time value."""

    _, config = write_for_run(
        planted_instance[0], tmp_path, max_iter_outer=1, max_iter=3
    )
    completed = subprocess.run(
        [sys.executable, "-m", "tests.test_run_state", *flags, str(config)],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )

    assert completed.returncode == 0, completed.stderr[-4000:]

    changed = set(json.loads(completed.stdout.strip().splitlines()[-1]))

    assert changed <= OUTLIVES, f"left behind: {sorted(changed - OUTLIVES)}"

    if not flags:
        assert changed == OUTLIVES, f"released: {sorted(OUTLIVES - changed)}"


if __name__ == "__main__":
    import matplotlib as mpl

    mpl.use("Agg")
    print(json.dumps(changed_by_a_run(sys.argv[1:])))
