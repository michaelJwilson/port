"""Module-level state `port` writes after import, declared and held to a run (#517 E2).

**A drop-in is a rebinding with no state that outlives the call.** Every
module-level name `port` writes after import is listed in `STATE` with what
it is, and the source is read to find them, so a new one arrives as a diff to
this file rather than as a line nobody reads as a switch.

The kinds:

- `switch`: a process-global option that changes what a drop-in returns.
  None remains: #517 steps 1 and 2 made each an option bound at install.
- `run`: what one run carries between calls that nothing threads through.
  Justified, and released when the run ends.
- `cache`: a memo keyed by its inputs.
- `rebind`: a name rebound outside a swap table.

`OUTLIVES` is what a whole run leaves behind, empty since step 1. It is
declared so that it can only shrink.

`infra` for the declaration, `smoke` for the run: the second executes
`cnaster`, and checks the run against itself rather than against a truth.
"""

from __future__ import annotations

import copy
import importlib
import pickle
from pathlib import Path
from typing import Any, Literal

import pytest

from tests.source_graph import ROOT, state_writes

Kind = Literal["switch", "run", "cache", "rebind"]

STATE: dict[str, Kind] = {
    "cnaster.hmm_initialize.GaussianMixture": "rebind",
    "port.extensions.copy_likelihood._FITS": "run",
    "port.extensions.samples._CURRENT": "run",
    "port.extensions.segments._CURRENT": "run",
    "port.patch.hmm_nophasing.shifted_emission.hmm_nophasing._row_shift": "run",
    "port.patch.hmrf.clone_assignment._BOUNDARY": "cache",
    "port.patch.hmrf.core_inference._NORMAL": "run",
    "port.patch.hmrf.core_inference._PROPAGATED": "run",
    "port.patch.hmrf.refinement._KEPT": "run",
    "port.patch.hmrf.run_core_inference": "rebind",
    "port.patch.integer_copy._RECORDERS": "run",
    "port.patch.integer_copy._SHARED": "cache",
    "port.patch.io.NORMAL_SPOTS": "run",
}
"""Every name `port` writes after import, by kind.

No `switch` since #517 steps 1 and 2: each of the 12 is an option a row
binds at install.
"""

OUTLIVES: frozenset[str] = frozenset()
"""What a whole run leaves changed: nothing since #517 step 1.

`port.pipeline.patched` calls each `RUN_STATE` release on exit.
"""


@pytest.mark.infra
def test_every_name_written_after_import_is_declared() -> None:
    """The source's writes against `STATE`, both ways.

    A new module-level name written by a function fails here until it is
    declared with its kind; a declared one nothing writes any more fails
    until its entry goes.
    """
    found = set(state_writes())

    assert found == set(STATE), (
        f"undeclared {sorted(found - set(STATE))}, stale {sorted(set(STATE) - found)}"
    )


@pytest.mark.infra
def test_what_outlives_a_run_is_declared_state() -> None:
    assert set(OUTLIVES) <= set(STATE), sorted(OUTLIVES - set(STATE))


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
    import port.scripts.run_cnaster as entry_point

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
    """One run of the entry point, then every declared name.

    Each is read before and after in a fresh interpreter, so what an earlier
    test left behind cannot hide a change: anything but `OUTLIVES` must be
    back at its import-time value, and under the default everything in
    `OUTLIVES` must still differ, so a release lands with its entry removed.
    """
    import json
    import subprocess
    import sys

    from port.sim.run_config import write_for_run

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
    import json
    import sys

    import matplotlib as mpl

    mpl.use("Agg")
    print(json.dumps(changed_by_a_run(sys.argv[1:])))
