"""Module-level state `port` writes after import, declared in `STATE` and held to a run
(#517 E2).

Kinds: `switch` (none remain), `run`, `cache`, `rebind`. `OUTLIVES` may only shrink.
`infra` for the declaration, `smoke` for the run.
"""

from __future__ import annotations

import copy
import importlib
import pickle
from pathlib import Path
from typing import Any, Literal

import pytest

from tests import ROOT
from tests.source_graph import state_writes

Kind = Literal["switch", "run", "cache", "rebind"]

STATE: dict[str, Kind] = {
    # NB studies run by hand, reached from no entry point (T- #673 G5).
    "port.extensions.label_solver.sweep_for": "rebind",
    "port.patch.hmrf.core_inference.UPSTREAM": "rebind",
    "port.patch.normal_spot.determine_normal_candidates": "rebind",
    "port.studies.clone_label_arms.HELD": "run",
    "port.studies.clone_label_arms.potts_graph": "rebind",
    "port.studies.copy_start_arms._CALLS": "run",
    "port.studies.copy_state_stream._WARM": "cache",
    "port.studies.potts_stream._GRAPHS": "cache",
    # NB `audit_truth` restores the name in its `finally` (T- #673 G3).
    "cnaster.scripts.run_cnaster.determine_normal_candidates": "rebind",
    "cnaster.hmm_initialize.GaussianMixture": "rebind",
    # NB #735: `port.qa.stage` wraps one `run_core_inference` call and restores it.
    "cnaster.hmrf.pipeline_clone_assignment": "rebind",
    "port.extensions.copy_likelihood._FITS": "run",
    "port.extensions.cnamaste._ACTIVE": "run",
    "port.extensions.run_record._HELD": "run",
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
"""Every name `port` writes after import, by kind; no `switch` since #517 steps 1-2."""

OUTLIVES: frozenset[str] = frozenset()
"""What a whole run leaves changed: nothing since #517 step 1."""


@pytest.mark.infra
def test_every_name_written_after_import_is_declared() -> None:
    """The source's writes match `STATE`, both ways."""
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
    """After one run, every declared name but `OUTLIVES` is back at its import-time
    value.
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
