"""`port.sim.fixtures.stage` writes into its target, never into the sample (#492)."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.infra
def test_a_sample_loaded_by_absolute_path_stages_into_the_target(
    tmp_path: Path,
) -> None:
    from port.sim.fixtures import EASY, SIM_ROOT, load_simulated, stage

    source = SIM_ROOT / EASY
    committed = sorted(p for p in source.rglob("*") if p.is_file())
    sample = load_simulated(str(source), Path("/"))

    target = stage(sample, tmp_path / "inputs")

    assert target.parent == tmp_path / "inputs"
    assert all(not p.is_symlink() for p in committed)
    assert sorted(p for p in source.rglob("*") if p.is_file()) == committed
