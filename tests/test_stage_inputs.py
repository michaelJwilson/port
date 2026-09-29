"""`tests.sim_fixtures.stage` writes into its target, never into the sample (#492).

A CalicoST sample loaded by absolute path carried the path as its name, and
`into / name` resolved to the sample itself: staging unlinked every committed
input and linked it to itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.infra
def test_a_sample_loaded_by_absolute_path_stages_into_the_target(
    tmp_path: Path,
) -> None:
    from tests.sim_fixtures import EASY, SIM_ROOT, load_simulated, stage

    source = SIM_ROOT / EASY
    committed = sorted(p for p in source.rglob("*") if p.is_file())
    sample = load_simulated(str(source), Path("/"))

    target = stage(sample, tmp_path / "inputs")

    assert target.parent == tmp_path / "inputs"
    assert all(not p.is_symlink() for p in committed)
    assert sorted(p for p in source.rglob("*") if p.is_file()) == committed
