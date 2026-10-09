"""`port.qa.benchmark.staged`: a kept root is staged once, so a rerun resumes CalicoST (#532)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from port.qa.benchmark import STAGED, staged


@pytest.mark.smoke
def test_a_staged_root_is_reused_not_restaged(tmp_path: Path) -> None:
    """With the marker present, `staged` returns what it records and writes nothing.

    The sample's path does not exist, so staging it again would raise.
    """
    config = tmp_path / "config.yaml"
    (tmp_path / STAGED).write_text(json.dumps({"config": str(config), "joint": True}))
    before = sorted(tmp_path.iterdir())

    sample = SimpleNamespace(path=tmp_path / "absent")

    assert staged(sample, tmp_path) == (config, True)
    assert sorted(tmp_path.iterdir()) == before
