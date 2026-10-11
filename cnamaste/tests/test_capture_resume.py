"""port's `scripts/capture_cnamaste.py --from-stage --base`: what it refuses before running anything.

The resume itself is checked by re-capturing easy from `05_baf` against its committed file, stage by
stage, in the pull request that added it; that is a capture (minutes), not a test. These are the
refusals, each a subprocess (the script patches numba's cache on import), none reaching the run.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT.parent / "scripts" / "capture_cnamaste.py"
BASE = PROJECT / "tests" / "data" / "sim_2d4ce9a9.hdf5"


def capture(tmp_path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--work", str(tmp_path / "work"), "--out", str(tmp_path / "out.hdf5"), *arguments],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "NUMBA_CACHE_DIR": str(tmp_path / "numba")},
    )


def test_from_stage_needs_a_base(tmp_path: Path) -> None:
    done = capture(tmp_path, "--from-stage", "05_baf/run_core_inference")
    assert done.returncode == 2
    assert "--from-stage and --base go together" in done.stderr


def test_a_mismatched_base_is_refused(tmp_path: Path) -> None:
    """easy's file, named as another sample's and resumed from a stage it never recorded: both reasons, nothing run."""
    done = capture(tmp_path, "--base", str(BASE), "--from-stage", "05_baf/nope", "--hash", "ffffffff")
    assert done.returncode == 1
    assert "--base refused" in done.stderr
    assert "fixture hash 2d4ce9a9, not ffffffff" in done.stderr
    assert "05_baf/nope is not a stage it recorded" in done.stderr
    assert not (tmp_path / "work").exists()
    assert not (tmp_path / "out.hdf5").exists()
