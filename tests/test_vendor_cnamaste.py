"""`cnamaste/` is `cnaster` at the lock's pin, renamed (#392).

Two referees. The pin itself, for every module no stage has rewritten: the
copy, renamed back, is the installed file byte for byte. And the installed
`cnaster`, for the whole: `run_cnamaste` writes what `run_cnaster` writes on
the round trip's instance, file for file.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from tests.vendor_cnamaste import (
    PACKAGE,
    ROOT,
    installed,
    installed_commit,
    manifest,
    pinned_files,
)


@pytest.mark.infra
def test_the_copy_is_the_pin_renamed() -> None:
    """Same commit as the lock, same files, and each unfolded file exact.

    The lock, the installed distribution and `VENDOR.toml` name one commit;
    the copy holds the pin's in-scope modules less the dropped ones; and
    every module no stage has folded reads, renamed back, as the pin's bytes.
    """
    vendor = manifest()
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    (cnaster,) = (p for p in lock["package"] if p["name"] == "cnaster")

    assert cnaster["source"]["git"].endswith("#" + vendor.commit)
    assert installed_commit() == vendor.commit

    pinned = pinned_files(vendor)
    copied = sorted(
        p.relative_to(PACKAGE).as_posix()
        for p in PACKAGE.rglob("*.py")
        if "__pycache__" not in p.parts
    )
    assert copied == sorted([vendor.copied(f) for f in pinned] + list(vendor.added))
    assert set(vendor.folded) <= set(copied)

    source = installed()
    for relative in pinned:
        name = vendor.copied(relative)
        text = (PACKAGE / name).read_bytes().decode()
        assert "cnaster" not in text, name
        if name not in vendor.folded:
            assert vendor.restore(text).encode() == (source / relative).read_bytes(), (
                name
            )


RUN = """
import sys, warnings

import matplotlib
import numpy as np

matplotlib.use("Agg")
warnings.simplefilter("ignore")
np.random.seed({seed})
from {module} import {function}

{function}(sys.argv[1])
"""
"""One whole run in a fresh interpreter, seeded as `tests.run_config.isolated_run` seeds."""


def _run(module: str, function: str, config: str) -> None:
    """Run one entry point in its own process.

    Its own process so that the two runs share no state -- neither the
    caches of the modules both import nor what an earlier test left -- and so
    that the test's peak memory is one run's (5.5 GB) rather than two (8.0
    GB), which is what three concurrent workers can hold on 15 GB.
    """
    from tests.run_config import ENTRY_POINT_SEED

    code = RUN.format(seed=ENTRY_POINT_SEED, module=module, function=function)
    environment = dict(os.environ, SOURCE_DATE_EPOCH="0")
    subprocess.run([sys.executable, "-c", code, config], check=True, env=environment)


@pytest.mark.patch
def test_run_cnamaste_writes_what_run_cnaster_writes(tmp_path: Path) -> None:
    """Every file of the round trip's run, byte for byte, through both entry points.

    Two clones of 500 spots over 40 bins (`tests/test_run_cnaster_round_trip.py`),
    seeded identically. `SOURCE_DATE_EPOCH` fixes the figures' creation date,
    which the two runs would otherwise differ in by construction.
    """
    from tests.fixtures import core_inference_truth
    from tests.run_config import write_run_cnaster_config
    from tests.tmp_inputs import write_tmp_inputs
    from tests.unsegment import unsegment

    truth = core_inference_truth(
        n_clones=2, n_states=3, lattice=(25, 40), n_obs=40, n_segments=3, seed=11
    )
    written = write_tmp_inputs(
        truth, unsegment(truth, flip_every=0, unassigned_genes=0), tmp_path
    )
    config = str(write_run_cnaster_config(written, truth, max_iter_outer=1, max_iter=3))
    output = written.root / "output"

    _run("cnaster.scripts.run_cnaster", "run_cnaster", config)
    reference = Path(shutil.move(output, written.root / "reference"))
    _run("cnamaste.scripts.run_cnamaste", "run_cnamaste", config)

    expected = _files(reference)
    produced = _files(output)
    assert expected, "the reference run wrote nothing"
    assert produced.keys() == expected.keys()
    differ = [name for name, data in produced.items() if data != expected[name]]
    assert not differ, f"{len(differ)} of {len(expected)} files differ: {differ[:5]}"


def _files(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
