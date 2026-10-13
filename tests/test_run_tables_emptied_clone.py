"""T- #855: `run_tables` reads a run whose RDR+BAF stage fits a clone the final assignment no longer holds."""

from __future__ import annotations

import dataclasses
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from tests import ROOT

MANIFEST = Path("sim/manifests/dev_tree_1s_dense.toml")


@pytest.mark.release
@pytest.mark.patch
def test_a_clone_emptied_before_the_reindex_reads_as_the_npz(tmp_path: Path) -> None:
    """dev_tree_1s_dense r0 (`33e3471e`): the stage fits 6 clones, the final assignment holds 5.

    `run_tables` from `cnamaste.h5` equals `run_tables` from `cnaster`'s
    tables and final npz in every field, and `score_sample` agrees on every
    metric: the parity `tests/test_run_plots.py` checks on `9ec90dc2`, where
    no clone empties.
    """
    from port.extensions import cnamaste
    from port.extensions.cnamaste import FILE
    from port.qa.audit import drawn_config, run_tables, score_sample
    from port.scripts.run_cnaster import main
    from port.studies import stage

    here = Path.cwd()
    os.chdir(ROOT)
    try:
        member = next(stage.members(MANIFEST, tmp_path / "sim", n=1))
        assert member.hash == "33e3471e"
        main(["--sal", "--no-plots", str(drawn_config(member.sample, tmp_path / "run", {}))])  # fmt: skip
    finally:
        os.chdir(here)

    output = tmp_path / "run" / "output"
    run = next(output.glob("clone*"))
    fitted, _ = cnamaste.read(output / FILE, "rdrbaf")
    final, _ = cnamaste.read(output / FILE, "clone_assignment")
    assert fitted["pred_cnv"].shape[1] == 6
    assert np.unique(final["assignment"]).size == 5

    tables = tmp_path / "tables"
    shutil.copytree(output, tables, ignore=shutil.ignore_patterns(FILE, "plots"))
    _, _, fit = run_tables(run)
    _, _, their_fit = run_tables(next(tables.glob("clone*")))
    assert fit.keys() == their_fit.keys()
    for key in fit:
        np.testing.assert_array_equal(
            fit[key], np.asarray(their_fit[key]).reshape(fit[key].shape)
        )

    ours = dataclasses.asdict(score_sample(member.sample, output, "sal", 0.0))
    theirs = dataclasses.asdict(score_sample(member.sample, tables, "sal", 0.0))
    assert ours.keys() == theirs.keys()
    for key in ours:
        assert repr(ours[key]) == repr(theirs[key]), key


@pytest.mark.critical
@pytest.mark.analytic
def test_the_survivors_are_the_one_increasing_choice() -> None:
    """`surviving` on `33e3471e`'s groups keeps fitted clone 1 for merged clone 1, and refuses an ambiguous merge.

    `merge_by_minspots` labels the group of its `i`-th surviving clone `i`:
    clone 3 (50 spots) failed the floor and joined clone 1, so merged clone 1
    holds fitted clones 1 and 3 and keeps clone 1's column.
    """
    from port.qa.audit import surviving

    fitted = np.array([0, 1, 2, 3, 4, 5, 1, 3])
    merged = np.array([0, 1, 2, 1, 3, 4, 1, 1])
    assert surviving(fitted, merged) == [0, 1, 2, 4, 5]
    assert surviving(np.array([2, 0, 1]), np.array([2, 0, 1])) == [0, 1, 2]
    with pytest.raises(ValueError, match="3 choices"):
        surviving(np.array([0, 1, 2, 3]), np.array([0, 1, 0, 1]))
