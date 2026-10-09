"""`port.extensions.copy_likelihood` against pseudobulks drawn from its own model (#327).

Referees: the planted `(A, B)` including totals above `cnaster`'s 6 (`end2end`), and the
likelihood's own maximum over single-state moves (`analytic`).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from port.extensions.copy_likelihood import (
    PURITY_GRID,
    _monotone,
    candidates,
)
from port.patch import integer_copy
from port.scripts.run_cnaster import main
from port.sim.run_config import isolated_run, write_for_run
from port.sim.truth import critical_instance


@pytest.mark.analytic
def test_the_candidates_are_every_pair_under_the_cap() -> None:
    lattice = candidates(12)

    assert len(lattice) == 90
    assert lattice.sum(axis=1).max() == 12
    assert ((lattice.sum(axis=1) > 0) & (lattice.min(axis=1) >= 0)).all()


def _entry_point_run(
    tmp_path: Path, argv: tuple[str, ...]
) -> tuple[Any, list[Any], Any]:
    """`run_cnaster_port` on the critical instance with `(1, 1)` and `(1, 2)` planted: segment table, decodes, output dir."""

    mpl.use("Agg")
    truth = critical_instance(copy_lattice=True)
    written, config = write_for_run(
        truth, tmp_path, max_iter_outer=1, max_iter=3, n_states=2
    )
    with isolated_run(), warnings.catch_warnings(), integer_copy.recorded() as decodes:
        warnings.simplefilter("ignore")
        assert main([*argv, str(config)]) == 0

    table = next((written.root / "output").rglob("cnv_seglevel.tsv"))
    return pd.read_csv(table, sep="\t"), decodes, table.parent


@pytest.mark.end2end
def test_the_entry_point_decodes_the_planted_pair_through_the_likelihood(
    tmp_path: Path,
) -> None:
    """The default `lattice` decode (#370) writes the planted pair per clone-bin, at fitted fractions 1 (#371)."""

    copies, seen, output = _entry_point_run(tmp_path, ())

    assert len(seen) == 1, "the decode ran once for the run, not once per clone"
    decoded = seen[0]
    pairs = {
        tuple(sorted(pair))
        for column in ("clone0", "clone1")
        for pair in zip(copies[f"{column} A"], copies[f"{column} B"], strict=True)
    }

    assert pairs == {(1, 1), (1, 2)}

    for clone, column in enumerate(("clone0", "clone1")):
        written = copies[[f"{column} A", f"{column} B"]].to_numpy()
        np.testing.assert_array_equal(written, decoded.pairs[clone])

    fitted = pd.read_csv(output / "copy_decode.tsv", sep="\t")

    assert fitted["tumour_fraction"].to_list() == pytest.approx([1.0, 1.0])


@pytest.mark.analytic
def test_the_fraction_step_never_goes_uphill() -> None:
    """The bounded fraction step never returns a value above its start's, on a minimum at the endpoint (#371)."""

    def objective(x: float) -> float:
        return min((x - 0.2) ** 2 + 0.05, 1.0 - x)

    for current in (0.3, 0.6, 1.0):
        chosen = _monotone(objective, current, (0.05, 1.0), PURITY_GRID)

        assert objective(chosen) <= objective(current)
        assert chosen == 1.0
