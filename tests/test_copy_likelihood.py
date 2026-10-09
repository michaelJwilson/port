"""`port.extensions.copy_likelihood` against pseudobulks drawn from its own model (#327).

Referees: the planted `(A, B)` including totals above `cnaster`'s 6 (`end2end`), and the
likelihood's own maximum over single-state moves (`analytic`).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from port.extensions.copy_likelihood import Pseudobulk

PLANTED = np.array([(1, 1), (2, 1), (1, 3), (2, 2), (4, 6), (5, 4)], dtype=np.int64)
"""`cnaster`'s convention, `p = A / (A + B)`; state 0 is the neutral one."""

OCCUPANCY = (700, 60, 60, 60, 60, 60)
"""A mostly normal genome, as a tumor's is, so the shift is realistic."""


def _draw(seed: int = 3, *, shift: bool = True) -> tuple[np.ndarray, Pseudobulk]:
    from port.extensions.copy_likelihood import Pseudobulk

    rng = np.random.default_rng(seed)
    path = np.repeat(np.arange(len(OCCUPANCY)), OCCUPANCY)
    rng.shuffle(path)

    base = rng.uniform(200.0, 600.0, path.size)
    log_lambda = np.log(base / base.sum())
    total = PLANTED.sum(axis=1)
    log_mu = np.log(total / 2.0)
    offset = np.logaddexp.reduce(log_mu[path] + log_lambda) if shift else 0.0
    mean = base * np.exp(log_mu[path] - offset)

    alpha, tau = 0.01, 300.0
    size = 1.0 / alpha
    counts_nb = rng.negative_binomial(size, size / (size + mean)).astype(float)

    p = PLANTED[:, 0] / total
    trials = rng.integers(150, 250, path.size).astype(float)
    success = rng.beta(p[path] * tau, (1 - p[path]) * tau)
    counts_bb = rng.binomial(trials.astype(int), success).astype(float)

    bulk = Pseudobulk(
        counts_nb=counts_nb,
        base_nb_mean=base,
        counts_bb=counts_bb,
        total_bb_RD=trials,
        normal_log_lambda=log_lambda,
        dispersion=alpha,
        taus=tau,
    )
    return path, bulk


def _offset(path: np.ndarray, bulk: Pseudobulk) -> float:
    """The planted clone's shift, `log Z_c`, as the draw applied it."""
    total = PLANTED.sum(axis=1)
    return float(
        np.logaddexp.reduce(np.log(total / 2.0)[path] + bulk.normal_log_lambda)
    )


@pytest.mark.end2end
@pytest.mark.parametrize("shift", [True, False], ids=["shifted", "unshifted"])
def test_the_shared_decode_recovers_every_planted_pair(shift: bool) -> None:
    """All six states exactly, `(4, 6)` and `(5, 4)` above cnaster's cap included."""
    from port.sandbox.extensions.shared_decode import shared_decode

    path, bulk = _draw(shift=shift)
    fitted = shared_decode(
        [(path, bulk, _offset(path, bulk) if shift else 0.0)],
        n_states=len(PLANTED),
        normal=0,
        max_total_copy=12,
    )

    np.testing.assert_array_equal(fitted.states, PLANTED)
    np.testing.assert_array_equal(fitted.pairs[0], PLANTED[path])


@pytest.mark.analytic
def test_the_shared_decode_is_each_states_likelihood_maximum() -> None:
    """With the path held, no other pair for any one state raises the likelihood."""
    from port.extensions.copy_likelihood import (
        candidates,
        pair_rate_and_share,
        pseudobulk_log_pmf,
    )
    from port.sandbox.extensions.shared_decode import shared_decode

    path, bulk = _draw()
    shift = _offset(path, bulk)
    fitted = shared_decode(
        [(path, bulk, shift)],
        n_states=len(PLANTED),
        normal=0,
        max_total_copy=12,
    )

    def likelihood(copies: np.ndarray) -> float:
        log_mu, p = pair_rate_and_share(copies)
        bins = np.arange(path.size)
        return float(
            np.sum(pseudobulk_log_pmf(log_mu[path] - shift, p[path], bulk, bins))
        )

    best = likelihood(fitted.states)
    assert best == pytest.approx(fitted.log_likelihood, rel=1e-12)

    for k in range(1, len(PLANTED)):
        for pair in candidates(12):
            trial = fitted.states.copy()
            trial[k] = pair
            assert likelihood(trial) <= best + 1e-9


@pytest.mark.infra
def test_the_candidates_are_every_pair_under_the_cap() -> None:
    from port.extensions.copy_likelihood import candidates

    lattice = candidates(12)

    assert len(lattice) == 90
    assert lattice.sum(axis=1).max() == 12
    assert ((lattice.sum(axis=1) > 0) & (lattice.min(axis=1) >= 0)).all()


def _entry_point_run(
    tmp_path: Path, argv: tuple[str, ...]
) -> tuple[Any, list[Any], Any]:
    """`run_cnaster_port` on the critical instance with `(1, 1)` and `(1, 2)` planted:
    segment table, decodes, output dir.
    """
    import warnings

    import matplotlib as mpl
    import pandas as pd
    from port.patch import integer_copy
    from port.scripts.run_cnaster import main
    from port.sim.run_config import isolated_run, write_for_run
    from port.sim.truth import critical_instance

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
    """The default `lattice` decode (#370) writes the planted pair per clone-bin, at fitted
    fractions 1 (#371).
    """
    import pandas as pd

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
    """The bounded fraction step never returns a value above its start's, on a minimum at
    the endpoint (#371).
    """
    from port.extensions.copy_likelihood import PURITY_GRID, _monotone

    def objective(x: float) -> float:
        return min((x - 0.2) ** 2 + 0.05, 1.0 - x)

    for current in (0.3, 0.6, 1.0):
        chosen = _monotone(objective, current, (0.05, 1.0), PURITY_GRID)

        assert objective(chosen) <= objective(current)
        assert chosen == 1.0
