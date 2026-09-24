"""`port.extensions.copy_likelihood` against pseudobulks drawn from its own model (#327).

The shared decode claims that, with the path held, the pseudobulk
likelihood the EM fits identifies each state's integer `(A, B)`. So the
referee is the truth the counts were drawn from: a clone-sized pseudobulk
under the shifted NB/BB model, planted `(A, B)` including totals above
`cnaster`'s 6 (`end2end` against the planted pairs). The decode is held to
be the likelihood's own maximum over single-state moves, and the tempered
E-step to reach Viterbi's path as its temperature falls (`analytic`).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

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
        total_bb_rd=trials,
        log_lambda=log_lambda,
        alpha=alpha,
        tau=tau,
    )
    return path, bulk


def _offset(path: np.ndarray, bulk: Pseudobulk) -> float:
    """The planted clone's shift, `log Z_c`, as the draw applied it."""
    total = PLANTED.sum(axis=1)
    return float(np.logaddexp.reduce(np.log(total / 2.0)[path] + bulk.log_lambda))


@pytest.mark.end2end
@pytest.mark.parametrize("shift", [True, False], ids=["shifted", "unshifted"])
def test_the_shared_decode_recovers_every_planted_pair(shift: bool) -> None:
    """All six states exactly, `(4, 6)` and `(5, 4)` above cnaster's cap included."""
    from port.extensions.copy_likelihood import SHARED, fit_copies

    path, bulk = _draw(shift=shift)
    fitted = fit_copies(
        [(path, bulk, _offset(path, bulk) if shift else 0.0)],
        SHARED,
        n_states=len(PLANTED),
        normal=0,
        normal_clone=0,
        max_total_copy=12,
        zero_normal=False,
    )

    np.testing.assert_array_equal(fitted.states, PLANTED)
    np.testing.assert_array_equal(fitted.pairs[0], PLANTED[path])


@pytest.mark.analytic
def test_the_shared_decode_is_each_states_likelihood_maximum() -> None:
    """With the path held, no other pair for any one state raises the likelihood."""
    from port.extensions.copy_likelihood import (
        SHARED,
        _emission,
        _parameters,
        candidates,
        fit_copies,
    )

    path, bulk = _draw()
    shift = _offset(path, bulk)
    fitted = fit_copies(
        [(path, bulk, shift)],
        SHARED,
        n_states=len(PLANTED),
        normal=0,
        normal_clone=0,
        max_total_copy=12,
        zero_normal=False,
    )

    def likelihood(copies: np.ndarray) -> float:
        log_mu, p = _parameters(copies)
        bins = np.arange(path.size)
        return float(np.sum(_emission(log_mu[path] - shift, p[path], bulk, bins)))

    best = likelihood(fitted.states)
    assert best == pytest.approx(fitted.log_likelihood, rel=1e-12)

    for k in range(1, len(PLANTED)):
        for pair in candidates(12):
            trial = fitted.states.copy()
            trial[k] = pair
            assert likelihood(trial) <= best + 1e-9


@pytest.mark.analytic
def test_tempering_to_a_low_temperature_is_viterbi() -> None:
    """At `T -> 0` each bin's responsibility is one-hot on Viterbi's state."""
    from port.extensions.copy_likelihood import (
        _forward_backward,
        _log_emissions,
        _viterbi,
        candidates,
    )

    path, bulk = _draw()
    lattice = candidates(6)
    n = len(lattice)
    transmat = np.log(np.full((n, n), 1e-4 / (n - 1)) + np.eye(n) * (1 - 1e-4))
    start = np.full(n, -np.log(n))
    emission = _log_emissions(lattice, _offset(path, bulk), 1.0, bulk)
    lengths = np.array([path.size])

    best, _ = _viterbi(emission, transmat, start, lengths)
    cold = _forward_backward(emission, transmat, start, lengths, 1e-3)

    np.testing.assert_array_equal(np.argmax(cold, axis=0), best)
    assert cold.max(axis=0).min() > 1.0 - 1e-9


@pytest.mark.infra
def test_the_candidates_are_every_pair_under_the_cap() -> None:
    from port.extensions.copy_likelihood import candidates

    lattice = candidates(12)

    assert len(lattice) == 90
    assert lattice.sum(axis=1).max() == 12
    assert ((lattice.sum(axis=1) > 0) & (lattice.min(axis=1) >= 0)).all()


@pytest.mark.end2end
def test_the_entry_point_decodes_the_planted_pair_through_the_likelihood(
    tmp_path: Path,
) -> None:
    """`run_cnaster_port` on a two-state copy lattice, decoding by likelihood.

    The critical instance with `(1, 1)` and `(1, 2)` planted: every altered
    clone-bin of the tumor clone is written as the planted pair, phase folded,
    and the likelihood decode -- the only one supported (#362) -- ran once
    per clone.
    """
    import warnings

    import matplotlib as mpl
    import pandas as pd
    from port.patch import integer_copy
    from port.scripts.run_cnaster import main

    from tests.fixtures import critical_instance
    from tests.run_config import isolated_run, write_run_cnaster_config
    from tests.tmp_inputs import write_tmp_inputs
    from tests.unsegment import unsegment

    mpl.use("Agg")
    truth = critical_instance(copy_lattice=True)
    written = write_tmp_inputs(
        truth, unsegment(truth, flip_every=0, unassigned_genes=0), tmp_path
    )
    config = write_run_cnaster_config(
        written, truth, max_iter_outer=1, max_iter=3, n_states=2
    )
    integer_copy.DECODED.clear()

    with isolated_run(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert main([str(config)]) == 0

    seen = list(integer_copy.DECODED)

    assert seen, "the likelihood refinement never ran"

    copies = pd.read_csv(
        next((written.root / "output").rglob("cnv_seglevel.tsv")), sep="\t"
    )
    pairs = {
        tuple(sorted(pair))
        for column in ("clone0", "clone1")
        for pair in zip(copies[f"{column} A"], copies[f"{column} B"], strict=True)
    }

    assert pairs == {(1, 1), (1, 2)}


@pytest.mark.analytic
def test_the_poisson_flag_is_the_zero_dispersion_limit() -> None:
    """`alpha = 0`, `tau = inf` agree with NB and BB at `alpha = 1e-9`, `tau = 1e9`."""
    from dataclasses import replace

    from port.extensions.copy_likelihood import _emission, _parameters

    path, bulk = _draw()
    log_mu, p = _parameters(PLANTED)
    bins = np.arange(path.size)
    exact = _emission(log_mu[path], p[path], replace(bulk, alpha=0.0, tau=np.inf), bins)
    near = _emission(log_mu[path], p[path], replace(bulk, alpha=1e-9, tau=1e9), bins)

    np.testing.assert_allclose(exact, near, rtol=1e-5)
