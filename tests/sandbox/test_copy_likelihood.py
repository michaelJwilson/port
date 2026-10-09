"""`port.sandbox.extensions.shared_decode` against pseudobulks drawn from `copy_likelihood`'s model (#327).

Referees: the planted `(A, B)` including totals above `cnaster`'s 6 (`end2end`), and the
likelihood's own maximum over single-state moves (`analytic`).
"""

from __future__ import annotations

import numpy as np
import pytest
from port.extensions.copy_likelihood import (
    Pseudobulk,
    candidates,
    pair_rate_and_share,
    pseudobulk_log_pmf,
)
from port.sandbox.extensions.shared_decode import shared_decode

PLANTED = np.array([(1, 1), (2, 1), (1, 3), (2, 2), (4, 6), (5, 4)], dtype=np.int64)
"""`cnaster`'s convention, `p = A / (A + B)`; state 0 is the neutral one."""


OCCUPANCY = (700, 60, 60, 60, 60, 60)
"""A mostly normal genome, as a tumor's is, so the shift is realistic."""


def _draw(seed: int = 3, *, shift: bool = True) -> tuple[np.ndarray, Pseudobulk]:
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
