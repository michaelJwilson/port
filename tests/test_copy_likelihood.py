"""`port.extensions.copy_likelihood` against pseudobulks drawn from its own model (#327).

The decoder claims that, with the path held, the pseudobulk likelihood the
EM fits identifies each state's integer `(A, B)`. So the referee is the
truth the counts were drawn from: a clone-sized pseudobulk under the
shifted NB/BB model, planted `(A, B)` including totals above `cnaster`'s 6,
decoded from a wrong start (`end2end` against the planted pairs). The
likelihood-ratio set is held to contain the truth, and the decode to be the
likelihood's own maximum over single-state moves (`analytic`).
"""

from __future__ import annotations

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


@pytest.mark.end2end
@pytest.mark.parametrize("shift", [True, False], ids=["shifted", "unshifted"])
def test_the_decode_recovers_every_planted_pair_from_a_wrong_start(shift: bool) -> None:
    """All six states exactly, `(4, 6)` and `(5, 4)` above cnaster's cap included."""
    from port.extensions.copy_likelihood import decode

    path, bulk = _draw(shift=shift)
    start = np.ones_like(PLANTED)

    decoded = decode(start, path, bulk, max_total_copy=12, neutral=0, shift=shift)

    np.testing.assert_array_equal(decoded.copies, PLANTED)
    assert decoded.passes <= 5


@pytest.mark.analytic
def test_the_decode_is_a_single_move_maximum_and_its_sets_hold_the_truth() -> None:
    """No single state's move raises the likelihood; each set contains the planted pair."""
    from port.extensions.copy_likelihood import candidates, decode, log_likelihood

    path, bulk = _draw()
    decoded = decode(
        np.ones_like(PLANTED), path, bulk, max_total_copy=12, neutral=0, shift=True
    )

    for k in range(1, len(PLANTED)):
        assert tuple(PLANTED[k]) in decoded.sets[k]

        for pair in candidates(12):
            trial = decoded.copies.copy()
            trial[k] = pair
            assert (
                log_likelihood(trial, path, bulk, shift=True)
                <= decoded.log_likelihood + 1e-9
            )


@pytest.mark.infra
def test_the_candidates_are_every_pair_under_the_cap() -> None:
    from port.extensions.copy_likelihood import candidates

    lattice = candidates(12)

    assert len(lattice) == 90
    assert lattice.sum(axis=1).max() == 12
    assert ((lattice.sum(axis=1) > 0) & (lattice.min(axis=1) >= 0)).all()
