"""#562: `sal`'s emission++ divergences floored at 0 around port's mixture start.

`run_cnaster_port --sal --hmm-start emission++...` seeds through
`sal.opt.emission_mixture.seed_scores` (public since sal #1236), whose negative-binomial Bregman
divergence is non-negative in exact arithmetic and about `-1.6e-15` in
float64 for a row a hair from a seed's mean. D-squared sampling handed those
to `rng.choice`, which refused them. Port floored them at 0 around
`sal_mixture.gmm_init` until sal #1136 floored them in the draw (T- #632).

The rows are the rate-space pairs `sal_mixture.instance_of` builds, seeded
through the same `CountPairSeeding` seam: 200 at `(2.0, 0.3)`; 200 whose
total is `2.0` to 1e-12 and whose B rate, 0.76034, puts their beta-binomial
deviance from a `(2.0, 0.3)` seed at its clamp of 0; and 5 far rows, so the
draw has positive mass to normalize.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.exact_densities import nb_divergence

SIZE = 10.0
SHARE = 0.7603373084367014
"""The B rate whose beta-binomial deviance from a `(2.0, 0.3)` seed is 0 at trials 50."""


def _rows() -> np.ndarray:
    z = np.random.default_rng(1).standard_normal(200)
    return np.vstack(
        [
            np.tile([2.0, 0.3], (200, 1)),
            np.column_stack([2.0 * (1.0 + 1e-12 * z), np.full(200, SHARE)]),
            np.column_stack([np.full(5, 20.0), np.full(5, 0.3)]),
        ]
    )


def _seam() -> Any:
    from sal.opt.emission_mixture import CountPairSeeding

    return CountPairSeeding(
        dispersion=SIZE, concentration=1_000.0, joint=False, trials=50.0
    )


def _refused(seed: int) -> bool:
    from sal.opt.emission_mixture import plus_plus_start

    try:
        plus_plus_start(_rows(), 3, _seam(), np.random.default_rng(seed))
    except ValueError as error:
        return "non-negative" in str(error)
    return False


@pytest.mark.bug
def test_sals_emission_plus_plus_no_longer_refuses_a_round_off_negative_divergence() -> (
    None
):
    """0 of 20 generators raise "Probabilities are not non-negative" through sal's own seam.

    11 of 20 did at sal `3ad4b04`; sal #1136 (`aa699a59`) floors round-off
    negatives in the D-squared draw. T- #632 PR B retires the workaround.
    """
    assert sum(_refused(seed) for seed in range(20)) == 0


@pytest.mark.oracle
def test_sals_scores_are_the_exact_divergence() -> None:
    """From a `(2.0, 0.3)` seed, each near row's score is the 50-digit divergence, to 2e-14 absolute.

    The exact negative-binomial divergence there is below 1e-24, and sal's
    negative-binomial part of each score lies within 1.6e-15 of it. At sal
    `3ad4b04` the beta-binomial deviance was clamped to 0 and 102 of the 200
    near rows scored as low as `-1.6e-15`. Since sal #1136 the deviance at
    concentration 1000 takes the Stirling-difference path above shape 100 and
    reads `+1.43e-14` where it is 0, so no near row is negative and the near
    scores are 1.28e-14 to 1.59e-14.
    """
    from sal.opt.emission_mixture import seed_scores

    rows = _rows()
    candidates = np.arange(rows.shape[0], dtype=np.float64)
    near = slice(200, 400)
    raw = seed_scores(rows, _seam())(0.0, candidates)
    exact = [nb_divergence(y, 2.0, SIZE) for y in rows[near, 0]]

    assert (raw >= 0.0).all()
    np.testing.assert_allclose(raw[near], exact, rtol=0, atol=2e-14)


def _port_floor(score: Any) -> Any:
    """The `_seed_scores` patch T- #632 retired: each divergence floored at 0."""

    def nonnegative(seed: float, candidates: np.ndarray) -> np.ndarray:
        return np.asarray(np.maximum(score(seed, candidates), 0.0))

    return nonnegative


@pytest.mark.patch
def test_sals_draw_is_the_retired_floors_draw() -> None:
    """sal's own D-squared draw against the same draw under port's retired floor: equal rows on 20 generators.

    sal #1136 floors each divergence at 0 before the running minimum, where
    port's patch floored it, so retiring the patch moves no seed.
    """
    from sal.opt.emission_mixture import seed_scores
    from sal.opt.mixture import emission_mixture_plus_plus

    rows = _rows()
    indices = np.arange(rows.shape[0], dtype=np.float64)

    for seed in range(20):
        sals = emission_mixture_plus_plus(
            indices, 3, seed_scores(rows, _seam()), np.random.default_rng(seed)
        )
        ported = emission_mixture_plus_plus(
            indices,
            3,
            _port_floor(seed_scores(rows, _seam())),
            np.random.default_rng(seed),
        )
        np.testing.assert_array_equal(sals, ported)
