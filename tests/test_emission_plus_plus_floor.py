"""`sal`'s emission++ divergences around port's mixture start (#562, sal #1136, T- #632).

Rows from `sal_mixture.instance_of`: 200 at `(2.0, 0.3)`, 200 at deviance 0 from that
seed,
and 5 far rows.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sal.opt.emission_mixture import CountPairSeeding, plus_plus_start, seed_scores

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
    return CountPairSeeding(
        dispersion=SIZE, concentration=1_000.0, joint=False, trials=50.0
    )


def _refused(seed: int) -> bool:
    try:
        plus_plus_start(_rows(), 3, _seam(), np.random.default_rng(seed))
    except ValueError as error:
        return "non-negative" in str(error)
    return False


@pytest.mark.bug
def test_sals_emission_plus_plus_no_longer_refuses_a_round_off_negative_divergence() -> (
    None
):
    """0 of 20 generators raise "Probabilities are not non-negative" through sal's own seam (sal #1136)."""
    assert sum(_refused(seed) for seed in range(20)) == 0


@pytest.mark.oracle
def test_sals_scores_are_the_exact_divergence() -> None:
    """From a `(2.0, 0.3)` seed each near row's score is the 50-digit divergence, to 2e-14 absolute."""

    rows = _rows()
    candidates = np.arange(rows.shape[0], dtype=np.float64)
    near = slice(200, 400)
    raw = seed_scores(rows, _seam())(0.0, candidates)
    exact = [nb_divergence(y, 2.0, SIZE) for y in rows[near, 0]]

    assert (raw >= 0.0).all()
    np.testing.assert_allclose(raw[near], exact, rtol=0, atol=2e-14)
