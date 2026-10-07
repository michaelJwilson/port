"""Rising factorials and dense emission scored on the distinct counts (#702).

The referee is the per-element evaluation each one replaces, bitwise: a
value taken on the distinct counts and gathered is the same arithmetic at
the same arguments, so `np.array_equal` is the bar (`patch`).
"""

from __future__ import annotations

import numpy as np
import pytest
from port.patch.hmm_nophasing.bb_logpmf import rises_on_distinct
from sal.emissions.rising import log_rising


def _counts(
    rng: np.random.Generator, n_bins: int, n_distinct: int, top: int
) -> np.ndarray:
    pool = rng.choice(np.arange(top + 1), n_distinct, replace=False).astype(float)
    return rng.choice(pool, n_bins)


@pytest.mark.patch
@pytest.mark.parametrize(
    "shape",
    [(25, 1), (4, 1), (1,), ()],
    ids=["states", "few-states", "one", "scalar"],
)
def test_distinct_counts_give_the_per_bin_rises_bitwise(shape: tuple[int, ...]) -> None:
    """A shape constant along the bins, below and above `log_rising`'s series threshold (10): equal to it."""
    rng = np.random.default_rng(702)
    m = _counts(rng, 2_829, 594, 2_797)
    x = np.asarray(rng.uniform(0.5, 4e3, shape) if shape else 37.5)

    assert np.array_equal(rises_on_distinct(x, m), log_rising(x, m))


@pytest.mark.patch
def test_a_shape_that_varies_along_the_bins_takes_rises_unchanged() -> None:
    """Per-bin shapes, a 2-D count array and all-distinct counts: `log_rising` itself."""
    rng = np.random.default_rng(703)
    m = _counts(rng, 600, 50, 400)
    per_bin = rng.uniform(1.0, 100.0, (3, m.size))
    grid = m.reshape(20, 30)

    assert np.array_equal(rises_on_distinct(per_bin, m), log_rising(per_bin, m))
    assert np.array_equal(
        rises_on_distinct(np.asarray(5.0), grid), log_rising(np.asarray(5.0), grid)
    )
    assert np.array_equal(
        rises_on_distinct(np.ones((2, 1)), np.arange(10.0)),
        log_rising(np.ones((2, 1)), np.arange(10.0)),
    )


@pytest.mark.patch
def test_the_dense_emission_with_a_held_cache_is_the_emission_without(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two iterations on the same counts: bitwise equal, and one `torch.unique` per array, not per call."""
    import torch
    from port.patch.hmm_nophasing.dense_emission import bb_states, nb_states

    rng = np.random.default_rng(704)
    obs = rng.negative_binomial(5, 0.1, 3_000).astype(float)
    exposure = rng.uniform(5.0, 80.0, obs.size)
    successes = rng.binomial(40, 0.3, obs.size).astype(float)
    trials = np.full(obs.size, 40.0)
    mu, alpha = np.array([0.6, 1.0, 1.5]), np.array([0.05, 0.08, 0.1])
    p, tau = np.array([0.2, 0.5, 0.7]), np.array([30.0, 50.0, 80.0])

    calls = {"n": 0}
    unique = torch.unique

    def counting(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        return unique(*args, **kwargs)

    monkeypatch.setattr(torch, "unique", counting)
    plain = [nb_states(obs, exposure, mu, alpha), bb_states(successes, trials, p, tau)]
    uncached = calls["n"]

    held: dict[object, object] = {}
    calls["n"] = 0
    for _ in range(2):
        cached = [
            nb_states(obs, exposure, mu, alpha, distinct=held),
            bb_states(successes, trials, p, tau, distinct=held),
        ]
        for a, b in zip(plain, cached, strict=True):
            assert np.array_equal(a, b)

    assert uncached > 0
    assert calls["n"] == uncached, (calls["n"], uncached)
