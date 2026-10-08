"""Each decoded segment's likelihood set, against analytic cases and a brute-force refit (#705)."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pytest


class _Decode(NamedTuple):
    states: np.ndarray
    paths: list[np.ndarray]
    shifts: np.ndarray
    purity: np.ndarray
    dispersion: float
    taus: float


def _clone(pairs: np.ndarray, depth: float, rho: float = 1.0) -> tuple[Any, Any]:
    """A pseudobulk whose counts are their expectation under `pairs` at `rho`."""
    from port.extensions.copy_likelihood import Pseudobulk, pair_rate_and_share

    log_mu, p = pair_rate_and_share(pairs, rho)
    base = np.full(pairs.shape[0], depth)
    total = np.full(pairs.shape[0], depth / 4)
    bulk = Pseudobulk(
        counts_nb=np.round(base * np.exp(log_mu)),
        base_nb_mean=base,
        counts_bb=np.round(total * p),
        total_bb_RD=total,
        normal_log_lambda=np.log(np.full(pairs.shape[0], 1.0 / pairs.shape[0])),
        dispersion=1e-4,
        taus=1e4,
    )
    return bulk, pairs


def _decode(pairs: np.ndarray, rho: float = 1.0) -> _Decode:
    from port.extensions.copy_likelihood import candidates

    states = candidates(6, 6)
    index = {tuple(row): k for k, row in enumerate(states.tolist())}
    path = np.array([index[tuple(row)] for row in pairs.tolist()])
    return _Decode(states, [path, path], np.zeros(2), np.array([1.0, rho]), 1e-4, 1e4)


@pytest.mark.analytic
def test_segments_are_runs_of_one_pair_split_at_contigs() -> None:
    """Path [0 0 1 1 1 0] over contigs of 4 and 2 is [0, 2), [2, 4), [4, 5), [5, 6)."""
    from port.sandbox.extensions.segment_sets import segments

    found = segments(np.array([0, 0, 1, 1, 1, 0]), np.array([4, 2]))

    assert found.tolist() == [[0, 2], [2, 4], [4, 5], [5, 6]]
    with pytest.raises(ValueError, match="sum"):
        segments(np.zeros(3), np.array([2]))


def _planted() -> np.ndarray:
    pairs = np.tile([1, 1], (60, 1))
    pairs[20:40] = (1, 2)
    return pairs


@pytest.mark.analytic
def test_counts_at_their_expectation_recover_the_planted_pair_alone(
    cnaster_config: None,
) -> None:
    """At 1e4 reads a bin only (1, 2) is admitted; at 10 reads the set widens but holds it."""
    from port.sandbox.extensions.segment_sets import segment_sets

    pairs = _planted()
    sets = {}
    for depth in (1e4, 10.0):
        bulk, _ = _clone(pairs, depth)
        found = segment_sets(
            _decode(pairs),
            [(pairs, bulk, 0.0), (pairs, bulk, 0.0)],
            0,
            np.array([60]),
            level=0.9973,
        )
        gain = next(f for f in found if f.clone == 1 and f.start == 20)
        assert (gain.start, gain.end, gain.decoded) == (20, 40, (1, 2))
        sets[depth] = [(a, b) for a, b, _ in gain.consistent]

    assert sets[1e4] == [(1, 2)]
    assert sets[10.0][0] == (1, 2)
    assert len(sets[10.0]) > 1


@pytest.mark.oracle
def test_each_deviance_is_the_brute_force_refit_s(cnaster_config: None) -> None:
    """Clone 1 at 0.8: every candidate's deviance against a plain loop over the grid."""
    from port.extensions.copy_likelihood import (
        pair_rate_and_share,
        pseudobulk_log_pmf,
        with_dispersions,
    )
    from port.sandbox.extensions.segment_sets import (
        FRACTION_STEPS,
        SHIFT_STEPS,
        segment_sets,
    )

    pairs = _planted()
    bulk, _ = _clone(pairs, 50.0, rho=0.8)
    decode = _decode(pairs, rho=0.8)
    found = segment_sets(
        decode,
        [(pairs, bulk, 0.0), (pairs, bulk, 0.0)],
        0,
        np.array([60]),
        level=0.9973,
    )
    gain = next(f for f in found if f.clone == 1 and f.start == 20)

    fitted = with_dispersions(bulk, decode.dispersion, decode.taus)
    states = decode.states

    def loglik(candidate: np.ndarray) -> float:
        best = -np.inf
        for d in SHIFT_STEPS:
            for f in np.unique(np.clip(0.8 + FRACTION_STEPS, 0.05, 1.0)):
                held = pairs.copy()
                held[20:40] = candidate
                log_mu, p = pair_rate_and_share(held, float(f))
                value = float(
                    np.sum(pseudobulk_log_pmf(log_mu - d, p, fitted, np.arange(60)))
                )
                best = max(best, value)
        return best

    scores = {tuple(row): loglik(row) for row in states.tolist()}
    top = max(scores.values())
    for a, b, deviance in gain.consistent:
        assert deviance == pytest.approx(2 * (top - scores[a, b]), abs=1e-6)
    assert {(a, b) for a, b, _ in gain.consistent} == {
        k for k, v in scores.items() if 2 * (top - v) <= gain.threshold
    }


@pytest.mark.oracle
def test_the_per_bin_table_is_each_bin_s_emission(cnaster_config: None) -> None:
    """`bin_loglik` at clone 1's 0.8, bin by bin, against `pseudobulk_log_pmf` per pair."""
    from port.extensions.copy_likelihood import (
        pair_rate_and_share,
        pseudobulk_log_pmf,
        with_dispersions,
    )
    from port.sandbox.extensions.segment_sets import bin_loglik

    pairs = _planted()
    bulk, _ = _clone(pairs, 50.0, rho=0.8)
    decode = _decode(pairs, rho=0.8)
    lattice, loglik, decoded = bin_loglik(
        decode, [(pairs, bulk, 0.0), (pairs, bulk, 0.0)]
    )

    fitted = with_dispersions(bulk, decode.dispersion, decode.taus)
    for k in (0, 7, len(lattice) - 1):
        log_mu, p = pair_rate_and_share(lattice[k : k + 1], 0.8)
        expected = pseudobulk_log_pmf(
            np.full(60, log_mu[0]), np.full(60, p[0]), fitted, np.arange(60)
        )
        np.testing.assert_array_equal(loglik[1, :, k], expected)
    assert [tuple(lattice[i]) for i in decoded[1, 19:22]] == [(1, 1), (1, 2), (1, 2)]
