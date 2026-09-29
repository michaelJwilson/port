"""`lattice_decode` recovers planted pairs, a tumour clone's shift and fraction (#362).

Two clones share four states' `(A, B)`: a normal clone and a tumour clone
whose depth is scaled by `exp(-SHIFT)`, as the shifted model's `logmu_shift`
scales it. The counts are drawn at depth, so the likelihood's argmax is the
truth; the start is the continuous fit's rounding with a fifth of each path
scrambled, which is what an E-step has to undo.
"""

from __future__ import annotations

import numpy as np
import pytest
from port.extensions.copy_likelihood import Pseudobulk, lattice_decode

PAIRS = np.array([[1, 1], [2, 1], [3, 1], [2, 2]])
SHIFT = 0.4
N_OBS = 400
DEPTH = 2.0e3
ALPHA = 1.0e-3
TAU = 2.0e2


def _planted(
    seed: int, pairs: np.ndarray = PAIRS, purity: float = 1.0
) -> tuple[list[np.ndarray], list[Pseudobulk]]:
    """A normal clone and a tumour clone of `purity`, both over `pairs`' states."""
    from port.extensions.copy_likelihood import _parameters

    rng = np.random.default_rng(seed)
    runs = np.repeat(np.arange(8) % 4, N_OBS // 8)
    paths = [np.where(runs == 2, 0, runs), runs]
    log_lambda = np.full(N_OBS, -np.log(N_OBS))
    bulks = []

    for path, shift, fraction in zip(paths, (0.0, SHIFT), (1.0, purity), strict=True):
        log_mu, share_of = _parameters(pairs, fraction)
        mean = DEPTH * np.exp(log_mu[path] - shift)
        counts = rng.negative_binomial(1.0 / ALPHA, 1.0 / (1.0 + ALPHA * mean))
        p = np.clip(share_of[path], 1e-6, 1.0 - 1e-6)
        share = rng.beta(p * TAU, (1.0 - p) * TAU)
        bulks.append(
            Pseudobulk(
                counts_nb=counts.astype(np.float64),
                base_nb_mean=np.full(N_OBS, DEPTH),
                counts_bb=rng.binomial(int(DEPTH), share).astype(np.float64),
                total_bb_rd=np.full(N_OBS, DEPTH),
                log_lambda=log_lambda,
                alpha=ALPHA * 3.0,
                tau=TAU / 3.0,
            )
        )

    return paths, bulks


def _scrambled(paths: list[np.ndarray], seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed + 100)
    return [
        np.where(rng.random(N_OBS) < 0.2, rng.integers(0, 4, N_OBS), path)
        for path in paths
    ]


LOG_TRANSMAT = np.log(np.full((4, 4), 1e-4 / 3) + np.eye(4) * (1 - 1e-4 - 1e-4 / 3))


LOH_PAIRS = np.array([[1, 1], [2, 1], [0, 1], [2, 2]])
"""`PAIRS` with LOH for `(3, 1)`: the one kind of state no fraction can mimic."""

PURITY = 0.8


@pytest.mark.analytic
@pytest.mark.parametrize("pair", [(1, 1), (2, 1), (3, 1), (2, 2), (4, 1)])
def test_half_purity_mimics_every_pair_with_both_alleles(pair: tuple[int, int]) -> None:
    """At `rho = 1/2`, `(2A - 1, 2B - 1)` has `(A, B)`'s depth and allele share.

    So the fraction is not identifiable from states that all carry both
    alleles; LOH, whose mimic would need `-1` copies, is what pins it.
    """
    from port.extensions.copy_likelihood import _parameters

    pure = _parameters(np.array([pair]), 1.0)
    mimic = _parameters(np.array([[2 * pair[0] - 1, 2 * pair[1] - 1]]), 0.5)

    np.testing.assert_allclose(mimic, pure, rtol=1e-12)


@pytest.mark.end2end
@pytest.mark.parametrize("seed", [0, 1])
def test_the_lattice_viterbi_em_recovers_every_bins_pair(seed: int) -> None:
    """One state per pair, fractions held at 1: each bin's pair and the shift."""
    paths, bulks = _planted(seed)
    fitted = lattice_decode(
        [(z, b, 0.0) for z, b in zip(paths, bulks, strict=True)],
        normal_clone=0,
        max_total_copy=6,
        lengths=np.array([N_OBS]),
        fit_purity=False,
    )

    for pairs, planted in zip(fitted.pairs, paths, strict=True):
        np.testing.assert_array_equal(pairs, PAIRS[planted])
    assert abs(fitted.shifts[1] - SHIFT) < 0.02


@pytest.mark.end2end
@pytest.mark.parametrize("seed", [0, 1])
def test_the_viterbi_em_recovers_a_planted_tumour_fraction(seed: int) -> None:
    """With LOH planted, `rho = 0.8` to 0.03, and every bin's pair."""
    paths, bulks = _planted(seed, LOH_PAIRS, PURITY)
    fitted = lattice_decode(
        [(z, b, 0.0) for z, b in zip(paths, bulks, strict=True)],
        normal_clone=0,
        max_total_copy=6,
        lengths=np.array([N_OBS]),
    )

    assert fitted.purity[0] == 1.0
    assert abs(fitted.purity[1] - PURITY) < 0.03
    for pairs, planted in zip(fitted.pairs, paths, strict=True):
        np.testing.assert_array_equal(pairs, LOH_PAIRS[planted])
