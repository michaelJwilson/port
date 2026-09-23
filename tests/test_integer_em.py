"""The integer EM recovers planted pairs, paths and a tumour clone's shift (#362).

Two clones share four states' `(A, B)`: a normal clone and a tumour clone
whose depth is scaled by `exp(-SHIFT)`, as the shifted model's `logmu_shift`
scales it. The counts are drawn at depth, so the likelihood's argmax is the
truth; the start is the continuous fit's rounding with a fifth of each path
scrambled, which is what an E-step has to undo.
"""

from __future__ import annotations

import numpy as np
import pytest
from port.extensions.copy_likelihood import Pseudobulk, integer_em

PAIRS = np.array([[1, 1], [2, 1], [3, 1], [2, 2]])
SHIFT = 0.4
N_OBS = 400
DEPTH = 2.0e3
ALPHA = 1.0e-3
TAU = 2.0e2


def _planted(seed: int) -> tuple[list[np.ndarray], list[Pseudobulk]]:
    rng = np.random.default_rng(seed)
    runs = np.repeat(np.arange(8) % 4, N_OBS // 8)
    paths = [np.where(runs == 2, 0, runs), runs]
    log_lambda = np.full(N_OBS, -np.log(N_OBS))
    bulks = []

    for path, shift in zip(paths, (0.0, SHIFT), strict=True):
        total = PAIRS[path].sum(axis=1)
        mean = DEPTH * total / 2.0 * np.exp(-shift)
        counts = rng.negative_binomial(1.0 / ALPHA, 1.0 / (1.0 + ALPHA * mean))
        p = PAIRS[path, 0] / total
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


@pytest.mark.end2end
@pytest.mark.parametrize("seed", [0, 1])
def test_the_integer_em_recovers_pairs_paths_and_the_shift(seed: int) -> None:
    """Every pair, every bin's state, and `SHIFT` to 0.02, from a scrambled start."""
    paths, bulks = _planted(seed)
    rng = np.random.default_rng(seed + 100)
    scrambled = [
        np.where(rng.random(N_OBS) < 0.2, rng.integers(0, 4, N_OBS), path)
        for path in paths
    ]
    start = np.array([[1, 1], [2, 1], [2, 1], [3, 3]])
    log_transmat = np.log(np.full((4, 4), 1e-4 / 3) + np.eye(4) * (1 - 1e-4 - 1e-4 / 3))

    fit = integer_em(
        [(z, b, 0.0) for z, b in zip(scrambled, bulks, strict=True)],
        start,
        normal_clone=0,
        log_transmat=log_transmat,
        log_startprob=np.full(4, -np.log(4)),
        lengths=np.array([N_OBS]),
        max_total_copy=6,
        normal=0,
    )

    np.testing.assert_array_equal(fit.copies, PAIRS)
    for fitted, planted in zip(fit.paths, paths, strict=True):
        np.testing.assert_array_equal(fitted, planted)
    assert fit.shifts[0] == 0.0
    assert abs(fit.shifts[1] - SHIFT) < 0.02
