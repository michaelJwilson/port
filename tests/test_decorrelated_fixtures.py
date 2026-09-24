"""`tests.sim_fixtures.gene_agreement`: which genes follow baseline x copy number (#372).

A synthetic sample with a known answer: every tumour spot draws its genes from
the normal baseline times the planted copy factor at `rho = 0.92`, and a
planted subset also carries a shared tumour program, a per-gene log offset
common to every tumour clone. The offset is what `decorrelate` selects on, so
it must rank the planted program genes last and recover the rest.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from tests.sim_fixtures import ADMIXED_PURITY, gene_agreement

N_GENES = 3_000
N_CLONES = 4
SPOTS = 60
PROGRAM = 0.6
"""The share of genes carrying the planted tumour program."""


def _sample(
    seed: int = 372,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    baseline = rng.lognormal(0.0, 1.0, N_GENES)
    total = np.full((N_GENES, N_CLONES), 2, dtype=np.int64)

    for clone in range(1, N_CLONES):
        start = rng.integers(0, N_GENES - 400)
        total[start : start + 400, clone] = rng.choice([1, 3, 4])

    program = rng.random(N_GENES) < PROGRAM
    offset = np.where(program, rng.normal(0.0, 1.5, N_GENES), 0.0)
    rows, labels = [], []

    for clone in range(N_CLONES):
        factor = ADMIXED_PURITY * total[:, clone] / 2 + 1 - ADMIXED_PURITY
        weights = baseline * (factor if clone else 1.0)

        if clone:
            weights = weights * np.exp(offset)

        for _ in range(SPOTS):
            rows.append(rng.poisson(40.0 * weights / weights.mean()))
            labels.append(clone)

    return sp.csr_matrix(np.array(rows)), np.array(labels), total, program


@pytest.mark.analytic
def test_the_offset_ranks_the_planted_program_last() -> None:
    """The fifth of genes with the smallest `|o_g|` carry no planted program.

    Fails if the offset mixes the program into the copy factor, or scores a
    gene against the wrong clone's copies.
    """
    counts, labels, total, program = _sample()
    agreement = gene_agreement(counts, labels, total)
    order = np.argsort(np.abs(agreement.offset))
    kept = agreement.genes[order[: order.size // 5]]

    assert agreement.genes.size > 0.9 * N_GENES
    assert program[kept].mean() < 0.05


@pytest.mark.analytic
def test_the_kept_fifth_follows_baseline_times_copy_number() -> None:
    """Kept genes correlate with baseline x copy factor at r >= 0.75 on log scale; all do not.

    The threshold #372 sets for the real samples, where the kept fifth reads
    0.788 (easy) and 0.766 (hard) against 0.266 and 0.253 over every tested
    gene. Fails if selection by `|o_g|` does not raise the agreement.
    """
    counts, labels, total, _ = _sample()
    agreement = gene_agreement(counts, labels, total)
    order = np.argsort(np.abs(agreement.offset))
    kept = order[: order.size // 5]

    def r(rows: np.ndarray) -> float:
        return float(
            np.corrcoef(
                np.log(agreement.realized[rows]).ravel(),
                np.log(agreement.expected[rows]).ravel(),
            )[0, 1]
        )

    every = r(np.arange(order.size))

    assert r(kept) >= 0.75
    assert r(kept) > every + 0.1
