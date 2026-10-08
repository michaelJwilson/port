"""Integer copies decoded at known states after fitting only the two dispersions (#362).

Referee: the planted `(A, B)` pairs, compared unordered.
"""

from __future__ import annotations

import numpy as np
import pytest
from port.extensions.copy_likelihood import (
    Pseudobulk,
    pseudobulk_log_pmf,
)
from port.sandbox.extensions.shared_decode import shared_decode
from port.sim.truth import COPY_LATTICE, CoreInferenceTruth, core_inference_truth
from scipy.optimize import minimize_scalar

N_STATES = 7
"""Seven of `COPY_LATTICE`'s nine: identifiable from BAF as well as RDR."""


def _pseudobulks(truth: CoreInferenceTruth) -> list[tuple[Pseudobulk, np.ndarray]]:
    rows = []
    for clone in np.unique(truth.labels):
        spots = truth.labels == clone
        base = truth.base_nb_mean[:, spots].sum(axis=1)
        bulk = Pseudobulk(
            counts_nb=truth.counts_nb[:, spots].sum(axis=1),
            base_nb_mean=base,
            counts_bb=truth.counts_bb[:, spots].sum(axis=1),
            total_bb_RD=truth.total_bb_RD[:, spots].sum(axis=1),
            normal_log_lambda=np.log(base / base.sum()),
            dispersion=1.0,
            taus=1.0,
        )
        rows.append((bulk, np.asarray(truth.states[clone], dtype=np.int64)))
    return rows


def _fit_dispersions(
    truth: CoreInferenceTruth, bulks: list[tuple[Pseudobulk, np.ndarray]]
) -> tuple[float, float]:
    """Shared `alpha` and `tau` by maximum likelihood, everything else planted."""
    log_mu = np.asarray(truth.log_mu, dtype=np.float64)
    p = np.asarray(truth.p_binom, dtype=np.float64)

    def total(alpha: float, tau: float) -> float:
        return -sum(
            float(
                np.sum(
                    pseudobulk_log_pmf(
                        log_mu[path],
                        p[path],
                        bulk._replace(dispersion=alpha, taus=tau),
                        np.arange(path.size),
                    )
                )
            )
            for bulk, path in bulks
        )

    # NB `alpha` enters only the NB term and `tau` only the BB term, so each fits independently.
    alpha = float(
        np.exp(
            minimize_scalar(
                lambda x: total(float(np.exp(x)), 30.0),
                bounds=(-12, 2),
                method="bounded",
            ).x
        )
    )
    tau = float(
        np.exp(
            minimize_scalar(
                lambda x: total(alpha, float(np.exp(x))),
                bounds=(0, 12),
                method="bounded",
            ).x
        )
    )
    return alpha, tau


@pytest.mark.end2end
def test_known_states_and_fitted_dispersions_decode_the_planted_pairs() -> None:
    """Every planted state's `(A, B)`, unordered, in every clone that visits it."""
    truth = core_inference_truth(
        n_clones=3,
        n_states=N_STATES,
        lattice=(20, 20),
        n_obs=200,
        n_segments=4,
        seed=11,
        copy_lattice=True,
    )
    bulks = _pseudobulks(truth)
    alpha, tau = _fit_dispersions(truth, bulks)

    for bulk, path in bulks:
        decoded = shared_decode(
            [(path, bulk._replace(dispersion=alpha, taus=tau), 0.0)],
            n_states=N_STATES,
            normal=0,
            max_total_copy=6,
        )
        for state in np.unique(path):
            assert sorted(decoded.states[state]) == sorted(COPY_LATTICE[state]), (
                state,
                decoded.states[state],
                COPY_LATTICE[state],
            )
