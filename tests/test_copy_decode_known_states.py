"""Integer copies decoded at known states, after fitting only the dispersions (#362).

The copy-lattice fixture plants integer `(A, B)` states. Given the planted
`mu`, `p`, clone labels and copy states, the only free parameters of the
HMM's emission are the two shared dispersions: the negative binomial's
`alpha` and the beta-binomial's `tau`. They are fitted here by maximum
likelihood on each clone's pseudobulk, with every other quantity held -- the
complete-data likelihood at the known path, which is what Baum-Welch
maximizes when the states are known.

The integer copies are then decoded by `port.extensions.copy_likelihood.
fit_copies` under `SHARED`, the decode `port.patch.integer_copy` runs, at those
states: the normal state `(1, 1)` by definition and the shift 0, because the
planted `mu` is already on the normal scale. Referee: the planted pairs.
The fixture writes the major allele second (`p = B / (A + B)`) and the
decoder first, so pairs are compared as unordered.

The dispersions fitted are the pseudobulk's, not the spots': on this instance
`alpha = 7.6e-4` and `tau = 3.0e3` against the planted per-spot `1/6` and
`30`, since each clone's pseudobulk sums 100 or more spots.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from port.extensions.copy_likelihood import Pseudobulk, _emission, shared_decode
from scipy.optimize import minimize_scalar

from tests.fixtures import COPY_LATTICE, CoreInferenceTruth, core_inference_truth

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
            total_bb_rd=truth.total_bb_RD[:, spots].sum(axis=1),
            log_lambda=np.log(base / base.sum()),
            alpha=1.0,
            tau=1.0,
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
                    _emission(
                        log_mu[path],
                        p[path],
                        replace(bulk, alpha=alpha, tau=tau),
                        np.arange(path.size),
                    )
                )
            )
            for bulk, path in bulks
        )

    # NB the two channels are separate terms, so each dispersion is fitted
    #    with the other at any value: `alpha` enters the NB term only, `tau`
    #    the BB term only.
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
            [(path, replace(bulk, alpha=alpha, tau=tau), 0.0)],
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
