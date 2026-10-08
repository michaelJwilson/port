"""`cnaster`'s coded NB/BB emission, scored by sal's dense log-emission (#425).

`_nb_logpmf_1d` and `_bb_logpmf_1d` score one state over a vector of counts,
`lgamma` per score. `sal.emissions.dense.log_emission` scores every state of
the same family at once from sal's tables in Rust (sal #1132), in the
family's own completion order (`Order.FAMILY`). The same edge behaviour as
`cnaster`'s:

- a zero exposure or trial count scores 0, which sal reads as unobserved;
- `alpha` floored at 1e-10 for the negative binomial, and the beta-binomial's
  `alpha`, `beta` floored at `EPS`;
- a non-positive rate `mu` scores 0 everywhere, as `exposure * mu <= 0` does.

sal's tables are `port.patch.emission`'s, the one evaluation every site
scores (T- #776): scaled rising factorials completed in sal's order (sal
#1334, #1336). The beta-binomial is that module's bit for bit at every
concentration, so no state is rescored (the `STABLE_TAU` hand-off of #561
is retired); the negative binomial is within 8.9e-14 over `max(|f|, 1)`.

To a tolerance, not bitwise, so it is `--sal`'s rather than a `SWAPS` row;
#244 is why a tolerance is measured end to end before it is anything else.
Selected by the `hmm_nophasing` row's `emission_kernels="sal"` option, not a
name rebind: `cnaster`'s compiled kernels call `_nb_logpmf_1d` as a global.
:func:`coded_emission` is upstream's coded method with every state scored in
one call per spot, which is where the speed is.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from port.patch.emission import bb_tables, nb_table

__all__ = ["bb_states", "coded_emission", "nb_states"]


def nb_states(
    obs: np.ndarray,
    exposure: np.ndarray,
    mu: np.ndarray,
    dispersions: np.ndarray,
) -> np.ndarray:
    """`(K, n)`: every state's `_nb_logpmf_1d` in one call; a rate <= 0 scores 0.

    `sal.emissions.dense.log_emission`'s kernel on `sal`'s table, built here
    once per distinct dispersion (`port.patch.emission.nb_table`): bitwise
    `log_emission`, which builds a row per state (T- #776).
    """
    from sal import oxisal
    from sal.emissions.dense import _counts

    mu = np.asarray(mu, dtype=np.float64)
    dead = mu <= 0.0
    counts = _counts(np.asarray(obs), "count")
    table, sizes = nb_table(dispersions, int(counts.max()) + 1 if counts.size else 1)
    out = np.empty(mu.size * counts.size)
    oxisal.dense_log_emission(
        mu.size,
        True,
        out,
        totals=counts,
        total_table=np.ascontiguousarray(table.T).reshape(-1),
        exposure=np.ascontiguousarray(exposure, dtype=np.float64).reshape(-1),
        dispersion=np.ascontiguousarray(sizes),
        mean=np.ascontiguousarray(np.where(dead, 1.0, mu)),
    )
    scores = out.reshape(mu.size, counts.size)
    scores[dead] = 0.0
    return scores


def bb_states(
    obs: np.ndarray,
    trials: np.ndarray,
    p_binom: np.ndarray,
    taus: np.ndarray,
) -> np.ndarray:
    """`(K, n)`: every state's `_bb_logpmf_1d` in one call.

    `sal.emissions.dense.log_emission`'s kernel on `sal`'s tables
    (`port.patch.emission.bb_tables`), each rising factorial taken once per
    distinct shape: bitwise `log_emission` (T- #776).
    """
    from sal import oxisal
    from sal.emissions.dense import _counts

    p = np.asarray(p_binom, dtype=np.float64).reshape(-1)
    successes = _counts(np.asarray(obs), "success count")
    per_observation = _counts(np.asarray(trials), "trial count")
    success_extent = int(successes.max()) + 1 if successes.size else 1
    trials_extent = int(per_observation.max()) + 1 if per_observation.size else 1
    tables = bb_tables(p, taus, max(success_extent, trials_extent))
    out = np.empty(p.size * successes.size)
    # NB `sal`'s stub still names `log_rate` `log_beta`; its own `dense.py`
    #    passes `log_rate`, as the compiled function takes (sal #1334).
    oxisal.dense_log_emission(  # type: ignore[call-arg]
        p.size,
        True,
        out,
        successes=successes,
        success_table=np.ascontiguousarray(
            tables.success[:, :success_extent].T
        ).reshape(-1),
        trials=per_observation,
        failure_table=np.ascontiguousarray(tables.failure[:, :trials_extent].T).reshape(
            -1
        ),
        trial_table=np.ascontiguousarray(tables.trial[:, :trials_extent].T).reshape(-1),
        log_factorial=tables.log_factorial[:trials_extent],
        log_rate=np.ascontiguousarray(np.stack([tables.log_p, tables.log_q])).reshape(
            -1
        ),
    )
    return out.reshape(p.size, successes.size)


def coded_emission(
    nbEncoder: Any,
    bbEncoder: Any,
    log_mu: np.ndarray,
    alphas: np.ndarray,
    p_binom: np.ndarray,
    taus: np.ndarray,
    *,
    clone_stack: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """`hmm_nophasing.compute_emission_probability_nb_betabinom_coded`, unshifted.

    Upstream's loop over spots, decode and stack, with each spot's states
    scored together: one NB and one BB family of `n_states` per spot.
    """
    n_spots = nbEncoder.n_spots

    if bbEncoder.n_spots != n_spots:  # invariant
        msg = "the encoders must cover the same spots"
        raise AssertionError(msg)

    rdr, baf = [], []

    for spot in range(n_spots):
        mu = np.exp(np.asarray(log_mu, dtype=np.float64)[:, spot])
        alpha = np.asarray(alphas, dtype=np.float64)[:, spot]
        p = np.asarray(p_binom, dtype=np.float64)[:, spot]
        tau = np.asarray(taus, dtype=np.float64)[:, spot]

        nb_endog = np.asarray(nbEncoder.get_unique_obs(spot), dtype=np.float64)
        nb_exposure = np.asarray(nbEncoder.get_unique_total(spot), dtype=np.float64)
        bb_endog = np.asarray(bbEncoder.get_unique_obs(spot), dtype=np.float64)
        bb_trials = np.asarray(bbEncoder.get_unique_total(spot), dtype=np.float64)

        rdr_unique = nb_states(nb_endog, nb_exposure, mu, alpha)
        baf_unique = bb_states(bb_endog, bb_trials, p, tau)

        rdr.append(nbEncoder.decode_array(rdr_unique, spot))
        baf.append(bbEncoder.decode_array(baf_unique, spot))

    if clone_stack:
        return np.concatenate(rdr, axis=1), np.concatenate(baf, axis=1)

    return np.stack(rdr, axis=2), np.stack(baf, axis=2)
