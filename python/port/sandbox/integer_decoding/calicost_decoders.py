"""CalicoST's integer-copy decoders, called as `calicost_main` calls them (#362).

CalicoST fits each clone's continuous `(mu, p)` and then decodes integer
copies per clone with a least-squares cost on them
(`find_integer_copynumber.py`):

- `hill_climbing_integer_copynumber_fixdiploid`, with `nonbalance_bafdist`
  1.0 and `nondiploid_rdrdist` 10.0 (`arg_parse.py:67-68`): its main answer,
  the `max_medploidy = None` pass;
- `hill_climbing_integer_copynumber_oneclone` with `max_medploidy` 2, 3 or 4:
  its three further passes (`calicost_main.py:329-331`).

Each is handed the clone's rates normalized as CalicoST normalizes them,
`log mu - log sum_g lambda_g mu_z(g)` (`calicost_main.py:327`), which is the
per-clone shift `port`'s fit carries. The decoders are CalicoST's, imported,
not copied: this module only lays out their inputs from a
:func:`port.extensions.copy_likelihood.fit_copies` input, so the two decode
the same fit. Set aside as a referee for the sim audit, not installed.
"""

from __future__ import annotations

import numpy as np

from port.extensions.copy_likelihood import Pseudobulk

__all__ = ["MEDPLOIDY", "calicost_pairs"]

MEDPLOIDY = (None, 2, 3, 4)
"""CalicoST's passes: `None` is `fixdiploid`, the rest `oneclone` at that cap."""

NONBALANCE_BAFDIST = 1.0
NONDIPLOID_RDRDIST = 10.0


def _adjusted(log_mu: np.ndarray, path: np.ndarray, bulk: Pseudobulk) -> np.ndarray:
    """`calicost_main.py:327`: the clone's rates, normalized to unit mean depth."""
    weights = bulk.base_nb_mean / np.sum(bulk.base_nb_mean)
    return np.asarray(
        log_mu - np.log(np.sum(np.exp(log_mu[path]) * weights)), dtype=np.float64
    )


def calicost_pairs(
    clones: list[tuple[np.ndarray, Pseudobulk, float]],
    log_mu: np.ndarray,
    p_binom: np.ndarray,
    max_medploidy: int | None = None,
) -> list[np.ndarray]:
    """Each clone's per-bin `(A, B)` by CalicoST's decoder for `max_medploidy`.

    `clones` is :func:`~port.extensions.copy_likelihood.fit_copies`' input;
    `log_mu` and `p_binom` are the shared continuous fit's, one per state.
    A clone whose decode raises -- `fixdiploid` does when it finds no
    balanced state above its occupancy threshold, and CalicoST then retries
    at `min_prop_threshold = 0.02` -- is retried the same way.
    """
    from calicost.find_integer_copynumber import (
        hill_climbing_integer_copynumber_fixdiploid,
        hill_climbing_integer_copynumber_oneclone,
    )

    pairs = []

    for path, bulk, _ in clones:
        adjusted = _adjusted(log_mu, path, bulk)

        if max_medploidy is not None:
            copies, _ = hill_climbing_integer_copynumber_oneclone(
                adjusted, bulk.base_nb_mean, p_binom, path, max_medploidy=max_medploidy
            )
        else:
            try:
                copies, _ = hill_climbing_integer_copynumber_fixdiploid(
                    adjusted,
                    bulk.base_nb_mean,
                    p_binom,
                    path,
                    nonbalance_bafdist=NONBALANCE_BAFDIST,
                    nondiploid_rdrdist=NONDIPLOID_RDRDIST,
                )
            except Exception:
                copies, _ = hill_climbing_integer_copynumber_fixdiploid(
                    adjusted,
                    bulk.base_nb_mean,
                    p_binom,
                    path,
                    nonbalance_bafdist=NONBALANCE_BAFDIST,
                    nondiploid_rdrdist=NONDIPLOID_RDRDIST,
                    min_prop_threshold=0.02,
                )

        pairs.append(np.asarray(copies, dtype=np.int64)[path])

    return pairs
