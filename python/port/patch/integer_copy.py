"""`cnaster.integer_copy`'s two decoders, both replaced by one: the HMM's likelihood.

`run_cnaster` calls `hill_climbing_integer_copynumber_oneclone` or
`hill_climbing_integer_copynumber_fixdiploid_milp` once per clone, handing it
the fitted `log mu` and `p`, the clone's pseudobulk baseline and its decode.
Both do the same job with an L1 cost on `(mu, p)` and a ploidy search, and
both re-derive the normal state per clone as the balanced state whose raw
`mu` is closest to 1 (`integer_copy.py:84`) -- on CalicoST's easy simulated
sample that chose the pinned state for every tumour clone and decoded their
`(2, 2)` gains as `(1, 1)` (#362).

**Only one decode is supported** (#362): the pseudobulk NB/BB likelihood the
HMM fitted (`port.extensions.copy_likelihood.decode_fixed`, #327), with the
pinned `mu`, the clone's propagated `logmu_shift`, its spots, its decoded
path and the fitted dispersions all held. So the one-candidate-per-state
MILP separates and is solved exactly, state by state. The normal state is
`(1, 1)` by definition: the pinned one, shared by every clone
(`port.patch.hmrf.core_inference`). Both of `cnaster`'s names return that
decode, their signatures kept so the swap is a drop-in.

The copy caps are `int_copy_num.max_total_copy` from the configuration, as
before (#313); without the key, `cnaster`'s `A + B <= 6`.

The clone's counts come from `copy_likelihood.capture`, which
`run_cnaster_port` installs with these rows; a clone it cannot identify is an
error, not a fallback to another decoder.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "DECODED",
    "configured_caps",
    "decode_clone",
    "hill_climbing_integer_copynumber_fixdiploid_milp",
    "hill_climbing_integer_copynumber_oneclone",
]

DECODED: list[Any] = []
"""Each clone's `port.extensions.copy_likelihood.Decoded`, in call order."""

MAX_ALLELE_COPY = 5
"""`cnaster`'s default, in both signatures."""

MAX_TOTAL_COPY = 6
"""`cnaster`'s default, in both signatures."""


def configured_caps() -> tuple[int, int]:
    """`(max_allele_copy, max_total_copy)`: the configured cap for both, else `cnaster`'s."""
    from cnaster.config import get_global_config

    section = getattr(get_global_config(), "int_copy_num", None)
    total = getattr(section, "max_total_copy", None)

    if total is None:
        return MAX_ALLELE_COPY, MAX_TOTAL_COPY

    return int(total), int(total)


def _caps(max_allele_copy: int, max_total_copy: int) -> tuple[int, int]:
    """The caps to decode under: the configuration's where the caller left the default."""
    allele, total = configured_caps()

    return (
        allele if max_allele_copy == MAX_ALLELE_COPY else max_allele_copy,
        total if max_total_copy == MAX_TOTAL_COPY else max_total_copy,
    )


def decode_clone(
    new_log_mu: Any, base_nb_mean: Any, new_p_binom: Any, pred_cnv: Any, total: int
) -> tuple[np.ndarray, float, int]:
    """One clone's `(copies, loss, ploidy)`, as `cnaster`'s decoders return them.

    `loss` is the negative log-likelihood reached; `ploidy` the median total
    copy over the clone's bins.
    """
    from port.extensions.copy_likelihood import decode_fixed, pseudobulk_for
    from port.patch.hmm_nophasing.shifted_emission import neutral_state
    from port.patch.hmrf.core_inference import shift_for

    bulk = pseudobulk_for(np.asarray(base_nb_mean))

    if bulk is None:
        msg = (
            "no captured clone matches this baseline: the likelihood decode needs "
            "copy_likelihood.capture() around the run (#362)"
        )
        raise RuntimeError(msg)

    log_mu = np.asarray(new_log_mu, dtype=np.float64).reshape(-1)
    path = np.asarray(pred_cnv, dtype=np.int64).reshape(-1) % log_mu.size
    shift, normal = shift_for(pred_cnv)

    if normal is None:
        normal = neutral_state(
            log_mu, np.asarray(new_p_binom).reshape(-1), path[:, None]
        )

    decoded = decode_fixed(
        path,
        bulk,
        n_states=log_mu.size,
        max_total_copy=total,
        normal=normal,
        log_shift=shift,
    )
    DECODED.append(decoded)
    ploidy = int(np.rint(np.median(decoded.copies[path].sum(axis=1))))

    return decoded.copies, -decoded.log_likelihood, ploidy


def hill_climbing_integer_copynumber_oneclone(
    new_log_mu: Any,
    base_nb_mean: Any,
    new_p_binom: Any,
    pred_cnv: Any,
    max_allele_copy: int = 5,
    max_total_copy: int = 6,
    **ignored: Any,  # noqa: ARG001 -- cnaster's other keywords, unused here
) -> Any:
    """`cnaster`'s name, decoding by :func:`decode_clone`."""
    _, total = _caps(max_allele_copy, max_total_copy)

    return decode_clone(new_log_mu, base_nb_mean, new_p_binom, pred_cnv, total)


def hill_climbing_integer_copynumber_fixdiploid_milp(
    new_log_mu: Any,
    base_nb_mean: Any,
    new_p_binom: Any,
    pred_cnv: Any,
    max_allele_copy: int = 5,
    max_total_copy: int = 6,
    **ignored: Any,  # noqa: ARG001 -- cnaster's other keywords, unused here
) -> Any:
    """`cnaster`'s name, decoding by :func:`decode_clone`."""
    _, total = _caps(max_allele_copy, max_total_copy)

    return decode_clone(new_log_mu, base_nb_mean, new_p_binom, pred_cnv, total)
