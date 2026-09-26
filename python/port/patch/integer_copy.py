"""`cnaster.integer_copy`'s two decoders, with their copy caps read from the config.

`run_cnaster` calls `hill_climbing_integer_copynumber_oneclone` and
`hill_climbing_integer_copynumber_fixdiploid_milp` without `max_total_copy` or
`max_allele_copy`, so both decode under their defaults, `A + B <= 6` and
`A, B <= 5`, whatever the configuration says. A state whose planted total is
above 6 cannot be decoded at all: #313's chr7 plants `2 mu = 10` and every
configuration returned 6 or less (`docs/audit-recovery.md`).

These read `int_copy_num.max_total_copy` from `cnaster`'s global
configuration and apply it as **both** caps: the total, and each allele, since
an allele cap below the total leaves totals above twice it unreachable -- at
`cnaster`'s 5 no pair beyond `(5, 5)` exists. Where the configuration states
no cap the decode is `cnaster`'s, so a configuration without the key decodes
exactly as `cnaster` does. The key is `port`'s: `cnaster` does not read it,
which is why this is its own table (`COPY_SWAPS`) rather than a `SWAPS` row --
a configuration that states it changes the output.

The signature is `cnaster`'s, defaults included, so the swap is a drop-in. A
caller passing the default value explicitly cannot be told from one passing
nothing, and takes the configured cap; `run_cnaster` passes neither.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from typing import Any

import numpy as np
from cnaster import integer_copy as upstream

__all__ = [
    "configured_caps",
    "hill_climbing_integer_copynumber_fixdiploid_milp",
    "hill_climbing_integer_copynumber_oneclone",
]

# NB bound at import, before any swap: `pipeline.patched` rebinds a name in
#    every module that holds it, `cnaster.integer_copy` included, so reading
#    `upstream.<name>` at call time would call this module back.
_ONECLONE = upstream.hill_climbing_integer_copynumber_oneclone
_MILP = upstream.hill_climbing_integer_copynumber_fixdiploid_milp

_LIKELIHOOD: list[bool] = [False]
"""Whether the decoders refine by the pseudobulk likelihood (#327)."""

DECODED: list[Any] = []
"""Each refinement's `port.extensions.copy_likelihood.Decoded`, in call order."""

logger = logging.getLogger(__name__)

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


@contextlib.contextmanager
def by_likelihood() -> Iterator[None]:
    """Refine each decode by the HMM's pseudobulk likelihood for the block (#327).

    Captures the RDR+BAF fit's inputs, and after `cnaster`'s decoder returns,
    re-decodes that clone's states by `port.extensions.copy_likelihood`,
    starting from the decoder's answer. A clone the capture cannot identify
    keeps the decoder's answer, and says so.
    """
    from port.extensions.copy_likelihood import capture

    previous = _LIKELIHOOD[0]
    _LIKELIHOOD[0] = True
    DECODED.clear()

    try:
        with capture():
            yield
    finally:
        _LIKELIHOOD[0] = previous


def _refine(
    result: Any,
    new_log_mu: Any,
    base_nb_mean: Any,
    new_p_binom: Any,
    pred_cnv: Any,
    total: int,
) -> Any:
    """The decoder's answer, re-decoded by likelihood when `by_likelihood` is on."""
    if not _LIKELIHOOD[0]:
        return result

    from port.extensions.copy_likelihood import decode, pseudobulk_for
    from port.patch.hmm_nophasing.shifted_emission import hmm_nophasing, neutral_state

    bulk = pseudobulk_for(np.asarray(base_nb_mean))

    if bulk is None:
        logger.warning("copy likelihood: no captured clone matches; keeping the MILP's")
        return result

    copies, _, ploidy = result
    copies = np.asarray(copies, dtype=np.int64)
    log_mu = np.asarray(new_log_mu, dtype=np.float64).reshape(-1)
    path = np.asarray(pred_cnv, dtype=np.int64).reshape(-1) % log_mu.size
    neutral = neutral_state(log_mu, np.asarray(new_p_binom).reshape(-1), path[:, None])

    decoded = decode(
        copies,
        path,
        bulk,
        max_total_copy=total,
        neutral=neutral,
        shift=bool(hmm_nophasing.apply_logmu_shift),
    )
    DECODED.append(decoded)

    return decoded.copies, -decoded.log_likelihood, ploidy


def _caps(max_allele_copy: int, max_total_copy: int) -> tuple[int, int]:
    """The caps to decode under: the configuration's where the caller left the default."""
    allele, total = configured_caps()

    return (
        allele if max_allele_copy == MAX_ALLELE_COPY else max_allele_copy,
        total if max_total_copy == MAX_TOTAL_COPY else max_total_copy,
    )


def hill_climbing_integer_copynumber_oneclone(
    new_log_mu: Any,
    base_nb_mean: Any,
    new_p_binom: Any,
    pred_cnv: Any,
    max_allele_copy: int = 5,
    max_total_copy: int = 6,
    max_medploidy: int = 4,
    enforce_states: Any = {},  # noqa: B006 -- cnaster's default, passed through
    EPS_BAF: float = 0.05,
    expression_weight: bool = False,
) -> Any:
    """`cnaster`'s hill climbing, under the configured caps."""
    allele, total = _caps(max_allele_copy, max_total_copy)

    result = _ONECLONE(
        new_log_mu,
        base_nb_mean,
        new_p_binom,
        pred_cnv,
        max_allele_copy=allele,
        max_total_copy=total,
        max_medploidy=max_medploidy,
        enforce_states=enforce_states,
        EPS_BAF=EPS_BAF,
        expression_weight=expression_weight,
    )

    return _refine(result, new_log_mu, base_nb_mean, new_p_binom, pred_cnv, total)


def hill_climbing_integer_copynumber_fixdiploid_milp(
    new_log_mu: Any,
    base_nb_mean: Any,
    new_p_binom: Any,
    pred_cnv: Any,
    max_allele_copy: int = 5,
    max_total_copy: int = 6,
    max_medploidy: int = 4,
    min_prop_threshold: float = 0.0,
    EPS_BAF: float = 0.05,
    nonbalance_bafdist: Any = None,
    nondiploid_rdrdist: Any = None,
    cost_type: str = "L1",
    enforce_order: bool = False,
    uniform_state_weights: bool = False,
    rdr_relative_weight: float = 0.3,
    enforce_states: Any = {},  # noqa: B006 -- cnaster's default, passed through
    max_samples: int = 20,
) -> Any:
    """`cnaster`'s MILP decoder, under the configured caps."""
    allele, total = _caps(max_allele_copy, max_total_copy)

    result = _MILP(
        new_log_mu,
        base_nb_mean,
        new_p_binom,
        pred_cnv,
        max_allele_copy=allele,
        max_total_copy=total,
        max_medploidy=max_medploidy,
        min_prop_threshold=min_prop_threshold,
        EPS_BAF=EPS_BAF,
        nonbalance_bafdist=nonbalance_bafdist,
        nondiploid_rdrdist=nondiploid_rdrdist,
        cost_type=cost_type,
        enforce_order=enforce_order,
        uniform_state_weights=uniform_state_weights,
        rdr_relative_weight=rdr_relative_weight,
        enforce_states=enforce_states,
        max_samples=max_samples,
    )

    return _refine(result, new_log_mu, base_nb_mean, new_p_binom, pred_cnv, total)
