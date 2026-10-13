"""Replaces `cnaster.hmm_initialize.gmm_init` on the BAF + RDR stage with `sal`'s mixture starts (#489, #540, #547).

Fits an NB-on-total times BB-on-B mixture, conditioned on each bin's exposure
and trials, seeded by `start` (`kmeans++x5+em` under `--sal`, or `lattice`),
polished by `sal`'s EM and returned as `(log_mu, p_binom)`. `only_minor` calls
keep their start.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from cnaster.hmm_initialize import gmm_init as UPSTREAM

from port.patch._signature import as_upstream

__all__ = [
    "DEFAULT",
    "EXPOSURE_SCALE",
    "POLISH_SECONDS",
    "checked",
    "gmm_init",
    "instance_of",
]

DEFAULT = "kmeans++x5+em"
"""The start `--sal` installs: kmeans++ in rate space, best of five, each polished by EM (#489)."""

POLISH_SECONDS = 160.0
"""The seconds the start and its polish may spend, `sal`'s notebook budget."""

DISPERSION = 10.0
"""The seam's negative-binomial size, `1 / alpha` at `backends.DEFAULT_ALPHA`."""

CONCENTRATION = 1_000.0
"""The seam's beta-binomial `alpha + beta`, `backends.DEFAULT_TAU`."""

EXPOSURE_SCALE = 100.0
"""Exposure divisor before `sal` reads it, so seeding's mean floor of 1 can place losses (#547)."""


def checked(start: str) -> str:
    """`start`, refused unless a lattice or `sal` copy-state start has that name."""
    from sal.search.mixture_starts import lookup

    from port.extensions.copy_starts import LATTICE

    if start not in LATTICE:
        try:
            lookup(start)
        except (KeyError, ValueError) as error:
            msg = f"no copy-state start {start!r}: {sorted(LATTICE)} or sal's mixture starts"
            raise ValueError(msg) from error
    return start


def instance_of(
    X: np.ndarray, base_nb_mean: np.ndarray, total_bb_RD: np.ndarray, n_states: int
) -> Any:
    """The call's pseudobulk as `sal`'s `MixtureInstance`, conditioned on exposure and trials.

    Zero-exposure bins are left out. Seeding rows are total per exposure (over
    `EXPOSURE_SCALE`) and the B fraction times the common trial count.
    """
    from sal.opt.emission_mixture import CountPairSeeding
    from sal.search.mixture_starts import MixtureInstance

    counts = np.asarray(X, dtype=np.float64).reshape(X.shape[0], 2, -1)[:, :, 0]
    exposure = (
        np.asarray(base_nb_mean, dtype=np.float64).reshape(X.shape[0], -1)[:, 0]
        / EXPOSURE_SCALE
    )
    trials = np.asarray(total_bb_RD, dtype=np.float64).reshape(X.shape[0], -1)[:, 0]
    kept = exposure > 0.0

    if not kept.any():
        msg = "every bin has zero exposure; there is no total to fit"
        raise ValueError(msg)

    observations = counts[kept]
    covariate = np.column_stack([exposure, trials])[kept]
    pooled = observations[:, 1].sum() / max(covariate[:, 1].sum(), 1.0)
    common = max(float(np.rint(np.median(covariate[:, 1]))), 1.0)
    rows = np.column_stack(
        [
            observations[:, 0] / covariate[:, 0],
            np.divide(
                observations[:, 1],
                covariate[:, 1],
                out=np.full(observations.shape[0], pooled),
                where=covariate[:, 1] > 0,
            )
            * common,
        ]
    )
    at = CountPairSeeding(
        dispersion=DISPERSION,
        concentration=CONCENTRATION,
        joint=False,
        trials=common,
    )
    return MixtureInstance(
        observations=observations,
        labels=np.zeros(observations.shape[0], dtype=np.int64),
        weights=np.full(n_states, 1.0 / n_states),
        # NB a start reads the truth for its component count alone.
        truth=at(rows[:n_states]),
        at=at,
        covariate=covariate,
        seeding_rows=rows,
    )


def _call(arguments: dict[str, Any], stage: str) -> Any:
    """The initializer's arguments as `copy_starts.CopyCall`; no start reads positions."""
    from port.extensions.copy_starts import CopyCall

    X = np.asarray(arguments["X"], dtype=np.float64)
    n_rows = X.shape[0]
    base = np.asarray(arguments["base_nb_mean"], dtype=np.float64).reshape(n_rows)
    trials = np.asarray(arguments["total_bb_RD"], dtype=np.float64).reshape(n_rows)
    unplaced = np.zeros(n_rows)
    return CopyCall(
        stage=stage,
        n_states=int(arguments["n_states"]),
        total=X[:, 0, 0],
        b=X[:, 1, 0],
        exposure=base if stage == "rdrbaf" else unplaced,
        trials=trials,
        clone=np.zeros(n_rows, dtype=np.int64),
        contig=unplaced.astype(str),
        start=unplaced,
        length=unplaced,
        planted=np.full((n_rows, 2), -1),
        raw={},
    )


@as_upstream(UPSTREAM, start=None, distinct=False)
def gmm_init(arguments: dict[str, Any], options: dict[str, Any]) -> Any:
    """`cnaster`'s initializer signature; `start` on the BAF + RDR call.

    Otherwise `cnaster`'s initializer, or `distinct`'s where `distinct` is set.
    """
    from port.extensions.copy_starts import run_start
    from port.patch.hmm_initialize import distinct

    params = str(arguments.get("params", ""))
    stage = "rdrbaf" if "m" in params else "baf"
    chosen = options["start"] if stage == "rdrbaf" else None

    if chosen is None or arguments.get("only_minor", True):
        fallback = distinct.gmm_init if options["distinct"] else distinct.UPSTREAM
        return fallback(**arguments)

    rng = np.random.default_rng([int(arguments.get("random_state") or 0), 0])
    result = run_start(
        checked(chosen), _call(arguments, stage), rng, seconds=POLISH_SECONDS
    )
    return result.log_mu.reshape(-1, 1), result.p_binom.reshape(-1, 1), None, None
