"""The read-depth + BAF stage's HMM start from `sal`'s count-pair mixture starts, under the covariate (#489).

`cnaster`'s start, and `distinct`'s (#348), fit Gaussians to log depth ratios
and BAFs. `sal.search.mixture_starts` fits the mixture **in the family the data
came from**: a negative binomial on each bin's total with its `base_nb_mean`
as exposure, times a beta-binomial on its B count out of `total_bb_RD`, each
bin's exposure and trials its covariate (sal #933/#1083). Seeded in rate space
by the named start and polished by `sal`'s EM, it hands the HMM its
components as `(log_mu, p_binom)`.

#489 measured every `sal` start on the call's own pseudobulk (60 x 50, easy,
hard). The polished best-of-five starts end nearest the best fit reached, and
`kmeans++x5+em`, installed by `--sal`, is the only start that is at least as
good end to end on every sample: CalicoST hard's clone ARI 0.8652 -> 0.9829,
copy ARI 0.8652 -> 0.9055; 60 x 50 and easy unchanged.

The BAF-only stage (`params` without `m`) and `only_minor=True` calls have no
exposure to condition on and keep the start they had: `distinct`'s where it is
installed, `cnaster`'s otherwise.

A seeding whose EM `sal` refuses is dropped rather than ending the start
(T- #596): on CalicoST hard (`1ae26365`) with the outlier filter off, one
`kmeans++` seeding of five collapses a component to weight 1e-41 and `sal`'s
dispersion M step raises. The seedings are independent, so the rest are run
again on the streams `sal` gave them and the best survivor kept; the start
fails only when every seeding does. A call where none fails is `sal`'s,
unchanged.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
from cnaster.hmm_initialize import gmm_init as UPSTREAM

from port.patch._signature import as_upstream

__all__ = [
    "DEFAULT",
    "POLISH_SECONDS",
    "checked",
    "dropped",
    "gmm_init",
    "instance_of",
]

logger = logging.getLogger(__name__)

_DROPPED: list[tuple[int, str]] = []

DEFAULT = "kmeans++x5+em"
"""The start `--sal` installs: kmeans++ in rate space, best of five, each polished by EM."""

POLISH_SECONDS = 160.0
"""The seconds the start and its polish may spend, `sal`'s notebook budget."""

DISPERSION = 10.0
"""The seam's negative-binomial size, `1 / alpha` at `backends.DEFAULT_ALPHA`."""

CONCENTRATION = 1_000.0
"""The seam's beta-binomial `alpha + beta`, `backends.DEFAULT_TAU`."""


def dropped() -> list[tuple[int, str]]:
    """Each seeding dropped since import: its index among the start's, and why `sal` refused it."""
    return list(_DROPPED)


def checked(start: str) -> str:
    """`start`, refused here if `sal` names no such start rather than hours in."""
    from sal.search.mixture_starts import lookup

    lookup(start)
    return start


def instance_of(
    X: np.ndarray, base_nb_mean: np.ndarray, total_bb_RD: np.ndarray, n_states: int
) -> Any:
    """The call's pseudobulk as `sal`'s `MixtureInstance`, conditioned on exposure and trials.

    Bins with zero exposure are left out, as `cnaster` scores their totals as
    uninformative. Starts seed in rate space: the total per unit exposure and
    the B count per trial, a bin of no trials read at the pooled rate.
    """
    from sal.opt.emission_mixture import CountPairSeeding
    from sal.search.mixture_starts import MixtureInstance

    counts = np.asarray(X, dtype=np.float64).reshape(X.shape[0], 2, -1)[:, :, 0]
    exposure = np.asarray(base_nb_mean, dtype=np.float64).reshape(X.shape[0], -1)[:, 0]
    trials = np.asarray(total_bb_RD, dtype=np.float64).reshape(X.shape[0], -1)[:, 0]
    kept = exposure > 0.0

    if not kept.any():
        msg = "every bin has zero exposure; there is no total to fit"
        raise ValueError(msg)

    observations = counts[kept]
    covariate = np.column_stack([exposure, trials])[kept]
    pooled = observations[:, 1].sum() / max(covariate[:, 1].sum(), 1.0)
    rows = np.column_stack(
        [
            observations[:, 0] / covariate[:, 0],
            np.divide(
                observations[:, 1],
                covariate[:, 1],
                out=np.full(observations.shape[0], pooled),
                where=covariate[:, 1] > 0,
            ),
        ]
    )
    at = CountPairSeeding(
        dispersion=DISPERSION,
        concentration=CONCENTRATION,
        joint=False,
        trials=max(float(np.rint(np.median(covariate[:, 1]))), 1.0),
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


def fitted(
    instance: Any, start: str, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """`(log_mu, p_binom)` of `start` on `instance`, polished by `sal`'s EM."""
    from sal.search.mixture_starts import BestOf, Selection, lookup, polish

    chosen = lookup(start)

    if isinstance(chosen, BestOf) and chosen.select is Selection.POLISHED:
        # NB `polished` spawns one stream per seeding from `rng`; a copy taken
        #    first hands the survivors the same streams.
        spare = copy.deepcopy(rng)

        try:
            _, best = chosen.polished(
                instance, rng, seconds=POLISH_SECONDS, passes=None, tolerance=1e-6
            )
            components = best.components
        except ValueError:
            components = _surviving(chosen, instance, spare)
    else:
        seeded = chosen(instance, rng)
        components = polish(
            instance, seeded.components, seconds=POLISH_SECONDS, tolerance=1e-6
        ).components

    from sal.emissions import CountPairEmission

    if not isinstance(components, CountPairEmission):  # invariant
        msg = f"expected a CountPairEmission, got {type(components).__name__}"
        raise TypeError(msg)

    depth = np.asarray(components.total.mean, dtype=np.float64).reshape(-1)
    rate = np.asarray(components.rate, dtype=np.float64).reshape(-1)
    return np.log(np.maximum(depth, 1e-12)), rate


def _surviving(chosen: Any, instance: Any, rng: np.random.Generator) -> Any:
    """The best polished seeding of `chosen` among those `sal` does not refuse.

    Each seeding runs on the stream `polished` would have spawned for it and
    is polished under the same budget, so a survivor is the fit `sal`'s own
    best-of computes for it.
    """
    from sal.search.mixture_starts import lookup, polish

    rounds = -(-chosen.n // min(chosen.workers, chosen.n))
    fits = []

    for index, stream in enumerate(rng.spawn(chosen.n)):
        try:
            seeded = lookup(chosen.name)(instance, stream)
            fits.append(
                polish(
                    instance,
                    seeded.components,
                    seconds=POLISH_SECONDS / rounds,
                    tolerance=1e-6,
                )
            )
        except ValueError as error:
            _DROPPED.append((index, str(error)))
            logger.info(f"{chosen.key}: seeding {index} dropped: {error}")

    if not fits:
        msg = f"{chosen.key}: sal refused every seeding"
        raise ValueError(msg)

    finals = [float(fit.log_likelihoods[-1]) for fit in fits]
    return fits[int(np.argmax(finals))].components


@as_upstream(UPSTREAM, start=None, distinct=False)
def gmm_init(arguments: dict[str, Any], options: dict[str, Any]) -> Any:
    """`cnaster`'s initializer signature; `sal`'s `start` on the read-depth + BAF call.

    Elsewhere, and with no `start`, `cnaster`'s initializer, or
    `port.patch.hmm_initialize.distinct`'s where `distinct` is set.
    `port.patch.hmrf.run_core_inference` binds both from its own options.
    """
    from port.patch.hmm_initialize import distinct

    params = str(arguments.get("params", ""))
    start = options["start"]

    if start is None or "m" not in params or arguments.get("only_minor", True):
        fallback = distinct.gmm_init if options["distinct"] else distinct.UPSTREAM
        return fallback(**arguments)

    n_states = int(arguments["n_states"])
    X = np.asarray(arguments["X"])
    base_nb_mean = np.asarray(arguments["base_nb_mean"])
    total_bb_RD = np.asarray(arguments["total_bb_RD"])
    rng = np.random.default_rng([int(arguments.get("random_state") or 0), 0])

    log_mu, p_binom = fitted(
        instance_of(X, base_nb_mean, total_bb_RD, n_states), start, rng
    )
    return log_mu.reshape(-1, 1), p_binom.reshape(-1, 1), None, None
