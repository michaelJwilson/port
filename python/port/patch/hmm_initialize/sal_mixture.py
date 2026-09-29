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
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT",
    "POLISH_SECONDS",
    "gmm_init",
    "installed",
    "instance_of",
    "sal_mixture",
]

DEFAULT = "kmeans++x5+em"
"""The start `--sal` installs: kmeans++ in rate space, best of five, each polished by EM."""

POLISH_SECONDS = 160.0
"""The seconds the start and its polish may spend, `sal`'s notebook budget."""

DISPERSION = 10.0
"""The seam's negative-binomial size, `1 / alpha` at `backends.DEFAULT_ALPHA`."""

CONCENTRATION = 1_000.0
"""The seam's beta-binomial `alpha + beta`, `backends.DEFAULT_TAU`."""

_START: list[str | None] = [None]


def installed() -> bool:
    """Whether a `sal` start is the read-depth stage's initializer."""
    return _START[0] is not None


@contextmanager
def sal_mixture(start: str = DEFAULT) -> Iterator[None]:
    """Hand `run_core_inference` this module's `gmm_init`, with `start`, for the block."""
    from sal.search.mixture_starts import lookup

    lookup(start)
    previous = _START[0]
    _START[0] = start

    try:
        yield
    finally:
        _START[0] = previous


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


def fitted(instance: Any, start: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """`(log_mu, p_binom)` of `start` on `instance`, polished by `sal`'s EM."""
    from sal.search.mixture_starts import BestOf, Selection, lookup, polish

    rng = np.random.default_rng([seed, 0])
    chosen = lookup(start)

    if isinstance(chosen, BestOf) and chosen.select is Selection.POLISHED:
        _, best = chosen.polished(
            instance, rng, seconds=POLISH_SECONDS, passes=None, tolerance=1e-6
        )
        components = best.components
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


def gmm_init(*args: Any, **kwargs: Any) -> Any:
    """`cnaster`'s initializer signature; `sal`'s start on the read-depth + BAF call."""
    from port.patch.hmm_initialize import distinct

    params = str(args[4] if len(args) > 4 else kwargs.get("params", ""))
    only_minor = kwargs.get("only_minor", args[10] if len(args) > 10 else True)
    start = _START[0]

    if start is None or "m" not in params or only_minor:
        fallback = distinct.gmm_init if distinct.installed() else distinct.UPSTREAM
        return fallback(*args, **kwargs)

    n_states = int(args[0] if args else kwargs["n_states"])
    X = np.asarray(args[1] if len(args) > 1 else kwargs["X"])
    base_nb_mean = np.asarray(args[2] if len(args) > 2 else kwargs["base_nb_mean"])
    total_bb_RD = np.asarray(args[3] if len(args) > 3 else kwargs["total_bb_RD"])
    seed = int(kwargs.get("random_state") or 0)

    log_mu, p_binom = fitted(
        instance_of(X, base_nb_mean, total_bb_RD, n_states), start, seed
    )
    return log_mu.reshape(-1, 1), p_binom.reshape(-1, 1), None, None
