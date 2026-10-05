"""The HMM's copy-state start from `sal`'s mixture starts or the integer lattice, polished by `sal`'s EM (#489, #540, #547).

`cnaster`'s start, and `distinct`'s (#348), fit Gaussians to log depth ratios
and BAFs. These fit the mixture **in the family the data came from**: a
negative binomial on each bin's total with its `base_nb_mean` as exposure,
times a beta-binomial on its B count out of `total_bb_RD`, each bin's
exposure and trials its covariate (sal #933/#1083). A start is seeded,
polished by `sal`'s EM and handed to the HMM as `(log_mu, p_binom)`, through
`port.extensions.copy_starts.run_start`.

Options, bound at install from `run_cnaster_port`'s flags:

- `start`: the BAF + RDR stage's start. `--sal` installs `kmeans++x5+em`
  (#489); `lattice` places the integer `(A, B)` lattice instead (#540).
- `baf_start`: the BAF-only stage's start, which otherwise keeps
  `distinct`'s where it is installed and `cnaster`'s elsewhere. That stage
  has no read depth, so its depth channel is a constant.

`only_minor=True` calls, the phasing's, keep the start they had.

**Seeding (#547).** `sal`'s `CountPairSeeding` floors a component's
negative-binomial mean at 1 and reads a row's second column as successes over
the instance's common trial count, as `sal.sim.count_pairs.rate_space` writes
it. `instance_of` divides the exposure by `EXPOSURE_SCALE`, so a loss's rate
is above that floor, and writes the B column over the common trial count.
The likelihood is the same model; only where a start can place a state
changes. Before #547 every BAF was seeded near 0.01 and no read-depth state
below neutral.

**`emission++` scores floored at 0 (#562).** A D-squared draw handed
`rng.choice` a round-off negative divergence (`-1.6e-15`) and the start was
refused. sal #1136 floors each divergence at 0 in the draw itself, where
port's patch of `_seed_scores` floored it, so the patch is retired (T- #632)
and the draws are unchanged.
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
"""Exposure is divided by this before `sal` reads it, so a state's rate per unit exposure is `mu` times it (#547).

`sal`'s seeding floors a component's negative-binomial mean at 1, which on
`cnaster`'s `base_nb_mean` -- `mu` of order 1 -- would seed a loss (`mu`
0.5) at neutral. The mean at each row is rate times exposure, so the
likelihood is unchanged."""


def checked(start: str) -> str:
    """`start`, refused here if no copy-state start has that name rather than hours in: the lattice's, or `sal`'s."""
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

    Bins with zero exposure are left out, as `cnaster` scores their totals as
    uninformative. Starts seed in `sal`'s rate space (`rate_space`): the
    total per unit of exposure over `EXPOSURE_SCALE`, and the B fraction over
    the common trial count, a bin of no trials read at the pooled fraction.
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


@as_upstream(UPSTREAM, start=None, distinct=False, baf_start=None)
def gmm_init(arguments: dict[str, Any], options: dict[str, Any]) -> Any:
    """`cnaster`'s initializer signature; `start` on the BAF + RDR call, `baf_start` on the BAF-only one.

    Elsewhere, and with neither, `cnaster`'s initializer, or
    `port.patch.hmm_initialize.distinct`'s where `distinct` is set.
    `port.patch.hmrf.run_core_inference` binds the options from its own.
    """
    from port.extensions.copy_starts import run_start
    from port.patch.hmm_initialize import distinct

    params = str(arguments.get("params", ""))
    stage = "rdrbaf" if "m" in params else "baf"
    chosen = options["start"] if stage == "rdrbaf" else options["baf_start"]

    if chosen is None or arguments.get("only_minor", True):
        fallback = distinct.gmm_init if options["distinct"] else distinct.UPSTREAM
        return fallback(**arguments)

    rng = np.random.default_rng([int(arguments.get("random_state") or 0), 0])
    result = run_start(
        checked(chosen), _call(arguments, stage), rng, seconds=POLISH_SECONDS
    )
    return result.log_mu.reshape(-1, 1), result.p_binom.reshape(-1, 1), None, None
