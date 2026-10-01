r"""Each clone's pseudobulk dispersion from one per-spot dispersion (#566, #100, #78).

`cnaster` fits one NB `alpha` and one BB `tau` to the clone-stacked
pseudobulks, so the fitted value mixes clone sizes: #78 measured the
aggregate's `alpha` falling 8.2x between clones of 8 and 64 spots. Here the
fitted parameter is the **per-spot** dispersion, and each pseudobulk row
scores at the moment-matched value of a sum of independent spots:

- NB, spots `s` of clone `c` with means `mu_s`: the sum's variance is
  `M + alpha sum_s mu_s^2`, so `alpha_c = alpha / S_eff,c`,
  `S_eff,c = (sum_s mu_s)^2 / sum_s mu_s^2`. `mu_s` is the spot's summed
  exposure, `sum_g base[g, s]`.
- BB, with `rho = 1 / (1 + tau)` and `n_s` trials per spot in the row's bin:
  `rho_c = rho * sum_s n_s (n_s - 1) / (N_c (N_c - 1))`, `N_c = sum_s n_s`.

**Conditions.** `S_eff,c` is one number per clone only where the exposure
factorizes as bin x spot, `base[g, s] = lambda_g T_s` (#100); `port.sim`'s
fixtures do, and elsewhere it is a moment-matched approximation that has to
be measured. The BB factor is per row and moment-exact, but #100 measured
the aggregate at 3x the Monte Carlo floor once trial counts span two orders
of magnitude. A row with `N_c <= 1` keeps the per-spot `rho`.

The factors are data: `merge_pseudobulk_by_index_mix` (port's row) records
them per clone assignment while :func:`recording` is open, keyed by the
pseudobulk's counts, and port's `hmm_nophasing` finds the record for the
rows it is handed. A fit whose rows match no record is refused, never fitted
unrescaled.
"""

from __future__ import annotations

import contextlib
import hashlib
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from typing import Any, NamedTuple

import numpy as np
from scipy.special import gammaln

from port.patch.hmm_nophasing.gradient import DISPERSION_FLOOR

__all__ = [
    "Components",
    "Rescale",
    "alpha_rows",
    "bb_factor",
    "bb_logpmf",
    "find",
    "key",
    "nb_factor",
    "nb_logpmf",
    "record",
    "recording",
    "rho_rows",
    "tau_rows",
]


class Rescale(NamedTuple):
    """One clone assignment's factors, on the clone-stacked rows."""

    nb: np.ndarray
    """`(n_clones,)` `1 / S_eff,c`: multiplies `alpha`."""
    bb: np.ndarray
    """`(n_rows,)` `sum n_s (n_s - 1) / (N (N - 1))`: multiplies `rho`."""
    lengths: tuple[int, ...]
    """Rows per clone, clone-major, as `clone_stack_obs` stacks them."""

    @property
    def nb_rows(self) -> np.ndarray:
        """`(n_rows,)` the NB factor repeated over each clone's rows."""
        return np.repeat(self.nb, self.lengths)


def nb_factor(exposure: np.ndarray) -> float:
    """`1 / S_eff = sum_s mu_s^2 / (sum_s mu_s)^2`, `mu_s` the spot's summed exposure; 1 without exposure."""
    spots = np.asarray(exposure, dtype=np.float64).sum(axis=0)
    total = float(spots.sum())
    return float(np.sum(spots * spots)) / (total * total) if total > 0.0 else 1.0


def bb_factor(trials: np.ndarray) -> np.ndarray:
    """`(n_bins,)` `sum_s n_s (n_s - 1) / (N (N - 1))` per bin; 1 where `N <= 1`."""
    n = np.asarray(trials, dtype=np.float64)
    total = n.sum(axis=1)
    pairs = np.sum(n * (n - 1.0), axis=1)
    safe = np.where(total > 1.0, total * (total - 1.0), 1.0)
    return np.where(total > 1.0, pairs / safe, 1.0)


class Components(NamedTuple):
    """The clone-shared dispersion beside the per-spot one: `--dispersion-two-component` (#566).

    `alpha_row = alpha_shared + alpha_k / S_eff,c` and `rho_row = rho_shared
    + rho_k g_row`: a clone-level effect -- #556's tumour gene program, sd
    1.78 about `lambda` against 0.35 for normal spots -- does not average
    away with clone size, and the per-spot term does. Held in logs, as the
    M step fits them; `rho_shared = 1 / (1 + tau_shared)`.
    """

    log_alpha: float
    log_tau: float

    @property
    def alpha(self) -> float:
        return float(np.exp(self.log_alpha))

    @property
    def rho(self) -> float:
        return float(1.0 / (1.0 + np.exp(self.log_tau)))


RHO_CEILING = 1.0 - 1e-9
"""`rho_row` held below 1, where `tau_row = 1 / rho - 1` reaches 0."""


def alpha_rows(alphas: np.ndarray, rescale: Rescale, shared: float = 0.0) -> np.ndarray:
    """`(K, n_rows)`: `shared + alpha_k f_c`, each state's per-spot `alpha` at each row's clone."""
    alpha = np.asarray(alphas, dtype=np.float64).reshape(-1, 1)
    out: np.ndarray = shared + alpha * rescale.nb_rows[None, :]
    return out


def rho_rows(taus: np.ndarray, rescale: Rescale, shared: float = 0.0) -> np.ndarray:
    """`(K, n_rows)`: `rho_row = shared + g_row / (1 + tau_k)`, below 1."""
    tau = np.asarray(taus, dtype=np.float64).reshape(-1, 1)
    out: np.ndarray = np.minimum(
        shared + rescale.bb[None, :] / (1.0 + tau), RHO_CEILING
    )
    return out


def tau_rows(taus: np.ndarray, rescale: Rescale, shared: float = 0.0) -> np.ndarray:
    """`(K, n_rows)` `tau_row = 1 / rho_row - 1`; `(1 + tau) / g - 1` without a shared part.

    `inf` where `rho_row = 0` -- no spot holds two trials and nothing is
    shared, so the sum is binomial.
    """
    rho = rho_rows(taus, rescale, shared)
    with np.errstate(divide="ignore"):
        out: np.ndarray = np.where(
            rho > 0.0, 1.0 / np.where(rho > 0.0, rho, 1.0) - 1.0, np.inf
        )
    return out


def nb_logpmf(obs: Any, mean: Any, dispersion: Any) -> np.ndarray:
    """`cnaster`'s `_nb_logpmf_1d`, broadcasting: 0 where `mean <= 0` or `p` rounds to 1."""
    size = 1.0 / np.maximum(dispersion, DISPERSION_FLOOR)
    success = 1.0 / (1.0 + dispersion * mean)
    live = (mean > 0.0) & (success < 1.0) & (success > 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        score = (
            gammaln(obs + size)
            - gammaln(size)
            - gammaln(obs + 1.0)
            + size * np.log(success)
            + obs * np.log1p(-success)
        )

    return np.where(live, score, 0.0)


def bb_logpmf(obs: Any, total: Any, p_binom: Any, taus: Any) -> np.ndarray:
    """`cnaster`'s `_bb_logpmf_1d`, broadcasting: `a`, `b` floored, 0 where `k > n`; binomial where `tau` is `inf`."""
    binomial = np.isinf(taus)
    finite = np.where(binomial, 1.0, taus)
    a = np.maximum(p_binom * finite, DISPERSION_FLOOR)
    b = np.maximum((1.0 - p_binom) * finite, DISPERSION_FLOOR)
    valid = (obs >= 0) & (total >= 0) & (obs <= total)
    share = np.clip(p_binom, DISPERSION_FLOOR, 1.0 - DISPERSION_FLOOR)

    with np.errstate(invalid="ignore", divide="ignore"):
        choose = gammaln(total + 1.0) - gammaln(obs + 1.0) - gammaln(total - obs + 1.0)
        beta = (
            gammaln(obs + a)
            + gammaln(total - obs + b)
            - gammaln(total + a + b)
            - (gammaln(a) + gammaln(b) - gammaln(a + b))
        )
        limit = obs * np.log(share) + (total - obs) * np.log1p(-share)
        score = choose + np.where(binomial, limit, beta)

    return np.where(valid, score, 0.0)


def key(counts: np.ndarray) -> bytes:
    """The record key: a digest of the clone-stacked `(n_rows, 2)` counts."""
    rows = np.ascontiguousarray(np.asarray(counts, dtype=np.float64).reshape(-1, 2))
    return hashlib.blake2b(rows.tobytes(), digest_size=16).digest()


_RECORDS: OrderedDict[bytes, Rescale] = OrderedDict()
_ACTIVE: list[bool] = []
_KEPT = 16
"""Records held: one per clone assignment, the oldest dropped first."""


def record(
    X: np.ndarray,
    single_base_nb_mean: np.ndarray,
    single_total_bb_RD: np.ndarray,
    clone_index: Sequence[Any],
) -> None:
    """Record `X`'s factors, if :func:`recording` is open; `clone_index` is the spots summed per clone."""
    if not _ACTIVE:
        return

    n_obs = int(X.shape[0])
    nb = np.ones(len(clone_index))
    bb = np.ones((len(clone_index), n_obs))

    for clone, idx in enumerate(clone_index):
        if len(idx) == 0:
            continue
        nb[clone] = nb_factor(single_base_nb_mean[:, idx])
        bb[clone] = bb_factor(single_total_bb_RD[:, idx])

    stacked = np.asarray(X).transpose(2, 0, 1).reshape(-1, 2)
    _RECORDS[key(stacked)] = Rescale(
        nb, bb.reshape(-1), tuple([n_obs] * len(clone_index))
    )

    while len(_RECORDS) > _KEPT:
        _RECORDS.popitem(last=False)


def find(X: np.ndarray) -> Rescale:
    """The record for the clone-stacked `X`, `(n_rows, 2, 1)`; refused if there is none."""
    found = _RECORDS.get(key(np.asarray(X)[:, :, 0]))

    if found is None:
        msg = (
            "no dispersion rescale recorded for these rows: the per-spot "
            "dispersion needs the spots behind each pseudobulk (#566)"
        )
        raise ValueError(msg)

    return found


def active() -> bool:
    """Whether a :func:`recording` block is open."""
    return bool(_ACTIVE)


@contextlib.contextmanager
def recording() -> Iterator[None]:
    """Record each pseudobulk's factors for the block; `run_cnaster_port --dispersion-rescale`."""
    _ACTIVE.append(True)

    try:
        yield
    finally:
        _ACTIVE.pop()

        if not _ACTIVE:
            _RECORDS.clear()
