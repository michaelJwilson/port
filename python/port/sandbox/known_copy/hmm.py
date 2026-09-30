"""The copy-state HMM's Baum-Welch at known clones: `cnaster`'s own, on the clones stacked along the genome.

Ticket: #540 -- copy-state starts at known clones, polished by Baum-Welch
  (`tests/studies/copy_state_stream.py`).
Measurement: `docs/study-copy-states.md`: each start's gap in log-likelihood
  and share of rows off their planted state, before and after Baum-Welch.
Exit: retire with the study; a start it finds better graduates through
  `port.extensions.copy_starts`.

`baum_welch` is `cnaster.hmm.pipeline_baum_welch` with `hmm_nophasing`, as
`cnaster.hmrf` calls it for the BAF + RDR stage: states' `log_mu` and
`p_binom` fitted (`params = "smp"`), the NB and beta-binomial dispersions
shared across states, the stickiness `T` of the configuration shipped for these
samples. The allele reads are phased by the truth, so no phase is modelled.
`decode` scores and labels the rows at given states without fitting: the same
forward-backward at `cnaster`'s initial dispersions.

Both run under `port.sandbox.patch.hmm_nophasing.nb_logpmf.patched` (#560):
`cnaster`'s negative binomial scores any count at probability 1 once its `p`
rounds to 1, and Baum-Welch drove a state there. `degenerate` still flags a
fit whose state would be scored so by the unpatched kernel.
"""

from __future__ import annotations

import time
from typing import Any, NamedTuple

import numpy as np

__all__ = ["Fit", "T", "baum_welch", "decode", "degenerate", "missed"]

T = 1.0 - 1e-7
"""`hmm.t` of `tests/data/zenodo_sim_config.yaml`: the probability a clone stays in its state from one bin to the next."""

ALPHA, TAU = 0.5, 1_000.0
"""`hmm_nophasing.get_initial_params`' NB and beta-binomial dispersions: what an unfitted start is scored at."""


class Fit(NamedTuple):
    """States, their log-likelihood on the stacked clones, each row's state, the seconds it took, and whether it is degenerate."""

    log_mu: np.ndarray
    p_binom: np.ndarray
    log_likelihood: float
    label: np.ndarray
    seconds: float
    degenerate: bool = False


def degenerate(
    log_mu: np.ndarray, alpha: float, exposure: np.ndarray, label: np.ndarray
) -> bool:
    """Whether a row's state scores it at probability 1 through `cnaster`'s negative binomial.

    `cnaster.hmm_nophasing.nbinom_logpmf_numba` returns 0 -- probability 1,
    for any count -- when `p = 1 / (1 + alpha * exposure * mu)` is at least
    1.0, which it is in float64 once `alpha * exposure * mu` is below about
    1e-16. Baum-Welch finds that: on dev_tree_1s_hard r0 one fit drove a
    state to log mu = -43, gave it 99% of rows, and scored -23,359 nats
    against the planted states' -76,306. The same arithmetic as the kernel,
    on each row at its own state.
    """
    mu = np.exp(np.asarray(log_mu, dtype=np.float64).ravel())[np.asarray(label)]
    lam = np.asarray(exposure, dtype=np.float64).ravel() * mu
    p = 1.0 / (1.0 + max(float(alpha), 1.0e-10) * lam)
    return bool(np.any((lam > 0.0) & (p >= 1.0)))


CONFIG = "tests/data/zenodo_sim_config.yaml"
"""The configuration shipped for these samples, relative to the repository: what `cnaster`'s encoder and M step read."""


def _configured() -> None:
    """`cnaster`'s global configuration, set from `CONFIG` unless one is set already."""
    from cnaster.config import YAMLConfig, get_global_config, set_global_config

    if get_global_config() is None:
        from pathlib import Path

        import yaml

        path = Path(__file__).resolve().parents[4] / CONFIG
        set_global_config(YAMLConfig(yaml.safe_load(path.read_text())))


def _arrays(problem: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.stack([problem.total, problem.b], axis=1)[:, :, None]
    return x, np.asarray(problem.exposure)[:, None], np.asarray(problem.trials)[:, None]


def baum_welch(
    problem: Any,
    log_mu: np.ndarray,
    p_binom: np.ndarray,
    *,
    max_iter: int = 100,
    tol: float = 1e-3,
) -> Fit:
    """`cnaster`'s Baum-Welch from `(log_mu, p_binom)` on `problem`'s stacked clones."""
    from cnaster.hmm import pipeline_baum_welch
    from cnaster.hmm_nophasing import hmm_nophasing

    from port.sandbox.patch.hmm_nophasing.nb_logpmf import patched

    _configured()
    x, exposure, trials = _arrays(problem)
    n_states = int(np.asarray(log_mu).size)
    opened = time.perf_counter()
    with patched():
        result = pipeline_baum_welch(
            None, x, np.asarray(problem.lengths), n_states, exposure, trials, np.zeros(x.shape[0]), None,
            hmmclass=hmm_nophasing, params="smp", t=T, shared_NB_dispersion=True, shared_BB_dispersion=True,
            is_diag=True, init_log_mu=np.asarray(log_mu, dtype=np.float64).reshape(-1, 1),
            init_p_binom=np.clip(np.asarray(p_binom, dtype=np.float64), 1e-4, 1 - 1e-4).reshape(-1, 1),
            max_iter=max_iter, tol=tol, clone_lengths=np.bincount(problem.clone),
        )  # fmt: skip
    log_mu = np.asarray(result.params.new_log_mu).ravel()
    label = np.asarray(result.profile.pred_cnv, dtype=np.int64).ravel()
    alpha = float(np.ravel(result.params.new_alphas)[0])
    return Fit(
        log_mu, np.asarray(result.params.new_p_binom).ravel(), float(result.llf), label, time.perf_counter() - opened,
        degenerate(log_mu, alpha, problem.exposure, label),
    )  # fmt: skip


def decode(problem: Any, log_mu: np.ndarray, p_binom: np.ndarray) -> Fit:
    """The log-likelihood and each row's most probable state at `(log_mu, p_binom)`, nothing fitted."""
    import scipy.special
    from cnaster.hmm_nophasing import get_log_transmat, hmm_nophasing

    x, exposure, trials = _arrays(problem)
    n_states = int(np.asarray(log_mu).size)
    from port.sandbox.patch.hmm_nophasing.nb_logpmf import patched

    opened = time.perf_counter()
    with patched():
        rdr, baf = hmm_nophasing.compute_emission_probability_nb_betabinom(
            x, exposure, np.asarray(log_mu, dtype=np.float64).reshape(-1, 1), np.full((n_states, 1), ALPHA), trials,
            np.clip(np.asarray(p_binom, dtype=np.float64), 1e-4, 1 - 1e-4).reshape(-1, 1), np.full((n_states, 1), TAU),
        )  # fmt: skip
    emission = rdr + baf
    lengths = np.asarray(problem.lengths)
    transmat = get_log_transmat(n_states, T)
    startprob = np.full(n_states, -np.log(n_states))
    sitewise = np.zeros(x.shape[0])
    alpha = hmm_nophasing.forward_lattice(
        lengths, transmat, startprob, emission, sitewise
    )
    beta = hmm_nophasing.backward_lattice(
        lengths, transmat, startprob, emission, sitewise
    )
    llf = float(
        np.sum(scipy.special.logsumexp(alpha[:, np.cumsum(lengths) - 1], axis=0))
    )
    label = np.argmax(alpha + beta, axis=0).astype(np.int64)
    return Fit(
        np.asarray(log_mu).ravel(), np.asarray(p_binom).ravel(), llf, label, time.perf_counter() - opened,
        degenerate(np.asarray(log_mu), ALPHA, problem.exposure, label),
    )  # fmt: skip


def missed(label: np.ndarray, truth: np.ndarray) -> int:
    """Rows whose state is not the planted one, under the 1-1 matching of fitted to planted states that misses fewest."""
    from scipy.optimize import linear_sum_assignment

    n = int(max(label.max(), truth.max())) + 1
    agree = np.zeros((n, n), dtype=np.int64)
    np.add.at(agree, (label, truth), 1)
    rows, cols = linear_sum_assignment(-agree)
    return int(label.size - agree[rows, cols].sum())
