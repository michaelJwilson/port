"""Set aside (#467): the read-depth stage's HMM start from `sal`'s covariate count-pair mixture.

Ticket: #467 -- `sal`'s covariate count-pair mixture as the read-depth HMM
  start, set aside as worse than `distinct.gmm_init`.
Measurement: copy ARI, `distinct.gmm_init` against this start, `--sal`:
  hard 0.919 / 0.864, easy 0.819 / 0.684; wall time 2-4x.
Exit: graduate to `patch/`, installed as `run_core_inference`'s
  `hmm_initializer`, if it matches `distinct`'s copy ARI on both samples;
  else retire.

**#424's backend as the initializer, measured and worse than `distinct`'s
GMM**, under `--sal` with #476's clone flags (copy ARI, phase-free exact
altered; clone ARI unchanged at 0.982 hard, 0.986 easy):

| sample | `distinct.gmm_init` | this start |
| --- | --- | --- |
| CalicoST hard | 0.919, 0.472 | 0.864, 0.307 |
| CalicoST easy | 0.819, 0.385 | 0.684, 0.339 |

Wall time rose 2-4x (5 restarts of EM on the clone-stacked counts). The
mixture places states by likelihood mass, as the GMM does before
`distinct` merges its near-duplicates, and nothing here merges them.

Installed by nothing. To rerun it, pass :func:`gmm_init` as
`run_core_inference`'s `hmm_initializer`.

`gmm_init` fits Gaussians to log depth ratios and BAFs, then keeps the most
populated components (`distinct` merges near-duplicates first). On CalicoST
hard that start leaves 3 of 7 states within 0.14 of neutral and 2 on empty
outliers, and no state at the planted gain or copy-neutral LOH (#471).

`gmm_init` here fits the mixture **in the family the data came from**:
`sal.opt.emission_mixture`, a negative binomial on each bin's total with its
`base_nb_mean` as exposure, times a beta-binomial on its B-allele count out
of `total_bb_RD` (`port.sandbox.patch.hmm_initialize.backends.sal_emission_backend`).
Seeded by D-squared sampling, `RESTARTS` times; the highest log-likelihood
is kept.

A call with no exposure -- the BAF-only stage -- has no total to fit, and is
`distinct.gmm_init`'s.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["RESTARTS", "gmm_init"]

RESTARTS = 5
"""Independent seedings; the fit of highest log-likelihood is kept."""


def gmm_init(
    n_states: int,
    X: np.ndarray,
    base_nb_mean: np.ndarray,
    total_bb_RD: np.ndarray,
    params: str,
    lengths: Any,
    log_transmat: Any,
    log_sitewise_transmat: Any,
    random_state: int | None = None,
    in_log_space: bool = True,
    only_minor: bool = True,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, Any, Any]:
    """`cnaster`'s initializer signature; `sal`'s mixture where there is exposure."""
    from port.patch.hmm_initialize import distinct
    from port.sandbox.patch.hmm_initialize.backends import sal_emission_backend

    base = np.asarray(base_nb_mean, dtype=np.float64)

    if "m" not in params or not np.any(base > 0):
        return distinct.gmm_init(  # type: ignore[no-any-return]
            n_states,
            X,
            base_nb_mean,
            total_bb_RD,
            params,
            lengths,
            log_transmat,
            log_sitewise_transmat,
            random_state=random_state,
            in_log_space=in_log_space,
            only_minor=only_minor,
            **kwargs,
        )

    seed = 0 if random_state is None else int(random_state)
    best = None

    for restart in range(RESTARTS):
        candidate = sal_emission_backend(
            np.asarray(X), base, np.asarray(total_bb_RD), n_states, seed=seed + restart
        )
        score = float(candidate.detail["log_likelihood"])

        if best is None or score > best[0]:
            best = (score, candidate)

    if best is None:  # invariant: RESTARTS >= 1
        msg = "no restart ran"
        raise AssertionError(msg)

    chosen = best[1]

    return (
        np.asarray(chosen.log_mu, dtype=np.float64).reshape(-1, 1),
        np.asarray(chosen.p_binom, dtype=np.float64).reshape(-1, 1),
        None,
        None,
    )
