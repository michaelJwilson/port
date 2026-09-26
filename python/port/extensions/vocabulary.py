"""The names `port`'s own API uses, and where each one comes from.

`CLAUDE.md`'s **API Conventions** refer here. A concept particular to the
application takes `cnaster`'s name for it; a concept that is not takes
`snakes_and_ladders`' (`sal`). Neither is coined here where one of the two
already names it, so a reader moving between the three repositories reads
one word for one thing.

A drop-in is exempt: a name `cnaster` defines keeps `cnaster`'s signature,
argument for argument, because rebinding it is how it installs. The terms
govern what `port` owns -- its extensions, its seams and its results.

`replaces` lists the words `port` has used for a term's concept. Its own API
does not use them; `tests/test_api_conventions.py` reads the source and
refuses a new one, and lists every current use with the ticket converging
it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = ["TERMS", "Term", "replaced_by"]


@dataclass(frozen=True)
class Term:
    """One concept, its one name, and the names it retires."""

    name: str
    meaning: str
    source: Literal["cnaster", "sal"]
    replaces: tuple[str, ...] = ()


TERMS: tuple[Term, ...] = (
    # --- the data, as cnaster names it ------------------------------------
    Term(
        "single_X",
        "counts per `(bin, channel, spot)`: channel 0 read depth, 1 B allele",
        "cnaster",
    ),
    Term("X", "`single_X` pooled over the spots of each clone", "cnaster"),
    Term(
        "base_nb_mean",
        "negative binomial exposure per `(bin, spot)`: the expected depth at rate one",
        "cnaster",
    ),
    Term(
        "total_bb_RD",
        "beta-binomial trials per `(bin, spot)`: the allele-informative depth",
        "cnaster",
        replaces=("total_bb_rd",),
    ),
    Term(
        "lengths",
        "bins per segment, summing to `n_obs`; the chain restarts at each",
        "cnaster",
    ),
    Term("n_obs", "bins", "cnaster", replaces=("n_bins",)),
    Term("n_clones", "clones, the labels of the spatial field", "cnaster"),
    Term("coords", "spot positions, `(n_spots, 2)`", "cnaster"),
    Term("sample_ids", "sample of each spot", "cnaster"),
    Term("single_tumor_prop", "tumour proportion per spot, or `None`", "cnaster"),
    Term("adjacency_mat", "spot adjacency, sparse `(n_spots, n_spots)`", "cnaster"),
    Term("smooth_mat", "spot pooling weights, sparse `(n_spots, n_spots)`", "cnaster"),
    # --- the model ---------------------------------------------------------
    Term(
        "log_mu",
        "log negative binomial rate per state, relative to `base_nb_mean`",
        "cnaster",
        replaces=("log_mus",),
    ),
    Term("alphas", "negative binomial dispersion per state", "cnaster"),
    Term("p_binom", "beta-binomial success probability per state", "cnaster"),
    Term(
        "taus",
        "beta-binomial concentration per state",
        "cnaster",
        replaces=("tau",),
    ),
    Term(
        "normal_log_lambda",
        "log share of the normal library per bin, the `log Z_c` weights",
        "cnaster",
        replaces=("log_lambda",),
    ),
    Term("log_transmat", "log transition matrix, `(n_states, n_states)`", "cnaster"),
    Term("log_startprob", "log initial state distribution", "cnaster"),
    Term(
        "log_sitewise_transmat",
        "log phase-switch probability per bin",
        "cnaster",
    ),
    Term(
        "log_emission",
        "log emission per `(state, bin, spot)`",
        "cnaster",
    ),
    Term(
        "pred_cnv",
        "decoded state per `(bin, clone)`",
        "cnaster",
    ),
    Term(
        "spatial_weight",
        "the Potts coupling: the weight of an edge whose ends agree",
        "cnaster",
        replaces=("beta",),
    ),
    Term(
        "single_llf",
        "log-likelihood per `(spot, clone)`, the field the labelling maximizes",
        "cnaster",
    ),
    Term(
        "res",
        "a fit's result dictionary, keyed as `cnaster` writes it",
        "cnaster",
        replaces=("fit", "result"),
    ),
    # --- what is not application specific, as sal names it -----------------
    Term(
        "energy",
        "a minimized objective; lower is better",
        "sal",
        replaces=("cost",),
    ),
    Term(
        "log_likelihood",
        "a maximized log probability of the data",
        "sal",
        replaces=("score", "llf"),
    ),
    Term("n_states", "hidden states", "sal"),
    Term(
        "exposure",
        "what a count family's mean is scaled by, in a kernel that knows no "
        "bins; `base_nb_mean` is the application's",
        "sal",
    ),
    Term(
        "dispersion",
        "negative binomial dispersion, elementwise, in a kernel that knows no "
        "states; `alphas` is the application's",
        "sal",
        replaces=("alpha",),
    ),
    Term(
        "trials",
        "beta-binomial trials, in a kernel that knows no bins; `total_bb_RD` "
        "is the application's",
        "sal",
    ),
    Term(
        "field",
        "a per-`(site, state)` array a labelling is scored against",
        "sal",
    ),
    Term(
        "rng",
        "`np.random.Generator`, keyword-only; a caller seeds it",
        "sal",
        replaces=("seed", "random_state"),
    ),
    Term("start", "the initial state of an iterative run, keyword-only", "sal"),
    Term(
        "max_iterations",
        "the loop cap; the unit of one iteration stated per loop",
        "sal",
        replaces=("max_iter", "max_passes", "n_iter"),
    ),
    Term(
        "tolerance",
        "the stopping tolerance, relative unless stated",
        "sal",
        replaces=("tol",),
    ),
    Term(
        "termination",
        "`Termination`: whether and why an iterative run stopped, required",
        "sal",
        replaces=("niter", "converged"),
    ),
)
"""Every term, in the order a reader meets them: data, model, generic."""


def replaced_by() -> dict[str, str]:
    """Each retired word, mapped to the term that replaces it."""
    return {old: term.name for term in TERMS for old in term.replaces}
