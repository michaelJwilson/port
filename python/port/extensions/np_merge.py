"""CalicoST's Neyman-Pearson merge of similar clones, which `cnaster` disabled (#497).

After each clone stage CalicoST merges clones whose decoded copy states agree
wherever the difference is long enough to test
(`calicost.hmm_NB_BB_phaseswitch.similarity_components_rdrbaf_neymanpearson`,
`np_threshold = 2.0`, `np_eventminlen = 10`). `cnaster` carries the call
commented out (`run_cnaster.py:743` and `:1172`) and the function is gone, so
nothing ever merges two clones that decode alike. On `dev_tree` 60 x 50 with
`cnaster`'s own loader that leaves clone_0 and normal one clone per slice: the
halves decode the same integer pair at 2,893 of 2,895 bins and the run scores
clone ARI 0.6686 for 0.9996.

**The test, as CalicoST states it.** For clones `c1, c2` and each pair of
states `(s1, s2)`, `s1 != s2`, the bins `B` where `c1` decodes `s1` and `c2`
decodes `s2` are an event. Its statistic is the mean log-likelihood per bin of
both clones' pseudobulks under their own states less under the swapped ones:

    t = mean_{b in B} [l_c1(s1, b), l_c2(s2, b)] - mean [l_c1(s2, b), l_c2(s1, b)]

The pair is mergeable when every event of at least `minlength` bins has
`t < threshold`; mergeable pairs are edges of a graph, and its maximal cliques,
largest first and then of least summed statistic, are the merged groups.
BAF-only stages score the beta-binomial alone (`params` without `m`).

**What differs from CalicoST, and why.** The emission is `cnaster`'s
unphased one, where a state is its own index (CalicoST's phased `2 n_states`
indices never occur), and each clone's read-depth rates carry its own
`new_log_mu_shift` (#435), which CalicoST's `tumor_prop is None` path does not
have. With every shift zero the groups and statistics are CalicoST's, which
`tests/test_np_merge.py` pins against CalicoST's function on the same inputs.
Maximal cliques are enumerated here (Bron-Kerbosch) rather than by `networkx`,
which port does not depend on; at ten clones the enumeration is immediate.
"""

from __future__ import annotations

import contextlib
import copy
from collections.abc import Iterator
from typing import Any

import numpy as np

__all__ = [
    "MINLENGTH",
    "THRESHOLD",
    "groups",
    "installed",
    "merged",
    "np_merge",
    "remember",
    "statistics",
]

THRESHOLD = 2.0
"""CalicoST's `np_threshold`: nats per bin an event must reach to keep two clones apart."""

MINLENGTH = 10
"""CalicoST's `np_eventminlen`: bins an event needs to be tested."""

_INSTALLED = [False]
_INPUTS: dict[str, Any] = {}


def installed() -> bool:
    """Whether the merge runs before `merge_by_minspots`."""
    return _INSTALLED[0]


@contextlib.contextmanager
def np_merge() -> Iterator[None]:
    """Run the merge before `cnaster`'s minimum-size merge for the block."""
    previous = _INSTALLED[0]
    _INSTALLED[0] = True

    try:
        yield
    finally:
        _INSTALLED[0] = previous
        _INPUTS.clear()


def remember(
    single_X: np.ndarray,
    single_base_nb_mean: np.ndarray,
    single_total_bb_RD: np.ndarray,
    params: str,
) -> None:
    """Hold the spot counts and parameters of the fit the next merge reads."""
    if _INSTALLED[0]:
        _INPUTS.update(
            single_X=single_X,
            single_base_nb_mean=single_base_nb_mean,
            single_total_bb_RD=single_total_bb_RD,
            params=params,
        )


def _column(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _emissions(
    X: np.ndarray,
    base_nb_mean: np.ndarray,
    total_bb_RD: np.ndarray,
    res: Any,
    shifts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """`(n_states, n_obs, n_clones)` read-depth and allele log-emissions, `cnaster`'s density."""
    from cnaster.hmm_nophasing import _bb_logpmf_1d, _nb_logpmf_1d

    log_mu = _column(res["new_log_mu"])
    alphas = _column(res["new_alphas"])
    p_binom = _column(res["new_p_binom"])
    taus = _column(res["new_taus"])
    n_obs, _, n_clones = X.shape
    n_states = p_binom.size
    rdr = np.zeros((n_states, n_obs, n_clones))
    baf = np.zeros((n_states, n_obs, n_clones))

    for c in range(n_clones):
        totals = np.ascontiguousarray(X[:, 0, c], dtype=np.float64)
        successes = np.ascontiguousarray(X[:, 1, c], dtype=np.float64)
        exposure = np.ascontiguousarray(base_nb_mean[:, c], dtype=np.float64)
        trials = np.ascontiguousarray(total_bb_RD[:, c], dtype=np.float64)

        for s in range(n_states):
            out = np.zeros(n_obs)
            _nb_logpmf_1d(
                totals,
                exposure,
                float(np.exp(log_mu[s] - shifts[c])),
                float(alphas[s % alphas.size]),
                out,
            )
            rdr[s, :, c] = out
            out = np.zeros(n_obs)
            _bb_logpmf_1d(
                successes,
                trials,
                float(p_binom[s]),
                float(taus[s % taus.size]),
                out,
            )
            baf[s, :, c] = out

    return rdr, baf


def statistics(
    X: np.ndarray,
    base_nb_mean: np.ndarray,
    total_bb_RD: np.ndarray,
    res: Any,
    params: str,
) -> dict[tuple[int, int], list[tuple[int, int, int, float]]]:
    """Per clone pair, each event's `(s1, s2, bins, t)`, CalicoST's statistic."""
    n_obs, _, n_clones = X.shape
    shifts = np.zeros(n_clones)

    if "m" in params and res.get("new_log_mu_shift") is not None:
        shifts = _column(res["new_log_mu_shift"])[:n_clones]

    rdr, baf = _emissions(X, base_nb_mean, total_bb_RD, res, shifts)
    score = baf + rdr if "m" in params else baf
    pred = np.asarray(res["pred_cnv"], dtype=np.int64).reshape(n_obs, n_clones)
    found: dict[tuple[int, int], list[tuple[int, int, int, float]]] = {}

    for c1 in range(n_clones):
        for c2 in range(c1 + 1, n_clones):
            events = []

            for s1, s2 in np.unique(pred[:, [c1, c2]], axis=0):
                if s1 == s2:
                    continue

                bins = np.where((pred[:, c1] == s1) & (pred[:, c2] == s2))[0]
                own = np.concatenate([score[s1, bins, c1], score[s2, bins, c2]])
                swapped = np.concatenate([score[s2, bins, c1], score[s1, bins, c2]])
                t = float(np.mean(own) - np.mean(swapped))
                events.append((int(s1), int(s2), int(bins.size), t))

            found[(c1, c2)] = events

    return found


def _cliques(nodes: list[int], edges: dict[tuple[int, int], float]) -> list[list[int]]:
    """Every maximal clique, by Bron-Kerbosch with a pivot."""
    neighbours: dict[int, set[int]] = {n: set() for n in nodes}

    for a, b in edges:
        neighbours[a].add(b)
        neighbours[b].add(a)

    found: list[list[int]] = []

    def expand(r: set[int], p: set[int], x: set[int]) -> None:
        if not p and not x:
            found.append(sorted(r))
            return

        pivot = max(p | x, key=lambda n: len(neighbours[n] & p))

        for v in sorted(p - neighbours[pivot]):
            expand(r | {v}, p & neighbours[v], x & neighbours[v])
            p = p - {v}
            x = x | {v}

    expand(set(), set(nodes), set())
    return found


def groups(
    X: np.ndarray,
    base_nb_mean: np.ndarray,
    total_bb_RD: np.ndarray,
    res: Any,
    params: str,
    *,
    threshold: float = THRESHOLD,
    minlength: int = MINLENGTH,
) -> list[list[int]]:
    """The merged groups of clone indices, CalicoST's rule, each sorted, ordered by least member."""
    n_clones = X.shape[2]
    edges: dict[tuple[int, int], float] = {}

    for (c1, c2), events in statistics(
        X, base_nb_mean, total_bb_RD, res, params
    ).items():
        tested = [t for _, _, n, t in events if n >= minlength]

        if not tested or max(tested) < threshold:
            edges[(c1, c2)] = max(tested) if tested else 1e-3

    def weight(clique: list[int]) -> float:
        return sum(edges[(a, b)] for i, a in enumerate(clique) for b in clique[i + 1 :])

    ranked = sorted(
        _cliques(list(range(n_clones)), edges), key=lambda c: (-len(c), weight(c))
    )
    covered: set[int] = set()
    chosen: list[list[int]] = []

    for clique in ranked:
        if not set(clique) & covered:
            chosen.append(clique)
            covered |= set(clique)

    chosen += [[c] for c in range(n_clones) if c not in covered]
    return sorted(chosen, key=min)


def merged(res: Any, chosen: list[list[int]]) -> Any:
    """`res` with each group one clone, CalicoST's way: the group's least clone keeps its path."""
    labels = np.unique(np.asarray(res["new_assignment"]))
    to_group = {int(labels[c]): g for g, members in enumerate(chosen) for c in members}
    first = [members[0] for members in chosen]
    out = copy.copy(res)
    out["new_assignment"] = np.array(
        [to_group[int(a)] for a in np.asarray(res["new_assignment"])], dtype=np.int64
    )
    pred = np.asarray(res["pred_cnv"])
    out["pred_cnv"] = pred.reshape(pred.shape[0], -1)[:, first]

    gamma = res.get("log_gamma")

    if gamma is not None and np.ndim(gamma) == 3:
        out["log_gamma"] = np.asarray(gamma)[:, :, first]

    shift = res.get("new_log_mu_shift")

    if shift is not None and np.size(shift) == len(labels):
        out["new_log_mu_shift"] = _column(shift)[first]

    out["total_llf"] = np.nan
    return out
