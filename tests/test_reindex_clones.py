"""`reindex_clones` against cnaster's, and the one-column contract it establishes
(#278).

The reorder is `patch`; the contract is `bug`, pinning upstream's two readings of one
axis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from cnaster.cna_hmrf_result import (
    CloneAssignment,
    CnaHMRFResult,
    HMMParams,
    HMMProfile,
)
from cnaster.hmrf import reindex_clones as upstream
from port.patch.hmrf.reindex import reindex_clones

from tests.builders import reindex_result


@pytest.mark.bug
def test_every_parameter_is_checked_not_only_p_binom() -> None:
    """Upstream asserts one column for `new_p_binom` only, then reorders all four (#267)."""
    for key in ("new_log_mu", "new_alphas", "new_taus"):
        widened = reindex_result()
        widened[key] = np.tile(widened[key], (1, 3))

        with pytest.raises(ValueError, match=f"{key} has shape"):
            reindex_clones(widened, posterior=None, single_tumor_prop=None)


@pytest.mark.bug
def test_upstream_accepts_the_widths_it_cannot_mean() -> None:
    """The three parameters upstream lets through widened are refused here."""

    widened = reindex_result()
    widened["new_log_mu"] = np.tile(widened["new_log_mu"], (1, 3))

    reindexed, _ = upstream(widened, posterior=None, single_tumor_prop=None)

    assert reindexed["new_log_mu"].shape[1] == 3, (
        "upstream now refuses a widened new_log_mu; #278's check can go"
    )


@pytest.mark.patch
def test_the_contract_makes_the_entry_point_branch_dead() -> None:
    """`idx = s if shape[1] > 1 else 0` (`run_cnaster.py:1366`) can only take the `else`."""
    reindexed, _ = reindex_clones(
        reindex_result(), posterior=None, single_tumor_prop=None
    )

    for key in ("new_log_mu", "new_alphas", "new_p_binom", "new_taus"):
        assert np.asarray(reindexed[key]).shape[1] == 1, key


def _by_rule(res: dict[str, Any], n_obs: int) -> tuple[np.ndarray, list[int]]:
    """The normal clone (BAF least outside `0.5 +- 0.05`) first, then by increasing spot count."""
    labels = sorted(set(res["new_assignment"].tolist()))
    p = res["new_p_binom"][:, 0]
    penalty = {}

    for c in labels:
        path = res["pred_cnv"][c * n_obs : (c + 1) * n_obs]
        penalty[c] = sum(max(abs(p[s] - 0.5) - 0.05, 0.0) for s in path)

    normal = min(labels, key=lambda c: (penalty[c], c))
    count = {c: int((res["new_assignment"] == c).sum()) for c in labels}
    order = [
        normal,
        *sorted((c for c in labels if c != normal), key=lambda c: count[c]),
    ]
    relabel = {old: new for new, old in enumerate(order)}

    return np.array([relabel[a] for a in res["new_assignment"]]), order


def _live(res: dict[str, Any], n_obs: int, n_clones: int) -> Any:
    """`res` as `run_cnaster` hands it over: a `CnaHMRFResult`, one column per clone."""

    n_states = res["new_log_mu"].shape[0]
    gamma = res["log_gamma"].reshape(n_states, n_clones, n_obs).transpose(0, 2, 1)

    return CnaHMRFResult(
        params=HMMParams(
            new_log_mu=res["new_log_mu"],
            new_alphas=res["new_alphas"],
            new_p_binom=res["new_p_binom"],
            new_taus=res["new_taus"],
            new_log_startprob=np.log(np.full(n_states, 1.0 / n_states)),
            new_log_transmat=np.log(np.full((n_states, n_states), 1.0 / n_states)),
        ),
        param_errors=None,
        profile=HMMProfile(
            log_gamma=np.ascontiguousarray(gamma),
            pred_cnv=res["pred_cnv"].reshape(n_clones, n_obs).T.copy(),
        ),
        llf=0.0,
        n_states=n_states,
        assignment=CloneAssignment(new_assignment=res["new_assignment"].copy()),
    )


@pytest.mark.oracle
def test_the_reorder_is_the_stated_rule_on_random_fits() -> None:
    """`reindex_clones` permutes as the loop rule says, exactly, over 50 draws, for `dict` and `CnaHMRFResult`."""
    rng = np.random.default_rng(517)

    for _ in range(50):
        n_states, n_obs, n_clones = 5, int(rng.integers(3, 9)), int(rng.integers(2, 6))
        sizes = rng.permutation(np.arange(1, n_clones + 1)) * 3
        res = {
            "new_assignment": rng.permutation(np.repeat(np.arange(n_clones), sizes)),
            "pred_cnv": rng.integers(0, n_states, size=n_obs * n_clones),
            "new_p_binom": rng.uniform(0.05, 0.95, size=(n_states, 1)),
            "new_log_mu": rng.normal(size=(n_states, 1)),
            "new_alphas": np.full((n_states, 1), 0.25),
            "new_taus": np.full((n_states, 1), 30.0),
            "log_gamma": rng.normal(size=(n_states, n_obs * n_clones)),
        }
        expected, order = _by_rule(res, n_obs)
        columns = np.concatenate([np.arange(c * n_obs, (c + 1) * n_obs) for c in order])

        reindexed, _ = reindex_clones(res)

        np.testing.assert_array_equal(reindexed["new_assignment"], expected)
        np.testing.assert_array_equal(reindexed["pred_cnv"], res["pred_cnv"][columns])
        np.testing.assert_array_equal(
            reindexed["log_gamma"], res["log_gamma"][:, columns]
        )

        # NB the live type and layout: path `(n_obs, n_clones)`, `log_gamma` `(n_states,
        # n_obs, n_clones)`.
        stacked = _live(res, n_obs, n_clones)
        paths = np.array(stacked["pred_cnv"])
        gamma = np.array(stacked["log_gamma"])

        reindexed, _ = reindex_clones(stacked)

        np.testing.assert_array_equal(reindexed["new_assignment"], expected)
        np.testing.assert_array_equal(reindexed["pred_cnv"], paths[:, order])
        np.testing.assert_array_equal(reindexed["log_gamma"], gamma[:, :, order])
