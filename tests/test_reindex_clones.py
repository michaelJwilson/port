"""`reindex_clones`, which is where the one-column contract gets established (#278).

`scripts/run_cnaster.py:1366` reads `shape[1] > 1` on a parameter, inside a
1,648-line function no patch can reach. `reindex_clones` runs ninety-seven
lines earlier and is 87 lines, so the contract is established there and the
branch below is dead by construction rather than by luck.

`patch`, because the claim is that the replacement reorders clones exactly as
`cnaster` does. The contract itself is `bug`: it pins that upstream holds two
readings of one axis in one function.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from port.patch.hmrf.reindex import reindex_clones


def _result(n_states: int = 4, n_obs: int = 12, n_clones: int = 3) -> dict[str, Any]:
    rng = np.random.default_rng(5)

    # clone 1 is the balanced one, so the reorder has something to do
    p_binom = np.array([[0.2], [0.5], [0.8], [0.35]])[:n_states]
    paths = np.concatenate(
        [
            np.full(n_obs, 0),  # clone 0: p = 0.2, far from balanced
            np.full(n_obs, 1),  # clone 1: p = 0.5, the normal one
            np.full(n_obs, 2),  # clone 2: p = 0.8
        ][:n_clones]
    )

    assignments = np.repeat(np.arange(n_clones), [5, 3, 7][:n_clones])

    return {
        "new_assignment": assignments,
        "pred_cnv": paths,
        "new_p_binom": p_binom,
        "new_log_mu": rng.normal(size=(n_states, 1)),
        "new_alphas": np.full((n_states, 1), 0.25),
        "new_taus": np.full((n_states, 1), 30.0),
        "log_gamma": rng.normal(size=(n_states, n_obs * n_clones)),
    }


@pytest.mark.patch
def test_the_replacement_reindexes_as_upstream_does() -> None:
    """Same normal clone, same order, same reindexed arrays.

    Refereed against `cnaster.hmrf.reindex_clones` on an instance where the
    reorder is non-trivial: the balanced clone is not already first, and the
    remaining two differ in spot count, so both halves of the ordering rule
    are exercised.
    """
    from cnaster.hmrf import reindex_clones as upstream

    theirs, _ = upstream(_result(), posterior=None, single_tumor_prop=None)
    ours, _ = reindex_clones(_result(), posterior=None, single_tumor_prop=None)

    assert set(ours) == set(theirs)

    for key in sorted(theirs):
        np.testing.assert_array_equal(
            np.asarray(ours[key]), np.asarray(theirs[key]), err_msg=key
        )


@pytest.mark.bug
def test_every_parameter_is_checked_not_only_p_binom() -> None:
    """Upstream asserts one column for one of the four, then reorders all four.

    **Written to fail when `cnaster` reconciles the two.** The assert at
    `hmrf.py:821` makes the reorder at `:859` unreachable for `new_p_binom`
    and leaves it reachable in principle for the other three -- #267's
    finding in miniature, inside a single function.
    """
    for key in ("new_log_mu", "new_alphas", "new_taus"):
        widened = _result()
        widened[key] = np.tile(widened[key], (1, 3))

        with pytest.raises(ValueError, match=f"{key} has shape"):
            reindex_clones(widened, posterior=None, single_tumor_prop=None)


@pytest.mark.bug
def test_upstream_accepts_the_widths_it_cannot_mean() -> None:
    """The three upstream lets through, which is why the check is here.

    `cnaster`'s assert names `new_p_binom` alone, so a widened `new_log_mu`
    reaches the reorder and is silently permuted by clone -- and every
    consumer downstream still reads column zero. Pinned against
    `cnaster.hmrf` directly so a swap row cannot make it pass.
    """
    from cnaster.hmrf import reindex_clones as upstream

    widened = _result()
    widened["new_log_mu"] = np.tile(widened["new_log_mu"], (1, 3))

    reindexed, _ = upstream(widened, posterior=None, single_tumor_prop=None)

    assert reindexed["new_log_mu"].shape[1] == 3, (
        "upstream now refuses a widened new_log_mu; #278's check can go"
    )


@pytest.mark.patch
def test_the_contract_makes_the_entry_point_branch_dead() -> None:
    """`idx = s if shape[1] > 1 else 0` can only take the `else`.

    That line is `scripts/run_cnaster.py:1366`, ninety-seven lines after
    `reindex_clones` is called. This is the whole reason the contract is
    established here: the branch is unreachable because the shape cannot
    reach it, not because nothing has produced one yet.
    """
    reindexed, _ = reindex_clones(_result(), posterior=None, single_tumor_prop=None)

    for key in ("new_log_mu", "new_alphas", "new_p_binom", "new_taus"):
        assert np.asarray(reindexed[key]).shape[1] == 1, key
