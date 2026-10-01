"""`--split-state`: one unbalanced state split by depth, refitted on fixed clones (#471)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

LOSS = np.log(0.5)
"""A one-copy loss against a copy-neutral LOH at the same BAF: half the depth."""


def _fit(loh: int, loss: int, p_unbalanced: float = 0.2) -> tuple[dict[str, Any], Any]:
    """A two-clone fit whose state 2 covers an LOH and a loss at one depth.

    States: 0 and 1 balanced and nearly equal (the redundant pair), 2 the
    unbalanced one, 3 a balanced gain. Bins run 100 neutral, `loh` LOH at
    neutral depth, `loss` losses at half depth, 20 gains.
    """
    level = np.concatenate(
        [np.zeros(100), np.zeros(loh), np.full(loss, LOSS), np.full(20, np.log(1.5))]
    )
    state = np.concatenate(
        [np.zeros(100), np.full(loh + loss, 2), np.full(20, 3)]
    ).astype(int)
    n_obs, n_spots = level.size, 10
    base = np.full((n_obs, n_spots), 10.0)
    X = np.zeros((n_obs, 2, n_spots))
    X[:, 0, :] = base * np.exp(level)[:, None]
    result = {
        "new_log_mu": np.array([[0.0], [0.05], [-0.3], [0.4]]),
        "new_p_binom": np.array([[0.5], [0.5], [p_unbalanced], [0.5]]),
        "pred_cnv": np.stack([state, state], axis=1),
        "new_assignment": np.repeat([0, 1], n_spots // 2),
    }

    return result, (X, [n_obs], base)


@pytest.mark.analytic
def test_a_state_covering_two_depths_splits_at_their_gap() -> None:
    """The split state and the freed one sit `log 2` apart, at the unbalanced BAF.

    The freed state is the less occupied of the closest pair (0 and 1), and
    the split keeps the majority's depth: LOH 50 bins, loss 30.
    """
    from port.patch.hmrf.split_state import split_init

    result, (X, lengths, base) = _fit(loh=50, loss=30)
    init = split_init(result, X, lengths, base)

    assert init is not None
    log_mu, p_binom, split, freed = init

    assert (split, freed) == (2, 1)
    assert log_mu[split, 0] - log_mu[freed, 0] == pytest.approx(-LOSS, abs=0.05)
    assert log_mu[split, 0] == pytest.approx(-0.3, abs=0.05)
    assert p_binom[freed, 0] == pytest.approx(0.2)
    np.testing.assert_array_equal(log_mu[[0, 3], 0], [0.0, 0.4])


@pytest.mark.analytic
def test_one_depth_or_a_balanced_state_does_not_split() -> None:
    """No gap, or a gap in a balanced state, leaves the fit alone.

    A balanced state holds the diploid reference the integer decode pins, and
    its depth is read by the RDR channel already.
    """
    from port.patch.hmrf.split_state import split_init

    single, (X, lengths, base) = _fit(loh=80, loss=0)
    balanced, (Xb, lengths_b, base_b) = _fit(loh=50, loss=30, p_unbalanced=0.5)

    assert split_init(single, X, lengths, base) is None
    assert split_init(balanced, Xb, lengths_b, base_b) is None


@pytest.mark.release
@pytest.mark.end2end
@pytest.mark.cnaster
def test_split_state_decodes_r0_s_losses_apart_from_its_loh() -> None:
    """r0 under `--sal --split-state`: altered bins at their planted `(A, B)`.

    Planted truth, `dev_tree` r0 (`93398396`). Without the flag on this tree:
    clone ARI 0.9997 and 0.707 of altered bins exact, because the HMM fits
    one-copy losses at LOH depth (#471). The clones are the first fit's, so
    clone ARI holds.
    """
    from tests.sim_audit import run_arm
    from tests.sim_fixtures import load_simulated
    from tests.sim_stages import r0

    recovery, _ = run_arm(load_simulated(str(r0())), ["--sal", "--split-state"])

    assert recovery.ari >= 0.99
    assert recovery.exact_altered >= 0.85
