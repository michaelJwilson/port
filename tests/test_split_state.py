"""One unbalanced state split by depth, refitted on fixed clones (#471, #481): the sandbox's.

`port.sandbox.split_state` is set aside, not installed by `run_cnaster_port`.
Referees: `analytic`, the split on drawn states of known depth; `infra`, the
context binds its one name and puts it back; `end2end` (`release`), `dev_tree`
r0's altered bins against the planted truth under the context.
"""

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
    from port.sandbox.split_state.split import split_init

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
    from port.sandbox.split_state.split import split_init

    single, (X, lengths, base) = _fit(loh=80, loss=0)
    balanced, (Xb, lengths_b, base_b) = _fit(loh=50, loss=30, p_unbalanced=0.5)

    assert split_init(single, X, lengths, base) is None
    assert split_init(balanced, Xb, lengths_b, base_b) is None


@pytest.mark.release
@pytest.mark.end2end
@pytest.mark.cnaster
def test_the_split_decodes_r0_s_losses_apart_from_its_loh() -> None:
    """r0 under `--sal` with the split installed: altered bins at their planted `(A, B)`.

    Planted truth, `dev_tree` r0 (`3381575a`). The clones are the first
    fit's, so clone ARI holds; the thresholds are this branch's measurement
    on `dev_tree` r0 42 x 42 (`93398396`), 0.896 exact altered
    (`port.sandbox.split_state.split`).
    """
    from port.sandbox.split_state import split_state

    from tests.sim_audit import run_arm
    from tests.sim_fixtures import load_simulated
    from tests.sim_stages import r0

    with split_state():
        recovery, _ = run_arm(load_simulated(str(r0())), ["--sal"])

    assert recovery.ari >= 0.99
    assert recovery.exact_altered >= 0.85


@pytest.mark.infra
@pytest.mark.cnaster
def test_the_sandbox_split_binds_its_one_name_and_restores_it() -> None:
    """Inside `split_state()`, `port.patch.hmrf.run_core_inference` is the sandbox's wrapper; after, the row's again."""
    from port.patch import hmrf
    from port.sandbox.split_state import installed, split_state

    before = hmrf.run_core_inference

    with split_state():
        assert installed()
        assert hmrf.run_core_inference is not before

    assert not installed()
    assert hmrf.run_core_inference is before


@pytest.mark.analytic
def test_the_kept_clones_name_the_refits_columns() -> None:
    """Labels `{0, 1, 2, 4}`, clone 3 emptied: renumbered `{0, 1, 2, 3}` in order, so label 4 names column 3 (#570)."""
    from port.sandbox.split_state import contiguous

    labels = np.array([4, 0, 0, 2, 1, 4, 2])

    np.testing.assert_array_equal(contiguous(labels), [3, 0, 0, 2, 1, 3, 2])
