"""CalicoST's Neyman-Pearson clone merge, as port runs it (#497).

Referee: `calicost.hmm_NB_BB_phaseswitch.similarity_components_rdrbaf_neymanpearson`
and `compute_neymanpearson_stats`, CalicoST's own functions, called on the same
pseudobulk and fit with `hmm_nophasing_v2`, CalicoST's unphased emission. With
every per-clone shift zero, port's statistics and groups must be CalicoST's.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

N_OBS = 120
N_STATES = 4


def _instance(
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Four clones drawn from the model, with planted decodes.

    Clone 1 is clone 0 but for 3 bins, under `MINLENGTH`, so it merges. Clone 2
    differs from clone 0 over 40 bins, a gain against a neutral state, so it
    does not. Clone 3 is decoded differently over 15 bins but drawn at clone
    0's states, so the statistic, not the decode, decides it.
    """
    rng = np.random.default_rng(seed)
    log_mu = np.log(np.array([1.0, 0.5, 1.5, 1.0]))
    p_binom = np.array([0.5, 0.2, 0.33, 0.8])
    alphas = np.full(N_STATES, 0.02)
    taus = np.full(N_STATES, 300.0)

    truth = np.zeros((N_OBS, 4), dtype=np.int64)
    truth[40:80, 0] = 1
    truth[:, 1] = truth[:, 0]
    truth[:, 2] = truth[:, 0]
    truth[:40, 2] = 2
    truth[:, 3] = truth[:, 0]
    pred = truth.copy()
    pred[100:103, 1] = 3
    pred[5:20, 3] = 3

    base = rng.uniform(200.0, 400.0, (N_OBS, 4))
    total = rng.integers(40, 80, (N_OBS, 4)).astype(float)
    X = np.zeros((N_OBS, 2, 4))

    for c in range(4):
        mean = base[:, c] * np.exp(log_mu[truth[:, c]])
        r = 1.0 / alphas[0]
        X[:, 0, c] = rng.negative_binomial(r, r / (r + mean))
        p = rng.beta(
            p_binom[truth[:, c]] * taus[0], (1 - p_binom[truth[:, c]]) * taus[0]
        )
        X[:, 1, c] = rng.binomial(total[:, c].astype(int), p)

    res = {
        "new_log_mu": log_mu.reshape(-1, 1),
        "new_alphas": alphas.reshape(-1, 1),
        "new_p_binom": p_binom.reshape(-1, 1),
        "new_taus": taus.reshape(-1, 1),
        "pred_cnv": pred,
        "new_log_mu_shift": np.zeros(4),
        "new_assignment": np.repeat(np.arange(4), 5),
    }
    return X, base, total, res


def _calicost_res(res: dict[str, Any]) -> dict[str, Any]:
    """The same fit in CalicoST's layout: one `log_gamma` column per (clone, bin)."""
    pred = res["pred_cnv"]
    gamma = np.full((N_STATES, pred.size), -50.0)
    flat = pred.flatten("F")
    gamma[flat, np.arange(flat.size)] = 0.0
    return {**res, "log_gamma": gamma, "pred_cnv": flat}


@pytest.mark.oracle
@pytest.mark.parametrize("params", ["smp", "sp"])
def test_the_statistics_are_calicosts(params: str) -> None:
    """Every event's statistic, pair for pair, to 1e-10."""
    from calicost.hmm_NB_BB_nophasing_v2 import hmm_nophasing_v2
    from calicost.hmm_NB_BB_phaseswitch import compute_neymanpearson_stats
    from port.extensions.np_merge import statistics

    X, base, total, res = _instance()
    theirs = compute_neymanpearson_stats(
        X, base, total, _calicost_res(res), params, None, hmm_nophasing_v2
    )
    ours = statistics(X, base, total, res, params)

    assert set(theirs) == set(ours)

    for pair, events in theirs.items():
        mine = {(s1, s2): t for s1, s2, _, t in ours[pair]}

        assert len(mine) == len(events), pair
        for s1, s2, t in events:
            assert mine[(int(s1), int(s2))] == pytest.approx(
                float(t), rel=1e-10, abs=1e-10
            )


@pytest.mark.oracle
@pytest.mark.parametrize("params", ["smp", "sp"])
def test_the_groups_are_calicosts(params: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The merged groups at CalicoST's defaults, 2.0 nats and 10 bins.

    CalicoST's function ends by writing `np.NAN`, which NumPy 2 removed; the
    alias is supplied for the call and nothing else of the referee is touched.
    """
    from calicost.hmm_NB_BB_nophasing_v2 import hmm_nophasing_v2
    from calicost.hmm_NB_BB_phaseswitch import (
        similarity_components_rdrbaf_neymanpearson,
    )
    from port.extensions.np_merge import groups

    monkeypatch.setattr(np, "NAN", np.nan, raising=False)
    X, base, total, res = _instance()
    theirs, _ = similarity_components_rdrbaf_neymanpearson(
        X,
        base,
        total,
        {**_calicost_res(res), "new_assignment": res["new_assignment"]},
        threshold=2.0,
        minlength=10,
        params=params,
        tumor_prop=None,
        hmmclass=hmm_nophasing_v2,
    )
    ours = groups(X, base, total, res, params)

    assert ours == [sorted(g) for g in theirs]
    assert [0, 1] in [g[:2] for g in ours if len(g) >= 2], "the 3-bin difference merges"
    assert all(2 not in g or len(g) == 1 for g in ours), "the 40-bin gain does not"


@pytest.mark.analytic
def test_a_clone_shift_scales_the_read_depth_as_its_rates() -> None:
    """A clone's shift enters as its rates less the shift: shifting a clone's
    exposure up by `e^d` and its shift by `d` leaves every statistic as it was."""
    from port.extensions.np_merge import statistics

    X, base, total, res = _instance()
    shifted_base = base.copy()
    shifted_base[:, 2] *= np.exp(0.3)
    shifted = {**res, "new_log_mu_shift": np.array([0.0, 0.0, 0.3, 0.0])}

    plain = statistics(X, base, total, res, "smp")
    moved = statistics(X, shifted_base, total, shifted, "smp")

    for pair in plain:
        for (a, b, n, t), (a2, b2, n2, t2) in zip(
            plain[pair], moved[pair], strict=True
        ):
            assert (a, b, n) == (a2, b2, n2)
            assert t2 == pytest.approx(t, rel=1e-9, abs=1e-9)


@pytest.mark.patch
def test_merged_keeps_each_groups_first_path_and_shift() -> None:
    """CalicoST keeps the group's least clone's decode; the shift follows it."""
    from port.extensions.np_merge import merged

    _, _, _, res = _instance()
    res = {**res, "new_log_mu_shift": np.array([0.1, 0.2, 0.3, 0.4])}
    out = merged(res, [[0, 1, 3], [2]])

    np.testing.assert_array_equal(out["pred_cnv"], res["pred_cnv"][:, [0, 2]])
    np.testing.assert_array_equal(out["new_log_mu_shift"], [0.1, 0.3])
    np.testing.assert_array_equal(np.unique(out["new_assignment"]), [0, 1])
    assert (out["new_assignment"][res["new_assignment"] == 2] == 1).all()


@pytest.mark.patch
@pytest.mark.cnaster
@pytest.mark.parametrize("installed", [False, True], ids=["off", "nothing-held"])
def test_the_swap_is_cnasters_merge_by_minspots_where_it_does_not_merge(
    installed: bool,
) -> None:
    """Not installed, or installed with no fit held, the swap is upstream's call.

    Referee: `cnaster.hmrf.merge_by_minspots`, called on the same arguments;
    the assignment and every array of the result bitwise.
    """
    import contextlib

    from port.extensions.np_merge import np_merge
    from port.patch.hmrf.merge import UPSTREAM, merge_by_minspots

    rng = np.random.default_rng(1)
    assignment = rng.integers(0, 3, 400)
    n_obs = 50
    res = {
        "new_assignment": assignment,
        "pred_cnv": rng.integers(0, 4, (n_obs, 3)),
        "log_gamma": rng.normal(size=(4, n_obs, 3)),
        "new_p_binom": np.full((4, 1), 0.5),
    }
    total = rng.integers(0, 30, (n_obs, 400)).astype(float)
    kwargs = {"min_spots_thresholds": 50, "min_umicount_thresholds": 0.0}

    theirs = UPSTREAM(assignment.copy(), dict(res), total, **kwargs)

    with np_merge() if installed else contextlib.nullcontext():
        ours = merge_by_minspots(assignment.copy(), dict(res), total, **kwargs)

    np.testing.assert_array_equal(ours[0], theirs[0])
    for key in theirs[1]:
        np.testing.assert_array_equal(
            np.asarray(ours[1][key]), np.asarray(theirs[1][key]), err_msg=key
        )
