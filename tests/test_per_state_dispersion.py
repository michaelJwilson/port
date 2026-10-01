"""Per-state NB/BB dispersions, and the shrinkage toward the pooled one (#566).

`run_cnaster_port --per-state-dispersion` fits one `alpha` and one `tau` per
state through `cnaster`'s own `shared_*_dispersion=False` path, bounded by
`DispersionBounds` unless `--no-dispersion-bounds`, and shrunk toward the
pooled value by `DispersionShrinkage` where `--dispersion-prior-rows` is
given. Three referees:

- `analytic`: states holding the same rows return the pooled fit; a state
  holding three rows stops at `alpha_min`, `tau_max` rather than running to
  `alpha -> 0`;
- `oracle`: the penalized gradient against central differences of the
  penalized objective `cnaster`'s `cost_fn` scores;
- `end2end`: planted per-state dispersions recovered from a fixture drawn
  with them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


def _drawn(
    alphas: tuple[float, ...],
    taus: tuple[float, ...],
    *,
    n_runs: int,
    run: int = 50,
    seed: int = 11,
) -> dict[str, Any]:
    """Runs of `run` bins per state, drawn from `cnaster`'s NB and BB laws.

    NB mean `base * mu_k`, size `1 / alpha_k`; BB `a = p_k tau_k`,
    `b = (1 - p_k) tau_k`, as `_nb_logpmf_1d` and `_bb_logpmf_1d` score them.
    """
    rng = np.random.default_rng(seed)
    n_states = len(alphas)
    mus = np.array([1.0, 2.0, 0.5])[:n_states]
    ps = np.array([0.5, 0.25, 0.4])[:n_states]

    states = np.repeat(np.arange(n_runs) % n_states, run)
    n_rows = states.size
    base = rng.uniform(150.0, 250.0, n_rows)
    total = rng.integers(30, 60, n_rows).astype(np.float64)

    alpha = np.asarray(alphas)[states]
    mean = base * mus[states]
    depth = rng.negative_binomial(1.0 / alpha, 1.0 / (1.0 + alpha * mean))

    tau = np.asarray(taus)[states]
    share = rng.beta(ps[states] * tau, (1.0 - ps[states]) * tau)
    allele = rng.binomial(total.astype(np.int64), share)

    X = np.stack([depth, allele], axis=1).astype(np.float64)[:, :, None]

    return {
        "X": X,
        "lengths": np.array([n_rows]),
        "base": base[:, None],
        "total": total[:, None],
        "states": states,
        "init_log_mu": np.log(mus)[:, None],
        "init_p_binom": ps[:, None],
    }


def _fit(data: dict[str, Any], **attributes: Any) -> dict[str, Any]:
    """`optimize` of port's class with `attributes` bound, unshifted, from `data`'s start."""
    from port.patch.hmm_nophasing import hmm_nophasing
    from port.pipeline import with_attributes

    model = with_attributes(hmm_nophasing, **attributes)(params="smp", t=1 - 1e-4)
    n_states = data["init_log_mu"].shape[0]

    result: dict[str, Any] = model.optimize(
        data["X"],
        data["lengths"],
        n_states,
        data["base"],
        data["total"],
        init_log_mu=data["init_log_mu"],
        init_p_binom=data["init_p_binom"],
        shared_NB_dispersion=True,
        shared_BB_dispersion=True,
        max_iter=200,
        clone_lengths=data["lengths"],
    )
    return result


@pytest.mark.analytic
@pytest.mark.usefixtures("cnaster_config")
@pytest.mark.parametrize("prior_rows", [0.0, 50.0])
def test_states_holding_the_same_rows_return_the_pooled_dispersions(
    prior_rows: float,
) -> None:
    """Three states with one start see one row set: each `alpha_k`, `tau_k` is the shared fit's.

    Identical starts keep the posteriors uniform, so every state holds every
    row; the per-state optimum is then the pooled one, and the shrinkage is
    zero there. To 1e-3 relative (BFGS's stopping, in `log alpha`).
    """
    data = _drawn((0.05,), (100.0,), n_runs=10, run=60)
    data["init_log_mu"] = np.zeros((3, 1))
    data["init_p_binom"] = np.full((3, 1), 0.5)

    shared = _fit(data)
    per_state = _fit(data, per_state_dispersion=True, dispersion_prior_rows=prior_rows)

    for key in ("new_alphas", "new_taus"):
        pooled = float(np.asarray(shared[key]).reshape(-1)[0])
        fitted = np.asarray(per_state[key]).reshape(-1)

        assert fitted.size == 3
        np.testing.assert_allclose(fitted, pooled, rtol=1e-3)


@pytest.mark.oracle
@pytest.mark.usefixtures("cnaster_config")
@pytest.mark.parametrize("shifted", [False, True])
def test_the_penalized_gradient_is_its_finite_difference(shifted: bool) -> None:
    """`d (cost_fn + P) / d x` against central differences, step 1e-6, to 1e-5 relative."""
    from port.patch.hmm_nophasing.gradient import DispersionShrinkage

    from tests.test_mstep_gradient import _cnaster_value, _problem

    model, gradient, x, _ = _problem(shifted=shifted, shared=False)
    penalty = DispersionShrinkage.for_fit(gradient, 40.0)
    penalty.hold_information(x, gradient)

    assert set(penalty.blocks) == {"nb", "bb"}
    assert all(h > 0.0 for h in penalty.information.values())

    def objective(point: np.ndarray) -> float:
        return _cnaster_value(model, gradient, point) + penalty.value(point)

    ours = gradient(x) + penalty(x)
    step = 1e-6
    theirs = np.array(
        [
            (objective(x + step * unit) - objective(x - step * unit)) / (2.0 * step)
            for unit in np.eye(x.size)
        ]
    )
    # NB the start probabilities are unread by `cost_fn` and by `P`.
    live = slice(gradient.n_states, None)

    np.testing.assert_allclose(
        ours[live], theirs[live], rtol=1e-5, atol=1e-5 * np.abs(theirs).max()
    )


@pytest.mark.end2end
@pytest.mark.usefixtures("cnaster_config")
def test_planted_per_state_dispersions_are_recovered() -> None:
    """Two states, 1,000 rows each, `alpha` 0.02 and 0.15, `tau` 100 and 15: each to 25 per cent.

    The shared fit returns one value between them, so it misses both by more
    than that; the per-state fit, shrunk with `n0 = 20` against `n_k = 1000`,
    separates them.
    """
    planted_alpha, planted_tau = np.array([0.02, 0.15]), np.array([100.0, 15.0])
    data = _drawn(tuple(planted_alpha), tuple(planted_tau), n_runs=40)

    per_state = _fit(data, per_state_dispersion=True, dispersion_prior_rows=20.0)
    shared = _fit(data)

    alphas = np.asarray(per_state["new_alphas"]).reshape(-1)
    taus = np.asarray(per_state["new_taus"]).reshape(-1)

    np.testing.assert_allclose(alphas, planted_alpha, rtol=0.25)
    np.testing.assert_allclose(taus, planted_tau, rtol=0.25)

    pooled = float(np.asarray(shared["new_alphas"]).reshape(-1)[0])
    assert np.all(np.abs(pooled / planted_alpha - 1.0) > 0.25)


@pytest.mark.analytic
@pytest.mark.usefixtures("cnaster_config")
def test_a_state_holding_three_rows_stops_at_the_bounds() -> None:
    """Three rows at their exact means: unbounded, `alpha` runs below `alpha_min / 10`; bounded, it stops at `alpha_min`.

    Realized: unbounded `alpha` 1.1e-10, `cnaster`'s own floor, and `tau`
    7.1e4; bounded `alpha_min` and `tau_max` exactly. Their allele counts sit
    at `p n`, so the likelihood's limit is `alpha -> 0`, `tau -> inf`.
    """
    from port.patch.hmm_nophasing.gradient import ALPHA_MIN, TAU_MAX

    data = _drawn((0.05, 0.05), (100.0, 100.0), n_runs=12, run=50)
    states = data["states"]
    first = int(np.flatnonzero(states == 1)[0])
    keep = np.r_[np.flatnonzero(states == 0), first : first + 3]
    X = data["X"][keep].copy()
    base, total = data["base"][keep], data["total"][keep]
    X[-3:, 0, 0] = base[-3:, 0] * 2.0
    total[-3:, 0] = 40.0
    X[-3:, 1, 0] = 10.0

    trimmed = dict(data, X=X, base=base, total=total, lengths=np.array([keep.size]))
    bounded = _fit(trimmed, per_state_dispersion=True)
    free = _fit(trimmed, per_state_dispersion=True, dispersion_bounds=False)

    alpha_bounded = np.asarray(bounded["new_alphas"]).reshape(-1)
    alpha_free = np.asarray(free["new_alphas"]).reshape(-1)

    assert alpha_free[1] < ALPHA_MIN / 10.0
    np.testing.assert_allclose(alpha_bounded[1], ALPHA_MIN, rtol=1e-9)
    assert alpha_bounded.min() >= ALPHA_MIN * (1.0 - 1e-12)
    assert np.asarray(bounded["new_taus"]).max() <= TAU_MAX * (1.0 + 1e-12)
