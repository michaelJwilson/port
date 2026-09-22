r"""The M step's gradient, against finite differences (#259 stage 5).

`port.patch.em_gradient` derives what `scipy` was differencing. The referee
is the objective itself: `check_grad`, and a central difference per variable
with the step chosen per variable.

**Forward differences are not good enough to referee this, and that is
measured rather than assumed.** `scipy.optimize.check_grad`'s default step
reads a relative error of 2.5e-02 on the `log tau` block -- where `tau` is
of order 1,000 and the objective's curvature in it is large -- while a
central difference at a fitted step reads 1.7e-07 on the same entry. A test
that trusted the first would have called a correct gradient wrong.

`patch`: this says the gradient is the derivative of the objective the code
optimizes. Whether that objective is the right one is a different claim with
a different referee.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

STEPS = (1e-6, 1e-5, 1e-4, 1e-3)
"""Central-difference steps tried per variable; the best is reported.

One step cannot serve `theta` of order 0.1 and `log tau` of order 7 at once,
so the referee is "there exists a step at which they agree", which is the
honest question for a finite-difference check.
"""

TOLERANCE = 1e-5
"""Relative, against the best central difference. Realized 1.7e-07."""

SETTINGS = {
    "optimize_nb": True,
    "fix_NB_dispersion": False,
    "shared_NB_dispersion": False,
    "fix_BB_dispersion": False,
    "shared_BB_dispersion": False,
    "use_logit": True,
}


def _case(seed: int, n_states: int = 3, n_obs: int = 200) -> dict[str, Any]:
    from cnaster.count_encoder import CountEncoder

    rng = np.random.default_rng(seed)

    nb_total = rng.integers(5, 40, size=(n_obs, 1)).astype(np.float64)
    bb_total = rng.integers(3, 20, size=(n_obs, 1)).astype(np.float64)

    return {
        "nb": CountEncoder(rng.poisson(nb_total).astype(np.float64), nb_total),
        "bb": CountEncoder(
            rng.binomial(bb_total.astype(int), 0.4).astype(np.float64), bb_total
        ),
        "gamma": rng.dirichlet(np.ones(n_states), size=n_obs).T,
        "lambdas": rng.normal(size=n_obs),
        "n_states": n_states,
        "n_obs": n_obs,
        "log_mu": np.linspace(-0.2, 0.2, n_states)[:, None],
        "p_binom": np.linspace(0.1, 0.4, n_states)[:, None],
        "alphas": np.full((n_states, 1), 0.5),
        "taus": np.full((n_states, 1), 800.0),
        "log_startprob": np.log(np.ones(n_states) / n_states),
    }


def _harness(case: dict[str, Any], *, shifted: bool) -> tuple[Any, Any, np.ndarray]:
    """`(cost, grad, x0)` for the objective the M step actually minimizes."""
    from port.patch.clone_shift import clone_state_weights, log_normalizers_from_weights
    from port.patch.hmm_nophasing import hmm_nophasing

    model = hmm_nophasing(params="smp")
    n_states = case["n_states"]
    lengths = (case["n_obs"],)

    if shifted:
        model.apply_logmu_shift = True
        model.state_posteriors = case["gamma"]

    initial = (
        case["log_startprob"],
        case["log_mu"],
        case["p_binom"],
        case["alphas"],
        case["taus"],
    )
    x0 = model.pack_params(*initial, **SETTINGS)
    weights = clone_state_weights(case["gamma"], case["lambdas"], lengths)

    def unpack(x: np.ndarray) -> tuple[np.ndarray, ...]:
        return model.unpack_params(x, n_states, *initial, **SETTINGS)  # type: ignore[no-any-return]

    def cost(x: np.ndarray) -> float:
        _, log_mu, p_binom, alphas, taus = unpack(x)
        extra = (
            {
                "normal_log_lambda": case["lambdas"],
                "num_segments_clones": list(lengths),
            }
            if shifted
            else {}
        )
        rdr, baf = model.compute_emission_probability_nb_betabinom_coded(
            case["nb"], case["bb"], log_mu, alphas, p_binom, taus, **extra
        )
        return -float(np.sum(case["gamma"] * (rdr + baf)))

    def grad(x: np.ndarray) -> np.ndarray:
        _, log_mu, p_binom, alphas, taus = unpack(x)
        shifts = log_normalizers_from_weights(weights, log_mu) if shifted else None

        return model._em_gradient(
            nbEncoder=case["nb"],
            bbEncoder=case["bb"],
            posteriors=case["gamma"],
            log_mu=log_mu,
            alphas=alphas,
            p_binom=p_binom,
            taus=taus,
            shifts=shifts,
            clone_weights=weights if shifted else None,
            lengths=lengths if shifted else None,
            **SETTINGS,
        )

    return cost, grad, x0


def _worst_relative(cost: Any, analytic: np.ndarray, x0: np.ndarray) -> float:
    """The largest relative gap to the best central difference, per variable."""
    worst = 0.0

    for index in range(x0.size):
        best = None

        for step in STEPS:
            up, down = x0.copy(), x0.copy()
            up[index] += step
            down[index] -= step

            central = (cost(up) - cost(down)) / (2.0 * step)
            relative = abs(analytic[index] - central) / max(abs(central), 1e-8)
            best = relative if best is None else min(best, relative)

        worst = max(worst, best or 0.0)

    return worst


@pytest.mark.patch
@pytest.mark.parametrize("shifted", [False, True], ids=["unshifted", "shifted"])
@pytest.mark.parametrize("seed", [11, 23])
def test_it_is_the_derivative_of_the_objective(
    cnaster_config: None, shifted: bool, seed: int
) -> None:
    """Every packed variable, against a central difference at a fitted step.

    The shifted case is the one the constraint lives in: `log Z` is in every
    state's term, so the rate block stops being separable and picks up
    `d log Z / d theta_j`. A gradient that ignored it would still pass the
    unshifted case, which is why both run.
    """
    case = _case(seed)
    cost, grad, x0 = _harness(case, shifted=shifted)

    worst = _worst_relative(cost, grad(x0), x0)

    assert worst < TOLERANCE, f"worst relative difference {worst:.3e}"


@pytest.mark.patch
def test_the_start_probability_block_is_exactly_zero(cnaster_config: None) -> None:
    """The EM cost carries no start term, so its gradient cannot be non-zero.

    Upstream packs the block anyway, and `scipy` was differencing it:
    `n_states` objective evaluations per gradient spent on a derivative that
    is zero by inspection.
    """
    case = _case(11)
    _, grad, x0 = _harness(case, shifted=False)

    assert np.array_equal(grad(x0)[: case["n_states"]], np.zeros(case["n_states"])), (
        "the start block is not exactly zero"
    )


@pytest.mark.patch
def test_the_shift_reaches_the_nb_blocks_and_not_the_baf_ones(
    cnaster_config: None,
) -> None:
    """`log Z` multiplies `mu`, so it moves the NB blocks and only those.

    The packed layout is `[startprob, log_mu, logit p, log alpha, log tau]`.
    The shift changes the rate the NB likelihood is evaluated at, so both
    `log_mu` **and** `log alpha` move -- the dispersion's derivative is a
    function of `mu`. `p_binom` and `tau` do not appear in the normalizer, so
    if their blocks moved the coupling would be leaking into the BAF channel.
    """
    case = _case(11)
    n_states = case["n_states"]

    _, plain, x0 = _harness(case, shifted=False)
    _, coupled, _ = _harness(case, shifted=True)

    without, with_shift = plain(x0), coupled(x0)

    blocks = {
        "log_mu": slice(n_states, 2 * n_states),
        "logit_p": slice(2 * n_states, 3 * n_states),
        "log_alpha": slice(3 * n_states, 4 * n_states),
        "log_tau": slice(4 * n_states, 5 * n_states),
    }

    for name in ("log_mu", "log_alpha"):
        assert not np.allclose(without[blocks[name]], with_shift[blocks[name]]), (
            f"{name} did not move; the coupling is doing nothing there"
        )

    for name in ("logit_p", "log_tau"):
        assert np.allclose(
            without[blocks[name]], with_shift[blocks[name]], rtol=0.0, atol=1e-12
        ), f"the shift reached {name}, which is not in the normalizer"
