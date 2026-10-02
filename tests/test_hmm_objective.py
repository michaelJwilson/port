"""The copy-state HMM's NLL as a `sal` `Objective`, and `sal`'s samplers on it (#634).

- `oracle`: the adapter's value is `cnaster`'s forward lattice normalizer at
  the same states, an independent implementation, and on the way it is
  port's JAX NLL (`jax_hmm`'s emission and forward, unjitted) to round-off.
- `analytic`: its gradient is the central difference of its value, and
  `sal`'s autograd route through `__call__` returns the declared gradient.
- `analytic`: each `sal` start, seeded, reports the NLL of the states it
  returns, spends no more forward-backward passes than port's deleted
  sampler did, reproduces itself from the same seed, and, where it counts
  its initial point (`anneal`, `tempering`), is no worse than it.
- `infra`: the registry's literal `HMM_SAMPLERS` names the module's.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


def _rows() -> dict[str, Any]:
    """3 states planted on 2 clones of 120 rows (#540's sampler test instance)."""
    rng = np.random.default_rng(540)
    n = 120
    state = (np.arange(2 * n) // 40) % 3
    exposure = rng.uniform(200.0, 400.0, 2 * n)
    trials = rng.integers(20, 60, 2 * n).astype(np.float64)
    total = rng.poisson(exposure * np.array([0.5, 1.0, 1.5])[state])
    b = rng.binomial(trials.astype(int), np.array([0.1, 0.5, 0.33])[state])
    return {
        "total": total.astype(np.float64),
        "b": b.astype(np.float64),
        "exposure": exposure,
        "trials": trials,
        "lengths": np.array([n, n]),
    }


def _objective(rows: dict[str, Any], theta: np.ndarray) -> Any:
    from port.sandbox.known_copy.hmm_objective import objective_for

    return objective_for(
        rows["total"], rows["b"], rows["exposure"], rows["trials"],
        rows["lengths"], 3, theta,
    )  # fmt: skip


@pytest.mark.oracle
def test_the_adapter_is_cnasters_forward_and_ports_nll() -> None:
    """At 5 random states: `cnaster`'s `forward_lattice` to 1e-9 relative, port's JAX NLL to 1e-12.

    `cnaster`'s emission and recursion are `numba` (`test_jax_hmm` realizes
    1e-15 relative on the forward); port's NLL is the same JAX arithmetic
    without `jit`, so only reduction order can differ. Realized 6.0e-15
    against port and 7.8e-14 against `cnaster`, on NLLs 2,222-6,453.
    """
    import torch
    from cnaster.hmm_nophasing import _bb_logpmf_1d, _nb_logpmf_1d, hmm_nophasing
    from port.extensions.jax_hmm import emission, marginal_negative_log_likelihood
    from port.sandbox.known_copy.hmm import ALPHA, TAU, T
    from scipy.special import logsumexp

    rows = _rows()
    n_obs, k = rows["total"].size, 3
    rng = np.random.default_rng(634)
    log_transmat = np.where(np.eye(k, dtype=bool), np.log(T), np.log((1 - T) / (k - 1)))
    for _ in range(5):
        log_mu = rng.uniform(-0.8, 0.5, k)
        p = rng.uniform(0.05, 0.95, k)
        theta = np.concatenate([log_mu, np.log(p / (1 - p))])
        ours = float(_objective(rows, theta)(torch.from_numpy(theta)))

        scores = emission(
            log_mu, np.full(k, ALPHA), p, np.full(k, TAU), rows["total"],
            rows["exposure"], rows["b"], rows["trials"],
        )  # fmt: skip
        port = float(
            marginal_negative_log_likelihood(
                scores, np.full(k, -np.log(k)), log_transmat, rows["lengths"]
            )
        )
        np.testing.assert_allclose(ours, port, rtol=1e-12, atol=0.0)

        dense = np.zeros((k, n_obs))
        for s in range(k):
            depth, allele = np.zeros(n_obs), np.zeros(n_obs)
            _nb_logpmf_1d(
                rows["total"], rows["exposure"], float(np.exp(log_mu[s])), ALPHA, depth
            )
            _bb_logpmf_1d(rows["b"], rows["trials"], float(p[s]), TAU, allele)
            dense[s] = depth + allele
        lattice = hmm_nophasing.forward_lattice(
            rows["lengths"].astype(np.int64), log_transmat, np.full(k, -np.log(k)),
            dense[:, :, None], np.zeros((n_obs, 2)),
        )  # fmt: skip
        ends = np.cumsum(rows["lengths"]) - 1
        theirs = -float(np.sum(logsumexp(np.asarray(lattice)[:, ends], axis=0)))
        np.testing.assert_allclose(ours, theirs, rtol=1e-9, atol=0.0)


@pytest.mark.analytic
def test_the_gradient_is_the_central_difference() -> None:
    """At 3 random states, each coordinate to 1e-6 relative of the gradient's norm (step 1e-5).

    The central difference's truncation error is O(h^2) times the third
    derivative; at h = 1e-5 on an NLL of about 4e3 that is below 1e-6 of the
    gradient; realized 1.9e-8. `sal`'s autograd route (`autograd_value_and_gradient` through
    `__call__`) must return the declared gradient bitwise.
    """
    import torch
    from sal.opt.objective import autograd_value_and_gradient, value_and_gradient

    rows = _rows()
    rng = np.random.default_rng(1)
    h = 1e-5
    for _ in range(3):
        theta = np.concatenate([rng.uniform(-0.8, 0.5, 3), rng.uniform(-2.0, 2.0, 3)])
        objective = _objective(rows, theta)
        value, grad = value_and_gradient(objective, torch.from_numpy(theta))
        grad_np = grad.numpy()
        central = np.array([
            (objective.energy(theta + h * e) - objective.energy(theta - h * e)) / (2 * h)
            for e in np.eye(theta.size)
        ])  # fmt: skip
        np.testing.assert_allclose(
            central, grad_np, rtol=0.0, atol=1e-6 * np.linalg.norm(grad_np)
        )
        carried_value, carried = autograd_value_and_gradient(
            objective, torch.from_numpy(theta)
        )
        assert float(carried_value) == float(value)
        assert torch.equal(carried, grad)


@pytest.mark.analytic
@pytest.mark.parametrize("name", ["hmc-hmm", "anneal-hmm", "tempering-hmm"])
def test_the_starts_keep_their_best_within_budget(name: str) -> None:
    """Seeds 0-2: the NLL re-evaluated at the returned states is the one reported, passes within budget, reproducible.

    The budget is what port's deleted samplers spent at their tuned
    schedules: 9 gradients per trajectory plus each chain's initial point --
    433 for `anneal-hmm`, 436 for `tempering-hmm`, 217 for `hmc-hmm`.
    `anneal` and `tempering` count the initial point among their best, so
    neither ends above it; `hmc` keeps the best of its draws alone.
    """
    from port.sandbox.known_copy.hmm_objective import negative_log_likelihood, sample

    budget = {"anneal-hmm": 433, "tempering-hmm": 436, "hmc-hmm": 217}[name]
    rows = _rows()
    for seed in range(3):
        found = sample(
            name, rows["total"], rows["b"], rows["exposure"], rows["trials"],
            rows["lengths"], 3, np.random.default_rng(seed),
        )  # fmt: skip
        if name != "hmc-hmm":
            assert found.nll <= found.initial_nll
        assert 0 < found.evaluations <= budget
        again = negative_log_likelihood(
            found.log_mu, found.p_binom, rows["total"], rows["b"], rows["exposure"],
            rows["trials"], rows["lengths"],
        )  # fmt: skip
        assert again == pytest.approx(found.nll, rel=1e-9)
        repeat = sample(
            name, rows["total"], rows["b"], rows["exposure"], rows["trials"],
            rows["lengths"], 3, np.random.default_rng(seed),
        )  # fmt: skip
        assert repeat.nll == found.nll
        np.testing.assert_array_equal(repeat.log_mu, found.log_mu)


@pytest.mark.infra
def test_the_registry_names_the_samplers_the_module_runs() -> None:
    """`sandbox.extensions.copy_starts.HMM_SAMPLERS` is a literal, so importing it imports no sampler; it must match `SAMPLERS`."""
    from port.sandbox.extensions.copy_starts import HMM_SAMPLERS
    from port.sandbox.known_copy.hmm_objective import SAMPLERS

    assert HMM_SAMPLERS == SAMPLERS
