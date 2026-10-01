"""The samplers on the HMM's own NLL keep a point no worse than where they started (#540).

`analytic`: a property of the sampler, whatever the data. Every sampler keeps
the lowest NLL it evaluated, the initial point included, so its best is never
above the initial point's; and the states it returns are that point, unsnapped,
so the objective re-evaluated at them is the NLL it reports.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.analytic
@pytest.mark.parametrize("name", ["hmc-hmm", "anneal-hmm", "tempering-hmm"])
def test_best_is_never_above_the_initial_point(name: str) -> None:
    from port.sandbox.known_copy.hmm_samplers import negative_log_likelihood, sample

    rng = np.random.default_rng(540)
    n = 120
    state = (np.arange(2 * n) // 40) % 3
    exposure = rng.uniform(200.0, 400.0, 2 * n)
    trials = rng.integers(20, 60, 2 * n).astype(np.float64)
    total = rng.poisson(exposure * np.array([0.5, 1.0, 1.5])[state]).astype(np.float64)
    b = rng.binomial(trials.astype(int), np.array([0.1, 0.5, 0.33])[state]).astype(
        np.float64
    )
    lengths = np.array([n, n])

    for seed in range(3):
        found = sample(
            name, total, b, exposure, trials, lengths, 3, np.random.default_rng(seed)
        )
        assert found.nll <= found.initial_nll
        again = negative_log_likelihood(
            found.log_mu, found.p_binom, total, b, exposure, trials, lengths
        )
        assert again == pytest.approx(found.nll, rel=1e-9)


@pytest.mark.infra
def test_the_registry_names_the_samplers_the_module_runs() -> None:
    """`sandbox.extensions.copy_starts.HMM_SAMPLERS` is a literal, so importing it imports no sampler; it must match `SAMPLERS`."""
    from port.sandbox.extensions.copy_starts import HMM_SAMPLERS
    from port.sandbox.known_copy.hmm_samplers import SAMPLERS

    assert HMM_SAMPLERS == SAMPLERS
