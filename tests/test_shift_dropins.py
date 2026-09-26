"""The shift's three drop-ins, each against the `cnaster` code it replaces (#293).

The per-clone `logmu_shift` reaches a run through `SHIFT_SWAPS` and through
`pipeline_clone_assignment`'s shifted branch. Each is pinned here against
`cnaster`'s own function, fed the input the shift says it should see:
the exposure rescaled by `exp(-shift)`. Tolerances rather than bitwise,
because the replacement forms `exp(log_mu - shift)` from recentred factors
and upstream from the raw ones, so the last bits of the product differ.
"""

from typing import Any

import numpy as np
import pytest

from tests.adapters import clone_assignment_arguments
from tests.fixtures import spot_clone_field


@pytest.mark.cnaster
@pytest.mark.patch
@pytest.mark.usefixtures("cnaster_config")
def test_a_shifted_clone_is_scored_as_upstream_scores_its_rescaled_exposure() -> None:
    """One clone, so upstream's single exposure can carry the clone's shift.

    `port`'s shifted field against `cnaster.hmrf.pipeline_clone_assignment`
    handed `base * exp(-shift)`, with `shift = log sum_g lambda_g mu_{s(g)}`
    and `lambda` from the unscaled baseline, as the replacement builds it.
    Stated to 1e-9 relative.
    """
    import scipy.special
    from port.patch.hmm_nophasing import hmm_nophasing, logmu_shift
    from port.patch.hmrf.clone_assignment import UPSTREAM, pipeline_clone_assignment

    fixture = spot_clone_field(n_states=3, n_obs=40, n_spots=16, n_clones=1)
    arguments = clone_assignment_arguments(fixture, width=4)

    profile = fixture.base_nb_mean.sum(axis=1)
    log_lambda = np.log(profile / profile.sum())
    shift = scipy.special.logsumexp(fixture.log_mu[fixture.pred[0]] + log_lambda)

    def call(function: Any, base: np.ndarray, hmmclass: Any) -> Any:
        return function(
            arguments["single_X"],
            base,
            arguments["single_total_bb_RD"],
            arguments["res"],
            arguments["pred"],
            arguments["adjacency_mat"],
            arguments["prev_assignment"].copy(),
            arguments["sample_ids"],
            arguments["spatial_weight"],
            hmmclass=hmmclass,
        )

    with logmu_shift():
        _, ours, _ = call(
            pipeline_clone_assignment, fixture.base_nb_mean, hmm_nophasing
        )

    _, theirs, _ = call(UPSTREAM, fixture.base_nb_mean * np.exp(-shift), hmm_nophasing)

    np.testing.assert_allclose(ours, theirs, rtol=1e-9)


def _stacked_instance(seed: int = 4) -> dict[str, Any]:
    """Two clones of 30 bins, stacked as `clone_stack_obs` stacks them."""
    rng = np.random.default_rng(seed)
    n_obs, n_clones = 30, 2
    n_segments = n_obs * n_clones

    base = rng.uniform(40.0, 80.0, (n_segments, 1))
    states = rng.integers(0, 2, n_segments)
    means = base[:, 0] * np.array([1.0, 2.0])[states]

    X = np.zeros((n_segments, 2, 1))
    X[:, 0, 0] = rng.poisson(means)
    X[:, 1, 0] = rng.binomial(20, np.array([0.5, 0.25])[states])

    return {
        "X": X,
        "lengths": np.array([n_obs] * n_clones),
        "base": base,
        "total": np.full((n_segments, 1), 20.0),
        "normal_lambda": base[:n_obs, 0] / base[:n_obs, 0].sum(),
        "clone_lengths": np.array([n_obs] * n_clones),
    }


@pytest.mark.cnaster
@pytest.mark.patch
@pytest.mark.usefixtures("cnaster_config")
def test_the_fit_is_upstreams_off_and_decodes_under_its_own_shift_on() -> None:
    """`optimize`, off and on.

    Off, the replacement returns what `cnaster`'s class returns, bitwise.
    On, its `log_gamma` is `cnaster`'s own `get_state_posteriors` on the
    emission of the rescaled exposure `base * exp(-shift)`, with the shift
    the replacement recorded, to 1e-9 -- so the decode it returns is the one
    its shift describes, which `hmm_nophasing.py:1085` alone would not give.
    """
    from cnaster.hmm_nophasing import hmm_nophasing as upstream
    from port.patch.hmm_nophasing import hmm_nophasing, logmu_shift

    instance = _stacked_instance()
    kwargs = {
        "init_log_mu": np.log(np.array([[1.0], [2.0]])),
        "init_p_binom": np.array([[0.5], [0.25]]),
        "max_iter": 20,
        "normal_lambda": instance["normal_lambda"],
        "clone_lengths": instance["clone_lengths"],
        "shared_NB_dispersion": True,
        "shared_BB_dispersion": True,
    }
    args = (instance["X"], instance["lengths"], 2, instance["base"], instance["total"])

    theirs = upstream(params="smp", t=0.99).optimize(*args, **kwargs)
    ours = hmm_nophasing(params="smp", t=0.99).optimize(*args, **kwargs)

    for key in ("new_log_mu", "new_p_binom", "log_gamma"):
        np.testing.assert_array_equal(ours[key], theirs[key])

    with logmu_shift():
        model = hmm_nophasing(params="smp", t=0.99)
        shifted = model.optimize(*args, **kwargs)
        row_shift = hmm_nophasing._row_shift

    assert row_shift is not None
    assert row_shift.size == instance["X"].shape[0]

    rdr, baf = upstream.compute_emission_probability_nb_betabinom(
        instance["X"],
        instance["base"] * np.exp(-row_shift)[:, None],
        shifted["new_log_mu"],
        shifted["new_alphas"],
        instance["total"],
        shifted["new_p_binom"],
        shifted["new_taus"],
    )
    expected = model.get_state_posteriors(
        instance["lengths"],
        shifted["new_log_transmat"],
        shifted["new_log_startprob"],
        rdr + baf,
        None,
    )

    np.testing.assert_allclose(shifted["log_gamma"], expected, rtol=1e-9, atol=1e-9)


@pytest.mark.patch
def test_the_pin_applies_to_a_shifted_rate_fit_only() -> None:
    """`run_core_inference` pins after upstream's inference, and only then.

    A shifted fit of `mu` is pinned so the balanced, lowest-`mu` state is 1;
    an unshifted one, or a fit with no `mu` (`params="sp"`, the BAF-only
    stage), is returned as upstream returned it.
    """
    import port.patch.hmrf.core_inference as module
    from port.patch.hmm_nophasing import hmm_nophasing, logmu_shift

    def fake(*_: Any, **__: Any) -> dict[str, np.ndarray]:
        return {
            "new_log_mu": np.log(np.array([[2.0], [0.8], [5.0]])),
            "new_p_binom": np.array([[0.5], [0.49], [0.1]]),
        }

    original = module.UPSTREAM
    module.UPSTREAM = fake

    try:
        unshifted = module.run_core_inference(hmmclass=hmm_nophasing, params="smp")

        with logmu_shift():
            pinned = module.run_core_inference(hmmclass=hmm_nophasing, params="smp")
            baf_only = module.run_core_inference(hmmclass=hmm_nophasing, params="sp")
    finally:
        module.UPSTREAM = original

    np.testing.assert_array_equal(unshifted["new_log_mu"], fake()["new_log_mu"])
    np.testing.assert_array_equal(baf_only["new_log_mu"], fake()["new_log_mu"])
    np.testing.assert_allclose(
        np.exp(pinned["new_log_mu"][:, 0]), [2.5, 1.0, 6.25], rtol=1e-12
    )
