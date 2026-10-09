"""The shift's three drop-ins, each against the `cnaster` code it replaces (#293).

Referee: `cnaster`'s own function on the exposure rescaled by `exp(-shift)`; to
tolerance,
since the product is formed from recentred factors.
"""

import inspect
from typing import Any

import numpy as np
import port.patch.hmrf.core_inference as module
import pytest
import scipy.special
from cnaster.hmm_nophasing import hmm_nophasing as upstream
from port.patch.hmm_nophasing import hmm_nophasing
from port.patch.hmrf.clone_assignment import UPSTREAM, pipeline_clone_assignment
from port.pipeline import with_attributes

from tests.adapters import clone_assignment_arguments, clone_assignment_call
from tests.fixtures import (
    spot_clone_field,
    two_clone_optimize_arguments,
    two_clone_stacked_instance,
)


@pytest.mark.cnaster
@pytest.mark.patch
@pytest.mark.usefixtures("cnaster_config")
def test_a_shifted_clone_is_scored_as_upstream_scores_its_rescaled_exposure() -> None:
    """One clone's shifted field against `pipeline_clone_assignment` on `base * exp(-shift)`, to 1e-9 relative."""

    fixture = spot_clone_field(n_states=3, n_obs=40, n_spots=16, n_clones=1)
    arguments = clone_assignment_arguments(fixture, width=4)

    profile = fixture.base_nb_mean.sum(axis=1)
    log_lambda = np.log(profile / profile.sum())
    shift = scipy.special.logsumexp(fixture.log_mu[fixture.pred[0]] + log_lambda)

    def call(function: Any, base: np.ndarray, hmmclass: Any) -> Any:
        return clone_assignment_call(
            function, arguments, single_base_nb_mean=base, hmmclass=hmmclass
        )

    shifted = with_attributes(hmm_nophasing, apply_logmu_shift=True)
    _, ours, _ = call(pipeline_clone_assignment, fixture.base_nb_mean, shifted)

    _, theirs, _ = call(UPSTREAM, fixture.base_nb_mean * np.exp(-shift), hmm_nophasing)

    np.testing.assert_allclose(ours, theirs, rtol=1e-9)


@pytest.mark.cnaster
@pytest.mark.patch
@pytest.mark.usefixtures("cnaster_config")
def test_the_fit_is_upstreams_off_and_decodes_under_its_own_shift_on() -> None:
    """`optimize` off is `cnaster`'s bitwise (#433); on, `log_gamma` is `get_state_posteriors` at the recorded shift, to 1e-9."""

    instance = two_clone_stacked_instance()
    args, kwargs = two_clone_optimize_arguments(instance)

    theirs = upstream(params="smp", t=0.99).optimize(*args, **kwargs)
    differenced = with_attributes(hmm_nophasing, analytic_gradient=False)
    ours = differenced(params="smp", t=0.99).optimize(*args, **kwargs)

    for key in ("new_log_mu", "new_p_binom", "log_gamma"):
        np.testing.assert_array_equal(ours[key], theirs[key])

    model = with_attributes(hmm_nophasing, apply_logmu_shift=True)(params="smp", t=0.99)
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
    """`run_core_inference` pins only a shifted `mu` fit; unshifted or BAF-only fits are upstream's."""

    def fake(*_: Any, **__: Any) -> dict[str, np.ndarray]:
        return {
            "new_log_mu": np.log(np.array([[2.0], [0.8], [5.0]])),
            "new_p_binom": np.array([[0.5], [0.49], [0.1]]),
        }

    original = module.UPSTREAM
    module.UPSTREAM = fake
    # NB `cnaster`'s required arguments, never read by the fake.
    blanks = dict.fromkeys(
        name
        for name, p in inspect.signature(original).parameters.items()
        if p.default is inspect.Parameter.empty
    )

    try:
        unshifted = module.run_core_inference(
            **blanks, hmmclass=hmm_nophasing, params="smp"
        )

        shifted = with_attributes(hmm_nophasing, apply_logmu_shift=True)
        pinned = module.run_core_inference(**blanks, hmmclass=shifted, params="smp")
        baf_only = module.run_core_inference(**blanks, hmmclass=shifted, params="sp")
    finally:
        module.UPSTREAM = original

    np.testing.assert_array_equal(unshifted["new_log_mu"], fake()["new_log_mu"])
    np.testing.assert_array_equal(baf_only["new_log_mu"], fake()["new_log_mu"])
    np.testing.assert_allclose(
        np.exp(pinned["new_log_mu"][:, 0]), [2.5, 1.0, 6.25], rtol=1e-12
    )
