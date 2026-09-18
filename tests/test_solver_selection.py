"""Solver selection, and the posterior the pipeline normalises with.

`hmm_utils` reads the solver from the global configuration and then builds
the keyword arguments that solver accepts. Passing an option a solver does
not take is not an error scipy reports usefully, so the mapping is the thing
worth pinning.
"""

import numpy as np
import pytest
from scipy.special import logsumexp

from tests.adapters import from_negative_binomial_chains
from tests.fixtures import negative_binomial_chains


@pytest.mark.infra
@pytest.mark.usefixtures("cnaster_config")
@pytest.mark.analytic
def test_the_configured_solver_is_returned() -> None:
    """The name comes from configuration rather than a default."""
    from cnaster.hmm_utils import get_solver

    assert get_solver() == "L-BFGS-B"


@pytest.mark.infra
@pytest.mark.usefixtures("cnaster_config")
@pytest.mark.analytic
def test_solver_options_are_the_ones_that_solver_takes() -> None:
    """Each solver gets its own keywords, with the `em_` prefix stripped.

    `L-BFGS-B` takes `maxiter` and `ftol`; handing it `xrtol`, which is
    `BFGS`'s, is silently ignored rather than rejected, so a mapping that
    returned the wrong set would go unnoticed until a fit stopped
    converging.
    """
    from cnaster.hmm_utils import get_em_solver_params

    params = get_em_solver_params()

    assert set(params) == {"maxiter", "ftol", "disp"}
    assert all(isinstance(value, float) for value in params.values())


@pytest.mark.infra
@pytest.mark.analytic
def test_an_unknown_solver_is_refused() -> None:
    """A name outside the supported set stops the fit rather than starting one."""
    from cnaster.config import YAMLConfig, get_global_config, set_global_config
    from cnaster.hmm_utils import get_solver

    previous = get_global_config()
    set_global_config(YAMLConfig({"hmm": {"solver": "Powell"}}))
    try:
        with pytest.raises(AssertionError):
            get_solver()
    finally:
        set_global_config(previous)


@pytest.mark.oracle
@pytest.mark.property
@pytest.mark.parametrize("n_states", [2, 4])
def test_copy_state_posterior_normalises_the_lattice(n_states: int) -> None:
    """`compute_copy_state_posterior` is the normalised forward-backward product.

    The same quantity `get_state_posteriors` returns, by a second route that
    takes the two lattices as arguments; the pipeline uses this one, so the
    two agreeing is what makes either safe to read.
    """
    from cnaster.hmm import compute_copy_state_posterior
    from cnaster.hmm_nophasing import hmm_nophasing

    fixture = negative_binomial_chains(n_states=n_states, sequence_length=30)
    inputs = from_negative_binomial_chains(fixture)
    log_emit_rdr, log_emit_baf = (
        hmm_nophasing.compute_emission_probability_nb_betabinom(
            inputs.single_X,
            inputs.base_nb_mean,
            inputs.log_mu,
            inputs.alphas,
            inputs.total_bb_RD,
            inputs.p_binom,
            inputs.taus,
        )
    )
    args = (
        inputs.lengths,
        inputs.log_transmat,
        inputs.log_startprob,
        log_emit_rdr + log_emit_baf,
        inputs.log_sitewise_transmat,
    )
    log_alpha = hmm_nophasing.forward_lattice(*args)
    log_beta = hmm_nophasing.backward_lattice(*args)

    log_gamma = compute_copy_state_posterior(log_alpha, log_beta)

    np.testing.assert_allclose(logsumexp(log_gamma, axis=0), 0.0, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(
        log_gamma,
        hmm_nophasing().get_state_posteriors(*args),
        rtol=0.0,
        atol=1e-9,
    )
