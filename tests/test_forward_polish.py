"""`port.studies.forward_polish`: the run's fit with its EM swapped for L-BFGS on the forward log-likelihood (#748).

The forward fit's gradient is #433's closed form read at the posteriors at
the point itself (Fisher's identity); it is pinned against central
differences of the forward log-likelihood, a route through no posterior.
Its end point is pinned against the run's own rescoring (`hmm.py:173`, the
dense emission), a second implementation of the likelihood it maximized.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


def _fit(method: str | None, capture: dict[str, Any] | None = None) -> tuple[Any, Any]:
    """`pipeline_baum_welch` on two stacked clones, shift on, fitted by `method` (`None`: no swap); with `capture`, the forward objective."""
    from cnaster.hmm import pipeline_baum_welch
    from port.patch.hmm_nophasing import hmm_nophasing, shifted_emission
    from port.studies import forward_polish

    from tests.test_shift_dropins import _stacked_instance

    emission: Any = shifted_emission
    instance = _stacked_instance()
    sitewise = np.zeros(instance["X"].shape[0])
    arguments = {
        "hmmclass": hmm_nophasing, "params": "smp", "t": 0.99,
        "init_log_mu": np.log(np.array([[1.0], [2.0]])), "init_p_binom": np.array([[0.5], [0.25]]),
        "normal_lambda": instance["normal_lambda"], "clone_lengths": instance["clone_lengths"],
    }  # fmt: skip
    args = (
        None,
        instance["X"],
        instance["lengths"],
        2,
        instance["base"],
        instance["total"],
        sitewise,
    )
    stage = type("S", (), {"lengths": instance["lengths"], "args": args})()
    shift = hmm_nophasing.apply_logmu_shift
    hmm_nophasing.apply_logmu_shift = True
    try:
        if method is None:
            return pipeline_baum_welch(*args, **arguments), None
        if capture is None:
            with forward_polish.polished(stage, method) as counts:
                return pipeline_baum_welch(*args, **arguments), counts
        installed = emission.analytic_bfgs

        def probe(gradient: Any) -> Any:
            counts = forward_polish.Counts()
            lattice = forward_polish.lattice_for(
                gradient, instance["lengths"], sitewise, counts
            )

            def method(
                fun: Any, x0: np.ndarray, args: tuple[Any, ...] = (), **_: Any
            ) -> Any:
                fun(x0)
                gradient.model.state_posteriors = lattice()[1]
                held = gradient.model._decode()
                gradient.model._decode = lambda: held
                capture["objective"] = forward_polish._Forward(
                    lambda x: float(fun(x)), gradient, lattice, x0, counts
                )
                capture["x0"] = np.asarray(x0, dtype=np.float64)
                raise StopIteration

            return method

        emission.analytic_bfgs = probe
        try:
            pipeline_baum_welch(*args, **arguments, max_iter=5)
        except StopIteration:
            pass
        finally:
            emission.analytic_bfgs = installed
        return capture["objective"], None
    finally:
        hmm_nophasing.apply_logmu_shift = shift


@pytest.mark.oracle
@pytest.mark.usefixtures("cnaster_config")
def test_the_forward_gradient_is_the_forward_likelihoods_derivative() -> None:
    """At 5 points about the start, the closed form at the point's own posteriors against central differences, to 1e-6 of the largest component.

    The decode the shift is taken at is held, as within a round of the fit;
    the start-probability block is the EM's and reads zero on both sides.
    """
    capture: dict[str, Any] = {}
    objective, _ = _fit("forward", capture)
    x0 = capture["x0"]
    rng = np.random.default_rng(748)
    for _ in range(5):
        x = x0 + rng.normal(0.0, 0.05, x0.size)
        _, grad = objective.evaluate(x)
        step = 1e-5
        numeric = np.array([
            (objective.evaluate(x + step * e)[0] - objective.evaluate(x - step * e)[0]) / (2 * step)
            for e in np.eye(x.size)
        ])  # fmt: skip
        np.testing.assert_allclose(grad, numeric, atol=1e-6 * np.abs(numeric).max())


@pytest.mark.oracle
@pytest.mark.usefixtures("cnaster_config")
def test_the_run_scores_the_forward_end_at_the_likelihood_it_maximized() -> None:
    """The run's own rescoring at the forward fit's end (dense emission, `hmm.py:173`) is the last round's value, to 1e-9 relative.

    Every bin is under `max_rdr` here, so the coded emission the fit read
    and the dense one the run rescores with are one likelihood; the fit
    ends where its decode is one it held before.
    """
    result, counts = _fit("forward")
    assert counts.rounds >= 1
    assert counts.trace[-1] == pytest.approx(float(result.llf), rel=1e-9)


@pytest.mark.patch
@pytest.mark.usefixtures("cnaster_config")
def test_the_em_arm_is_the_runs_fit_bitwise() -> None:
    """`polished(stage, "em")` counts the run's fit and changes nothing in it."""
    counted, counts = _fit("em")
    plain, _ = _fit(None)
    assert counts.passes > 0
    assert float(plain.llf) == float(counted.llf)
    np.testing.assert_array_equal(plain.params.new_log_mu, counted.params.new_log_mu)
