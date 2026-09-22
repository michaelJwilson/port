"""The rewritten pipeline is upstream's, with the spot axis gone (#259 stage 3).

`port.patch.optimization_pipeline` replaces
`_run_optimization_pipeline` outright: no `range(n_spots)`, two scratch
buffers instead of two one-element lists, `(n_states, 1)` parameters instead
of `(n_states, n_spots)`, and the coded emission where upstream calls the
dense one under its own `# TODO call coded`.

That is a lot of moved code for a claim of "nothing changed", so the referee
is the whole fit: both classes on one fixture, every returned array compared
**bitwise**. `patch`, because it says the rewrite agrees with `cnaster` and
not that `cnaster` is right.

#259 stage 5's derived gradient is **off by default**, so that claim is the
default arm's. The test below that turns it on compares the two arms at the
tolerance each reaches, which is the whole of what the flag changes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.adapters import from_core_inference_truth
from tests.fixtures import core_inference_truth

FIELDS = (
    "new_log_mu",
    "new_alphas",
    "new_p_binom",
    "new_taus",
    "new_log_startprob",
    "new_log_transmat",
)
"""Every fitted array the result carries, by the name it carries it under."""


def _baseline() -> type:
    """`cnaster`'s class, with the least that makes it run at all.

    `GuardedShift` and `RenamedKeywords` are #259 stage 1's compatibility
    rows: without them the unpatched class raises a `TypeError` and a numba
    `TypingError` before it fits anything, so there would be nothing to
    compare against. Neither changes a number -- that is what stage 1
    established -- so this is upstream's arithmetic with upstream's defects
    stepped around.
    """
    from port.patch.hmm_nophasing import UPSTREAM, GuardedShift, RenamedKeywords

    class Baseline(GuardedShift, RenamedKeywords, UPSTREAM):  # type: ignore[misc]
        pass

    return Baseline


def _fit(hmmclass: type, truth: Any) -> Any:
    import warnings

    from cnaster.hmrf import run_core_inference

    with warnings.catch_warnings():
        # `scipy` rejects the `ftol` the shipped solver options pass (#46).
        warnings.simplefilter("ignore")
        return run_core_inference(
            **from_core_inference_truth(truth).as_kwargs(),
            hmmclass=hmmclass,
            max_iter_outer=1,
            max_iter=5,
        )


def _truth() -> Any:
    return core_inference_truth(
        n_clones=2, n_states=3, lattice=(6, 5), n_obs=60, n_segments=2
    )


@pytest.mark.xfail(
    reason=(
        "the referee is gone: `cnaster@port#e4e8739` upstreamed this very "
        "pipeline but not the named-parameter object it reads, so its own "
        "`_run_optimization_pipeline` raises `AttributeError: 'tuple' object "
        "has no attribute 'log_mu'` at hmm_nophasing.py:1205 before it fits "
        "anything. There is no upstream arm left to compare against. "
        "**strict**, so this goes red the moment upstream can run again and "
        "the bitwise claim is restored rather than quietly dropped."
    ),
    strict=True,
)
@pytest.mark.patch
def test_the_rewritten_pipeline_is_upstreams_bitwise(cnaster_config: None) -> None:
    """Both classes, one fixture, every returned array to the bit.

    The fit is the referee rather than a unit of it: the rewrite moves the
    scratch buffers, the parameter rank, the emission call and the final
    posterior, and only running the whole thing exercises all four together.
    """
    from port.patch.hmm_nophasing import hmm_nophasing as REWRITE

    truth = _truth()

    theirs = _fit(_baseline(), truth)
    ours = _fit(REWRITE, truth)

    assert ours.llf == theirs.llf, f"llf {ours.llf!r} against {theirs.llf!r}"

    for field in FIELDS:
        mine = np.asarray(getattr(ours.params, field))
        upstream = np.asarray(getattr(theirs.params, field))

        assert mine.shape == upstream.shape, field
        assert np.array_equal(mine, upstream), (
            f"{field}: max |difference| {np.max(np.abs(mine - upstream)):.3e}"
        )

    assert np.array_equal(
        np.asarray(ours.assignment.new_assignment),
        np.asarray(theirs.assignment.new_assignment),
    ), "the clone assignment moved"


@pytest.mark.patch
def test_the_analytic_gradient_reaches_the_same_fit(cnaster_config: None) -> None:
    """The derived gradient is the same objective by a different route.

    BFGS with `jac=None` differences the objective; with the gradient it
    steps on the derived one. Both maximize the same likelihood, so the
    fit agrees to a tolerance rather than to the bit: on this fixture the
    likelihood agrees to 5.9e-06 relative, the fitted parameters to
    4.8e-03, and the clone assignment is identical. `log_startprob` and
    `log_transmat` are bitwise because they come from the E step's counts
    rather than from the solver.

    **That the fit moves at all is why the flag is off by default.** The
    tolerances asserted below carry 2x headroom over those figures; the
    gradient itself is refereed against central differences (1.7e-07) in
    `test_em_gradient.py`. That `new_taus` moves furthest in absolute terms
    while the likelihood moves 5.9e-06 is the weak identification of tau,
    not an error in either arm.
    """
    from port.patch.hmm_nophasing import hmm_nophasing as REWRITE
    from port.patch.optimization_pipeline import analytic_jac

    truth = _truth()

    differenced = _fit(REWRITE, truth)

    with analytic_jac():
        derived = _fit(REWRITE, truth)

    assert abs(derived.llf - differenced.llf) <= 1.2e-05 * abs(differenced.llf), (
        f"llf {derived.llf!r} against {differenced.llf!r}"
    )

    for field in ("new_log_mu", "new_alphas", "new_p_binom", "new_taus"):
        mine = np.asarray(getattr(derived.params, field))
        theirs = np.asarray(getattr(differenced.params, field))
        scale = max(float(np.max(np.abs(theirs))), 1e-12)

        assert np.max(np.abs(mine - theirs)) <= 1.0e-02 * scale, (
            f"{field}: max |difference| {np.max(np.abs(mine - theirs)):.3e}"
        )

    for field in ("new_log_startprob", "new_log_transmat"):
        assert np.array_equal(
            np.asarray(getattr(derived.params, field)),
            np.asarray(getattr(differenced.params, field)),
        ), f"{field} is the E step's, and moved"

    assert np.array_equal(
        np.asarray(derived.assignment.new_assignment),
        np.asarray(differenced.assignment.new_assignment),
    ), "the clone assignment moved"


@pytest.mark.patch
def test_the_gradient_is_off_unless_asked_for(cnaster_config: None) -> None:
    """The default is `cnaster`'s arm, and the switch restores it.

    A class attribute left set by a block that raised would make every later
    fit in the process the other arm, and the two exist to be compared. So
    the default is pinned here as well as the restore, in both directions.
    """
    from port.patch.hmm_nophasing import hmm_nophasing as REWRITE
    from port.patch.optimization_pipeline import analytic_jac

    assert REWRITE.use_analytic_jac is False, "the default arm is cnaster's"

    with analytic_jac():
        assert REWRITE.use_analytic_jac is True

        with analytic_jac(enabled=False):
            assert REWRITE.use_analytic_jac is False

        assert REWRITE.use_analytic_jac is True, "nesting restored to a literal"

    assert REWRITE.use_analytic_jac is False

    with pytest.raises(RuntimeError), analytic_jac():
        raise RuntimeError

    assert REWRITE.use_analytic_jac is False, "a raising block left it set"


@pytest.mark.patch
def test_more_than_one_column_is_refused(cnaster_config: None) -> None:
    """The assert upstream states and then carries the axis past.

    Checked where the code relies on it, so a caller that passes an instance
    the rewrite cannot reduce is told rather than silently given the first
    column.
    """
    from port.patch.hmm_nophasing import hmm_nophasing as REWRITE

    model = REWRITE()

    with pytest.raises(ValueError, match="expected one column"):
        model._run_optimization_pipeline(
            "em",
            np.zeros((10, 2, 3)),
            [10],
            2,
            np.ones((10, 3)),
            np.ones((10, 3)),
        )


@pytest.mark.patch
def test_the_initial_parameters_are_the_vstack_written_out(
    cnaster_config: None,
) -> None:
    """`np.vstack([linspace(...) for _ in range(1)]).T` is `linspace(...)[:, None]`.

    Pinned against upstream's own method rather than against the literal, so
    a change to `cnaster`'s defaults turns this red instead of going unseen.
    """
    from port.patch.hmm_nophasing import UPSTREAM
    from port.patch.hmm_nophasing import hmm_nophasing as REWRITE

    n_states = 5

    theirs = UPSTREAM.get_initial_params(UPSTREAM(), n_states, 1)
    ours = REWRITE.get_initial_params(REWRITE(), n_states, 1)

    for mine, upstream in zip(ours, theirs, strict=True):
        assert np.array_equal(np.asarray(mine), np.asarray(upstream))

    with pytest.raises(ValueError, match="expected one column"):
        REWRITE.get_initial_params(REWRITE(), n_states, 2)
