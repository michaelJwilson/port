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


def _baseline() -> Any:
    """`cnaster`'s class, with nothing wrapped around it.

    It used to need two compatibility rows to run at all -- `GuardedShift`
    and `RenamedKeywords`, without which it raised a `TypeError` and a numba
    `TypingError` before fitting anything. `cnaster@port#23cae59` fixed both
    and added `CnaHmmParams`, so upstream runs on its own and the referee is
    the unwrapped class. That is a stronger comparison than the shimmed one
    this test opened with.
    """
    from port.patch.hmm_nophasing import UPSTREAM

    return UPSTREAM


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
