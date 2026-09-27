"""`port`'s restated outer loop against `cnaster`'s, bitwise (#433).

`port.patch.hmrf.core_inference.inference` states the schedule upstream's
`run_core_inference` reaches by reassigning its counter. The referee is the
function it replaces, so the marker is `patch`, and the bar is equality of
every array the result carries.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from tests.adapters import from_core_inference_truth
from tests.fixtures import core_inference_truth


def _run(entry: Any, max_iter_outer: int) -> Any:
    from cnaster.hmm_nophasing import hmm_nophasing

    truth = core_inference_truth(n_obs=120)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # NB `cnaster`'s ICM draws its queue from the legacy global state (#45).
        np.random.seed(0)  # noqa: NPY002
        return entry(
            **from_core_inference_truth(truth).as_kwargs(),
            hmmclass=hmm_nophasing,
            max_iter_outer=max_iter_outer,
            max_iter=3,
        )


def _arrays(result: Any) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(result[key])
        for key in (
            "new_log_mu",
            "new_p_binom",
            "new_alphas",
            "new_taus",
            "log_gamma",
            "pred_cnv",
            "new_assignment",
            "prev_assignment",
        )
    } | {"total_llf": np.asarray(result["total_llf"])}


@pytest.mark.patch
@pytest.mark.parametrize("max_iter_outer", [1, 4])
def test_the_restated_loop_is_upstreams_bitwise(
    max_iter_outer: int, cnaster_config: None
) -> None:
    """Every array equal, at a schedule with no merge-by-count and one with."""
    from cnaster.hmrf import run_core_inference
    from port.patch.hmrf.core_inference import inference

    theirs = _arrays(_run(run_core_inference, max_iter_outer))
    ours = _arrays(_run(inference, max_iter_outer))

    for key, value in theirs.items():
        np.testing.assert_array_equal(ours[key], value, err_msg=key)


@pytest.mark.patch
@pytest.mark.parametrize(
    ("max_iter_outer", "schedule"),
    [
        (1, [False, False]),
        (4, [False, False, True, True]),
        (5, [False, False, False, True, True]),
    ],
)
def test_the_schedule_without_convergence_is_upstreams_bitwise(
    max_iter_outer: int,
    schedule: list[bool],
    cnaster_config: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ARI never reaches the tolerance: merging waits on `max_iter_outer - 2`.

    Below `max_iter_outer = 3` that trigger is out of reach, so the run never
    merges -- the case the restated schedule makes visible. The merge flags
    each round passes are the schedule, and the arrays are upstream's.
    """
    from cnaster import hmrf
    from cnaster.config import get_global_config
    from cnaster.hmrf import run_core_inference
    from port.patch.hmrf.core_inference import inference

    monkeypatch.setattr(get_global_config().hmrf, "ari_tolerance", 2.0)
    flags: list[bool] = []
    assign = hmrf.pipeline_clone_assignment

    def recorded(*args: Any, **kwargs: Any) -> Any:
        flags.append(bool(kwargs["merge"]))
        return assign(*args, **kwargs)

    monkeypatch.setattr(hmrf, "pipeline_clone_assignment", recorded)

    theirs = _arrays(_run(run_core_inference, max_iter_outer))
    assert flags == schedule
    flags.clear()
    ours = _arrays(_run(inference, max_iter_outer))
    assert flags == schedule

    for key, value in theirs.items():
        np.testing.assert_array_equal(ours[key], value, err_msg=key)
