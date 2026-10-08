"""The staged filter reproduces the array `gmm_init` hands `GaussianMixture.fit`, bitwise
(#229 stage 1).

Referee: `cnaster` itself, with `fit` intercepted. `patch`: agreement, not correctness
(#229, #230).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from port.sandbox.patch.hmm_initialize.filtering import (
    Standardize,
    design_matrix,
    filter_observations,
)
from port.sim.truth import CoreInferenceTruth, core_inference_truth
from sklearn.mixture import GaussianMixture

from tests.adapters import stacked_clones

MIN_BINOM, MAX_BINOM = 0.01, 0.99
"""The bounds `tests/conftest.py` configures, which `gmm_init` reads globally."""


@pytest.fixture
def planted() -> CoreInferenceTruth:
    return core_inference_truth(
        n_clones=2, n_states=3, lattice=(5, 6), n_obs=30, n_segments=2, seed=29
    )


def _stacked(truth: CoreInferenceTruth) -> tuple[Any, ...]:
    """`gmm_init`'s positional arguments, as `run_core_inference` builds them."""
    from cnaster.hmm_nophasing import get_log_transmat

    stacked = stacked_clones(truth)

    return (
        truth.n_states,
        stacked.X,
        stacked.base_nb_mean,
        stacked.total_bb_RD,
        "smp",
        stacked.lengths,
        get_log_transmat(truth.n_states, 1.0 - 1e-6),
        stacked.sitewise,
    )


def _capture_design(monkeypatch: pytest.MonkeyPatch, arguments: tuple[Any, ...]) -> Any:
    """Run `gmm_init` and return the array it passed to `GaussianMixture.fit`."""
    import cnaster.hmm_initialize as initialize

    seen: dict[str, Any] = {}
    original = GaussianMixture.fit

    def capturing(self: GaussianMixture, X: Any, y: Any = None) -> Any:
        seen.setdefault("design", np.array(X, copy=True))

        return original(self, X, y)

    monkeypatch.setattr(initialize.GaussianMixture, "fit", capturing)
    initialize.gmm_init(*arguments, random_state=0)

    assert "design" in seen, "gmm_init did not reach the mixture fit"

    return seen["design"]


@pytest.mark.patch
@pytest.mark.usefixtures("cnaster_config", "cnaster_perf_sink")
def test_the_staged_filter_reproduces_the_design_matrix_bitwise(
    planted: CoreInferenceTruth, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The staged design matrix equals the top (unaugmented) half of what reaches `fit`,
    bitwise.
    """
    arguments = _stacked(planted)
    captured = _capture_design(monkeypatch, arguments)

    _, stack_X, stack_base, stack_total, params, *_ = arguments

    observations = filter_observations(
        stack_X,
        stack_base,
        stack_total,
        params,
        min_binom=MIN_BINOM,
        max_binom=MAX_BINOM,
    )
    design, standardize, record = design_matrix(observations)

    assert design.shape[0] == record.rows_kept

    # NB the mirrored augmentation doubles the rows, so ours is the top half.
    assert captured.shape[0] in {design.shape[0], 2 * design.shape[0]}, (
        f"captured {captured.shape} against staged {design.shape}"
    )

    theirs = captured[: design.shape[0], :]

    assert np.array_equal(theirs, design), (
        "the staged filter does not reproduce gmm_init's design matrix; "
        f"max |difference| {np.max(np.abs(theirs - design)):.3e}"
    )

    assert standardize is not None
    assert standardize.fitted


@pytest.mark.patch
@pytest.mark.usefixtures("cnaster_config", "cnaster_perf_sink")
def test_the_transform_inverts_itself(planted: CoreInferenceTruth) -> None:
    """The inverse transform undoes the forward one (`cnaster` lines 351 and 519)."""
    arguments = _stacked(planted)
    _, stack_X, stack_base, stack_total, params, *_ = arguments

    observations = filter_observations(
        stack_X,
        stack_base,
        stack_total,
        params,
        min_binom=MIN_BINOM,
        max_binom=MAX_BINOM,
    )

    assert observations.rdr is not None

    values = np.log(observations.rdr)
    standardize = Standardize().fit(values)

    round_trip = standardize.invert(standardize.apply(values))

    assert np.allclose(round_trip, values, rtol=0.0, atol=1e-12), (
        f"max |difference| {np.max(np.abs(round_trip - values)):.3e}"
    )


@pytest.mark.patch
@pytest.mark.usefixtures("cnaster_config", "cnaster_perf_sink")
def test_the_record_counts_what_the_filters_did(
    planted: CoreInferenceTruth,
) -> None:
    """The record counts what each filter did (#230)."""
    arguments = _stacked(planted)
    _, stack_X, stack_base, stack_total, params, *_ = arguments

    observations = filter_observations(
        stack_X,
        stack_base,
        stack_total,
        params,
        min_binom=MIN_BINOM,
        max_binom=MAX_BINOM,
    )
    _, _, record = design_matrix(observations)

    assert record.rows_kept + record.rows_dropped == record.n_bins
    assert 0.0 <= record.retained <= 1.0
    assert record.baf_clipped >= 0

    # NB the conftest bounds clip nothing a fixture plants; this pins that intent.
    assert record.baf_clipped == 0, (
        "the fixture's allele shares now reach the clip, so the initializer's "
        "start is the clip's rather than the data's"
    )

    assert str(record).startswith(f"{record.rows_kept}/")


@pytest.mark.patch
@pytest.mark.usefixtures("cnaster_config", "cnaster_perf_sink")
def test_imputation_none_is_a_policy_rather_than_a_step(
    planted: CoreInferenceTruth,
) -> None:
    """Imputation `"none"` agrees with `cnaster` on a fixture with no missing data."""
    arguments = _stacked(planted)
    _, stack_X, stack_base, stack_total, params, *_ = arguments

    observations = filter_observations(
        stack_X,
        stack_base,
        stack_total,
        params,
        min_binom=MIN_BINOM,
        max_binom=MAX_BINOM,
    )

    filled, _, filled_record = design_matrix(observations, imputation="ffill_bfill")
    bare, _, bare_record = design_matrix(observations, imputation="none")

    if filled_record.imputed == 0:
        assert np.array_equal(filled, bare)
        assert filled_record.rows_kept == bare_record.rows_kept
    else:
        assert bare_record.rows_kept <= filled_record.rows_kept
