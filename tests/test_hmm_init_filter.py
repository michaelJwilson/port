"""`port`'s staged filter reproduces what `gmm_init` feeds the mixture, bitwise.

**#229 stage 1.** `cnaster.hmm_initialize.gmm_init` performs four filters, two
transforms and an imputation in one 235-line function, interleaved across
three blocks. `port.patch.hmm_initialize.filtering` separates them. A refactor is only a
refactor if the result is the same, so the claim asserted here is the strong
one: the array `cnaster` hands `GaussianMixture.fit` is **the same array**,
not one within a tolerance.

The referee is `cnaster` itself rather than a re-derivation. `gmm_init` does
not return its design matrix, so the fit is intercepted and the argument
captured -- which is what makes this a comparison against the code that ships
instead of against a second copy of the arithmetic.

`patch` throughout: this says the two agree, not that either is right. What
the filtering *should* be is #229, and which initializer is best is #230.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from port.patch.hmm_initialize.filtering import (
    Standardize,
    design_matrix,
    filter_observations,
)
from sklearn.mixture import GaussianMixture

from tests.fixtures import CoreInferenceTruth, core_inference_truth

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
    from cnaster.hmrf_utils import clone_stack_obs
    from cnaster.pseudobulk import merge_pseudobulk_by_index_mix

    counts = np.stack([truth.counts_nb, truth.counts_bb], axis=1)
    X, base, total, _ = merge_pseudobulk_by_index_mix(
        counts, truth.base_nb_mean, truth.total_bb_RD, truth.clone_index
    )
    stack_X, stack_base, stack_total, lengths, sitewise, _ = clone_stack_obs(
        X, base, total, truth.lengths, np.zeros((truth.n_obs, 2)), None
    )

    return (
        truth.n_states,
        stack_X,
        stack_base,
        stack_total,
        "smp",
        lengths,
        get_log_transmat(truth.n_states, 1.0 - 1e-6),
        sitewise,
    )


def _capture_design(monkeypatch: pytest.MonkeyPatch, arguments: tuple[Any, ...]) -> Any:
    """Run `gmm_init` and return the array it passed to `GaussianMixture.fit`.

    Intercepting `fit` rather than reimplementing the eleven steps is the
    point: a re-derivation would agree with itself.
    """
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
    """The array the mixture is fitted to is the same array.

    `gmm_init` augments the design by mirroring the BAF columns and stacking,
    so what reaches `fit` is `2N` rows for `N` observations. The staged
    pipeline produces the unaugmented half, which is the top block -- the
    augmentation is step 8 and belongs to the fit, not to the filtering.
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
    """`cnaster` writes the forward map at line 351 and the inverse at 519.

    Nothing there checks that the second undoes the first. Here they are one
    object, and this is the check.
    """
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
    """`cnaster` logs three of the five counts as free text and two not at all.

    The record is what makes a run's filtering recoverable afterwards, and
    what #230 needs to compare backends under one filter policy.
    """
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

    # NB the conftest bounds are wide enough to clip nothing a fixture plants,
    #    which is stated there on purpose, so this pins that intent.
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
    """`cnaster` fills NaNs by genomic adjacency and states no assumption.

    `"none"` is what measures its cost: the rows it would have saved are
    dropped instead, and the record says how many. On a fixture that plants
    no missing data the two agree, which is what this pins -- the policy is
    available, and it is not silently changing a clean run.
    """
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
