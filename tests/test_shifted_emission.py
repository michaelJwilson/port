"""The emission with the library normalizer `compute_logmu_shifts` folded in (#276).

Referee: `cnaster`'s own `compute_logmu_shifts` and `_nb_logpmf_1d`. Off it is upstream
bitwise; on, each clone takes its own per-segment shift.
"""

from __future__ import annotations

import numpy as np
import pytest
from cnaster.hmm_nophasing import _nb_logpmf_1d, compute_logmu_shifts
from cnaster.hmm_nophasing import hmm_nophasing as upstream
from port.patch.hmm_nophasing import hmm_nophasing
from port.patch.hmm_nophasing.logmu_shift import shifts
from port.patch.hmm_nophasing.shifted_emission import (
    current_clone_lengths,
    neutral_state,
    stacked_log_lambda,
)
from port.patch.hmrf.clone_assignment import _clone_shifts
from port.patch.hmrf.core_inference import pin_neutral
from port.pipeline import with_attributes

from tests.fixtures import (
    divergent_clone_instance,
    shifted_emission_call,
    shifted_replacement,
)


@pytest.mark.cnaster
@pytest.mark.patch
def test_off_it_is_upstreams_emission_bitwise(cnaster_config: None) -> None:
    """With the flag off the emission is upstream's, bitwise."""

    instance = divergent_clone_instance()

    their_rdr, their_baf = shifted_emission_call(upstream(), instance)
    our_rdr, our_baf = shifted_emission_call(shifted_replacement(instance), instance)

    np.testing.assert_array_equal(our_rdr, their_rdr)
    np.testing.assert_array_equal(our_baf, their_baf)


@pytest.mark.cnaster
@pytest.mark.patch
def test_off_is_the_default_and_a_missing_decode_still_delegates(
    cnaster_config: None,
) -> None:
    """A missing decode, exposure or clone lengths delegates to upstream rather than guessing."""

    instance = divergent_clone_instance()
    expected = shifted_emission_call(upstream(), instance)

    # no decode at all
    bare = shifted_replacement(instance, shifted=True)
    bare.state_posteriors = None
    np.testing.assert_array_equal(shifted_emission_call(bare, instance)[0], expected[0])

    # no exposures
    without_lambda = dict(instance, normal_log_lambda=None)
    np.testing.assert_array_equal(
        shifted_emission_call(
            shifted_replacement(instance, shifted=True), without_lambda
        )[0],
        expected[0],
    )

    # no clone lengths
    without_lengths = dict(instance, clone_lengths=None)
    np.testing.assert_array_equal(
        shifted_emission_call(
            shifted_replacement(instance, shifted=True), without_lengths
        )[0],
        expected[0],
    )


@pytest.mark.cnaster
@pytest.mark.patch
def test_on_it_applies_cnasters_own_shift(cnaster_config: None) -> None:
    """`exp(log_mu - log Z_c)` against `cnaster`'s `compute_logmu_shifts` and `_nb_logpmf_1d`, bitwise."""

    instance = divergent_clone_instance()
    per_clone = instance["per_clone"]

    shifts = compute_logmu_shifts(
        np.ascontiguousarray(instance["log_mu"][:, 0]),
        np.ascontiguousarray(instance["decode"]),
        np.ascontiguousarray(instance["normal_log_lambda"]),
        instance["clone_lengths"],
    )

    rdr, _ = shifted_emission_call(
        shifted_replacement(instance, shifted=True), instance
    )

    observed = np.asarray(instance["nbEncoder"].obs_count).reshape(-1)
    exposure = np.asarray(instance["nbEncoder"].total_count).reshape(-1)

    for clone in range(instance["n_clones"]):
        start = clone * per_clone
        stop = start + per_clone

        for state in range(instance["n_states"]):
            expected = np.zeros(per_clone)

            _nb_logpmf_1d(
                observed[start:stop],
                exposure[start:stop],
                float(np.exp(instance["log_mu"][state, 0] - shifts[start])),
                float(instance["alphas"][state, 0]),
                expected,
            )

            np.testing.assert_array_equal(
                rdr[state, start:stop],
                expected,
                err_msg=f"clone {clone}, state {state}",
            )


@pytest.mark.bug
def test_each_clone_takes_its_own_shift(cnaster_config: None) -> None:
    """Each clone scores under its own shift, not clone zero's: fails if indexed by clone rather than segment."""

    instance = divergent_clone_instance()
    per_clone = instance["per_clone"]

    shifts = compute_logmu_shifts(
        np.ascontiguousarray(instance["log_mu"][:, 0]),
        np.ascontiguousarray(instance["decode"]),
        np.ascontiguousarray(instance["normal_log_lambda"]),
        instance["clone_lengths"],
    )

    distinct = {round(float(shifts[c * per_clone]), 12) for c in range(3)}

    assert len(distinct) == 3, (
        f"the fixture must give the clones different shifts, got {distinct}"
    )

    rdr, _ = shifted_emission_call(
        shifted_replacement(instance, shifted=True), instance
    )

    # NB the same `(obs, total)` pair in several clones must score differently;
    #    equal scores mean one shift was used for all three.
    observed = np.asarray(instance["nbEncoder"].obs_count).reshape(-1)
    exposure = np.asarray(instance["nbEncoder"].total_count).reshape(-1)

    scored: dict[tuple[float, float], set[float]] = {}

    for segment in range(len(observed)):
        scored.setdefault((observed[segment], exposure[segment]), set()).add(
            round(float(rdr[0, segment]), 12)
        )

    shared = [values for values in scored.values() if len(values) > 1]

    assert shared, (
        "no (obs, total) pair scored differently across clones, so either the "
        "fixture shares none or every clone took the same shift"
    )


@pytest.mark.bug
def test_stale_clone_lengths_are_retiled_to_the_decoded_sequence() -> None:
    """Stale `clone_lengths` from `hmrf.py:564` are retiled to the decode, or refused (#293, #292)."""

    assert current_clone_lengths((300,) * 6, 900) == (300,) * 3
    assert current_clone_lengths((300,) * 3, 900) == (300,) * 3

    with pytest.raises(ValueError, match="do not tile"):
        current_clone_lengths((300, 200), 900)


@pytest.mark.analytic
def test_a_per_bin_lambda_is_repeated_over_the_clone_stack() -> None:
    """A per-bin `normal_lambda` (`hmrf.py:476`) is repeated over the clone stack; other lengths are refused (#293)."""

    profile = np.log(np.array([0.2, 0.3, 0.5]))

    np.testing.assert_array_equal(
        stacked_log_lambda(profile, (3, 3)), np.tile(profile, 2)
    )
    np.testing.assert_array_equal(
        stacked_log_lambda(np.tile(profile, 2), (3, 3)), np.tile(profile, 2)
    )

    with pytest.raises(ValueError, match="expected one per genome"):
        stacked_log_lambda(profile, (4, 4))


@pytest.mark.analytic
def test_the_neutral_state_is_the_balanced_one_with_the_lowest_mu() -> None:
    """The neutral state is balanced within 0.05 of 0.5, then lowest `mu` (#293)."""

    log_mu = np.log(np.array([2.0, 0.9, 0.5, 1.1]))
    p_binom = np.array([0.5, 0.52, 0.12, 0.47])

    assert neutral_state(log_mu, p_binom) == 1
    assert neutral_state(log_mu, 1.0 - p_binom) == 1


@pytest.mark.analytic
def test_the_pin_leaves_every_shifted_rate_as_it_was() -> None:
    """The pin `mu -> c mu` leaves every shifted rate unchanged to 1e-12; the pinned state reads 0 in log."""

    rng = np.random.default_rng(3)
    states = rng.integers(0, 3, 40).astype(np.int64)
    log_lambda = np.log(rng.random(40) / 40)
    lengths = np.array([20, 20], dtype=np.int64)

    result = {
        "new_log_mu": np.log(np.array([[1.7], [0.6], [3.1]])),
        "new_p_binom": np.array([[0.49], [0.51], [0.12]]),
    }
    before = result["new_log_mu"][:, 0]
    shifted_before = (
        before[None, :] - shifts(before, states, log_lambda, lengths)[:, None]
    )

    assert pin_neutral(result) == 1

    after = result["new_log_mu"][:, 0]

    assert after[1] == 0.0

    shifted_after = after[None, :] - shifts(after, states, log_lambda, lengths)[:, None]

    np.testing.assert_allclose(shifted_after, shifted_before, rtol=0.0, atol=1e-12)


@pytest.mark.patch
@pytest.mark.parametrize("offset", [0.0, -7024.0])
def test_the_dense_emission_applies_the_recorded_shift_to_the_mean(
    offset: float,
) -> None:
    """`base * exp(log_mu - shift)` against upstream on the shifted mean, to 1e-9 relative, finite at offset -7,024 (#292)."""

    shifted = with_attributes(hmm_nophasing, apply_logmu_shift=True)
    rng = np.random.default_rng(5)
    n_obs = 12
    X = np.stack(
        [rng.poisson(40, (n_obs, 1)), rng.integers(0, 10, (n_obs, 1))], axis=1
    ).astype(float)
    base = rng.uniform(20, 60, (n_obs, 1))
    total = np.full((n_obs, 1), 10.0)
    rates = np.log(np.array([1.0, 2.0]))
    alphas, p_binom, taus = (
        np.full((2, 1), 0.1),
        np.array([[0.5], [0.2]]),
        np.full((2, 1), 30.0),
    )
    shift = rng.normal(0.0, 0.3, n_obs)

    previous = hmm_nophasing._row_shift
    hmm_nophasing._row_shift = shift + offset

    try:
        ours = shifted.compute_emission_probability_nb_betabinom(
            X, base, (rates + offset)[:, None], alphas, total, p_binom, taus
        )

        for state in range(2):
            theirs = upstream.compute_emission_probability_nb_betabinom(
                X,
                base * np.exp(rates[state] - shift)[:, None],
                np.zeros((2, 1)),
                alphas,
                total,
                p_binom,
                taus,
            )

            assert np.all(np.isfinite(ours[0][state]))
            np.testing.assert_allclose(ours[0][state], theirs[0][state], rtol=1e-9)
            np.testing.assert_array_equal(ours[1][state], theirs[1][state])

        off = hmm_nophasing.compute_emission_probability_nb_betabinom(
            X, base, rates[:, None], alphas, total, p_binom, taus
        )
        plain = upstream.compute_emission_probability_nb_betabinom(
            X, base, rates[:, None], alphas, total, p_binom, taus
        )

        for mine, reference in zip(off, plain, strict=True):
            np.testing.assert_array_equal(mine, reference)
    finally:
        hmm_nophasing._row_shift = previous


@pytest.mark.analytic
def test_each_candidate_clone_is_scored_under_its_own_normalizer() -> None:
    """`_clone_shifts` against `log sum_g lambda_g mu_{s_c(g)}` written out, to 1e-12; `None` when unshifted."""

    rng = np.random.default_rng(9)
    base = rng.uniform(1.0, 5.0, (30, 7))
    decoded = rng.integers(0, 3, (30, 2)).astype(np.int64)
    res = {"new_log_mu": np.log(np.array([[1.0], [1.5], [3.0]]))}

    assert _clone_shifts(hmm_nophasing, res, decoded, base) is None

    ours = _clone_shifts(
        with_attributes(hmm_nophasing, apply_logmu_shift=True), res, decoded, base
    )

    weights = base.sum(axis=1) / base.sum()
    mu = np.array([1.0, 1.5, 3.0])
    expected = np.log([(weights * mu[decoded[:, c]]).sum() for c in range(2)])

    assert ours is not None
    np.testing.assert_allclose(ours, expected, rtol=0.0, atol=1e-12)


@pytest.mark.analytic
def test_the_pinned_state_is_the_normal_clones_dominant_one() -> None:
    """The pinned state is the normal clone's dominant balanced state, not the lowest `mu` (#299)."""

    log_mu = np.log(np.array([2.4, 1.0, 4.0]))
    p_binom = np.array([0.50, 0.50, 0.17])

    path = np.zeros((100, 2), dtype=np.int64)
    path[:5, :] = 1
    path[60:, 1] = 2

    assert neutral_state(log_mu, p_binom) == 1
    assert neutral_state(log_mu, p_binom, path) == 0
