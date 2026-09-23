"""The emission with the library normalizer folded in (#276).

`cnaster` computes the shift at `hmm_nophasing.py:133` and discards it: the
only call site is commented out at `:279` under a `# TODO fold in
logmu_shifts`. `port.patch.hmm_nophasing.hmm_nophasing` applies it, behind a
flag that is off by default.

Three claims, and they are different claims:

*Off, it is upstream.* Bitwise, through the same encoder, so the default path
is the one upstream refereed rather than a re-derivation that happens to
agree.

*On, it is upstream's own shift.* `compute_logmu_shifts` is called rather
than reimplemented, so what is checked is that the quantity reaches the
kernel -- `exp(log_mu - log Z_c)` against the rate scored by hand.

*On, each clone gets its own.* The one that matters, because the failure is
silent: `compute_logmu_shifts` returns one value per **segment**, and reading
it per clone hands every clone clone zero's shift on any instance whose first
clone is longer than the clone count.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


def _instance(
    n_states: int = 3, n_clones: int = 3, per_clone: int = 8, seed: int = 41
) -> dict[str, Any]:
    """A clone-stacked instance whose clones decode differently.

    The clones must decode to different states for the shift to differ
    between them -- a fixture where every clone takes the same path would
    pass the per-clone test while indexing by clone, which is the defect
    being pinned.

    `per_clone` exceeds `n_clones`, deliberately: that is the regime where
    `shifts[clone]` reads inside clone zero's block rather than out of
    bounds, so a wrong index is silent rather than an `IndexError`.
    """
    from cnaster.count_encoder import CountEncoder

    generator = np.random.default_rng(seed)
    n_segments = n_clones * per_clone

    exposure = generator.integers(20, 60, n_segments).astype(np.float64)
    trials = generator.integers(10, 40, n_segments).astype(np.float64)

    observed = generator.poisson(exposure).astype(np.float64)
    successes = generator.binomial(trials.astype(int), 0.4).astype(np.float64)

    # NB one state per clone, so the three shifts are distinct by
    #    construction rather than by luck of the draw.
    decode = np.repeat(np.arange(n_clones) % n_states, per_clone).astype(np.int64)

    return {
        "nbEncoder": CountEncoder(observed.reshape(-1, 1), exposure.reshape(-1, 1)),
        "bbEncoder": CountEncoder(successes.reshape(-1, 1), trials.reshape(-1, 1)),
        "log_mu": generator.normal(0.0, 0.3, size=(n_states, 1)),
        "alphas": np.full((n_states, 1), 0.2),
        "p_binom": generator.uniform(0.2, 0.8, size=(n_states, 1)),
        "taus": np.full((n_states, 1), 25.0),
        "normal_log_lambda": generator.normal(0.0, 0.1, size=n_segments),
        "clone_lengths": np.full(n_clones, per_clone, dtype=np.int64),
        "decode": decode,
        "n_states": n_states,
        "n_clones": n_clones,
        "per_clone": per_clone,
    }


def _call(model: Any, instance: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    scored: tuple[np.ndarray, np.ndarray]
    scored = model.compute_emission_probability_nb_betabinom_coded(
        instance["nbEncoder"],
        instance["bbEncoder"],
        instance["log_mu"],
        instance["alphas"],
        instance["p_binom"],
        instance["taus"],
        normal_log_lambda=instance["normal_log_lambda"],
        clone_lengths=instance["clone_lengths"],
    )
    return scored


def _replacement(instance: dict[str, Any]) -> Any:
    """The drop-in, carrying the decode the shift is taken at."""
    from port.patch.hmm_nophasing import hmm_nophasing

    model = hmm_nophasing()
    model.state_posteriors = np.eye(instance["n_states"])[instance["decode"]].T

    return model


@pytest.mark.cnaster
@pytest.mark.patch
def test_off_it_is_upstreams_emission_bitwise(cnaster_config: None) -> None:
    """The default path is upstream's, not a re-derivation that agrees.

    Bitwise: the flag off means the call reaches `super()`, so the only way
    this fails is if the override computed something on the way past.
    """
    from cnaster.hmm_nophasing import hmm_nophasing as upstream

    instance = _instance()

    their_rdr, their_baf = _call(upstream(), instance)
    our_rdr, our_baf = _call(_replacement(instance), instance)

    np.testing.assert_array_equal(our_rdr, their_rdr)
    np.testing.assert_array_equal(our_baf, their_baf)


@pytest.mark.cnaster
@pytest.mark.patch
def test_off_is_the_default_and_a_missing_decode_still_delegates(
    cnaster_config: None,
) -> None:
    """Three ways to be unable to shift, each handed on rather than guessed.

    Upstream warns rather than guessing when it cannot compute the shift;
    this carries that, and the test is here because a patch that defaulted
    any of the three would produce a number nobody asked for and it would
    look like a working run.
    """
    from cnaster.hmm_nophasing import hmm_nophasing as upstream
    from port.patch.hmm_nophasing import logmu_shift

    instance = _instance()
    expected = _call(upstream(), instance)

    with logmu_shift():
        # no decode at all
        bare = _replacement(instance)
        bare.state_posteriors = None
        np.testing.assert_array_equal(_call(bare, instance)[0], expected[0])

        # no exposures
        without_lambda = dict(instance, normal_log_lambda=None)
        np.testing.assert_array_equal(
            _call(_replacement(instance), without_lambda)[0], expected[0]
        )

        # no clone lengths
        without_lengths = dict(instance, clone_lengths=None)
        np.testing.assert_array_equal(
            _call(_replacement(instance), without_lengths)[0], expected[0]
        )


@pytest.mark.cnaster
@pytest.mark.patch
def test_on_it_applies_cnasters_own_shift(cnaster_config: None) -> None:
    """`exp(log_mu - log Z_c)`, against the rate scored by hand.

    The referee is `cnaster`'s `compute_logmu_shifts` and `_nb_logpmf_1d`,
    both called directly here, so what is checked is that the quantity
    reaches the kernel rather than that two implementations of it agree.
    Bitwise, because nothing is reassociated between the two.
    """
    from cnaster.hmm_nophasing import _nb_logpmf_1d, compute_logmu_shifts
    from port.patch.hmm_nophasing import logmu_shift

    instance = _instance()
    per_clone = instance["per_clone"]

    shifts = compute_logmu_shifts(
        np.ascontiguousarray(instance["log_mu"][:, 0]),
        np.ascontiguousarray(instance["decode"]),
        np.ascontiguousarray(instance["normal_log_lambda"]),
        instance["clone_lengths"],
    )

    with logmu_shift():
        rdr, _ = _call(_replacement(instance), instance)

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
    """**Written to fail if the shift is indexed by clone rather than segment.**

    `compute_logmu_shifts` returns one value per segment, constant within a
    clone. Indexing it as `shifts[clone]` reads indices 0, 1, 2 -- all inside
    clone zero's block whenever the first clone is longer than the clone
    count -- and hands every clone clone zero's shift. No exception, no
    warning, and the fit reports debiased rates that were all debiased by the
    same wrong constant.

    The fixture's clones decode to different states, so the three shifts are
    distinct; a patch with the wrong index makes the three clones' rates
    equal where they should differ.
    """
    from cnaster.hmm_nophasing import compute_logmu_shifts
    from port.patch.hmm_nophasing import logmu_shift

    instance = _instance()
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

    with logmu_shift():
        rdr, _ = _call(_replacement(instance), instance)

    # NB the same `(obs, total)` pair appears in more than one clone, which is
    #    the whole reason the encoder has to be split; those entries must now
    #    score differently, and equal scores are the signature of one shift
    #    having been used for all three.
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
    """`hmrf.py:564` sets `clone_lengths` once, before clones merge (#293).

    On #292's genome `cnaster` passes six clones of 300 while the fit is over
    three: 1,800 against 900. The shift re-derives the current count from
    the decode, and refuses lengths that do not tile it. Written to fail if
    `cnaster` starts passing current lengths, when `_current` has nothing
    left to repair.
    """
    from port.patch.hmm_nophasing.shifted_emission import _current

    assert _current((300,) * 6, 900) == (300,) * 3
    assert _current((300,) * 3, 900) == (300,) * 3

    with pytest.raises(ValueError, match="do not tile"):
        _current((300, 200), 900)


@pytest.mark.analytic
def test_a_per_bin_lambda_is_repeated_over_the_clone_stack() -> None:
    """`hmrf.py:476` builds `normal_lambda` per genome bin (#293).

    The reduction walks the stacked sequence without bounds checks, so the
    per-bin profile is repeated clone after clone, which is exact because
    every clone shares the one normal profile. A per-segment one passes
    through, and any other length is refused.
    """
    from port.patch.hmm_nophasing.shifted_emission import _stacked

    profile = np.log(np.array([0.2, 0.3, 0.5]))

    np.testing.assert_array_equal(_stacked(profile, (3, 3)), np.tile(profile, 2))
    np.testing.assert_array_equal(
        _stacked(np.tile(profile, 2), (3, 3)), np.tile(profile, 2)
    )

    with pytest.raises(ValueError, match="expected one per genome"):
        _stacked(profile, (4, 4))


@pytest.mark.analytic
def test_the_neutral_state_is_the_balanced_one_with_the_lowest_mu() -> None:
    """Balanced within 0.05 of 0.5, in either allele's convention, then lowest.

    A gain that is balanced is not neutral when a lower balanced state
    exists, and an unbalanced state at a low `mu` is never chosen while a
    balanced one is available (#293).
    """
    from port.patch.hmm_nophasing.shifted_emission import neutral_state

    log_mu = np.log(np.array([2.0, 0.9, 0.5, 1.1]))
    p_binom = np.array([0.5, 0.52, 0.12, 0.47])

    assert neutral_state(log_mu, p_binom) == 1
    assert neutral_state(log_mu, 1.0 - p_binom) == 1


@pytest.mark.analytic
def test_the_pin_leaves_every_shifted_rate_as_it_was() -> None:
    """`mu -> c mu` changes no shifted rate, so the pin changes no emission.

    `log mu - log sum lambda mu` against the same after the pin, to 1e-12,
    and the pinned state reads exactly 0 in log.
    """
    from port.patch.hmm_nophasing.logmu_shift import shifts
    from port.patch.hmrf.core_inference import pin_neutral

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
    """`base * exp(log_mu - shift)`, finite at any common offset of the two.

    Against upstream called on the shifted mean directly -- exposure
    `base * exp(log_mu_k - shift_g)` per state, with `log_mu = 0` -- to
    1e-9 relative. The offset is the gauge the fit drifts along: at -7,024,
    which #292's realization 3 reached, forming `exp(-shift)` and
    `exp(log_mu)` separately is `inf * 0`. Unshifted when the flag is off.
    """
    from cnaster.hmm_nophasing import hmm_nophasing as upstream
    from port.patch.hmm_nophasing import hmm_nophasing, logmu_shift

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
        with logmu_shift():
            ours = hmm_nophasing.compute_emission_probability_nb_betabinom(
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
    """`_clone_shifts` is `log sum_g lambda_g mu_{s_c(g)}`, per clone.

    Against the sum written out, to 1e-12, with `lambda` the baseline summed
    over spots and normalized as `hmrf.py:476` builds it; `None` when the fit
    was not shifted, so the unshifted path is the fused field as before.
    """
    from port.patch.hmm_nophasing import hmm_nophasing, logmu_shift
    from port.patch.hmrf.clone_assignment import _clone_shifts

    rng = np.random.default_rng(9)
    base = rng.uniform(1.0, 5.0, (30, 7))
    decoded = rng.integers(0, 3, (30, 2)).astype(np.int64)
    res = {"new_log_mu": np.log(np.array([[1.0], [1.5], [3.0]]))}

    assert _clone_shifts(hmm_nophasing, res, decoded, base) is None

    with logmu_shift():
        ours = _clone_shifts(hmm_nophasing, res, decoded, base)

    weights = base.sum(axis=1) / base.sum()
    mu = np.array([1.0, 1.5, 3.0])
    expected = np.log([(weights * mu[decoded[:, c]]).sum() for c in range(2)])

    assert ours is not None
    np.testing.assert_allclose(ours, expected, rtol=0.0, atol=1e-12)
