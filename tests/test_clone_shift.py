r"""The posterior-weighted normalizer, and the emission that folds it in.

**#259 stage 4.** `cnaster` computes a per-clone shift with a hard decode and
discards it. `port.patch.clone_shift.log_normalizers` is the same quantity
with the decode replaced by the state posterior the M step holds fixed, and
`port.patch.shifted_emission` applies it.

`patch` for the reduction and for the off path -- those are claims about
agreement with `cnaster`. The on path changes every fitted RDR parameter and
**is not refereed here**: what it computes is judged against planted truth in
#259 stage 5, and these only pin that it is the formula it says it is.
"""

from __future__ import annotations

import numpy as np
import pytest
from cnaster.count_encoder import CountEncoder
from cnaster.hmm_nophasing import compute_logmu_shifts
from port.patch.clone_shift import (
    clone_state_weights,
    log_normalizers,
    log_normalizers_from_weights,
)
from port.patch.hmm_nophasing import hmm_nophasing as PATCHED


def _one_hot(copy_states: np.ndarray, n_states: int) -> np.ndarray:
    gamma = np.zeros((n_states, copy_states.size))
    gamma[copy_states, np.arange(copy_states.size)] = 1.0
    return gamma


@pytest.mark.patch
@pytest.mark.parametrize(
    "lengths",
    [
        pytest.param([40, 25, 55], id="unequal-no-view-can-exist"),
        pytest.param([10], id="single-clone"),
        pytest.param([1, 99], id="one-segment-clone"),
        pytest.param([7, 3, 10], id="three-short-clones"),
    ],
)
def test_a_one_hot_posterior_is_cnasters_hard_decode(lengths: list[int]) -> None:
    r"""The reduction that makes this a rewrite rather than a new formula.

    :math:`\sum_k \gamma_{g,k} \exp(\theta_k)` is :math:`\exp(\theta_{k(g)})`
    when :math:`\gamma` is one-hot at the decode, so the two normalizers are
    the same number. **Bitwise**, not to a tolerance: the terms are identical
    and only the summation order could differ.
    """
    rng = np.random.default_rng(17)
    n_segments, n_states = sum(lengths), 4

    log_mu = rng.normal(size=n_states)
    copy_states = rng.integers(0, n_states, size=n_segments)
    normal_log_lambda = rng.normal(size=n_segments)

    theirs = compute_logmu_shifts(log_mu, copy_states, normal_log_lambda, lengths)
    ours = log_normalizers(
        log_mu, _one_hot(copy_states, n_states), normal_log_lambda, lengths
    )

    assert ours.shape == theirs.shape == (len(lengths),)
    assert np.array_equal(ours, theirs), (
        f"max |difference| {np.max(np.abs(ours - theirs)):.3e}, not bitwise"
    )


FACTORED = 1e-12
"""The two-pass form against the one-pass one; realized 8.88e-16.

Not bitwise, and the reason is stated rather than tolerated: the fast path
reduces over segments and then over states, where the reference sums every
`(segment, state)` term at once. The same terms in a different order.
"""


@pytest.mark.patch
@pytest.mark.parametrize(
    "lengths",
    [
        pytest.param([40, 25, 55], id="unequal"),
        pytest.param([1, 99], id="one-segment-clone"),
        pytest.param([30, 30], id="equal"),
    ],
)
def test_the_factored_fast_path_agrees_with_the_single_pass(
    lengths: list[int],
) -> None:
    r"""`Z_c = sum_k W_ck exp(theta_k)` is the same sum, regrouped.

    The emission takes this path because :math:`W` does not depend on
    :math:`\theta` and can be cached across an M step, which is what makes
    the shift cost `(n_clones, n_states)` per call instead of
    `(n_states, n_segments)`. It has to be the same number.
    """
    rng = np.random.default_rng(23)
    n_segments, n_states = sum(lengths), 5

    gamma = rng.dirichlet(np.ones(n_states), size=n_segments).T
    log_mu = rng.normal(size=n_states)
    normal_log_lambda = rng.normal(size=n_segments)

    reference = log_normalizers(log_mu, gamma, normal_log_lambda, lengths)
    factored = log_normalizers_from_weights(
        clone_state_weights(gamma, normal_log_lambda, lengths), log_mu
    )

    assert factored.shape == reference.shape
    assert np.allclose(factored, reference, rtol=0.0, atol=FACTORED), (
        f"max |difference| {np.max(np.abs(factored - reference)):.3e}"
    )


@pytest.mark.patch
def test_the_weights_do_not_depend_on_the_rates() -> None:
    """The property the cache rests on, asserted rather than assumed.

    If `W` moved with `theta`, caching it across an M step would freeze a
    stale normalizer into every iteration after the first.
    """
    rng = np.random.default_rng(29)
    n_states, n_segments = 4, 80

    gamma = rng.dirichlet(np.ones(n_states), size=n_segments).T
    normal_log_lambda = rng.normal(size=n_segments)

    weights = clone_state_weights(gamma, normal_log_lambda, [40, 40])

    for rates in (np.zeros(n_states), rng.normal(size=n_states) * 10.0):
        assert np.array_equal(
            weights, clone_state_weights(gamma, normal_log_lambda, [40, 40])
        )
        assert log_normalizers_from_weights(weights, rates).shape == (2,)


@pytest.mark.patch
def test_a_soft_posterior_is_a_different_number() -> None:
    """Otherwise the reduction above would be pinning nothing.

    A posterior that never leaves a vertex of the simplex would make the two
    forms interchangeable, and the stage would be a rename.
    """
    rng = np.random.default_rng(3)
    n_states, n_segments = 4, 60

    gamma = rng.dirichlet(np.ones(n_states), size=n_segments).T
    log_mu = rng.normal(size=n_states)
    normal_log_lambda = rng.normal(size=n_segments)

    soft = log_normalizers(log_mu, gamma, normal_log_lambda, [n_segments])
    hard = log_normalizers(
        log_mu,
        _one_hot(gamma.argmax(axis=0), n_states),
        normal_log_lambda,
        [n_segments],
    )

    assert soft[0] != hard[0]


@pytest.mark.patch
def test_a_clone_of_minus_infinities_stays_minus_infinity() -> None:
    """The branch `cnaster`'s loop takes, and the one a rewrite gets wrong.

    A clone whose every term is ruled out returns `-inf` rather than a `nan`.
    Here the posterior rules out the only state the clone's segments carry.
    """
    n_states, n_segments = 2, 4

    gamma = np.zeros((n_states, n_segments))
    gamma[1, :] = 1.0

    shifts = log_normalizers(
        np.array([0.0, -np.inf]), gamma, np.zeros(n_segments), [2, 2]
    )

    assert np.all(np.isneginf(shifts))
    assert not np.any(np.isnan(shifts)), "a nan here would be a silent wrong answer"


@pytest.mark.patch
def test_mismatched_shapes_are_refused() -> None:
    """A caller error, named rather than broadcast into a wrong answer."""
    gamma = np.full((3, 20), 0.25)

    with pytest.raises(ValueError, match="n_states, n_segments"):
        log_normalizers(np.zeros(4), gamma, np.zeros(20), [20])

    with pytest.raises(ValueError, match="sum to"):
        log_normalizers(np.zeros(3), gamma, np.zeros(20), [10, 5])


def _encoders(n_segments: int, seed: int) -> tuple[CountEncoder, CountEncoder]:
    rng = np.random.default_rng(seed)

    nb_total = rng.integers(1, 9, size=(n_segments, 1)).astype(np.float64)
    bb_total = rng.integers(1, 15, size=(n_segments, 1)).astype(np.float64)

    return (
        CountEncoder(
            rng.integers(0, 12, size=(n_segments, 1)).astype(np.float64), nb_total
        ),
        CountEncoder(
            rng.integers(0, 1 + bb_total.astype(int)).astype(np.float64), bb_total
        ),
    )


def _parameters(n_states: int, seed: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    return (
        rng.normal(scale=0.3, size=(n_states, 1)),
        rng.uniform(0.05, 0.45, size=(n_states, 1)),
        rng.uniform(0.2, 0.8, size=(n_states, 1)),
        rng.uniform(500.0, 1500.0, size=(n_states, 1)),
    )


@pytest.mark.patch
@pytest.mark.parametrize(
    ("flag", "lambdas", "lengths", "posteriors", "why"),
    [
        pytest.param(False, True, True, True, "the flag is off", id="flag-off"),
        pytest.param(True, False, True, True, "no exposures", id="no-lambda"),
        pytest.param(True, True, False, True, "no clone lengths", id="no-lengths"),
        pytest.param(True, True, True, False, "no posterior yet", id="no-posterior"),
    ],
)
def test_the_off_path_is_the_refereed_one_bitwise(
    cnaster_config: None,
    flag: bool,
    lambdas: bool,
    lengths: bool,
    posteriors: bool,
    why: str,
) -> None:
    """Every way of not shifting returns exactly what #262 pinned.

    A shift needs the flag, the exposures, the clone boundaries and a
    posterior. Missing any one, the call has to fall through to
    `port.patch.coded_emission` rather than substitute a default -- a shift
    computed from a default is a number nobody asked for.
    """
    n_states, n_segments = 4, 120
    nb, bb = _encoders(n_segments, seed=11)
    log_mu, p_binom, alphas, taus = _parameters(n_states, seed=3)

    baseline = PATCHED()
    shifted = PATCHED()
    shifted.apply_logmu_shift = flag

    if posteriors:
        shifted.state_posteriors = np.full((n_states, n_segments), 1.0 / n_states)

    theirs = baseline.compute_emission_probability_nb_betabinom_coded(
        nb, bb, log_mu, alphas, p_binom, taus
    )
    ours = shifted.compute_emission_probability_nb_betabinom_coded(
        nb,
        bb,
        log_mu,
        alphas,
        p_binom,
        taus,
        normal_log_lambda=np.zeros(n_segments) if lambdas else None,
        num_segments_clones=[60, 60] if lengths else None,
    )

    for theirs_part, ours_part in zip(theirs, ours, strict=True):
        assert np.array_equal(ours_part, theirs_part), why


@pytest.mark.patch
def test_the_on_path_shifts_the_rdr_and_leaves_the_baf_alone(
    cnaster_config: None,
) -> None:
    """The shift multiplies `mu`, which enters the NB channel and nothing else.

    So the BAF must come back identical -- that is what lets the on path keep
    the whole-genome encoder for it -- and the RDR must not.
    """
    n_states, n_segments = 4, 120
    nb, bb = _encoders(n_segments, seed=11)
    log_mu, p_binom, alphas, taus = _parameters(n_states, seed=3)

    rng = np.random.default_rng(5)

    baseline = PATCHED()
    shifted = PATCHED()
    shifted.apply_logmu_shift = True
    shifted.state_posteriors = rng.dirichlet(np.ones(n_states), size=n_segments).T

    theirs_rdr, theirs_baf = baseline.compute_emission_probability_nb_betabinom_coded(
        nb, bb, log_mu, alphas, p_binom, taus
    )
    ours_rdr, ours_baf = shifted.compute_emission_probability_nb_betabinom_coded(
        nb,
        bb,
        log_mu,
        alphas,
        p_binom,
        taus,
        normal_log_lambda=rng.normal(size=n_segments),
        num_segments_clones=[60, 60],
    )

    assert np.array_equal(ours_baf, theirs_baf), "the shift reached the BAF channel"
    assert ours_rdr.shape == theirs_rdr.shape
    assert not np.allclose(ours_rdr, theirs_rdr), (
        "the shift changed no RDR value; the flag is doing nothing"
    )


@pytest.mark.patch
def test_a_zero_shift_is_the_unshifted_emission_bitwise(cnaster_config: None) -> None:
    """The on path at `log Z_c = 0` must be the off path, to the bit.

    This is what says the per-clone re-encoding is a regrouping and not a
    different calculation: `exp(theta - 0)` is `exp(theta)`, so only the
    encoder split differs, and the split must not move a value.
    """
    n_states, n_segments = 4, 120
    nb, bb = _encoders(n_segments, seed=11)
    log_mu, p_binom, alphas, taus = _parameters(n_states, seed=3)

    baseline = PATCHED()
    shifted = PATCHED()
    shifted.apply_logmu_shift = True

    # NB one state carries the whole posterior, `log_mu` is -inf there and
    #    every exposure is zero, so `log Z_c` is exactly 0.0 for both clones.
    gamma = np.zeros((n_states, n_segments))
    gamma[0, :] = 1.0
    shifted.state_posteriors = gamma

    rates = log_mu.copy()
    rates[0, 0] = 0.0

    zero = shifted.compute_emission_probability_nb_betabinom_coded(
        nb,
        bb,
        rates,
        alphas,
        p_binom,
        taus,
        normal_log_lambda=np.full(n_segments, -np.log(n_segments / 2)),
        num_segments_clones=[60, 60],
    )
    plain = baseline.compute_emission_probability_nb_betabinom_coded(
        nb, bb, rates, alphas, p_binom, taus
    )

    for zero_part, plain_part in zip(zero, plain, strict=True):
        assert np.array_equal(zero_part, plain_part), (
            f"max |difference| {np.max(np.abs(zero_part - plain_part)):.3e}"
        )


@pytest.mark.patch
def test_the_per_clone_encoders_are_built_once(cnaster_config: None) -> None:
    """`optimize_params` builds its encoders once per fit; so must these.

    Rebuilding them inside `cost_fn` would put an `np.unique` over the whole
    genome on the optimizer's hot path, which is the cost this stage is
    affordable without.
    """
    n_states, n_segments = 3, 90
    nb, bb = _encoders(n_segments, seed=2)
    log_mu, p_binom, alphas, taus = _parameters(n_states, seed=1)

    shifted = PATCHED()
    shifted.apply_logmu_shift = True
    shifted.state_posteriors = np.full((n_states, n_segments), 1.0 / n_states)

    def call() -> None:
        shifted.compute_emission_probability_nb_betabinom_coded(
            nb,
            bb,
            log_mu,
            alphas,
            p_binom,
            taus,
            normal_log_lambda=np.zeros(n_segments),
            num_segments_clones=[30, 60],
        )

    call()
    first = [
        id(encoder) for encoder in shifted._clone_encoder_cache[(id(nb), (30, 60))][1]
    ]

    call()
    again = [
        id(encoder) for encoder in shifted._clone_encoder_cache[(id(nb), (30, 60))][1]
    ]

    assert first == again, "the per-clone encoders were rebuilt"
    assert len(first) == 2


@pytest.mark.infra
def test_the_block_sets_the_flag_and_puts_it_back() -> None:
    """A class attribute left set would shift every later run in the process.

    `port`'s own tests put a shifted emission beside an unshifted one, so a
    leak here would make the pair agree for the wrong reason.
    """
    from port.patch.shifted_emission import logmu_shift

    assert PATCHED.apply_logmu_shift is False, "the flag must default to off"

    with logmu_shift():
        assert PATCHED.apply_logmu_shift is True

    assert PATCHED.apply_logmu_shift is False


@pytest.mark.infra
def test_the_flag_is_refused_rather_than_ignored_under_no_patch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--logmu-shift --no-patch` cannot do what it was asked, so it stops.

    The shift lives on the patched class; with nothing rebound there is no
    class to set it on. Running anyway would hand back an undebiased answer
    to someone who asked for a debiased one, which is the failure mode
    `--track` refuses for the same reason (#251).
    """
    from port.scripts.run_cnaster import main

    assert main(["--logmu-shift", "--no-patch", "--list"]) == 0, (
        "--list exits before the check, and should keep doing so"
    )

    assert main(["--logmu-shift", "--no-patch", "missing.yaml"]) == 2

    assert "--logmu-shift needs the patched HMM" in capsys.readouterr().err
