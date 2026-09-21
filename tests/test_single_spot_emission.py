"""The collapsed emission is upstream's, at the one spot the assert allows.

**#259 stage 2.** `optimize_params` opens with
`assert X.shape[-1] == 1`; the emission it then calls loops `for s in
range(n_spots)`, keeps a scratch buffer per `s`, and branches on whether to
concatenate its one-element list or stack it. `port.patch.hmm_single_spot`
spends that assert instead. These pin that spending it changes no value.

`patch` throughout: the claim is that the two bodies agree, not that either
computes the right emission. The referee is upstream's own method on the
same encoders and the same parameters, which is the only thing that can
settle a refactor of it.
"""

from __future__ import annotations

import numpy as np
import pytest
from cnaster.count_encoder import CountEncoder
from port.patch.hmm_nophasing import UPSTREAM
from port.patch.hmm_nophasing import hmm_nophasing as PATCHED

EXACT = 0.0
"""Bitwise. The same kernels on the same buffers; a tolerance would hide a
reordering that is not supposed to be there."""


def _encoders(
    n_obs: int, n_spots: int, seed: int
) -> tuple[CountEncoder, CountEncoder, np.ndarray, np.ndarray]:
    """A pair of encoders over a draw with real repeats, so decoding matters.

    The counts are small on purpose: `CountEncoder` compresses to unique
    `(obs, total)` pairs, and a draw with no repeats makes `decode_array` an
    identity that would hide an indexing error.
    """
    rng = np.random.default_rng(seed)

    nb_obs = rng.integers(0, 12, size=(n_obs, n_spots)).astype(np.float64)
    nb_total = rng.integers(1, 9, size=(n_obs, n_spots)).astype(np.float64)

    bb_total = rng.integers(1, 15, size=(n_obs, n_spots)).astype(np.float64)
    bb_obs = rng.integers(0, 1 + bb_total.astype(int)).astype(np.float64)

    return (
        CountEncoder(nb_obs, nb_total),
        CountEncoder(bb_obs, bb_total),
        nb_obs,
        bb_obs,
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
@pytest.mark.parametrize("clone_stack", [True, False], ids=["concatenate", "stack"])
@pytest.mark.parametrize("n_states", [1, 3, 5])
def test_it_is_upstreams_emission_bitwise(
    cnaster_config: None, clone_stack: bool, n_states: int
) -> None:
    """Both branches of the return, over three state counts.

    `clone_stack` is the branch the collapse rewrites as two identities --
    `np.concatenate([a], axis=1)` is `a` copied and `np.stack([a], axis=2)`
    is `a[:, :, None]` -- so both have to be checked or half the rewrite is
    unrefereed. `n_states=1` is the case where a `(n_states, n_obs)` result
    and an `(n_obs,)` one would be confusable.
    """
    nb, bb, _, _ = _encoders(n_obs=60, n_spots=1, seed=11)
    log_mu, p_binom, alphas, taus = _parameters(n_states, seed=3)

    theirs = UPSTREAM.compute_emission_probability_nb_betabinom_coded(
        UPSTREAM(), nb, bb, log_mu, alphas, p_binom, taus, clone_stack=clone_stack
    )
    ours = PATCHED.compute_emission_probability_nb_betabinom_coded(
        PATCHED(), nb, bb, log_mu, alphas, p_binom, taus, clone_stack=clone_stack
    )

    for theirs_part, ours_part, name in zip(theirs, ours, ("rdr", "baf"), strict=True):
        assert ours_part.shape == theirs_part.shape, name
        assert np.array_equal(ours_part, theirs_part), (
            f"{name}: max |difference| "
            f"{np.max(np.abs(ours_part - theirs_part)):.3e}, not bitwise"
        )


@pytest.mark.patch
def test_the_scratch_buffers_are_used_and_reused(cnaster_config: None) -> None:
    """The optimizer passes them per iteration; both bodies must write there.

    A collapse that quietly allocated instead would be bitwise correct and
    would cost an allocation per call on the hottest loop in the fit, which
    is the kind of regression a value comparison cannot see.
    """
    n_states = 4
    nb, bb, _, _ = _encoders(n_obs=45, n_spots=1, seed=5)
    log_mu, p_binom, alphas, taus = _parameters(n_states, seed=7)

    scratch_rdr = [np.zeros((n_states, len(nb.get_unique_obs(0))))]
    scratch_baf = [np.zeros((n_states, len(bb.get_unique_obs(0))))]

    rdr, _ = PATCHED.compute_emission_probability_nb_betabinom_coded(
        PATCHED(),
        nb,
        bb,
        log_mu,
        alphas,
        p_binom,
        taus,
        scratch_rdr=scratch_rdr,
        scratch_baf=scratch_baf,
    )

    assert np.any(scratch_rdr[0] != 0.0), "the rdr scratch was not written"
    assert np.any(scratch_baf[0] != 0.0), "the baf scratch was not written"

    # NB and the decode is a fresh array rather than a view of the scratch,
    #    which is what lets the optimizer reuse the buffer between calls.
    assert not np.shares_memory(rdr, scratch_rdr[0])


@pytest.mark.patch
def test_more_than_one_spot_is_refused_by_name(cnaster_config: None) -> None:
    """Upstream would loop; this body would read spot 0 and drop the rest.

    So the assert `optimize_params` holds two frames up is restated where the
    collapse is. The message quotes upstream's, because a reader hitting this
    needs to know the assert exists rather than that `port` invented a limit.
    """
    nb, bb, _, _ = _encoders(n_obs=30, n_spots=3, seed=2)
    log_mu, p_binom, alphas, taus = _parameters(3, seed=1)

    with pytest.raises(ValueError, match="one spot only"):
        PATCHED.compute_emission_probability_nb_betabinom_coded(
            PATCHED(), nb, bb, log_mu, alphas, p_binom, taus
        )


@pytest.mark.patch
def test_the_phased_class_keeps_upstreams_loop() -> None:
    """`hmm_phased` reaches this method too, and is not collapsed.

    It calls with `clone_stack=False` on a path nothing here has established
    is single-spot, so it must **not** inherit `SingleSpot`. Pinned on the
    MRO rather than on behaviour: the difference is which body runs, and a
    single-spot fixture would agree either way.
    """
    from port.patch.hmm_phased import hmm_phased
    from port.patch.hmm_single_spot import SingleSpot

    assert SingleSpot in PATCHED.__mro__
    assert SingleSpot not in hmm_phased.__mro__
