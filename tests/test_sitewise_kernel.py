"""cnaster's sitewise rates as the varying kernel upstream now takes (#70).

`cnaster` never materializes a transition kernel. It rebuilds a `(2K, 2K)`
block per position inside its forward and backward passes, from two things:
the copy-state kernel `get_log_transmat(n_states, t)`, constant along the
chain, and the per-site phase-switch probability `get_sitewise_transmat`
derives from a genetic map.

Upstream's `forward_backward` and `baum_welch_family` take `(T - 1, K, K)`
since #656 and #660. This builds the array that carries `cnaster`'s model
across that signature, and referees it against `cnaster`'s own builder so the
bridge is pinned rather than assumed.

The referee is `update_combined_transmat` (`hmm_phased.py:44`) and the bar is
bitwise: the same two terms are added in the same order, so nothing may move.
"""

import numpy as np
import pytest

SELF_TRANSITION = 1.0 - 1e-4
"""The copy-state diagonal. `run_core_inference` ships `1 - 1e-6`; this is
looser so the off-diagonal is representable and a wrong block shows."""


def sitewise_kernel(
    log_transmat: np.ndarray,
    switch_prob: np.ndarray,
    *,
    penalize_phase_only_on_same_cnv: bool = False,
) -> np.ndarray:
    """`(T - 1, 2K, 2K)`, one block per transition, in upstream's layout.

    Parameters
    ----------
    log_transmat : np.ndarray
        `(K, K)`, the copy-state kernel, constant along the chain.
    switch_prob : np.ndarray
        Per-site phase-switch probability, `(T,)`. Only the first `T - 1`
        entries are transitions; `cnaster` indexes it the same way, `idx`
        running over the steps rather than the positions.

    Notes
    -----
    The doubling is the phase latent: state `k` and state `k + K` are the same
    copy state on opposite haplotypes, which is the `2K` representation
    `cnaster` already uses (`n_paired_states = 2 * n_states`) and the one #70
    settles on rather than adding a field.
    """
    from cnaster.hmm_phased import update_combined_transmat

    n_states = log_transmat.shape[0]
    steps = switch_prob.shape[0] - 1

    log_switch = np.log(switch_prob)
    log_self = np.log1p(-switch_prob)
    log_half = np.log(0.5)

    kernels = np.empty((steps, 2 * n_states, 2 * n_states))
    for step in range(steps):
        update_combined_transmat(
            kernels[step],
            n_states,
            log_transmat,
            log_self[step],
            log_switch[step],
            penalize_phase_only_on_same_cnv,
            log_half,
        )
    return kernels


@pytest.mark.snapshot
@pytest.mark.parametrize("n_states", [2, 5])
def test_every_block_is_cnasters_own(n_states: int) -> None:
    """Each slice is what `cnaster` builds at that site, bitwise.

    Built through `cnaster`'s own function rather than reimplemented, so what
    this pins is the **indexing**: that step `s` of the kernel carries site
    `s`'s switch probability and not `s + 1`'s, which is the error a
    reimplementation would make and a shape check would miss.
    """
    from cnaster.hmm_nophasing import get_log_transmat

    rng = np.random.default_rng(7)
    switch_prob = rng.uniform(0.01, 0.2, 40)
    log_transmat = get_log_transmat(n_states, SELF_TRANSITION)

    kernels = sitewise_kernel(log_transmat, switch_prob)

    assert kernels.shape == (39, 2 * n_states, 2 * n_states)

    from cnaster.hmm_phased import update_combined_transmat

    for step in (0, 17, 38):
        expected = np.empty((2 * n_states, 2 * n_states))
        update_combined_transmat(
            expected,
            n_states,
            log_transmat,
            float(np.log1p(-switch_prob[step])),
            float(np.log(switch_prob[step])),
            False,
            float(np.log(0.5)),
        )
        np.testing.assert_array_equal(kernels[step], expected)


@pytest.mark.analytic
@pytest.mark.parametrize("n_states", [2, 5])
def test_every_block_is_a_transition_kernel(n_states: int) -> None:
    """Each row exponentiates to one, at every site.

    The property upstream's recursion assumes and never checks: a row that did
    not normalize would give a likelihood that drifts with the chain's length
    rather than a wrong answer at one position, which is the failure hardest
    to attribute later.
    """
    from cnaster.hmm_nophasing import get_log_transmat

    rng = np.random.default_rng(11)
    switch_prob = rng.uniform(0.01, 0.45, 30)

    kernels = sitewise_kernel(get_log_transmat(n_states, SELF_TRANSITION), switch_prob)
    rows = np.exp(kernels).sum(axis=-1)

    np.testing.assert_allclose(rows, 1.0, rtol=0, atol=1e-12)


@pytest.mark.smoke
def test_a_constant_rate_gives_a_constant_kernel() -> None:
    """With one switch probability everywhere, every step is the same block.

    So the varying signature reduces to the constant one exactly, which is
    what makes a `(T - 1, K, K)` fit comparable with the `(K, K)` fit it
    replaces rather than merely similar.
    """
    from cnaster.hmm_nophasing import get_log_transmat

    switch_prob = np.full(25, 0.07)
    kernels = sitewise_kernel(get_log_transmat(4, SELF_TRANSITION), switch_prob)

    for step in range(1, kernels.shape[0]):
        np.testing.assert_array_equal(kernels[step], kernels[0])


@pytest.mark.smoke
def test_upstream_accepts_the_kernel_and_scores_with_it() -> None:
    """The array reaches upstream's recursion, and the rate changes the answer.

    `forward_log_likelihood_from_density` takes `(length - 1, K, K)` since
    #656. Accepting it is half the claim; the other half is that the sitewise
    rate is **read**, so the same emission under a different genetic map gives
    a different evidence. A signature that accepted the array and ignored it
    would pass the first and fail this.
    """
    import torch
    from cnaster.hmm_nophasing import get_log_transmat
    from sal.opt.hmm import forward_log_likelihood_from_density

    n_states, length = 3, 24
    rng = np.random.default_rng(5)
    log_transmat = get_log_transmat(n_states, SELF_TRANSITION)

    density = torch.as_tensor(rng.normal(-2.0, 1.0, (1, length, 2 * n_states)))
    log_initial = torch.log(torch.full((2 * n_states,), 1.0 / (2 * n_states)))

    quiet = sitewise_kernel(log_transmat, np.full(length, 1e-4))
    busy = sitewise_kernel(log_transmat, np.full(length, 0.4))

    evidence_quiet = float(
        forward_log_likelihood_from_density(
            density, log_initial, torch.as_tensor(quiet)
        )
    )
    evidence_busy = float(
        forward_log_likelihood_from_density(density, log_initial, torch.as_tensor(busy))
    )

    assert np.isfinite(evidence_quiet)
    assert np.isfinite(evidence_busy)
    assert abs(evidence_quiet - evidence_busy) > 1e-6, (
        f"the sitewise rate did not reach the recursion: {evidence_quiet:.6f} "
        f"against {evidence_busy:.6f}"
    )
