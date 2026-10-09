"""`cnaster`'s sitewise rates as upstream's `(T - 1, K, K)` kernel (#70, #656, #660).

Referee: `update_combined_transmat` (`hmm_phased.py:44`), bitwise.
"""

import numpy as np
import pytest
import torch
from cnaster.hmm_nophasing import get_log_transmat
from cnaster.hmm_phased import update_combined_transmat
from sal.opt.hmm import forward_log_likelihood_from_density

SELF_TRANSITION = 1.0 - 1e-4
"""The copy-state diagonal, looser than the shipped `1 - 1e-6` so a wrong block shows."""


def sitewise_kernel(
    log_transmat: np.ndarray,
    switch_prob: np.ndarray,
    *,
    penalize_phase_only_on_same_cnv: bool = False,
) -> np.ndarray:
    """`(T - 1, 2K, 2K)` from the `(K, K)` copy kernel and per-site `switch_prob`; step `s` reads site `s`."""

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
    """Each slice is `cnaster`'s block at that site, bitwise."""

    rng = np.random.default_rng(7)
    switch_prob = rng.uniform(0.01, 0.2, 40)
    log_transmat = get_log_transmat(n_states, SELF_TRANSITION)

    kernels = sitewise_kernel(log_transmat, switch_prob)

    assert kernels.shape == (39, 2 * n_states, 2 * n_states)

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
    """Each row exponentiates to one, at every site."""

    rng = np.random.default_rng(11)
    switch_prob = rng.uniform(0.01, 0.45, 30)

    kernels = sitewise_kernel(get_log_transmat(n_states, SELF_TRANSITION), switch_prob)
    rows = np.exp(kernels).sum(axis=-1)

    np.testing.assert_allclose(rows, 1.0, rtol=0, atol=1e-12)


@pytest.mark.analytic
def test_a_constant_rate_gives_a_constant_kernel() -> None:
    """A constant switch probability gives the same block at every step."""

    switch_prob = np.full(25, 0.07)
    kernels = sitewise_kernel(get_log_transmat(4, SELF_TRANSITION), switch_prob)

    for step in range(1, kernels.shape[0]):
        np.testing.assert_array_equal(kernels[step], kernels[0])


@pytest.mark.smoke
def test_upstream_accepts_the_kernel_and_scores_with_it() -> None:
    """Upstream's recursion accepts the kernel, and a different genetic map changes the evidence."""

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
