"""`lattice_decode` under #566's dispersion models: per-state, rescaled, two-component.

The decode fits the HMM's dispersion model (`copy_likelihood.DISPERSION_MODELS`).
Two clones share four states' `(A, B)`, drawn at depth from the model under
test, so the likelihood's argmax is the planted dispersion. Referees:

- `end2end`: each model recovers what it planted -- per-state values that
  differ 25x, a per-spot `alpha` behind clones whose pseudobulks differ 8x,
  and a clone-shared part beside it;
- `analytic`: the rescaled model at unit factors is the shared one; the
  per-state values are inside their bounds, and a large `prior_rows` shrinks
  every state onto the pooled value.
"""

from __future__ import annotations

import numpy as np
import pytest
from port.extensions.copy_likelihood import (
    CopyFit,
    Pseudobulk,
    candidates,
    lattice_decode,
)

PAIRS = np.array([[1, 1], [2, 1], [3, 1], [2, 2]])
N_OBS = 600
DEPTH = 2.0e3
TAU = 2.0e2


def _planted(
    alphas: np.ndarray,
    nb_factors: tuple[float, float] = (1.0, 1.0),
    shared_alpha: float = 0.0,
    seed: int = 0,
) -> tuple[list[np.ndarray], list[Pseudobulk]]:
    """Two clones at shift 0 over `PAIRS`, state `k`'s rows at `shared_alpha + alphas[k] f_c`."""
    from port.extensions.copy_likelihood import _parameters

    rng = np.random.default_rng(seed)
    runs = np.repeat(np.arange(12) % 4, N_OBS // 12)
    paths = [runs, np.roll(runs, N_OBS // 24)]
    bulks = []

    for path, factor in zip(paths, nb_factors, strict=True):
        log_mu, share_of = _parameters(PAIRS, 1.0)
        mean = DEPTH * np.exp(log_mu[path])
        alpha = shared_alpha + alphas[path] * factor
        counts = rng.negative_binomial(1.0 / alpha, 1.0 / (1.0 + alpha * mean))
        share = rng.beta(share_of[path] * TAU, (1.0 - share_of[path]) * TAU)
        bulks.append(
            Pseudobulk(
                counts_nb=counts.astype(np.float64),
                base_nb_mean=np.full(N_OBS, DEPTH),
                counts_bb=rng.binomial(int(DEPTH), share).astype(np.float64),
                total_bb_RD=np.full(N_OBS, DEPTH),
                normal_log_lambda=np.full(N_OBS, -np.log(N_OBS)),
                dispersion=float(np.mean(alphas)),
                taus=TAU,
                nb_factor=factor,
            )
        )

    return paths, bulks


def _decode(
    paths: list[np.ndarray], bulks: list[Pseudobulk], model: str, **options: float
) -> CopyFit:
    return lattice_decode(
        [(z, b, 0.0) for z, b in zip(paths, bulks, strict=True)],
        normal_clone=0,
        max_total_copy=6,
        lengths=np.array([N_OBS]),
        fit_purity=False,
        fit_shifts=False,
        model=model,  # type: ignore[arg-type]
        **options,  # type: ignore[arg-type]
    )


def _lattice_index() -> np.ndarray:
    """Each of `PAIRS`' rows in `candidates(6)`."""
    states = candidates(6)
    return np.array(
        [int(np.flatnonzero((states == pair).all(axis=1))[0]) for pair in PAIRS]
    )


@pytest.mark.end2end
def test_per_state_dispersions_recover_values_25x_apart() -> None:
    """Planted 0.002, 0.002, 0.05, 0.05: each within 40 per cent, every pair recovered.

    Realized: 0.0019, 0.0023, 0.0485, 0.0441. The shared model fits one
    value between them, 0.0244.
    """
    planted = np.array([0.002, 0.002, 0.05, 0.05])
    paths, bulks = _planted(planted)
    fitted = _decode(paths, bulks, "per-state")
    alphas = np.asarray(fitted.dispersion)[_lattice_index()]

    np.testing.assert_allclose(alphas, planted, rtol=0.4)
    for pairs, path in zip(fitted.pairs, paths, strict=True):
        np.testing.assert_array_equal(pairs, PAIRS[path])


@pytest.mark.analytic
def test_per_state_dispersions_hold_their_bounds_and_shrink_to_the_pool() -> None:
    """Every state inside `[alpha_min, ...]`, `[..., tau_max]`; `prior_rows` 1e9 puts every state at one value."""
    planted = np.array([0.002, 0.002, 0.05, 0.05])
    paths, bulks = _planted(planted)
    bounded = _decode(paths, bulks, "per-state", alpha_min=0.01, tau_max=50.0)
    pooled = _decode(paths, bulks, "per-state", prior_rows=1.0e9)

    assert np.min(bounded.dispersion) >= 0.01 * (1.0 - 1e-12)
    assert np.max(bounded.taus) <= 50.0 * (1.0 + 1e-12)
    np.testing.assert_allclose(
        pooled.dispersion,
        np.full_like(pooled.dispersion, pooled.dispersion[0]),
        rtol=1e-6,
    )


@pytest.mark.end2end
def test_the_rescaled_decode_recovers_the_per_spot_alpha() -> None:
    """Clones of `S_eff` 8 and 64 at per-spot `alpha` 0.3: recovered within 25 per cent.

    Their pseudobulks score at 0.0375 and 0.0047, which one shared value
    cannot both fit. Realized: 0.285, and a log-likelihood 243 nats above the
    shared fit's 0.0194.
    """
    planted = np.full(4, 0.3)
    paths, bulks = _planted(planted, nb_factors=(1.0 / 8.0, 1.0 / 64.0))
    rescaled = _decode(paths, bulks, "rescale")
    shared = _decode(paths, bulks, "shared")

    assert abs(float(rescaled.dispersion) / 0.3 - 1.0) < 0.25
    assert rescaled.log_likelihood > shared.log_likelihood


@pytest.mark.analytic
def test_at_unit_factors_the_rescaled_decode_is_the_shared_one() -> None:
    """`nb_factor` 1 and no `bb_factor`: the same pairs, `alpha` within 1e-3, as the shared decode."""
    paths, bulks = _planted(np.full(4, 0.01))
    rescaled = _decode(paths, bulks, "rescale")
    shared = _decode(paths, bulks, "shared")

    for left, right in zip(rescaled.pairs, shared.pairs, strict=True):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_allclose(
        float(rescaled.dispersion), float(shared.dispersion), rtol=1e-3
    )


@pytest.mark.end2end
def test_the_two_component_decode_finds_the_clone_shared_part() -> None:
    """Planted 0.02 shared beside 0.3 per spot: shared within a factor 2, and no worse a fit than rescale alone.

    Realized: 0.0205 shared and 0.259 per spot, 121 nats above rescale alone.
    """
    planted = np.full(4, 0.3)
    paths, bulks = _planted(
        planted, nb_factors=(1.0 / 8.0, 1.0 / 64.0), shared_alpha=0.02
    )
    both = _decode(paths, bulks, "two-component")
    rescaled = _decode(paths, bulks, "rescale")

    assert 0.01 < both.shared_alpha < 0.04
    assert both.log_likelihood >= rescaled.log_likelihood - 1e-6
