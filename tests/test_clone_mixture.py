"""`port.extensions.clone_mixture` against `cnaster`'s emission and a planted blend (#380)."""

from __future__ import annotations

import numpy as np
import pytest

MU = np.array([1.0, 1.5, 0.5, 2.0])
P = np.array([0.5, 0.67, 0.5, 0.75])
ALPHA, TAU = 0.02, 200.0


def _planted(
    weights: np.ndarray, seed: int = 0, n_bins: int = 400
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Pure paths in blocks, and pseudobulk counts drawn from the mixed model."""
    from port.extensions.clone_mixture import mixed_parameters

    rng = np.random.default_rng(seed)
    k = weights.shape[0]
    paths = np.zeros((k, n_bins), dtype=np.int64)

    for j in range(1, k):
        paths[j, (j - 1) * 80 + 20 : (j - 1) * 80 + 90] = j

    base = np.full((k, n_bins), 400.0) * rng.uniform(0.5, 1.5, n_bins)
    total = np.full((k, n_bins), 300.0)
    mixed, share = mixed_parameters(MU, P, paths, weights)
    weight = base / base.sum(axis=1, keepdims=True)
    mean = base * mixed / np.sum(weight * mixed, axis=1, keepdims=True)
    r = 1.0 / ALPHA
    rdr = rng.negative_binomial(r, r / (r + mean)).astype(np.float64)
    baf = rng.binomial(
        total.astype(np.int64), rng.beta(share * TAU, (1 - share) * TAU)
    ).astype(np.float64)

    return paths, (rdr, baf, total, base)


@pytest.mark.oracle
def test_the_pure_mixture_is_cnasters_emission() -> None:
    """At `W = I` each clone's score is `cnaster`'s NB + BB at its own path.

    `cnaster`'s dense kernels at the library-normalized exposure
    `base / sum(lambda mu)`, which is the shifted emission's mean (#276).
    """
    from cnaster.hmm_nophasing import _dense_bb_logpmf, _dense_nb_logpmf
    from port.extensions.clone_mixture import mixed_parameters, score

    weights = np.eye(3)
    paths, bulks = _planted(weights)
    rdr, baf, total, base = bulks
    mixed, share = mixed_parameters(MU, P, paths, weights)
    ours = score(bulks, mixed, share, ALPHA, TAU)
    states = MU.size
    alphas = np.full((states, 1), ALPHA)
    taus = np.full((states, 1), TAU)

    for i in range(3):
        weight = base[i] / base[i].sum()
        exposure = base[i] / np.sum(weight * MU[paths[i]])
        nb = _dense_nb_logpmf(
            rdr[i][:, None], exposure[:, None], np.log(MU)[:, None], alphas
        )[:, :, 0]
        bb = _dense_bb_logpmf(baf[i][:, None], total[i][:, None], P[:, None], taus)[
            :, :, 0
        ]
        bins = np.arange(paths.shape[1])
        theirs = nb[paths[i], bins] + bb[paths[i], bins]

        np.testing.assert_allclose(ours[i], theirs, rtol=1e-12, atol=1e-9)


@pytest.mark.end2end
def test_the_fit_recovers_a_planted_blend_and_never_goes_downhill() -> None:
    """Three clones, clone 1's pseudobulk 25 per cent clone 2's spots.

    From `W = I` and the pure paths, the fit finds the off-diagonal weight to
    0.05 and ends at a log-likelihood no lower than it started.
    """
    from port.extensions.clone_mixture import fit_mixture

    planted = np.eye(3)
    planted[1] = [0.0, 0.75, 0.25]
    paths, bulks = _planted(planted, seed=3)
    n = MU.size
    transmat = np.log(
        np.full((n, n), 0.01 / (n - 1)) + np.eye(n) * (0.99 - 0.01 / (n - 1))
    )

    fit = fit_mixture(bulks, MU, P, ALPHA, TAU, paths, transmat)

    assert fit.end >= fit.start
    np.testing.assert_allclose(fit.weights[1], planted[1], atol=0.05)
    np.testing.assert_allclose(fit.weights[0], planted[0], atol=0.05)
    np.testing.assert_array_equal(fit.paths, paths)


@pytest.mark.analytic
def test_the_identity_is_the_start_and_a_pure_sample_stays_there() -> None:
    """No blend planted: the fit keeps `W` within 0.02 of the identity."""
    from port.extensions.clone_mixture import fit_mixture

    paths, bulks = _planted(np.eye(3), seed=5)
    n = MU.size
    transmat = np.log(
        np.full((n, n), 0.01 / (n - 1)) + np.eye(n) * (0.99 - 0.01 / (n - 1))
    )

    fit = fit_mixture(bulks, MU, P, ALPHA, TAU, paths, transmat)

    np.testing.assert_allclose(fit.weights, np.eye(3), atol=0.02)
    assert fit.end >= fit.start


@pytest.mark.analytic
def test_a_cap_bounds_the_mixing_and_binds_on_a_larger_blend() -> None:
    """A 25 per cent blend under a 0.2 cap: the row's off-diagonal mass is 0.2.

    The cap is `sum_{j != i} W_ij <= cap`; the likelihood wants 0.25, so the
    bound binds, and the weight goes to the planted contaminant alone.
    """
    from port.extensions.clone_mixture import fit_mixture

    planted = np.eye(3)
    planted[1] = [0.0, 0.75, 0.25]
    paths, bulks = _planted(planted, seed=3)
    n = MU.size
    transmat = np.log(
        np.full((n, n), 0.01 / (n - 1)) + np.eye(n) * (0.99 - 0.01 / (n - 1))
    )

    fit = fit_mixture(bulks, MU, P, ALPHA, TAU, paths, transmat, cap=0.2)

    assert 1.0 - fit.weights[1, 1] == pytest.approx(0.2, abs=1e-6)
    assert fit.weights[1, 2] == pytest.approx(0.2, abs=1e-6)
    assert np.all(1.0 - np.diag(fit.weights) <= 0.2 + 1e-9)


@pytest.mark.analytic
def test_the_anneal_is_linear_and_holds_at_its_end() -> None:
    """0.1 to 0.5 over 4 outer iterations: 0.1, 0.2, 0.3, 0.4, 0.5, then 0.5."""
    from port.extensions.clone_mixture import annealed

    caps = [annealed(0.1, 0.5, t, 4) for t in range(6)]

    np.testing.assert_allclose(caps, [0.1, 0.2, 0.3, 0.4, 0.5, 0.5])
