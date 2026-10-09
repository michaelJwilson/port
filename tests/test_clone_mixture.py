"""`port.sandbox.admixture.clone_mixture` against `cnaster`'s emission and a planted blend (#380)."""

from __future__ import annotations

import numpy as np
import pytest
from cnaster.hmm_nophasing import _dense_bb_logpmf, _dense_nb_logpmf
from port.sandbox.admixture.clone_mixture import (
    ADMIXTURE_STARTS,
    PARSIMONY,
    fit_mixture,
    lattice,
    mixed_parameters,
    score,
)

MU = np.array([1.0, 1.5, 0.5, 2.0])
P = np.array([0.5, 0.67, 0.5, 0.75])
ALPHA, TAU = 0.02, 200.0


def _drawn(
    rng: np.random.Generator,
    mu: np.ndarray,
    p: np.ndarray,
    paths: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pseudobulk `(rdr, baf, total, base)` drawn from the mixed model on `paths`."""
    k, n_bins = paths.shape
    base = np.full((k, n_bins), 400.0) * rng.uniform(0.5, 1.5, n_bins)
    total = np.full((k, n_bins), 300.0)
    mixed, share = mixed_parameters(mu, p, paths, weights)
    weight = base / base.sum(axis=1, keepdims=True)
    mean = base * mixed / np.sum(weight * mixed, axis=1, keepdims=True)
    r = 1.0 / ALPHA
    rdr = rng.negative_binomial(r, r / (r + mean)).astype(np.float64)
    baf = rng.binomial(
        total.astype(np.int64), rng.beta(share * TAU, (1 - share) * TAU)
    ).astype(np.float64)

    return rdr, baf, total, base


def _sticky(n: int) -> np.ndarray:
    """A log transition matrix holding 0.99 on its diagonal over `n` states."""
    return np.log(np.full((n, n), 0.01 / (n - 1)) + np.eye(n) * (0.99 - 0.01 / (n - 1)))


def _planted(
    weights: np.ndarray, seed: int = 0, n_bins: int = 400
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Pure paths in blocks, and pseudobulk counts drawn from the mixed model."""

    rng = np.random.default_rng(seed)
    k = weights.shape[0]
    paths = np.zeros((k, n_bins), dtype=np.int64)

    for j in range(1, k):
        paths[j, (j - 1) * 80 + 20 : (j - 1) * 80 + 90] = j

    return paths, _drawn(rng, MU, P, paths, weights)


@pytest.mark.oracle
def test_the_pure_mixture_is_cnasters_emission() -> None:
    """At `W = I` each clone's score is `cnaster`'s NB + BB at the shifted-emission mean (#276)."""

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
    """The fit recovers a planted 25% blend to 0.05 and never lowers the log-likelihood."""

    planted = np.eye(3)
    planted[1] = [0.0, 0.75, 0.25]
    paths, bulks = _planted(planted, seed=3)
    transmat = _sticky(MU.size)

    fit = fit_mixture(bulks, MU, P, ALPHA, TAU, paths, transmat)

    assert fit.end >= fit.start
    np.testing.assert_allclose(fit.weights[1], planted[1], atol=0.05)
    np.testing.assert_allclose(fit.weights[0], planted[0], atol=0.05)
    np.testing.assert_array_equal(fit.paths, paths)


@pytest.mark.analytic
def test_the_identity_is_the_start_and_a_pure_sample_stays_there() -> None:
    """With no blend planted, the fit keeps `W` within 0.03 of the identity (#380)."""

    paths, bulks = _planted(np.eye(3), seed=5)
    transmat = _sticky(MU.size)

    fit = fit_mixture(bulks, MU, P, ALPHA, TAU, paths, transmat)

    np.testing.assert_allclose(fit.weights, np.eye(3), atol=0.03)
    assert fit.end >= fit.start


@pytest.mark.end2end
def test_the_lattice_finds_a_uniform_normal_admixture() -> None:
    """On integer pairs each tumour clone's diploid weight is 0.08 to 0.03 and its pure pair planted (#380)."""

    pairs, mu, p = lattice(6)
    index = {tuple(x): i for i, x in enumerate(pairs.tolist())}
    n_bins = 400
    planted = np.full((3, n_bins), index[(1, 1)], dtype=np.int64)
    planted[1, 50:200] = index[(1, 0)]
    planted[2, 150:300] = index[(2, 1)]
    weights = np.eye(3)
    weights[1] = [0.08, 0.92, 0.0]
    weights[2] = [0.08, 0.0, 0.92]

    bulks = _drawn(np.random.default_rng(7), mu, p, planted, weights)
    transmat = _sticky(mu.size)
    prior = -PARSIMONY * np.abs(pairs.sum(axis=1) - 2).astype(np.float64)
    start = np.full((4, n_bins), index[(1, 1)], dtype=np.int64)

    fit = max(
        (
            fit_mixture(
                bulks,
                mu,
                p,
                ALPHA,
                TAU,
                start,
                transmat,
                log_prior=prior,
                diploid=3,
                admixture=a,
            )
            for a in ADMIXTURE_STARTS
        ),
        key=lambda fit: fit.end,
    )

    np.testing.assert_allclose(fit.weights[1:, 3], [0.08, 0.08], atol=0.03)
    np.testing.assert_allclose(fit.weights[0, [0, 3]].sum(), 1.0, atol=0.02)
    np.testing.assert_array_equal(fit.paths[:3], planted)
