"""#445: the fitted laws and admixture share `port.sim.draw` and `normal_fit` use.

The fitters are judged by recovering planted parameters, and `allele_share`
by its analytic values at the planted copies.
"""

from __future__ import annotations

import numpy as np
import pytest
from port.sim.laws import allele_share, counted, fit_lognormal, fit_negative_binomial


@pytest.mark.end2end
def test_the_fitters_recover_planted_laws() -> None:
    """20,000 draws: lognormal `(mu, sigma)` and NB `(mean, dispersion)` to 4 SE.

    The SE of the NB dispersion by moments is taken from 200 repeat fits.
    """
    rng = np.random.default_rng(1)
    n = 20_000
    mu, sigma, ks = fit_lognormal(rng.lognormal(6.0, 0.4, n))
    assert mu == pytest.approx(6.0, abs=4 * 0.4 / np.sqrt(n))
    assert sigma == pytest.approx(0.4, abs=4 * 0.4 / np.sqrt(2 * n))
    assert ks < 0.02

    mean, dispersion = 40.0, 0.2
    r = 1 / dispersion

    def draw() -> np.ndarray:
        return rng.negative_binomial(r, r / (r + mean), n)

    repeats = [fit_negative_binomial(draw())[1] for _ in range(200)]
    found_mean, found, _ = fit_negative_binomial(draw())
    assert found_mean == pytest.approx(
        mean, abs=4 * np.sqrt(mean * (1 + dispersion * mean) / n)
    )
    assert found == pytest.approx(dispersion, abs=4 * float(np.std(repeats)))
    assert counted(draw(), "negative_binomial").family == "negative_binomial"


@pytest.mark.analytic
def test_the_allele_share_at_planted_copies() -> None:
    """Read law: `(1 - f) a/(a + b) + f/2`; cell law: `((1 - f) a + f)/((1 - f)(a + b) + 2f)`."""
    a, b = np.array([1.0, 2.0, 0.0, 2.0]), np.array([1.0, 0.0, 2.0, 1.0])
    f = 0.2

    np.testing.assert_allclose(
        allele_share(a, b, f, "read"), 0.8 * a / (a + b) + 0.1, rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(
        allele_share(a, b, f, "cell"), (0.8 * a + 0.2) / (0.8 * (a + b) + 0.4),
        rtol=0, atol=1e-15,
    )  # fmt: skip
    np.testing.assert_array_equal(allele_share(a, b, 0.0, "read"), a / (a + b))
