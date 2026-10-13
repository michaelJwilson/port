"""The #380 design study's set-aside variants: the mixing cap and its anneal."""

from __future__ import annotations

import numpy as np
import pytest
from port.sandbox.admixture.variants import annealed, fit_mixture

from tests.test_clone_mixture import ALPHA, MU, TAU, P, _planted, _sticky


@pytest.mark.analytic
def test_a_cap_bounds_the_mixing_and_binds_on_a_larger_blend() -> None:
    """A 25% blend under a 0.2 cap binds at 0.2 off-diagonal mass, all on the planted contaminant."""

    planted = np.eye(3)
    planted[1] = [0.0, 0.75, 0.25]
    paths, bulks = _planted(planted, seed=3)
    transmat = _sticky(MU.size)

    fit = fit_mixture(bulks, MU, P, ALPHA, TAU, paths, transmat, cap=0.2)

    assert 1.0 - fit.weights[1, 1] == pytest.approx(0.2, abs=1e-6)
    assert fit.weights[1, 2] == pytest.approx(0.2, abs=1e-6)
    assert np.all(1.0 - np.diag(fit.weights) <= 0.2 + 1e-9)


@pytest.mark.analytic
def test_the_anneal_is_linear_and_holds_at_its_end() -> None:
    """0.1 to 0.5 over 4 outer iterations: 0.1, 0.2, 0.3, 0.4, 0.5, then 0.5."""

    caps = [annealed(0.1, 0.5, t, 4) for t in range(6)]

    np.testing.assert_allclose(caps, [0.1, 0.2, 0.3, 0.4, 0.5, 0.5])
