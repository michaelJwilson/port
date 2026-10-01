"""#562: `sal`'s emission++ divergences floored at 0 around port's mixture start.

`run_cnaster_port --sal --hmm-start emission++...` seeds through
`sal.opt.emission_mixture._seed_scores`, whose negative-binomial Bregman
divergence is non-negative in exact arithmetic and about `-1.6e-15` in
float64 for a row a hair from a seed's mean. D-squared sampling hands those
to `rng.choice`, which refuses them. `sal_mixture.fitted` floors them at 0.

The rows are the rate-space pairs `sal_mixture.instance_of` builds, seeded
through the same `CountPairSeeding` seam: 200 at `(2.0, 0.3)`; 200 whose
total is `2.0` to 1e-12 and whose B rate, 0.76034, puts their beta-binomial
deviance from a `(2.0, 0.3)` seed at its clamp of 0; and 5 far rows, so the
draw has positive mass to normalize.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.exact_densities import nb_divergence

SIZE = 10.0
SHARE = 0.7603373084367014
"""The B rate whose beta-binomial deviance from a `(2.0, 0.3)` seed is 0 at trials 50."""


def _rows() -> np.ndarray:
    z = np.random.default_rng(1).standard_normal(200)
    return np.vstack(
        [
            np.tile([2.0, 0.3], (200, 1)),
            np.column_stack([2.0 * (1.0 + 1e-12 * z), np.full(200, SHARE)]),
            np.column_stack([np.full(5, 20.0), np.full(5, 0.3)]),
        ]
    )


def _seam() -> Any:
    from sal.opt.emission_mixture import CountPairSeeding

    return CountPairSeeding(
        dispersion=SIZE, concentration=1_000.0, joint=False, trials=50.0
    )


def _refused(seed: int) -> bool:
    from sal.opt.emission_mixture import plus_plus_start

    try:
        plus_plus_start(_rows(), 3, _seam(), np.random.default_rng(seed))
    except ValueError as error:
        return "non-negative" in str(error)
    return False


@pytest.mark.bug
def test_sals_emission_plus_plus_refuses_a_round_off_negative_divergence() -> None:
    """11 of 20 generators raise "Probabilities are not non-negative" through sal's own seam."""
    assert sum(_refused(seed) for seed in range(20)) == 11


@pytest.mark.oracle
def test_the_floored_scores_are_the_exact_divergence() -> None:
    """From a `(2.0, 0.3)` seed, each near row's floored score is the 50-digit divergence, to 1e-14 absolute.

    The exact negative-binomial divergence there is below 1e-24 and the
    beta-binomial deviance is clamped to 0 by `sal`; unfloored, 194 of the
    200 near rows score as low as `-1.6e-15`.
    """
    from port.patch.hmm_initialize.sal_mixture import clamped_divergence
    from sal.opt.emission_mixture import _seed_scores as unfloored

    rows = _rows()
    candidates = np.arange(rows.shape[0], dtype=np.float64)
    near = slice(200, 400)
    raw = unfloored(rows, _seam())(0.0, candidates)

    with clamped_divergence():
        import sal.opt.emission_mixture as upstream

        floored = upstream._seed_scores(rows, _seam())(0.0, candidates)

    exact = [nb_divergence(y, 2.0, SIZE) for y in rows[near, 0]]

    assert (raw[near] < 0.0).sum() > 100
    assert (floored >= 0.0).all()
    np.testing.assert_allclose(floored[near], exact, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(floored[raw >= 0.0], raw[raw >= 0.0])


@pytest.mark.patch
def test_the_floor_draws_sals_seeds_wherever_sal_draws_any() -> None:
    """Under the floor every generator seeds, and the 9 sal does not refuse draw the same rows.

    Flooring moves only scores below 0, which `rng.choice` never accepted, so
    a draw sal completes is the same draw.
    """
    import sal.opt.emission_mixture as upstream
    from port.patch.hmm_initialize.sal_mixture import clamped_divergence
    from sal.opt.mixture import emission_mixture_plus_plus

    rows = _rows()
    indices = np.arange(rows.shape[0], dtype=np.float64)
    completed = 0

    for seed in range(20):
        with clamped_divergence():
            floored = emission_mixture_plus_plus(
                indices,
                3,
                upstream._seed_scores(rows, _seam()),
                np.random.default_rng(seed),
            )
        try:
            raw = emission_mixture_plus_plus(
                indices,
                3,
                upstream._seed_scores(rows, _seam()),
                np.random.default_rng(seed),
            )
        except ValueError:
            continue
        completed += 1
        np.testing.assert_array_equal(floored, raw)

    assert completed == 9


@pytest.mark.infra
def test_fitted_floors_the_scores_for_the_call_and_restores_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`sal_mixture.fitted` seeds under the floor; `sal`'s own name is back afterwards."""
    import sal.opt.emission_mixture as upstream
    import sal.search.mixture_starts as starts
    from port.patch.hmm_initialize import sal_mixture

    original = upstream._seed_scores
    seen: list[bool] = []

    class Stop(Exception):
        pass

    def start(instance: Any, rng: np.random.Generator) -> Any:
        seen.append(upstream._seed_scores is not original)
        raise Stop

    monkeypatch.setattr(starts, "lookup", lambda _: start)

    with pytest.raises(Stop):
        sal_mixture.fitted(object(), "emission++", np.random.default_rng(0))

    assert seen == [True]
    assert upstream._seed_scores is original
