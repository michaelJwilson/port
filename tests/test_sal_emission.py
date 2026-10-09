"""sal's dense log-emission against cnaster's 1-D kernels (#425, sal #1132).

Referees: `cnaster.hmm_nophasing._nb_logpmf_1d` and `_bb_logpmf_1d`, over zero exposures
and trials, at parameters from the fit's bounds.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import cnaster.hmm_nophasing
import cnaster.hmrf
import cnaster.scripts.run_cnaster as pipeline
import numpy as np
import pytest
from cnaster.count_encoder import CountEncoder
from cnaster.hmm_nophasing import _bb_logpmf_1d, _nb_logpmf_1d
from cnaster.hmm_nophasing import hmm_nophasing as upstream
from port.patch.hmm_nophasing.dense_emission import bb_states, coded_emission, nb_states
from port.scripts.run_cnaster import main

from tests.fixtures import (
    divergent_clone_instance,
    shifted_emission_call,
    shifted_replacement,
)

CASES = [
    (1.3, 0.05, 0.4, 30.0, 1e-10),
    (0.2, 1e-6, 1e-6, 50.0, 1e-8),
    (0.0, 0.1, 0.5, 5.0, 1e-10),
    (3.0, 2.0, 0.999, 1000.0, 1e-10),
]
"""`(mu, alpha, p_binom, tau, atol)`: typical, the fit's lower bounds, a zero rate, heavy dispersion."""


def _counts(
    rng: np.random.Generator | None = None,
    shape: int | tuple[int, int] = 5000,
    every: int = 50,
) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(0) if rng is None else rng
    exposure = rng.uniform(0, 150, shape)
    exposure[::every] = 0.0
    totals = rng.poisson(exposure).astype(float)
    trials = rng.integers(0, 60, shape).astype(float)
    successes = np.minimum(rng.poisson(trials * 0.4), trials).astype(float)
    return totals, exposure, successes, trials


@pytest.mark.patch
@pytest.mark.parametrize("case", CASES)
def test_the_sal_kernels_are_cnasters_to_a_stated_tolerance(
    case: tuple[float, float, float, float, float],
) -> None:
    """Each state's row of `nb_states`, `bb_states` against `cnaster`'s 1-D kernel."""

    mu, alpha, p_binom, tau, atol = case
    totals, exposure, successes, trials = _counts()
    # NB two states, so a row is read per state.
    mus, alphas = np.array([mu, 1.0]), np.array([alpha, 0.1])
    ps, taus = np.array([p_binom, 0.5]), np.array([tau, 20.0])

    ours_rdr = nb_states(totals, exposure, mus, alphas)
    ours_baf = bb_states(successes, trials, ps, taus)

    for state in range(2):
        rdr, baf = np.zeros(totals.size), np.zeros(totals.size)
        _nb_logpmf_1d(totals, exposure, mus[state], alphas[state], rdr)
        _bb_logpmf_1d(successes, trials, ps[state], taus[state], baf)

        np.testing.assert_allclose(ours_rdr[state], rdr, rtol=0, atol=atol)
        np.testing.assert_allclose(ours_baf[state], baf, rtol=0, atol=atol)
        np.testing.assert_array_equal(ours_rdr[state][exposure == 0], 0.0)
        np.testing.assert_array_equal(ours_baf[state][trials == 0], 0.0)


@pytest.mark.bug
def test_cnasters_nb_mean_is_not_lambda_below_its_dispersion_floor() -> None:
    """Below `alpha = 1e-10` cnaster floors `r` but not `p`; sal keeps the mean at `lambda`."""
    alpha, exposure = 1e-12, 150.0
    r = 1.0 / max(alpha, 1.0e-10)
    p = 1.0 / (1.0 + alpha * exposure)
    implied = r * (1.0 - p) / p

    assert implied == pytest.approx(alpha / 1e-10 * exposure, rel=1e-6)
    assert implied != pytest.approx(exposure, rel=0.5)


def _coded(clone_stack: bool) -> tuple[Any, Any]:
    """Upstream's coded method and sal's, through real `CountEncoder`s: (ours, theirs)."""
    rng = np.random.default_rng(3)
    n_obs, n_spots, n_states = 400, 2, 8
    totals, exposure, successes, trials = _counts(rng, (n_obs, n_spots), 40)

    nb_encoder = CountEncoder(totals, exposure)
    bb_encoder = CountEncoder(successes, trials)
    log_mu = rng.normal(0, 0.4, (n_states, n_spots))
    log_mu[0] = -np.inf
    alphas = rng.uniform(0.01, 0.3, (n_states, n_spots))
    p_binom = rng.uniform(0.1, 0.9, (n_states, n_spots))
    taus = rng.uniform(10, 100, (n_states, n_spots))

    theirs = upstream().compute_emission_probability_nb_betabinom_coded(
        nb_encoder, bb_encoder, log_mu, alphas, p_binom, taus, clone_stack=clone_stack
    )
    ours = coded_emission(
        nb_encoder, bb_encoder, log_mu, alphas, p_binom, taus, clone_stack=clone_stack
    )
    return ours, theirs


def _swapped_class(shifted: bool) -> tuple[Any, Any]:
    """The swapped class's emission with the `sal` option on, and off: (ours, theirs)."""
    instance = divergent_clone_instance()

    def scored(kernels: str) -> tuple[np.ndarray, np.ndarray]:
        model = shifted_replacement(instance, shifted=shifted, kernels=kernels)
        return shifted_emission_call(model, instance)

    return scored("sal"), scored("cnaster")


@pytest.mark.patch
@pytest.mark.parametrize(
    "emissions",
    [
        partial(_coded, True),
        partial(_coded, False),
        partial(_swapped_class, False),
        partial(_swapped_class, True),
    ],
    ids=["coded-clone-stack", "coded", "class", "class-shifted"],
)
def test_the_sal_emission_is_upstreams_to_a_stated_tolerance(
    emissions: Callable[[], tuple[Any, Any]], cnaster_config: None
) -> None:
    """Upstream's coded method against sal's, and the swapped class shifted and unshifted, option on against off, to 1e-10."""

    ours, theirs = emissions()

    for mine, reference in zip(ours, theirs, strict=True):
        assert mine.shape == reference.shape
        np.testing.assert_allclose(mine, reference, rtol=0, atol=1e-10)


@pytest.mark.infra
@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], ("sal", True)),
        (["--no-sal-emission"], ("cnaster", True)),
        (["--no-shift", "--no-copy-cap"], ("cnaster", False)),
        (["--no-patch", "--shift"], ("sal", False)),
    ],
    ids=["default", "no-sal-emission", "no-shift", "no-patch-shift"],
)
def test_run_cnaster_scores_with_sal_s_kernels_where_the_shift_reads_them(
    argv: list[str],
    expected: tuple[str, bool],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--sal-emission` and `--distinct-init` follow the shift rows that read them (#466)."""

    config = tmp_path / "config.yaml"
    config.write_text("{}\n")
    seen: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        pipeline,
        "run_cnaster",
        lambda *_: seen.append(
            (
                # NB cnaster's class, where no shift row installed port's.
                getattr(
                    cnaster.hmm_nophasing.hmm_nophasing, "emission_kernels", "cnaster"
                ),
                getattr(cnaster.hmrf.run_core_inference, "keywords", {}).get(
                    "distinct_init", False
                ),
            )
        ),
    )
    main([*argv, "--no-rust", str(config)])

    assert seen == [expected]


@pytest.mark.smoke
@pytest.mark.parametrize("flag", ["--sal-emission", "--distinct-init"])
def test_a_flag_the_shift_rows_read_is_refused_without_them(
    flag: str, tmp_path: Path
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("{}\n")

    with pytest.raises(SystemExit):
        main(["--no-shift", flag, "--no-rust", str(config)])
