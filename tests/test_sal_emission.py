"""sal's dense log-emission as `cnaster`'s 1-D kernels (#425, sal #1132).

Referees: `cnaster.hmm_nophasing._nb_logpmf_1d` and `_bb_logpmf_1d`
themselves, over counts with zero exposures and zero trials, at parameters
from the fit's bounds. Realized at most 3.2e-12 absolute at typical
dispersions and 3.5e-9 at the fit's lower bound `alpha = 1e-6`, where
`lgamma(k + r) - lgamma(r)` cancels at `r = 1e6`. Measured once against the
density at 50 digits (#425's pull request, not a test dependency): sal is
the nearer of the two in every case, 1.6e-9 against 2.6e-9 at that bound.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CASES = [
    (1.3, 0.05, 0.4, 30.0, 1e-10),
    (0.2, 1e-6, 1e-6, 50.0, 1e-8),
    (0.0, 0.1, 0.5, 5.0, 1e-10),
    (3.0, 2.0, 0.999, 1000.0, 1e-10),
]
"""`(mu, alpha, p_binom, tau, atol)`: typical, at the fit's lower bounds
(`min_alpha = 1e-6`), a zero rate, and heavy dispersion near the boundary."""


def _counts() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(0)
    n = 5000
    exposure = rng.uniform(0, 150, n)
    exposure[::50] = 0.0
    totals = rng.poisson(exposure).astype(float)
    trials = rng.integers(0, 60, n).astype(float)
    successes = np.minimum(rng.poisson(trials * 0.4), trials).astype(float)
    return totals, exposure, successes, trials


@pytest.mark.patch
@pytest.mark.parametrize("case", CASES)
def test_the_sal_kernels_are_cnasters_to_a_stated_tolerance(
    case: tuple[float, float, float, float, float],
) -> None:
    """Each state's row of `nb_states`, `bb_states` against `cnaster`'s 1-D kernel."""
    from cnaster.hmm_nophasing import _bb_logpmf_1d, _nb_logpmf_1d
    from port.patch.hmm_nophasing.dense_emission import bb_states, nb_states

    mu, alpha, p_binom, tau, atol = case
    totals, exposure, successes, trials = _counts()
    # NB two states, the case's and a typical one, so a row is read per state.
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
    """`r = 1 / max(alpha, 1e-10)` but `p = 1 / (1 + alpha * lambda)`, unfloored.

    Below the floor the implied mean `r (1 - p) / p` is `alpha / 1e-10 *
    lambda`: at `alpha = 1e-12`, a hundredth of the exposure's. sal's family
    keeps the mean at `lambda`, which is why the swap departs from `cnaster`
    there. Unreachable from a fit: the optimizer bounds `alpha` at 1e-6.
    """
    alpha, exposure = 1e-12, 150.0
    r = 1.0 / max(alpha, 1.0e-10)
    p = 1.0 / (1.0 + alpha * exposure)
    implied = r * (1.0 - p) / p

    assert implied == pytest.approx(alpha / 1e-10 * exposure, rel=1e-6)
    assert implied != pytest.approx(exposure, rel=0.5)


@pytest.mark.patch
@pytest.mark.parametrize("clone_stack", [True, False])
def test_the_coded_emission_is_upstreams_to_a_stated_tolerance(
    clone_stack: bool, cnaster_config: None
) -> None:
    """Upstream's coded method against sal's, through real `CountEncoder`s.

    Two spots, eight states, a zero-rate state and zero-exposure bins: each
    entry of both channels to 1e-10, and the shapes upstream returns.
    """
    from cnaster.count_encoder import CountEncoder
    from cnaster.hmm_nophasing import hmm_nophasing as upstream
    from port.patch.hmm_nophasing.dense_emission import coded_emission

    rng = np.random.default_rng(3)
    n_obs, n_spots, n_states = 400, 2, 8
    exposure = rng.uniform(0, 150, (n_obs, n_spots))
    exposure[::40] = 0.0
    totals = rng.poisson(exposure).astype(float)
    trials = rng.integers(0, 60, (n_obs, n_spots)).astype(float)
    successes = np.minimum(rng.poisson(trials * 0.4), trials).astype(float)

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

    for mine, reference in zip(ours, theirs, strict=True):
        assert mine.shape == reference.shape
        np.testing.assert_allclose(mine, reference, rtol=0, atol=1e-10)


@pytest.mark.patch
@pytest.mark.parametrize("shifted", [False, True])
def test_the_class_under_sal_emission_scores_as_it_does_under_cnasters(
    shifted: bool, cnaster_config: None
) -> None:
    """The swapped class, both branches, with the option on against off, to 1e-10.

    Unshifted: `coded_emission` against upstream's coded method. Shifted:
    the batched per-clone rows against the per-state `cnaster` kernels on
    the same `(clone, obs, total)` triples.
    """
    from tests.test_shifted_emission import _call, _instance, _replacement

    instance = _instance()

    def scored(kernels: str) -> tuple[np.ndarray, np.ndarray]:
        model = _replacement(instance, shifted=shifted, kernels=kernels)
        return _call(model, instance)

    theirs = scored("cnaster")
    ours = scored("sal")

    for mine, reference in zip(ours, theirs, strict=True):
        assert mine.shape == reference.shape
        np.testing.assert_allclose(mine, reference, rtol=0, atol=1e-10)


@pytest.mark.infra
@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], ("sal", True)),
        (["--no-sal-emission"], ("cnaster", True)),
        (["--no-shift"], ("cnaster", False)),
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
    """`--sal-emission` and `--distinct-init` follow the shift rows that read them.

    Under `--no-shift` both were entered and neither read (#466); the run
    said "distinct initial states" regardless.
    """
    import cnaster.hmm_nophasing
    import cnaster.hmrf
    import cnaster.scripts.run_cnaster as pipeline
    from port.scripts.run_cnaster import main

    config = tmp_path / "config.yaml"
    config.write_text("{}\n")
    seen: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        pipeline,
        "run_cnaster",
        lambda *_: seen.append(
            (
                # NB `cnaster`'s class, where no shift row installed port's.
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


@pytest.mark.infra
@pytest.mark.parametrize("flag", ["--sal-emission", "--distinct-init"])
def test_a_flag_the_shift_rows_read_is_refused_without_them(
    flag: str, tmp_path: Path
) -> None:
    from port.scripts.run_cnaster import main

    config = tmp_path / "config.yaml"
    config.write_text("{}\n")

    with pytest.raises(SystemExit):
        main(["--no-shift", flag, "--no-rust", str(config)])
