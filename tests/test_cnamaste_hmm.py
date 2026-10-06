"""`cnamaste`'s HMM fit chain, `docs/port-forward.md` rows 25-26 (T- #670 PR5).

`cnamaste.hmm_phased` and `cnamaste.hmm_nophasing` are `port`'s drop-ins
moved in: the coded emission read by state (#269), the analytic M-step
gradient (#433) and the per-clone shift (#276). Referees, one per test:

- `oracle`: `sal.likelihood.forward_backward`, with emissions from
  `scipy.stats`, for the evidence and the posterior of both classes;
- `analytic`: central finite differences of `cost_fn`, for the gradient;
- `patch`: `port`'s classes under `LOG_SPACE_SWAPS` (`cnamaste`'s kernels
  since PR4), bitwise;
- `bug`: #135, #244 and #269, each against `cnaster`.

The end-to-end equivalence is `test_cnamaste_copy.py`'s, where `ABSORBED`
now carries `FIT_CHAIN`. Fixtures are drawn here from fixed seeds, small
enough for the gate; the clone-stacked one, 3 clones x 400 bins at seed 1,
is `_stacked`.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import stats

N_STATES = 5
"""States in the clone-stacked fixture."""


@pytest.fixture
def _configs(tmp_path: Path) -> Iterator[None]:
    """`cnaster`'s test configuration, installed in both packages and restored."""
    from cnamaste import config as own
    from port.sim.inputs import written_config

    from tests.conftest import SHIPPED_EM_FTOL, cnaster_test_config

    document = cnaster_test_config(tmp_path, SHIPPED_EM_FTOL, 100)
    saved = own._global_config
    with written_config(document):
        own.set_global_config(own.YAMLConfig(document))
        try:
            yield
        finally:
            own.set_global_config(saved)


def _stacked(seed: int = 1, per_clone: int = 400) -> dict[str, Any]:
    """3 clones of `per_clone` bins, `N_STATES` planted in blocks of 40 bins.

    Negative binomial depth at dispersion 0.05 over a shared normal profile,
    beta-binomial allele counts at concentration 40 over 5 to 59 trials.
    """
    generator = np.random.default_rng(seed)
    n_clones = 3
    states = np.repeat(
        generator.integers(0, N_STATES, (n_clones, per_clone // 40)), 40, axis=1
    ).reshape(-1)
    log_mu = np.linspace(-0.7, 0.6, N_STATES)
    p_binom = np.linspace(0.15, 0.5, N_STATES)

    profile = generator.uniform(0.5, 1.5, per_clone)
    profile /= profile.sum()
    base = np.tile(profile, n_clones) * 20_000.0
    mean = base * np.exp(log_mu[states])
    size = 1.0 / 0.05
    depth = generator.negative_binomial(size, size / (size + mean))
    trials = generator.integers(5, 60, n_clones * per_clone)
    fraction = generator.beta(p_binom[states] * 40.0, (1.0 - p_binom[states]) * 40.0)
    successes = generator.binomial(trials, fraction)

    X = np.stack([depth, successes], axis=1)[:, :, None].astype(np.float64)
    return {
        "X": X,
        "lengths": np.full(n_clones, per_clone),
        "base": base[:, None],
        "total": trials[:, None].astype(np.float64),
        "settings": {
            "log_sitewise_transmat": np.zeros(X.shape[0]),
            "shared_NB_dispersion": True,
            "shared_BB_dispersion": True,
            "init_log_mu": np.linspace(-0.5, 0.5, N_STATES)[:, None],
            "init_p_binom": np.linspace(0.2, 0.45, N_STATES)[:, None],
            "max_iter": 30,
            "normal_lambda": profile,
            "clone_lengths": np.full(n_clones, per_clone),
        },
    }


def _fit(cls: type, problem: dict[str, Any], **attributes: Any) -> dict[str, Any]:
    """`cls(params="smp").optimize` on `problem`, with class `attributes` set."""
    model = type(cls.__name__, (cls,), attributes)(params="smp", t=1 - 1e-4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result: dict[str, Any] = model.optimize(
            problem["X"],
            problem["lengths"],
            N_STATES,
            problem["base"],
            problem["total"],
            **problem["settings"],
        )
    return result


FITTED = ("new_log_mu", "new_alphas", "new_p_binom", "new_taus")
"""The emission parameters a fit returns."""


def _moved(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    """Each fitted parameter's largest relative difference between two fits."""
    return {
        name: float(
            np.max(np.abs(left[name] - right[name]) / np.abs(left[name]).clip(1e-300))
        )
        for name in FITTED
    }


# --- forward-backward against `sal` ----------------------------------------------

POSTERIOR_TOLERANCE = 1e-12
"""Absolute, on posteriors in [0, 1]; 1.1e-13 realized."""

EVIDENCE_TOLERANCE = 1e-12
"""Relative, on a log evidence of -360 to -373 nats; 4.6e-15 realized."""


def _chains(seed: int) -> dict[str, Any]:
    """Two chains of 30 positions, 3 states, both channels observed.

    Whole-number exposures: `CountEncoder`, which `hmm_phased`'s emission goes
    through, rounds a fractional one to `hmm.compression_decimals`.
    """
    generator = np.random.default_rng(seed)
    n_states, n_obs = 3, 60
    base = generator.integers(20, 80, n_obs).astype(np.float64)
    total = generator.integers(0, 40, n_obs).astype(np.float64)
    X = np.stack(
        [
            generator.poisson(base).astype(np.float64),
            generator.binomial(total.astype(int), 0.35).astype(np.float64),
        ],
        axis=1,
    )[:, :, None]
    transition = np.full((n_states, n_states), 0.1 / (n_states - 1))
    np.fill_diagonal(transition, 0.9)
    return {
        "X": X,
        "lengths": np.array([30, 30]),
        "base": base[:, None],
        "total": total[:, None],
        "log_mu": np.array([[-0.4], [0.0], [0.5]]),
        "alphas": np.array([[0.05], [0.1], [0.2]]),
        "p_binom": np.array([[0.2], [0.35], [0.5]]),
        "taus": np.array([[30.0], [60.0], [200.0]]),
        "log_startprob": np.log(np.array([0.5, 0.3, 0.2])),
        "log_transmat": np.log(transition),
        "log_switch": np.log(generator.uniform(0.01, 0.2, n_obs)),
    }


def _scipy_emission(chains: dict[str, Any], *, phased: bool) -> np.ndarray:
    """`(n_obs, K)` log emissions from `scipy.stats`; `2K` columns when phased."""
    depth, alleles = chains["X"][:, 0, 0], chains["X"][:, 1, 0]
    mean = chains["base"] * np.exp(chains["log_mu"][:, 0])
    size = 1.0 / chains["alphas"][:, 0]
    nb = stats.nbinom.logpmf(depth[:, None], size, size / (size + mean))
    a = chains["p_binom"][:, 0] * chains["taus"][:, 0]
    b = (1.0 - chains["p_binom"][:, 0]) * chains["taus"][:, 0]
    trials = chains["total"]
    bb = stats.betabinom.logpmf(alleles[:, None], trials, a, b)

    if not phased:
        return np.asarray(nb + bb)

    switched = stats.betabinom.logpmf(alleles[:, None], trials, b, a)
    return np.concatenate([nb + bb, nb + switched], axis=1)


def _sal(chains: dict[str, Any], *, phased: bool) -> list[Any]:
    """`sal`'s forward-backward on each chain."""
    from sal.likelihood.forward_backward import forward_backward

    emission = _scipy_emission(chains, phased=phased)
    start = np.exp(chains["log_startprob"])
    transition = np.exp(chains["log_transmat"])
    bounds = np.concatenate(([0], np.cumsum(chains["lengths"])))
    runs = []

    for first, last in itertools.pairwise(bounds):
        if phased:
            # NB `cnaster`'s combined kernel at each step: the copy state's
            #    transition, times staying in phase or switching, from the
            #    site's switch probability (`update_combined_transmat`).
            switch = np.exp(chains["log_switch"][first : last - 1])[:, None, None]
            kernels = np.block(
                [
                    [(1.0 - switch) * transition, switch * transition],
                    [switch * transition, (1.0 - switch) * transition],
                ]
            )
            log_initial = np.log(0.5 * np.concatenate([start, start]))
            runs.append(
                forward_backward(emission[first:last], log_initial, np.log(kernels))
            )
        else:
            runs.append(
                forward_backward(
                    emission[first:last], np.log(start), np.log(transition)
                )
            )
    return runs


@pytest.mark.oracle
@pytest.mark.cnamaste
@pytest.mark.usefixtures("_configs")
@pytest.mark.parametrize("phased", [False, True], ids=["nophasing", "phased"])
@pytest.mark.parametrize("seed", [0, 1])
def test_the_evidence_and_posterior_are_sals(phased: bool, seed: int) -> None:
    """Rows 25-26's emission, recursions and posterior against `sal`'s.

    `cnamaste`'s emission (the dense one for `hmm_nophasing`, `port`'s coded
    one for `hmm_phased`), its forward lattice and `get_state_posteriors`,
    on two chains: the evidence summed over chains, and every posterior.
    Phased, with a switch probability drawn per site.
    """
    from cnamaste.hmm_nophasing import hmm_nophasing
    from cnamaste.hmm_phased import hmm_phased
    from scipy.special import logsumexp

    chains = _chains(seed)
    model = (hmm_phased if phased else hmm_nophasing)(params="smp")
    rdr, baf = model.compute_emission_probability_nb_betabinom(
        chains["X"],
        chains["base"],
        chains["log_mu"],
        chains["alphas"],
        chains["total"],
        chains["p_binom"],
        chains["taus"],
    )
    emission = rdr + baf
    log_alpha = model.forward_lattice(
        chains["lengths"],
        chains["log_transmat"],
        chains["log_startprob"],
        emission,
        chains["log_switch"],
    )
    posterior = np.exp(
        model.get_state_posteriors(
            chains["lengths"],
            chains["log_transmat"],
            chains["log_startprob"],
            emission,
            chains["log_switch"],
        )
    )

    runs = _sal(chains, phased=phased)
    ends = np.cumsum(chains["lengths"]) - 1

    ours = float(sum(logsumexp(log_alpha[:, end]) for end in ends))
    theirs = float(sum(float(run.log_evidence) for run in runs))
    assert ours == pytest.approx(theirs, rel=EVIDENCE_TOLERANCE)

    np.testing.assert_allclose(
        posterior.T,
        np.concatenate([np.asarray(run.posterior) for run in runs]),
        rtol=0.0,
        atol=POSTERIOR_TOLERANCE,
    )


# --- the analytic gradient ---------------------------------------------------------


def _gradient_at(shifted: bool) -> tuple[Any, Any, np.ndarray]:
    """`cnamaste`'s gradient mid-EM on `_stacked(per_clone=80)`: posteriors held."""
    from cnamaste.gradient import EmGradient
    from cnamaste.hmm_nophasing import hmm_nophasing

    problem = _stacked(per_clone=80)
    generator = np.random.default_rng(3)
    model = type("shifted", (hmm_nophasing,), {"apply_logmu_shift": shifted})(
        params="smp", t=1 - 1e-4
    )
    model.state_posteriors = generator.dirichlet(
        np.ones(N_STATES), size=problem["X"].shape[0]
    ).T

    settings = dict(problem["settings"])
    settings["shared_NB_dispersion"] = False
    gradient = EmGradient.for_fit(
        model, problem["X"], N_STATES, problem["base"], problem["total"], **settings
    )
    log_startprob, log_mu, p_binom, _, _ = gradient.initial
    x = model.pack_params(
        log_startprob,
        log_mu + generator.normal(0.0, 0.1, log_mu.shape),
        p_binom,
        generator.uniform(0.03, 0.2, (N_STATES, 1)),
        np.full((N_STATES, 1), 35.0),
        **gradient.flags,
    )
    return model, gradient, x


def _cost(model: Any, gradient: Any, x: np.ndarray) -> float:
    """`cost_fn`'s value: `-sum(gamma * emission)` through the coded emission."""
    _, log_mu, p_binom, alphas, taus = model.unpack_params(
        x, gradient.n_states, *gradient.initial, **gradient.flags
    )
    rdr, baf = model.compute_emission_probability_nb_betabinom_coded(
        gradient.nb,
        gradient.bb,
        log_mu,
        alphas,
        p_binom,
        taus,
        normal_log_lambda=gradient.normal_log_lambda,
        clone_lengths=gradient.clone_lengths,
    )
    return float(-np.sum(model.state_posteriors * (rdr + baf)))


@pytest.mark.analytic
@pytest.mark.cnamaste
@pytest.mark.usefixtures("_configs")
@pytest.mark.parametrize("shifted", [False, True], ids=["unshifted", "shifted"])
def test_the_gradient_is_the_cost_functions_derivative(shifted: bool) -> None:
    """Every coordinate against a central difference of `cost_fn`, step 1e-5.

    Shifted, the decode the shift is taken at is the held posteriors'. The
    difference's own error is `O(h^2)` plus round-off over `h`; 2.0e-9 of
    the largest component realized, asserted at 1e-7 of it.
    """
    model, gradient, x = _gradient_at(shifted)
    if shifted:
        assert gradient._shift_inputs() is not None

    step = 1e-5
    differences = np.array(
        [
            (
                _cost(model, gradient, x + step * unit)
                - _cost(model, gradient, x - step * unit)
            )
            / (2.0 * step)
            for unit in np.eye(x.size)
        ]
    )
    closed = gradient(x)

    assert closed.shape == differences.shape
    np.testing.assert_allclose(
        closed, differences, rtol=0.0, atol=1e-7 * np.abs(differences).max()
    )


# --- `port`'s drop-ins, bitwise ----------------------------------------------------


@pytest.mark.patch
@pytest.mark.cnamaste
@pytest.mark.usefixtures("_configs")
@pytest.mark.parametrize("shift", [False, True], ids=["unshifted", "shifted"])
def test_the_fit_is_ports(shift: bool) -> None:
    """`cnamaste.hmm_nophasing`'s fit against `port`'s, bitwise.

    `port`'s under `LOG_SPACE_SWAPS`, the kernels `cnamaste` holds since PR4;
    the shift as `port`'s `SHIFT_SWAPS` row binds it. Every returned array.
    """
    from cnamaste.hmm_nophasing import hmm_nophasing as own
    from port.patch.hmm_nophasing import hmm_nophasing as ports
    from port.pipeline import LOG_SPACE_SWAPS, patched

    problem = _stacked(per_clone=200)
    ours = _fit(own, problem, apply_logmu_shift=shift)
    with patched(LOG_SPACE_SWAPS):
        theirs = _fit(ports, problem, apply_logmu_shift=shift)

    assert sorted(ours) == sorted(theirs)
    for name, value in ours.items():
        np.testing.assert_array_equal(value, theirs[name], err_msg=name)


@pytest.mark.patch
@pytest.mark.cnamaste
@pytest.mark.usefixtures("_configs")
def test_the_phased_coded_emission_is_ports_on_pooled_spots() -> None:
    """`cnamaste.hmm_phased`'s coded emission against `port`'s, three spots, bitwise."""
    from cnamaste.count_encoder import CountEncoder
    from cnamaste.hmm_phased import hmm_phased as own
    from port.patch.hmm_phased import hmm_phased as ports
    from port.pipeline import LOG_SPACE_SWAPS, patched

    arguments = _pooled()
    ours = own.compute_emission_probability_nb_betabinom_coded(
        CountEncoder(*arguments["nb"]),
        CountEncoder(*arguments["bb"]),
        *arguments["params"],
    )
    with patched(LOG_SPACE_SWAPS):
        from cnaster.count_encoder import CountEncoder as Theirs

        theirs = ports.compute_emission_probability_nb_betabinom_coded(
            Theirs(*arguments["nb"]), Theirs(*arguments["bb"]), *arguments["params"]
        )

    for left, right in zip(ours, theirs, strict=True):
        assert left.shape == (2 * 3, 3 * 50)
        np.testing.assert_array_equal(left, right)


def _pooled() -> dict[str, Any]:
    """Three spots of 50 bins and one fitted parameter column, `(3, 1)`."""
    generator = np.random.default_rng(11)
    base = generator.uniform(10.0, 40.0, (50, 3))
    total = generator.integers(0, 30, (50, 3)).astype(np.float64)
    return {
        "nb": (generator.poisson(base).astype(np.float64), base),
        "bb": (generator.binomial(total.astype(int), 0.3).astype(np.float64), total),
        "params": (
            np.array([[-0.3], [0.0], [0.4]]),
            np.array([[0.1], [0.1], [0.1]]),
            np.array([[0.2], [0.35], [0.5]]),
            np.array([[40.0], [40.0], [40.0]]),
        ),
    }


@pytest.mark.bug
@pytest.mark.cnamaste
@pytest.mark.usefixtures("_configs")
def test_cnasters_phased_emission_raises_on_pooled_spots() -> None:
    """#269: `cnaster` indexes the `(3, 1)` parameter by spot; `cnamaste` scores all three."""
    from cnamaste.count_encoder import CountEncoder
    from cnamaste.hmm_phased import _cnaster_hmm_phased, hmm_phased

    arguments = _pooled()

    def encoded() -> tuple[Any, Any]:
        return CountEncoder(*arguments["nb"]), CountEncoder(*arguments["bb"])

    with pytest.raises(IndexError, match="out of bounds"):
        _cnaster_hmm_phased.compute_emission_probability_nb_betabinom_coded(
            *encoded(), *arguments["params"]
        )

    rdr, baf = hmm_phased.compute_emission_probability_nb_betabinom_coded(
        *encoded(), *arguments["params"]
    )
    assert np.isfinite(rdr).all()
    assert np.isfinite(baf).all()


# --- #135 and #244 --------------------------------------------------------------


@pytest.mark.bug
@pytest.mark.cnamaste
@pytest.mark.usefixtures("_configs")
def test_a_tumour_proportion_is_refused_by_the_fit_and_warned_by_the_driver() -> None:
    """#135: `cnaster`'s fit returns the same bytes with a proportion as without.

    `cnamaste`'s `optimize` takes none, and `pipeline_baum_welch` warns that
    the one it is given is not fitted, then fits as without.
    """
    from cnamaste.hmm import pipeline_baum_welch
    from cnamaste.hmm_nophasing import hmm_nophasing
    from cnaster.hmm_nophasing import hmm_nophasing as cnasters

    problem = _stacked(per_clone=80)
    problem["settings"]["max_iter"] = 5
    proportion = np.full((problem["X"].shape[0], 1), 0.3)

    ignored = _fit(cnasters, problem)
    problem["settings"]["tumor_prop"] = proportion
    with_proportion = _fit(cnasters, problem)
    for name in FITTED:
        np.testing.assert_array_equal(with_proportion[name], ignored[name])

    with pytest.raises(TypeError, match="tumor_prop"):
        _fit(hmm_nophasing, problem)
    del problem["settings"]["tumor_prop"]

    settings = {
        key: problem["settings"][key]
        for key in ("init_log_mu", "init_p_binom", "max_iter", "normal_lambda")
    }

    def drive(tumor_prop: np.ndarray | None) -> Any:
        return pipeline_baum_welch(
            None,
            problem["X"],
            problem["lengths"],
            N_STATES,
            problem["base"],
            problem["total"],
            problem["settings"]["log_sitewise_transmat"],
            tumor_prop,
            hmmclass=hmm_nophasing,
            params="smp",
            clone_lengths=problem["lengths"],
            **settings,
        )

    with pytest.warns(UserWarning, match="#135"):
        warned = drive(proportion)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        plain = drive(None)

    np.testing.assert_array_equal(warned.params.new_log_mu, plain.params.new_log_mu)


@pytest.mark.bug
@pytest.mark.cnamaste
@pytest.mark.usefixtures("_configs")
def test_a_kernel_change_at_round_off_moves_the_differenced_fit_and_not_the_analytic_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#244: the finite-difference M step amplifies round-off; the analytic one does not.

    `cnaster`'s per-bin kernels in place of the log-space ones `cnamaste`
    holds (at most 3.9e-10 relative apart, PR4), on `_stacked()`. With
    differences (`analytic_gradient=False`, `cnaster`'s M step) the fitted
    `log_mu` moves by 4.5e-4 relative; with the analytic gradient by 5.1e-12.
    Asserted at above 1e-5 and below 1e-9.
    """
    import cnamaste.hmm_nophasing as module
    from cnamaste.hmm_nophasing import hmm_nophasing
    from cnaster import hmm_nophasing as cnasters

    problem = _stacked()

    def both(analytic: bool) -> dict[str, float]:
        own = _fit(hmm_nophasing, problem, analytic_gradient=analytic)
        with monkeypatch.context() as kernels:
            kernels.setattr(module, "_nb_logpmf_1d", cnasters._nb_logpmf_1d)
            kernels.setattr(module, "_bb_logpmf_1d", cnasters._bb_logpmf_1d)
            other = _fit(hmm_nophasing, problem, analytic_gradient=analytic)
        return _moved(own, other)

    differenced, analytic = both(analytic=False), both(analytic=True)

    assert differenced["new_log_mu"] > 1e-5
    assert max(analytic.values()) < 1e-9
