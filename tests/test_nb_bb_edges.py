"""#560 on the live path: every port site that scores the negative binomial, at a vanishing mean.

`p = 1 / (1 + a)`, `a = alpha * mean`, rounds to exactly 1 in float64 below
`a` of about 1.1e-16. `cnaster`'s kernel then scores every count 0; port's
NumPy and `jax` copies scored a count of 0 NaN and any other `-inf`, and lost
digits of `log(1 - p)` below `a` of about 1e-4. Each site `run_cnaster_port
--sal` reaches is judged here against `tests.exact_densities`, the density
as 50-digit sums of logs:

- the HMM emission, `LOG_SPACE_SWAPS`' rows wherever `cnaster` binds them,
  and `dense_emission.nb_states` (sal's kernel) under `--sal`;
- the clone field, which compiles its kernel in and takes `log_space`;
- the M-step gradient, the copy decode and the `--copy-errors` `jax` model.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.special import logsumexp

from tests.exact_densities import nb_logpmf, nb_partials
from tests.fixtures import SpotCloneField, spot_clone_field

COUNTS = np.array([0, 1, 7, 42, 300, 1000, 2500], dtype=np.float64)
VANISHING = 1e-17
"""A mean at which `alpha * mean = 1e-19` for `alpha = 0.01`: `p` rounds to 1."""

DEGENERATE = (-43.22, 0.1184, 1000.0, 1000.0)
"""`(log mu, alpha, exposure, count)` of dev_tree_1s_hard r0's degenerate state (#560)."""

Scorer = Callable[[np.ndarray, float, float], np.ndarray]


def _exact(counts: np.ndarray, mean: float, alpha: float) -> np.ndarray:
    return np.array([nb_logpmf(int(k), mean, alpha) for k in counts])


def _copy_likelihood(counts: np.ndarray, mean: float, alpha: float) -> np.ndarray:
    """The decode's emission with no trials, so the allele channel adds 0."""
    from port.extensions.copy_likelihood import Pseudobulk, _emission

    zeros = np.zeros_like(counts)
    bulk = Pseudobulk(
        counts_nb=counts,
        base_nb_mean=np.ones_like(counts),
        counts_bb=zeros,
        total_bb_RD=zeros,
        normal_log_lambda=zeros,
        dispersion=alpha,
        taus=np.inf,
    )
    bins = np.arange(counts.size)
    return np.asarray(_emission(np.log(mean), np.array(0.5), bulk, bins))


def _jax(counts: np.ndarray, mean: float, alpha: float) -> np.ndarray:
    """`jax_hmm.emission` with no trials, so the beta-binomial adds 0 to within 1e-15."""
    from port.extensions.jax_hmm import emission

    zeros = np.zeros_like(counts)
    scores = emission(
        np.array([[np.log(mean)]]),
        np.array([[alpha]]),
        np.array([[0.5]]),
        np.array([[20.0]]),
        counts,
        np.ones_like(counts),
        zeros,
        zeros,
    )
    return np.asarray(np.asarray(scores)[0])


def _row(counts: np.ndarray, mean: float, alpha: float) -> np.ndarray:
    """What `LOG_SPACE_SWAPS` installs over `cnaster.hmm_nophasing._nb_logpmf_1d`."""
    from port.patch.hmm_nophasing.nb_logpmf import _nb_logpmf_1d

    out = np.full(counts.size, np.nan)
    _nb_logpmf_1d(counts, np.ones(counts.size), mean, alpha, out)
    return out


def _sal(counts: np.ndarray, mean: float, alpha: float) -> np.ndarray:
    """`dense_emission.nb_states`, sal's Rust kernel, the HMM's under `--sal`."""
    from port.patch.hmm_nophasing.dense_emission import nb_states

    return np.asarray(
        nb_states(counts, np.ones(counts.size), np.array([mean]), np.array([alpha]))[0]
    )


KERNELS: dict[str, Scorer] = {
    "row": _row,
    "sal": _sal,
    "copy_likelihood": _copy_likelihood,
    "jax_hmm": _jax,
}

NORMAL = [
    (mean, alpha) for mean in (0.5, 30.0, 1000.0) for alpha in (1e-3, 0.07, 0.5, 3.0)
] + [(1e-10, 0.01)]
"""Realistic means and dispersions, and `a = 1e-12`, where forming `p` lost 0.089 nats at a count of 1000."""


@pytest.mark.oracle
@pytest.mark.parametrize("name", KERNELS)
@pytest.mark.parametrize(("mean", "alpha"), NORMAL)
def test_each_live_kernel_is_the_negative_binomial(
    name: str, mean: float, alpha: float
) -> None:
    """The 50-digit sums of logs, to 1e-9 relative and 1e-12 absolute."""
    np.testing.assert_allclose(
        KERNELS[name](COUNTS, mean, alpha),
        _exact(COUNTS, mean, alpha),
        rtol=1e-9,
        atol=1e-12,
    )


@pytest.mark.oracle
@pytest.mark.parametrize("name", KERNELS)
def test_a_vanishing_mean_scores_a_large_count_as_impossible(name: str) -> None:
    """At `alpha * mean = 1e-19` a count of 1000 is -43,420 nats, and 0 is finite; 1e-9 relative.

    The old NumPy form scored the count `-inf` and 0 NaN; `cnaster`'s scores
    both 0.
    """
    counts = np.array([0.0, 1000.0])
    scores = KERNELS[name](counts, VANISHING, 0.01)

    assert scores[1] < -30_000.0
    np.testing.assert_allclose(
        scores, _exact(counts, VANISHING, 0.01), rtol=1e-9, atol=1e-12
    )


@pytest.mark.analytic
@pytest.mark.parametrize("name", KERNELS)
@pytest.mark.parametrize("mean", [1e-3, VANISHING])
def test_each_live_kernel_sums_to_one_at_a_small_mean(name: str, mean: float) -> None:
    """`logsumexp` over counts 0..200 is 0 to 1e-12: a pmf, not a score of 1 per count."""
    counts = np.arange(201, dtype=np.float64)

    assert abs(logsumexp(KERNELS[name](counts, mean, 0.5))) < 1e-12


@pytest.mark.oracle
@pytest.mark.parametrize(("mean", "alpha"), [*NORMAL, (VANISHING, 0.01)])
def test_the_closed_form_gradient_is_the_exact_derivative(
    mean: float, alpha: float
) -> None:
    """`nb_partials` against 50-digit central differences, to 1e-9 relative, 1e-9 absolute.

    At the vanishing mean `d ell / d log mean` is the count, 1000 for 1000,
    where the old gradient read 0 because `cnaster`'s score does not move.
    """
    from port.patch.hmm_nophasing.gradient import nb_partials as closed_form

    d_eta, d_alpha = closed_form(COUNTS, np.full(COUNTS.shape, mean), np.array(alpha))
    exact = np.array([nb_partials(int(k), mean, alpha) for k in COUNTS])

    np.testing.assert_allclose(d_eta, exact[:, 0], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(d_alpha, exact[:, 1], rtol=1e-9, atol=1e-9)


# --- the field -------------------------------------------------------------


def _degenerate_field(kernel: Any, *, log_space: bool) -> float:
    """One bin, one spot, one clone; no allele trials, so the field is the NB score."""
    log_mu, alpha, exposure, count = DEGENERATE
    field = kernel(
        np.full((1, 1), count),
        np.full((1, 1), exposure),
        np.zeros((1, 1)),
        np.zeros((1, 1)),
        np.array([log_mu]),
        np.array([alpha]),
        np.array([0.5]),
        np.array([30.0]),
        np.zeros((1, 1), dtype=np.int64),
        np.ones(1),
        np.zeros((1, 1)),
        log_space=log_space,
    )
    return float(field[0, 0])


@pytest.mark.oracle
@pytest.mark.parametrize("name", ["fused", "tabulated"])
def test_the_field_scores_the_degenerate_state_exactly_in_log_space(name: str) -> None:
    """At `log mu = -43.22`, exposure 1000, a count of 1000 is -38,403.9 nats, to 1e-9 relative.

    Without `log_space` the field is `cnaster`'s, bitwise, and scores it 0 --
    probability 1 -- because `p` rounds to 1 (`a = 2.0e-17`).
    """
    from port.patch.hmrf.fused_field import fused_spot_clone_field
    from port.patch.hmrf.tabulated_field import tabulated_spot_clone_field

    kernel = {"fused": fused_spot_clone_field, "tabulated": tabulated_spot_clone_field}
    log_mu, alpha, exposure, count = DEGENERATE
    exact = nb_logpmf(int(count), exposure * float(np.exp(log_mu)), alpha)

    assert _degenerate_field(kernel[name], log_space=False) == 0.0
    np.testing.assert_allclose(
        _degenerate_field(kernel[name], log_space=True), exact, rtol=1e-9
    )


def _two_step_under_the_table(fixture: SpotCloneField) -> np.ndarray:
    """`cnaster`'s producer under `LOG_SPACE_SWAPS`, then the field: what the fused kernel replaces."""
    import cnaster.hmm_nophasing as upstream
    from port.patch.hmrf.field import compute_loglike_spot_assignment_strided
    from port.pipeline import LOG_SPACE_SWAPS, patched
    from scipy.sparse import eye as sparse_eye

    with patched(LOG_SPACE_SWAPS):
        rdr = upstream._dense_nb_logpmf(
            fixture.counts_nb,
            fixture.base_nb_mean,
            fixture.log_mu[:, None],
            fixture.alphas[:, None],
        )
        baf = upstream._dense_bb_logpmf(
            fixture.counts_bb,
            fixture.total_bb_RD,
            fixture.p_binom[:, None],
            fixture.taus[:, None],
        )

    smooth = sparse_eye(fixture.n_spots, format="csr")
    field: np.ndarray = compute_loglike_spot_assignment_strided(
        fixture.n_spots,
        np.ones(fixture.n_spots),
        np.ones(fixture.n_spots),
        np.empty(0),
        False,
        rdr,
        baf,
        fixture.pred,
        fixture.n_obs,
        fixture.n_clones,
        smooth_indices=smooth.indices,
        smooth_indptr=smooth.indptr,
    )
    return field


@pytest.mark.patch
@pytest.mark.parametrize(("n_states", "n_clones"), [(5, 5), (3, 1)])
def test_the_log_space_field_is_bitwise_the_two_step_under_the_table(
    n_states: int, n_clones: int
) -> None:
    """Fused and tabulated with `log_space=True`, against `cnaster`'s two steps under the rows: identical.

    The chain the default field keeps with `cnaster`, kept with the table.
    """
    from port.patch.hmrf.fused_field import fused_spot_clone_field
    from port.patch.hmrf.tabulated_field import tabulated_spot_clone_field

    fixture = spot_clone_field(n_states=n_states, n_clones=n_clones)
    fixture.taus[0] = 1e13
    arguments = (
        fixture.counts_nb,
        fixture.base_nb_mean,
        fixture.counts_bb,
        fixture.total_bb_RD,
        fixture.log_mu,
        fixture.alphas,
        fixture.p_binom,
        fixture.taus,
        fixture.pred,
        np.ones(fixture.n_spots),
    )
    shape = (fixture.n_spots, fixture.n_clones)
    fused: Any = fused_spot_clone_field
    tabulated: Any = tabulated_spot_clone_field
    reference = _two_step_under_the_table(fixture)

    np.testing.assert_array_equal(
        fused(*arguments, np.empty(shape), log_space=True), reference
    )
    np.testing.assert_array_equal(
        tabulated(*arguments, np.empty(shape), log_space=True), reference
    )


# --- installation ----------------------------------------------------------


@pytest.mark.infra
def test_the_rows_reach_every_binding_called_from_python_and_restore_it() -> None:
    """`cnaster`'s two modules and port's two Python callers; not the compiled field.

    The field imports `cnaster`'s kernels under other names, so a first
    compile under the table cannot cache the table's kernel into it.
    """
    # NB a site is a module once imported; a run imports both.
    import port.patch.hmm_nophasing.shifted_emission
    import port.patch.hmm_phased.coded_emission  # noqa: F401
    from port.pipeline import LOG_SPACE_SWAPS, patched, swap_sites

    reached = {site.module for site in swap_sites(LOG_SPACE_SWAPS)}

    assert {
        "cnaster.hmm_nophasing",
        "cnaster.hmm_phased",
        "port.patch.hmm_nophasing.shifted_emission",
        "port.patch.hmm_phased.coded_emission",
    } <= reached
    assert "port.patch.hmrf.fused_field" not in reached

    import cnaster.hmm_phased as phased
    from port.patch.hmm_nophasing import bb_logpmf, nb_logpmf

    before = (phased._nb_logpmf_1d, phased._bb_logpmf_1d)
    with patched(LOG_SPACE_SWAPS):
        assert phased._nb_logpmf_1d is nb_logpmf._nb_logpmf_1d
        assert phased._bb_logpmf_1d is bb_logpmf._bb_logpmf_1d
    assert (phased._nb_logpmf_1d, phased._bb_logpmf_1d) == before


@pytest.mark.infra
@pytest.mark.parametrize(
    ("argv", "expected"),
    [([], True), (["--sal"], True), (["--no-shift", "--no-copy-cap"], False)],
    ids=["default", "sal", "no-shift"],
)
def test_run_cnaster_installs_the_rows_and_the_field_option_with_the_shift(
    argv: list[str],
    expected: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both kernels rebound and `log_space` bound into the clone assignment, or neither."""
    import cnaster.hmm_nophasing
    import cnaster.hmrf
    import cnaster.scripts.run_cnaster as pipeline
    from port.patch.hmm_nophasing import bb_logpmf, nb_logpmf
    from port.scripts.run_cnaster import main

    config = tmp_path / "config.yaml"
    config.write_text("{}\n")
    seen: list[tuple[bool, bool, bool]] = []
    monkeypatch.setattr(
        pipeline,
        "run_cnaster",
        lambda *_: seen.append(
            (
                cnaster.hmm_nophasing._nb_logpmf_1d is nb_logpmf._nb_logpmf_1d,
                cnaster.hmm_nophasing._bb_logpmf_1d is bb_logpmf._bb_logpmf_1d,
                getattr(cnaster.hmrf.pipeline_clone_assignment, "keywords", {}).get(
                    "log_space", False
                ),
            )
        ),
    )
    main([*argv, "--no-rust", str(config)])

    assert seen == [(expected, expected, expected)]


# --- cnaster ---------------------------------------------------------------


@pytest.mark.bug
def test_cnasters_kernel_scores_every_count_zero_below_the_dispersion_floor() -> None:
    """At `alpha = 1e-17`, `mu = 10`, counts 0 and 1000 score 0; exact: -10.0 and -3,619.5.

    The #560 defect reached through `alpha` rather than the mean: `r` is
    floored at `1 / 1e-10` but `p = 1 / (1 + alpha * lambda)` takes the raw
    `alpha` and rounds to 1. `cnaster` passes `bounds=None` to BFGS, so
    `log alpha` is unbounded. The row floors `alpha` in both.
    """
    from cnaster.hmm_nophasing import _nb_logpmf_1d

    out = np.full(2, np.nan)
    _nb_logpmf_1d(np.array([0.0, 1000.0]), np.ones(2), 10.0, 1e-17, out)

    np.testing.assert_array_equal(out, [0.0, 0.0])


# --- sandbox ---------------------------------------------------------------
# NB `port.sandbox`'s own copies of the negative binomial (#540): no console
#    script reaches them, so they are judged here beside the live sites rather
#    than among them.


def _clone_mixture(counts: np.ndarray, mean: float, alpha: float) -> np.ndarray:
    from port.sandbox.admixture.clone_mixture import _nb

    return _nb(counts, np.full(counts.shape, mean), alpha)


def _variants(counts: np.ndarray, mean: float, alpha: float) -> np.ndarray:
    from port.sandbox.admixture.variants import _nb

    return _nb(counts, np.full(counts.shape, mean), alpha)


SANDBOX_KERNELS: dict[str, Scorer] = {
    "clone_mixture": _clone_mixture,
    "variants": _variants,
}


@pytest.mark.oracle
@pytest.mark.parametrize("name", SANDBOX_KERNELS)
@pytest.mark.parametrize(("mean", "alpha"), NORMAL)
def test_each_sandbox_kernel_is_the_negative_binomial(
    name: str, mean: float, alpha: float
) -> None:
    """The 50-digit sums of logs, to 1e-9 relative and 1e-12 absolute."""
    np.testing.assert_allclose(
        SANDBOX_KERNELS[name](COUNTS, mean, alpha),
        _exact(COUNTS, mean, alpha),
        rtol=1e-9,
        atol=1e-12,
    )


@pytest.mark.oracle
@pytest.mark.parametrize("name", SANDBOX_KERNELS)
def test_a_sandbox_kernel_scores_a_vanishing_mean_as_impossible(name: str) -> None:
    """At `alpha * mean = 1e-19` a count of 1000 is -43,420 nats, and 0 is finite; 1e-9 relative."""
    counts = np.array([0.0, 1000.0])
    scores = SANDBOX_KERNELS[name](counts, VANISHING, 0.01)

    assert scores[1] < -30_000.0
    np.testing.assert_allclose(
        scores, _exact(counts, VANISHING, 0.01), rtol=1e-9, atol=1e-12
    )


@pytest.mark.analytic
@pytest.mark.parametrize("name", SANDBOX_KERNELS)
@pytest.mark.parametrize("mean", [1e-3, VANISHING])
def test_each_sandbox_kernel_sums_to_one_at_a_small_mean(
    name: str, mean: float
) -> None:
    """`logsumexp` over counts 0..200 is 0 to 1e-12: a pmf, not a score of 1 per count."""
    counts = np.arange(201, dtype=np.float64)

    assert abs(logsumexp(SANDBOX_KERNELS[name](counts, mean, 0.5))) < 1e-12


def _bound_kernel(module: str) -> float:
    """The `_nb_logpmf_1d` `module` compiles in by name, on one degenerate bin."""
    import importlib

    log_mu, alpha, exposure, count = DEGENERATE
    out = np.full(1, np.nan)
    importlib.import_module(module)._nb_logpmf_1d(
        np.array([count]), np.array([exposure]), float(np.exp(log_mu)), alpha, out
    )
    return float(out[0])


def _np_merge() -> float:
    from port.sandbox.np_merge import _emissions

    log_mu, alpha, exposure, count = DEGENERATE
    X = np.zeros((1, 2, 1))
    X[0, 0, 0] = count
    res = {
        "new_log_mu": [log_mu],
        "new_alphas": [alpha],
        "new_p_binom": [0.5],
        "new_taus": [30.0],
    }
    rdr, _ = _emissions(
        X, np.full((1, 1), exposure), np.zeros((1, 1)), res, np.zeros(1)
    )
    return float(rdr[0, 0, 0])


SANDBOX_SITES: dict[str, Callable[[], float]] = {
    "np_merge": _np_merge,
    "hmm_initialize_backends": lambda: _bound_kernel(
        "port.sandbox.patch.hmm_initialize.backends"
    ),
}
"""The sandbox sites that compile `_nb_logpmf_1d` in by name, out of `LOG_SPACE_SWAPS`' reach."""


@pytest.mark.oracle
@pytest.mark.parametrize("site", sorted(SANDBOX_SITES))
def test_every_sandbox_site_scores_the_degenerate_state_in_log_space(
    site: str,
) -> None:
    """At `log mu = -43.22`, `alpha = 0.1184`, exposure 1000, a count of 1000 is -38,403.9 nats, to 1e-9.

    `cnaster`'s kernel scores it 0 -- probability 1 -- because `p` rounds to
    1 (`a = 2.0e-17`); each site imports the log-space kernel (#560) instead.
    """
    log_mu, alpha, exposure, count = DEGENERATE
    score = SANDBOX_SITES[site]()

    assert score < -30_000.0
    np.testing.assert_allclose(
        score, nb_logpmf(int(count), exposure * float(np.exp(log_mu)), alpha), rtol=1e-9
    )
