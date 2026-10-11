"""Per-function rows for normal spots: the candidates, the baseline, the BAF bin filter, the expression filter, and the beta-binomial fit they use.

Inputs: the normal stages' recorded inputs (replayed where they are counts);
else a synthetic layout with a planted answer.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
import scipy.optimize
import scipy.stats
from cnamaste.config import set_global_config
from cnamaste.hmm_emission import OptimizationResult, Weighted_BetaBinom, Weighted_BetaBinom_mix, betabinom_logpmf, betabinom_logpmf_zp, compute_bb_ab, get_betabinom_start_params
from cnamaste.hmm_utils import get_em_solver_params, get_solver
from cnamaste.normal_spot import determine_normal_baseline, determine_normal_candidates, filter_normal_diffexp, normal_baf_bin_filter

from audit.fn import Replay, Row, deep, run, table, unchanged


def beta_draws(n: int = 800, p: float = 0.3, tau: float = 50.0, seed: int = 21) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    depth = rng.integers(20, 80, size=n)
    return {"k": rng.binomial(depth, rng.beta(p * tau, (1 - p) * tau, size=n)).astype(float), "n": depth.astype(float), "p": p, "tau": tau}


def mle(d: dict[str, np.ndarray]) -> np.ndarray:
    """The beta-binomial MLE by scipy alone: Nelder-Mead on log-odds and log-tau, tight tolerances."""
    def nll(x: np.ndarray) -> float:
        p, tau = 1 / (1 + np.exp(-x[0])), np.exp(x[1])
        return -float(np.sum(scipy.stats.betabinom.logpmf(d["k"], d["n"], p * tau, (1 - p) * tau)))
    r = scipy.optimize.minimize(nll, [0.0, np.log(100.0)], method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 20_000})
    return np.array([1 / (1 + np.exp(-r.x[0])), np.exp(r.x[1])])


def model_of(d: dict[str, np.ndarray]) -> Any:
    return Weighted_BetaBinom(d["k"], np.ones(d["k"].size), weights=np.ones(d["k"].size), exposure=d["n"])


# --- oracle --------------------------------------------------------------------


def _nll(d: dict[str, np.ndarray]) -> None:
    model = model_of(d)
    params = np.array([0.35, 70.0])
    want = -np.sum(scipy.stats.betabinom.logpmf(d["k"], d["n"], 0.35 * 70, 0.65 * 70))
    assert np.isclose(model.nloglikeobs(params), want), "scipy path (no zero point)"
    model.zero_point = betabinom_logpmf_zp(model.endog, model.exposure)
    assert np.isclose(model.nloglikeobs(params), want), "numba path with the binomial-coefficient zero point"


def _kernel(_: Any) -> None:
    k, n = np.array([0.0, 3.0, 10.0]), np.array([10.0, 10.0, 10.0])
    exog = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    a, b = compute_bb_ab(exog, np.array([0.2, 0.6, 30.0, 5.0]))
    assert np.allclose(a, [6.0, 3.0, 6.0]) and np.allclose(b, [24.0, 2.0, 24.0]), "a = p tau, b = (1-p) tau per row's state"
    found = betabinom_logpmf(k, n, a, b, betabinom_logpmf_zp(k, n))
    assert np.allclose(found, scipy.stats.betabinom.logpmf(k, n, a, b))


def _fit_recovers(d: dict[str, np.ndarray]) -> None:
    fit = model_of(d).fit(**get_em_solver_params())
    want = mle(d)
    assert abs(fit.params[0] - want[0]) < 0.005, f"p {fit.params[0]:.4f} against scipy's {want[0]:.4f}"
    assert abs(np.log(fit.params[1] / want[1])) < 0.2, f"tau {fit.params[1]:.1f} against scipy's MLE {want[1]:.1f}"


def _candidates(ctx: Any) -> dict[str, Any]:
    return {"in": ctx("06_normal/determine_normal_candidates/in"), "out": np.asarray(ctx.sim.stored("06_normal/determine_normal_candidates/out"))}


def _candidates_recomputed(d: dict[str, Any]) -> None:
    config, res, profiles, x, rdr, smooth = d["in"]["args"]
    labels = np.asarray(res["new_assignment"])
    clone = np.argmin(np.sum(np.maximum(np.abs(profiles - 0.5) - 0.05, 0), axis=1))
    stds = np.std(np.log1p(rdr @ smooth), axis=0)
    prior, percent = np.inf, 40
    while True:
        threshold = np.percentile(stds[labels == clone], percent)
        want = (stds < threshold) & (labels == clone)
        if threshold > 1.5 * prior or percent == 100:
            break
        percent += 10
    assert np.array_equal(want, d["out"]), "the spots below the 40th percentile of log1p-RDR spread in the most balanced BAF clone"
    assert np.array_equal(unchanged(determine_normal_candidates, config, res, profiles, x, rdr, smooth), d["out"])


def _baseline(ctx: Any) -> dict[str, Any]:
    given = ctx("07_rebin/determine_normal_baseline/in")
    return {"rdr": given["args"][0], "normal": np.asarray(given["args"][1]), "floor": ctx.config.quality.min_normal_count_perbin}


def _baseline_recomputed(d: dict[str, Any]) -> None:
    rdr = d["rdr"].copy()
    low = rdr[:, d["normal"]].sum(axis=1) < d["floor"]
    lam = np.where(low, 0, rdr[:, d["normal"]].sum(axis=1)).astype(float)
    lam /= lam.sum()
    zeroed = np.where(low[:, None], 0, rdr)
    rdr_normal, out_rdr, base = determine_normal_baseline(rdr, d["normal"])
    assert np.allclose(rdr_normal, lam) and np.array_equal(out_rdr, zeroed) and np.allclose(base, np.outer(lam, zeroed.sum(axis=0)))
    assert out_rdr is rdr, "it zeroes the low rows of the array it is handed, in place, and returns it"


def _diffexp(_: Any) -> None:
    rng = np.random.default_rng(31)
    n = 200
    normal = np.arange(n) < 100
    counts = pd.DataFrame({f"g{i}": rng.poisson(20, size=n) for i in range(6)}, index=[f"s{i}" for i in range(n)])
    counts["up"] = np.where(normal, rng.poisson(5, size=n), rng.poisson(400, size=n))
    table_ = pd.DataFrame({"INCLUDED_GENES": [f"g{i}" for i in range(6)] + ["up"]})
    out = unchanged(filter_normal_diffexp, counts, table_, normal, use_kmeans=False)
    assert out.shape == (7, n)
    assert np.array_equal(out[:6], counts.iloc[:, :6].to_numpy().T), "a flat gene's bin keeps every UMI"
    assert not out[6].any(), "a 2^6-fold tumour gene above the UMI quantile leaves its bin"


def _bin_filter(_: Any) -> None:
    rng = np.random.default_rng(41)
    n_bins, n_spots = 12, 40
    total = rng.integers(30, 60, size=(n_bins, n_spots))
    p = np.full(n_bins, 0.5)
    p[4] = 0.1  # NB allele-specific in the normal spots
    x = np.stack([rng.poisson(50, size=(n_bins, n_spots)), rng.binomial(total, p[:, None])], axis=1)
    frame = pd.DataFrame({"CHR": np.repeat([1, 2], [8, 4]), "START": np.arange(n_bins), "END": np.arange(n_bins) + 1, "bin_id": np.arange(n_bins)})
    out_frame, counts = normal_baf_bin_filter(deep(frame), x, np.zeros((n_bins, n_spots)), total, 1.0, -2.0, np.arange(20), None, confidence_interval=(0.01, 0.99))
    kept = np.delete(np.arange(n_bins), 4)
    assert np.array_equal(counts.X, x[kept]) and np.array_equal(counts.total_bb_RD, total[kept]) and counts.lengths.tolist() == [7, 4]
    assert out_frame["bin_id"].isna().tolist() == [i == 4 for i in range(n_bins)]
    assert out_frame["bin_id"].dropna().tolist() == list(range(11)), "kept bins re-ranked 0..K'-1"


ORACLE: list[Row] = table(
    "oracle",
    ("hmm_emission:Weighted_BetaBinom_mix.nloglikeobs", "synthetic: 800 beta-binomial draws", lambda c: beta_draws(), _nll, "the negative log-likelihood is scipy's, with and without the zero point"),
    ("hmm_emission:nloglikeobs_bb", "synthetic: 800 beta-binomial draws", lambda c: beta_draws(), _nll, "scipy's betabinom.logpmf summed, negated"),
    ("hmm_emission:betabinom_logpmf", "synthetic: 3 counts", lambda c: None, _kernel, "zero point + lgamma terms is scipy's log-pmf"),
    ("hmm_emission:betabinom_logpmf_zp", "synthetic: 3 counts", lambda c: None, _kernel, "the log binomial coefficient completes scipy's log-pmf"),
    ("hmm_emission:compute_bb_ab", "synthetic: 2 states", lambda c: None, _kernel, "a = p tau, b = (1 - p) tau for each row's state"),
    ("hmm_emission:Weighted_BetaBinom_mix.fit", "synthetic: 800 draws, p 0.3, tau 50", lambda c: beta_draws(), _fit_recovers, "the fit is scipy's MLE: p to 0.005, tau to 20%",
     "Ticket#30: The emission M step stops short of its maximum and reports that it converged (tau stays at its start, 1000, against an MLE near 50)"),
    ("normal_spot:determine_normal_candidates", Replay("06_normal/determine_normal_candidates in"), _candidates, _candidates_recomputed, "recomputed from its input: the low-spread spots of the most balanced clone; no input mutation"),
    ("normal_spot:determine_normal_baseline", Replay("07_rebin/determine_normal_baseline in"), _baseline, _baseline_recomputed, "lambda = normal sums floored and normalized; base = lambda x spot totals"),
    ("normal_spot:filter_normal_diffexp", "synthetic: 200 spots, 6 flat genes and one tumour gene", lambda c: None, _diffexp, "a flat gene's bin keeps its UMIs; a strongly tumour-up gene's bin loses them"),
    ("normal_spot:normal_baf_bin_filter", "synthetic: 12 bins, one allele-specific in the normal spots", lambda c: None, _bin_filter, "removes exactly the planted allele-specific bin; re-ranks the rest; lengths per contig"),
)


# --- invariants ---------------------------------------------------------------------


def _solver(ctx: Any) -> Any:
    return ctx.config


def _solver_params(config: Any) -> None:
    assert get_solver() == config.hmm.solver == "L-BFGS-B"
    assert get_em_solver_params() == {"maxiter": float(config.hmm.em_maxiter), "ftol": float(config.hmm.em_ftol), "disp": float(config.hmm.em_disp)}
    config.hmm.solver = "Powell"
    set_global_config(config)
    with pytest.raises(AssertionError):
        get_solver()
    with pytest.raises(ValueError):
        get_em_solver_params()


def _defaults(_: Any) -> None:
    ps, disp = get_betabinom_start_params()
    assert ps == [0.5] * 7 and disp == 1000.0
    m = Weighted_BetaBinom_mix(np.ones(4), np.ones(4), np.ones(4), np.ones(4) * 3)
    assert m.get_default_params().tolist() == [0.5, 1000.0] and len(m.get_bounds()) == 2
    m2 = Weighted_BetaBinom_mix(np.ones(4), np.eye(4)[:, :2], np.ones(4), np.ones(4) * 3, shared_dispersion=False)
    assert m2.num_states == 2 and m2.get_default_params().tolist() == [0.5, 0.5, 1000.0, 1000.0] and len(m2.get_bounds()) == 4
    assert m2.exog.ndim == 2 and m.exog.shape == (4, 1), "a 1-d exog becomes one column"


def _result(_: Any) -> None:
    r = OptimizationResult("L-BFGS-B", np.ones(2), -1.0, True, 7, 9)
    assert r.mle_retvals == {"converged": True, "iterations": 7, "fcalls": 9} and r.mle_settings == {"optimizer": "L-BFGS-B"}


def _perf(d: dict[str, Any]) -> None:
    configured = d["tmp"] / "configured.perf"
    d["config"].paths.perf_path = str(configured)
    cwd = os.getcwd()
    os.chdir(d["tmp"])
    try:
        model_of(beta_draws(100)).fit(**get_em_solver_params())
    finally:
        os.chdir(cwd)
    assert configured.exists() and not (d["tmp"] / "cnamaste.perf").exists(), "the timing row goes to paths.perf_path"


def _options(_: Any) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model_of(beta_draws(100)).fit(**get_em_solver_params())
    assert not [w for w in caught if "Unknown solver options" in str(w.message)], "scipy drops an option the fit sends"


def _bin_filter_pure(_: Any) -> None:
    rng = np.random.default_rng(41)
    total = rng.integers(30, 60, size=(6, 10))
    x = np.stack([rng.poisson(50, size=(6, 10)), rng.binomial(total, 0.5)], axis=1)
    frame = pd.DataFrame({"CHR": [1] * 6, "START": np.arange(6), "END": np.arange(6) + 1, "bin_id": np.arange(6)})
    unchanged(normal_baf_bin_filter, frame, x, np.zeros((6, 10)), total, 1.0, -2.0, np.arange(10), None, confidence_interval=(0.0, 1.0))


INVARIANT: list[Row] = table(
    "invariant",
    ("hmm_utils:get_solver", "the staged configuration", _solver, _solver_params, "the configured solver; an unknown one is refused"),
    ("hmm_utils:get_em_solver_params", "the staged configuration", _solver, _solver_params, "L-BFGS-B's keys from hmm.em_*, as floats; an unknown solver raises"),
    ("hmm_emission:get_betabinom_start_params", "the staged configuration", lambda c: None, _defaults, "betabinom.start_params and start_disp, parsed"),
    ("hmm_emission:Weighted_BetaBinom_mix.get_default_params", "the staged configuration", lambda c: None, _defaults, "one p per state and one shared (or per-state) dispersion"),
    ("hmm_emission:Weighted_BetaBinom_mix.get_bounds", "synthetic", lambda c: None, _defaults, "one bound per parameter"),
    ("hmm_emission:Weighted_BetaBinom_mix.__init__", "synthetic", lambda c: None, _defaults, "a 1-d exog is one state column"),
    ("hmm_emission:OptimizationResult.__post_init__", "synthetic", lambda c: None, _result, "statsmodels-style mle_retvals and mle_settings"),
    ("hmm_emission:flush_perf", "synthetic fit in a scratch directory", lambda c: {"tmp": c.tmp_path, "config": c.config}, _perf, "the fit's timing row goes to paths.perf_path",
     "Ticket#46: Weighted_BetaBinom_mix.fit writes to the working directory, and sends scipy an option it rejects"),
    ("hmm_emission:Weighted_BetaBinom_mix.fit", "synthetic: 100 draws", lambda c: None, _options, "every option sent to scipy is one it takes",
     "Ticket#46: Weighted_BetaBinom_mix.fit writes to the working directory, and sends scipy an option it rejects"),
    ("normal_spot:normal_baf_bin_filter", "synthetic: 6 bins", lambda c: None, _bin_filter_pure, "the caller's gene table is not rewritten",
     "new: normal_baf_bin_filter rewrites its caller's df_gene_snp (bin_id set to None, re-mapped, cast) in place"),
)


# --- captured -------------------------------------------------------------------------


def _diffexp_captured(ctx: Any) -> tuple[np.ndarray, np.ndarray]:
    return ctx.replayed.run("06_normal/filter_normal_diffexp"), ctx("06_normal/normal_baf_bin_filter/out/1/X")[:, 0, :]


def _diffexp_drops(pair: tuple[np.ndarray, np.ndarray]) -> None:
    filtered, unfiltered = pair
    assert filtered.shape == unfiltered.shape and np.all(filtered <= unfiltered + 1e-9), "the filter only removes UMIs"
    single = np.isclose(filtered, unfiltered).all(axis=1)
    assert 0 < single.mean() < 1


CAPTURED: list[Row] = table(
    "captured",
    ("normal_spot:filter_normal_diffexp", Replay("06_normal/filter_normal_diffexp, replayed; 06_normal/normal_baf_bin_filter out"), _diffexp_captured, _diffexp_drops,
     "on easy it only removes UMIs, from some bins and not others"),
)


@pytest.mark.parametrize("row", ORACLE, ids=[r.id for r in ORACLE])
def test_oracle(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """scipy's beta-binomial, its MLE, the candidate and baseline rules recomputed, planted filters."""
    run(row, ctx, request)


@pytest.mark.parametrize("row", INVARIANT, ids=[r.id for r in INVARIANT])
def test_invariant(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """Configuration parsing, parameter layout, side effects."""
    run(row, ctx, request)


@pytest.mark.parametrize("row", CAPTURED, ids=[r.id for r in CAPTURED])
def test_captured(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """The run's filter output against its input."""
    run(row, ctx, request)
