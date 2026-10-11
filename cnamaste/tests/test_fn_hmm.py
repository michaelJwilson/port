"""Per-function rows for the HMM: emissions, lattices, posteriors, the M step, initialization, the count encoder, the result type.

Inputs, cheapest first: the fitted runs' final clone stacks (`audit.fn.final_stack`,
rebuilt from `run_core_inference`'s input and its final labels) at the stored
fitted parameters, sliced where a brute force is quadratic; else a few-element
synthetic input with a closed form. `audit.fn` defines the row and its kinds.
"""

from __future__ import annotations

import ast
import inspect
import itertools
import textwrap
from typing import Any

import numpy as np
import pytest
import scipy.special
import scipy.stats
from cnamaste.cna_hmrf_result import CloneAssignment
from cnamaste.count_encoder import CountEncoder
from cnamaste.hmm import compute_copy_state_posterior, pipeline_baum_welch
from cnamaste.hmm_initialize import gmm_init
from cnamaste.hmm_nophasing import (
    _bb_logpmf_1d,
    _dense_bb_logpmf,
    _dense_nb_logpmf,
    _nb_logpmf_1d,
    betabinom_logpmf_numba,
    get_log_transmat,
    hmm_nophasing,
    nbinom_logpmf_numba,
    np_sum_ax_squeeze,
    numba_logsumexp,
)

from audit.fn import Replay, Row, final_estep, final_stack, fitted, run, stacked_gamma, table, unchanged

BAF, RDR = "05_baf", "08_rdr"
STACK = Replay("final clone stack (run_core_inference in + final labels), fitted parameters")


def close(a: Any, b: Any, atol: float = 1e-9, rtol: float = 0.0) -> None:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape, f"shape {a.shape} != {b.shape}"
    assert np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True), f"max |diff| {np.nanmax(np.abs(a - b)):.3e}"


# --- builders ------------------------------------------------------------------


def stack_slice(run_: str, n: int = 300) -> Any:
    """The first `n` stacked bins of `run_`'s final stack, with its fitted parameters."""
    def build(ctx: Any) -> dict[str, Any]:
        s, res = final_stack(ctx, run_), fitted(ctx, run_)["res"]
        return {"X": s["X"][:n], "base": s["base_nb_mean"][:n], "total": s["total_bb_RD"][:n],
                "log_mu": np.asarray(res["new_log_mu"]), "alphas": np.asarray(res["new_alphas"]),
                "p": np.asarray(res["new_p_binom"]), "taus": np.asarray(res["new_taus"]),
                "transmat": np.asarray(res["new_log_transmat"]), "start": np.asarray(res["new_log_startprob"])}
    return build


def scipy_nb(x: np.ndarray, base: np.ndarray, log_mu: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """log NB(x; r = 1/alpha, p = 1/(1 + alpha mu base)) per state; 0 where the baseline is 0 (no information)."""
    out = np.zeros((log_mu.shape[0],) + x.shape)
    for i in range(log_mu.shape[0]):
        lam = base * np.exp(log_mu[i, 0])
        with np.errstate(divide="ignore", invalid="ignore"):
            v = scipy.stats.nbinom.logpmf(x, 1.0 / alphas[i, 0], 1.0 / (1.0 + alphas[i, 0] * lam))
        out[i] = np.where(lam > 0, v, 0.0)
    return out


def scipy_bb(x: np.ndarray, total: np.ndarray, p: np.ndarray, taus: np.ndarray) -> np.ndarray:
    out = np.zeros((p.shape[0],) + x.shape)
    for i in range(p.shape[0]):
        out[i] = scipy.stats.betabinom.logpmf(x, total, p[i, 0] * taus[i, 0], (1 - p[i, 0]) * taus[i, 0])
    return out


def brute_forward(emission: np.ndarray, transmat: np.ndarray, start: np.ndarray) -> np.ndarray:
    """log alpha by the textbook recursion, one segment, emission (states, obs)."""
    alpha = np.zeros_like(emission)
    alpha[:, 0] = start + emission[:, 0]
    for t in range(1, emission.shape[1]):
        alpha[:, t] = scipy.special.logsumexp(alpha[:, t - 1][:, None] + transmat, axis=0) + emission[:, t]
    return alpha


def brute_backward(emission: np.ndarray, transmat: np.ndarray) -> np.ndarray:
    beta = np.zeros_like(emission)
    for t in range(emission.shape[1] - 2, -1, -1):
        beta[:, t] = scipy.special.logsumexp(transmat + (emission[:, t + 1] + beta[:, t + 1])[None, :], axis=1)
    return beta


def tiny_hmm() -> dict[str, Any]:
    """Three states, six observations, one segment: small enough to enumerate every path (3^6)."""
    rng = np.random.default_rng(7)
    emission = rng.normal(size=(3, 6))

    return {"emission": emission, "transmat": get_log_transmat(3, 0.8), "start": np.log(np.array([0.5, 0.3, 0.2]))}


def enumerate_paths(h: dict[str, Any]) -> tuple[float, np.ndarray]:
    """log P(obs) and the posterior marginals, by summing over all 3^6 state paths."""
    e, a, s = h["emission"], h["transmat"], h["start"]
    n_states, n_obs = e.shape
    logp, marg = [], np.full((n_states, n_obs), -np.inf)
    for path in itertools.product(range(n_states), repeat=n_obs):
        lp = s[path[0]] + e[path[0], 0] + sum(a[path[t - 1], path[t]] + e[path[t], t] for t in range(1, n_obs))
        logp.append(lp)
        for t, k in enumerate(path):
            marg[k, t] = np.logaddexp(marg[k, t], lp)
    total = scipy.special.logsumexp(logp)
    return float(total), marg - total


def planted_hmm(params: str, n: int = 600, seed: int = 11) -> dict[str, Any]:
    """Two copy states, one clone, planted mu (0.5, 1.5), p (0.2, 0.5), alpha 0.02, tau 200; states in two halves."""
    rng = np.random.default_rng(seed)
    state = (np.arange(n) >= n // 2).astype(int)
    mu, p = np.array([0.5, 1.5]), np.array([0.2, 0.5])
    base = rng.uniform(80, 120, size=n)
    alpha, tau = 0.02, 200.0
    lam = base * mu[state]
    rdr = rng.negative_binomial(1 / alpha, 1 / (1 + alpha * lam))
    total = rng.integers(40, 80, size=n)
    b = rng.binomial(total, rng.beta(p[state] * tau, (1 - p[state]) * tau))
    x = np.stack([rdr if "m" in params else np.zeros(n), b], axis=1)[:, :, None].astype(float)
    return {"X": x, "lengths": np.array([n]), "base": (base if "m" in params else np.zeros(n))[:, None], "total": total[:, None].astype(float),
            "mu": mu, "p": p, "state": state, "params": params}


def optimized(h: dict[str, Any]) -> dict[str, Any]:
    n = h["X"].shape[0]
    model = hmm_nophasing(params=h["params"], t=1 - 1e-3)
    out = model.optimize(h["X"], h["lengths"], 2, h["base"], h["total"], log_sitewise_transmat=np.full(n, np.log(1e-3)),
                         shared_NB_dispersion=True, shared_BB_dispersion=True, init_log_mu=np.log(np.array([[0.8], [1.2]])),
                         init_p_binom=np.array([[0.3], [0.45]]), clone_lengths=np.array([n]), max_iter=200)
    return out


def result_of(ctx: Any) -> Any:
    return ctx.sim.stored(f"{RDR}/run_core_inference/out")


# --- oracle ----------------------------------------------------------------------


def _dense_nb(d: dict[str, Any]) -> None:
    close(_dense_nb_logpmf(d["X"][:, 0, :], d["base"], d["log_mu"], d["alphas"]), scipy_nb(d["X"][:, 0, :], d["base"], d["log_mu"], d["alphas"]), 1e-8)


def _dense_bb(d: dict[str, Any]) -> None:
    close(_dense_bb_logpmf(d["X"][:, 1, :], d["total"], d["p"], d["taus"]), scipy_bb(d["X"][:, 1, :], d["total"], d["p"], d["taus"]), 1e-8)


def _nb_1d(d: dict[str, Any]) -> None:
    x, base = d["X"][:, 0, 0].astype(float), d["base"][:, 0]
    out = np.empty(x.size)
    _nb_logpmf_1d(x, base, float(np.exp(d["log_mu"][3, 0])), float(d["alphas"][3, 0]), out)
    close(out, scipy_nb(x[:, None], base[:, None], d["log_mu"][3:4], d["alphas"][3:4])[0, :, 0], 1e-8)


def _bb_1d(d: dict[str, Any]) -> None:
    x, total = d["X"][:, 1, 0].astype(float), d["total"][:, 0]
    out = np.empty(x.size)
    _bb_logpmf_1d(x, total, float(d["p"][2, 0]), float(d["taus"][2, 0]), out)
    close(out, scipy_bb(x[:, None], total[:, None], d["p"][2:3], d["taus"][2:3])[0, :, 0], 1e-8)


def _nb_scalar(_: Any) -> None:
    for k, r, p in itertools.product((0.0, 3.0, 40.0), (0.5, 5.0, 50.0), (0.05, 0.5, 0.95)):
        assert np.isclose(nbinom_logpmf_numba(k, r, p), scipy.stats.nbinom.logpmf(k, r, p), atol=1e-10), (k, r, p)


def _bb_scalar(_: Any) -> None:
    for k, n, a, b in itertools.product((0.0, 4.0, 9.0), (9.0, 30.0), (0.5, 20.0), (1.5, 300.0)):
        assert np.isclose(betabinom_logpmf_numba(k, n, a, b), scipy.stats.betabinom.logpmf(k, n, a, b), atol=1e-10), (k, n, a, b)


def _logsumexp(_: Any) -> None:
    for a in (np.array([0.0, -1.0, -700.0]), np.array([1e3, 1e3 - 2]), np.array([-np.inf, -np.inf]), np.array([-np.inf, 0.0])):
        assert np.isclose(numba_logsumexp(a), scipy.special.logsumexp(a)) or (np.isneginf(numba_logsumexp(a)) and np.isneginf(scipy.special.logsumexp(a)))


def _squeeze(_: Any) -> None:
    a = np.arange(24.0).reshape(4, 6)
    assert np.array_equal(np_sum_ax_squeeze(a, 1), a.sum(axis=1)) and np.array_equal(np_sum_ax_squeeze(a, 0), a.sum(axis=0))


def _transmat(_: Any) -> None:
    t = np.exp(get_log_transmat(4, 0.97))
    assert np.allclose(np.diag(t), 0.97) and np.allclose(t[~np.eye(4, dtype=bool)], 0.01) and np.allclose(t.sum(axis=1), 1)
    assert np.array_equal(get_log_transmat(1, 0.5), np.zeros((1, 1)))


def _forward(d: dict[str, Any]) -> None:
    e = scipy_nb(d["X"][:, 0, :], d["base"], d["log_mu"], d["alphas"]) + scipy_bb(d["X"][:, 1, :], d["total"], d["p"], d["taus"])
    lengths = np.array([100, d["X"].shape[0] - 100])
    found = hmm_nophasing.forward_lattice(lengths, d["transmat"], d["start"], e, np.zeros(d["X"].shape[0]))
    want = np.concatenate([brute_forward(e[:, :100, 0], d["transmat"], d["start"]), brute_forward(e[:, 100:, 0], d["transmat"], d["start"])], axis=1)
    close(found, want, 1e-8)


def _backward(d: dict[str, Any]) -> None:
    e = scipy_nb(d["X"][:, 0, :], d["base"], d["log_mu"], d["alphas"]) + scipy_bb(d["X"][:, 1, :], d["total"], d["p"], d["taus"])
    lengths = np.array([100, d["X"].shape[0] - 100])
    found = hmm_nophasing.backward_lattice(lengths, d["transmat"], d["start"], e, np.zeros(d["X"].shape[0]))
    want = np.concatenate([brute_backward(e[:, :100, 0], d["transmat"]), brute_backward(e[:, 100:, 0], d["transmat"])], axis=1)
    close(found, want, 1e-8)


def _posteriors_enumerated(h: dict[str, Any]) -> None:
    _, marginals = enumerate_paths(h)
    gamma = hmm_nophasing().get_state_posteriors(np.array([6]), h["transmat"], h["start"], h["emission"][:, :, None], np.zeros(6))
    close(gamma, marginals, 1e-10)


def _copy_posterior(h: dict[str, Any]) -> None:
    args = (np.array([6]), h["transmat"], h["start"], h["emission"][:, :, None], np.zeros(6))
    total, marginals = enumerate_paths(h)
    alpha = hmm_nophasing.forward_lattice(*args)
    assert np.isclose(scipy.special.logsumexp(alpha[:, -1]), total), "the forward pass' last column sums to log P(obs)"
    close(compute_copy_state_posterior(alpha, hmm_nophasing.backward_lattice(*args)), marginals, 1e-10)


def _coded_vs_dense(d: dict[str, Any]) -> None:
    nb, bb = CountEncoder(d["X"][:, 0, :], d["base"]), CountEncoder(d["X"][:, 1, :], d["total"])
    rdr, baf = hmm_nophasing().compute_emission_probability_nb_betabinom_coded(nb, bb, d["log_mu"], d["alphas"], d["p"], d["taus"])
    dense = hmm_nophasing.compute_emission_probability_nb_betabinom(d["X"], d["base"], d["log_mu"], d["alphas"], d["total"], d["p"], d["taus"])
    close(rdr, dense[0][:, :, 0], 1e-10)
    close(baf, dense[1][:, :, 0], 1e-10)


def _dense_emission(d: dict[str, Any]) -> None:
    rdr, baf = hmm_nophasing.compute_emission_probability_nb_betabinom(d["X"], d["base"], d["log_mu"], d["alphas"], d["total"], d["p"], d["taus"])
    close(rdr, scipy_nb(d["X"][:, 0, :], d["base"], d["log_mu"], d["alphas"]), 1e-8)
    close(baf, scipy_bb(d["X"][:, 1, :], d["total"], d["p"], d["taus"]), 1e-8)


def _recovers(field_: str, want: np.ndarray, atol: float) -> Any:
    def check(out: dict[str, Any]) -> None:
        found = np.sort(np.asarray(out[field_]).ravel())
        found = np.exp(found) if field_ == "new_log_mu" else found
        close(found, np.sort(want), atol)
    return check


def _param_errors(_: Any) -> None:
    model = hmm_nophasing(params="smp")
    x = np.array([0.0, 0.0, 0.1, -0.2, 0.3, -1.0, np.log(2.0), np.log(50.0)])
    hess_inv = np.diag([0.0, 0.0, 0.04, 0.09, 0.16, 0.25, 0.01, 0.04])
    start, mu, p, alpha, tau = model.unpack_param_errors(x, hess_inv, 2, shared_NB_dispersion=True, shared_BB_dispersion=True)
    pv = scipy.special.expit(np.array([0.3, -1.0]))
    close(mu.ravel(), [0.2, 0.3])
    close(p.ravel(), pv * (1 - pv) * np.array([0.4, 0.5]))
    close(alpha.ravel(), [2.0 * 0.1] * 2)
    close(tau.ravel(), [50.0 * 0.2] * 2)
    assert start.shape == (2,) and np.all(start >= 0)


ORACLE: list[Row] = table(
    "oracle",
    ("hmm_nophasing:_dense_nb_logpmf", STACK, stack_slice(RDR), _dense_nb, "the dense NB log-pmf is scipy's nbinom.logpmf(x; 1/alpha, 1/(1 + alpha mu base)), 0 at no baseline"),
    ("hmm_nophasing:_dense_bb_logpmf", STACK, stack_slice(RDR), _dense_bb, "the dense BB log-pmf is scipy's betabinom.logpmf(x; n, p tau, (1-p) tau)"),
    ("hmm_nophasing:_nb_logpmf_1d", STACK, stack_slice(RDR), _nb_1d, "the 1-d NB kernel is scipy's, one state"),
    ("hmm_nophasing:_bb_logpmf_1d", STACK, stack_slice(RDR), _bb_1d, "the 1-d BB kernel is scipy's, one state"),
    ("hmm_nophasing:nbinom_logpmf_numba", "synthetic grid", lambda c: None, _nb_scalar, "the scalar NB log-pmf is scipy's on a 27-point grid"),
    ("hmm_nophasing:betabinom_logpmf_numba", "synthetic grid", lambda c: None, _bb_scalar, "the scalar BB log-pmf is scipy's on a 24-point grid"),
    ("hmm_nophasing:numba_logsumexp", "synthetic", lambda c: None, _logsumexp, "logsumexp is scipy's, with -inf and overflow-scale inputs"),
    ("hmm_nophasing:np_sum_ax_squeeze", "synthetic", lambda c: None, _squeeze, "the axis sum is numpy's"),
    ("hmm_nophasing:get_log_transmat", "synthetic", lambda c: None, _transmat, "closed form: t on the diagonal, (1-t)/(K-1) off it; a 1-state chain is [[0]]"),
    ("hmm_nophasing:hmm_nophasing.forward_lattice", STACK, stack_slice(RDR), _forward, "log alpha is the textbook recursion, restarted at each segment"),
    ("hmm_nophasing:hmm_nophasing.backward_lattice", STACK, stack_slice(RDR), _backward, "log beta is the textbook recursion, restarted at each segment"),
    ("hmm_nophasing:hmm_nophasing.get_state_posteriors", "synthetic: 3 states, 6 observations", lambda c: tiny_hmm(), _posteriors_enumerated, "the posteriors are the marginals of all 3^6 paths, enumerated"),
    ("hmm:compute_copy_state_posterior", "synthetic: 3 states, 6 observations", lambda c: tiny_hmm(), _copy_posterior, "alpha + beta normalized is the enumerated marginal; forward sums to log P(obs)"),
    ("hmm_nophasing:hmm_nophasing.compute_emission_probability_nb_betabinom_coded", STACK, stack_slice(BAF), _coded_vs_dense, "BAF fit (integer depths): the coded emission equals the dense one (a second implementation), to 1e-10"),
    ("hmm_nophasing:hmm_nophasing.compute_emission_probability_nb_betabinom_coded", STACK, stack_slice(RDR), _coded_vs_dense, "RDR fit: the coded emission (the M step's) equals the dense one (the E step's), to 1e-10",
     "new: CountEncoder rounds the NB exposure base_nb_mean to hmm.compression_decimals (0), so the M step's coded emission differs from the dense E step's by up to 0.94 nats per bin on easy's RDR fit"),
    ("hmm_nophasing:hmm_nophasing.compute_emission_probability_nb_betabinom", STACK, stack_slice(RDR), _dense_emission, "the dense emission is scipy's NB and BB log-pmfs"),
    ("hmm_nophasing:hmm_nophasing._run_optimization_pipeline", "synthetic: planted 2-state NB+BB chain, 600 bins", lambda c: optimized(planted_hmm("smp")), _recovers("new_p_binom", np.array([0.2, 0.5]), 0.02), "EM recovers the planted p (0.2, 0.5) to 0.02"),
    ("hmm_nophasing:hmm_nophasing.optimize", "synthetic: planted 2-state NB+BB chain, 600 bins", lambda c: optimized(planted_hmm("smp")), _recovers("new_log_mu", np.array([0.5, 1.5]), 0.05), "EM recovers the planted mu (0.5, 1.5) to 0.05"),
    ("hmm_nophasing:hmm_nophasing.unpack_param_errors", "synthetic: a diagonal inverse Hessian", lambda c: None, _param_errors, "delta method: sd(p) = p(1-p) sd(logit p), sd(alpha) = alpha sd(log alpha)"),
)


# --- invariants -------------------------------------------------------------------


def _pack_roundtrip(_: Any) -> None:
    rng = np.random.default_rng(3)
    k = 3
    start = np.log(np.array([0.2, 0.3, 0.5]))
    mu, p = rng.normal(size=(k, 1)), rng.uniform(0.1, 0.9, size=(k, 1))
    for params, shared in itertools.product(("smp", "sp", "mp", "p"), (True, False)):
        model = hmm_nophasing(params=params)
        alphas = np.full((k, 1), 0.3) if shared else rng.uniform(0.1, 1, size=(k, 1))
        taus = np.full((k, 1), 40.0) if shared else rng.uniform(10, 100, size=(k, 1))
        flags = {"shared_NB_dispersion": shared, "shared_BB_dispersion": shared}
        x = model.pack_params(start, mu, p, alphas, taus, **flags)
        assert len(model.get_bounds(k, **flags)) == x.size, (params, shared)
        s2, mu2, p2, a2, t2 = model.unpack_params(x, k, start, mu, p, alphas, taus, **flags)
        for got, want in ((s2, start), (mu2, mu), (p2, p), (a2, alphas), (t2, taus)):
            close(got, want, 1e-12)


def _initial(_: Any) -> None:
    log_mu, p, alphas, taus, start, transmat = hmm_nophasing(t=0.9).get_initial_params(4, 1)
    close(log_mu.ravel(), np.linspace(-0.1, 0.1, 4))
    close(p.ravel(), np.linspace(0.05, 0.45, 4))
    assert np.allclose(np.exp(start).sum(), 1) and np.allclose(np.exp(transmat).sum(axis=1), 1) and np.allclose(alphas, 0.5) and np.allclose(taus, 1000)
    given = np.ones((4, 1))
    assert hmm_nophasing().get_initial_params(4, 1, init_log_mu=given)[0] is given


def _bounds(_: Any) -> None:
    b = hmm_nophasing(params="smp").get_bounds(5, shared_NB_dispersion=True, shared_BB_dispersion=False)
    assert len(b) == 5 + 5 + 5 + 1 + 5
    assert b[15] == (np.log(1e-6), np.log(1000.0)) and b[-1] == (np.log(1e-4), np.log(5000.0))


def _init(_: Any) -> None:
    m = hmm_nophasing(params="sp", t=0.25)
    assert (m.params, m.t) == ("sp", 0.25)


def _em_is_optimize(h: dict[str, Any]) -> None:
    n = h["X"].shape[0]
    kw = {"log_sitewise_transmat": np.full(n, np.log(1e-3)), "init_p_binom": np.array([[0.3], [0.45]]), "max_iter": 30}
    a = hmm_nophasing(params="sp").optimize(h["X"], h["lengths"], 2, h["base"], h["total"], **kw)
    b = hmm_nophasing(params="sp").run_baum_welch_nb_bb(h["X"], h["lengths"], 2, h["base"], h["total"], **kw)
    for key in ("new_p_binom", "new_taus", "log_gamma"):
        assert np.array_equal(a[key], b[key]), key
    g = np.asarray(a["log_gamma"])
    assert np.allclose(scipy.special.logsumexp(g, axis=0), 0) and np.array_equal(a["pred_cnv"], np.argmax(g, axis=0))


def _em_objective(h: dict[str, Any]) -> None:
    """The fitted parameters score the data at least as well as the start (log-likelihood by the forward pass)."""
    n = h["X"].shape[0]
    model = hmm_nophasing(params="sp", t=1 - 1e-3)
    start_p = np.array([[0.3], [0.45]])
    out = model.optimize(h["X"], h["lengths"], 2, h["base"], h["total"], log_sitewise_transmat=np.full(n, np.log(1e-3)), init_p_binom=start_p, max_iter=50)

    def loglik(p: np.ndarray, taus: np.ndarray) -> float:
        e = scipy_bb(h["X"][:, 1, :], h["total"], p, taus)
        a = hmm_nophasing.forward_lattice(h["lengths"], out["new_log_transmat"], out["new_log_startprob"], e, np.zeros(n))
        return float(scipy.special.logsumexp(a[:, -1]))

    assert loglik(out["new_p_binom"], out["new_taus"]) >= loglik(start_p, np.full((2, 1), 1000.0))


def _baum_welch(h: dict[str, Any]) -> None:
    n = h["X"].shape[0]
    transmat = np.full(n, np.log(1e-3))
    res = unchanged(pipeline_baum_welch, None, h["X"], h["lengths"], 2, h["base"], h["total"], transmat, hmmclass=hmm_nophasing, params="sp",
                    init_p_binom=np.array([[0.3], [0.45]]), init_log_mu=np.zeros((2, 1)), max_iter=50)
    g = np.asarray(res["log_gamma"])
    assert np.allclose(scipy.special.logsumexp(g, axis=0), 0, atol=1e-10)
    assert np.array_equal(res["pred_cnv"], np.argmax(g, axis=0) % 2)
    e = scipy_bb(h["X"][:, 1, :], h["total"], res["new_p_binom"], res["new_taus"])
    alpha = brute_forward(e[:, :, 0], res["new_log_transmat"], res["new_log_startprob"])
    assert np.isclose(res["llf"], scipy.special.logsumexp(alpha[:, -1])), "llf is the forward pass' total"
    assert np.mean(res["pred_cnv"] == (h["p"][h["state"]] == 0.5)) > 0.95 or np.mean(res["pred_cnv"] == (h["p"][h["state"]] != 0.5)) > 0.95


def _gmm_minor(h: dict[str, Any]) -> None:
    n = h["X"].shape[0]
    log_mu, p, *_ = unchanged(gmm_init, 2, h["X"], h["base"], h["total"], "sp", h["lengths"], get_log_transmat(2, 0.99), np.zeros(n), random_state=0, only_minor=True)
    assert log_mu is None and p.shape == (2, 1) and np.all((p >= 0) & (p <= 0.5))
    close(np.sort(p.ravel()), [0.2, 0.5], 0.06)  # NB folding a BAF spread about 1/2 biases its mean below 1/2


def _gmm_rdr(h: dict[str, Any]) -> None:
    n = h["X"].shape[0]
    log_mu, p, *_ = gmm_init(2, h["X"], h["base"], h["total"], "smp", h["lengths"], get_log_transmat(2, 0.99), np.zeros(n), random_state=0, in_log_space=False, only_minor=False)
    assert log_mu.shape == (2, 1) and p.shape == (2, 1)
    close(np.sort(np.exp(log_mu.ravel())), [0.5, 1.5], 0.1)


def _estep_normalized(e: dict[str, Any]) -> None:
    assert np.allclose(scipy.special.logsumexp(e["gamma"], axis=0), 0, atol=1e-9)
    assert np.all(np.isfinite(e["gamma"])) and e["gamma"].shape == e["emission"].shape[:2]


def _encoder(d: dict[str, Any]) -> None:
    obs, total = d["X"][:, 1, :], d["total"]
    enc = unchanged(CountEncoder, obs, total)
    pairs = enc.unique_counts[0]
    assert np.array_equal(pairs, np.unique(np.stack([obs[:, 0], total[:, 0]], axis=1), axis=0)), "the unique (obs, total) pairs, sorted"
    assert np.array_equal(enc.get_unique_obs(0), pairs[:, 0]) and np.array_equal(enc.get_unique_total(0), pairs[:, 1])
    values = np.stack([pairs[:, 0] * 10 + pairs[:, 1], -pairs[:, 1]])
    decoded = enc.decode_array(values, 0)
    assert np.array_equal(decoded, np.stack([obs[:, 0] * 10 + total[:, 0], -total[:, 0]])), "decode maps each pair's value back to its observations"
    assert np.isclose(enc.compression_rate, 1 - pairs.shape[0] / obs.shape[0])


def _zero_depth(_: Any) -> None:
    obs, total = np.array([[3], [0], [5], [3]]), np.array([[0], [0], [9], [0]])
    pairs, _ = CountEncoder.construct_unique_encoding(obs, total, common_zero_depth=True)
    assert pairs[0].tolist() == [[0, 0], [5, 9]], "every zero-depth entry collapses to one (0, 0) state"
    pairs, mapping = CountEncoder.construct_unique_encoding(obs, total, common_zero_depth=False)
    assert pairs[0].tolist() == [[0, 0], [3, 0], [5, 9]] and np.array_equal(mapping[0].sum(axis=0).A.ravel(), [1, 2, 1])


def _result_access(res: Any) -> None:
    assert res["new_log_mu"] is res.params.new_log_mu and res["log_gamma"] is res.profile.log_gamma and res["new_assignment"] is res.assignment.new_assignment
    keys = res.keys()
    assert {"llf", "n_states", "new_log_mu", "log_gamma", "pred_cnv", "new_assignment"} <= set(keys)
    assert [k for k, _ in res.items()] == keys and len(res.values()) == len(keys)
    with pytest.raises(KeyError):
        res["no_such_key"]
    text = str(res)
    assert all(k in text for k in ("new_log_mu", "log_gamma", "new_assignment"))


def _clone_counts(res: Any) -> None:
    labels = np.asarray(res["new_assignment"])
    assert np.array_equal(res.assignment.unique_clone_labels, np.unique(labels)) and res.assignment.num_clones == np.unique(labels).size
    assert CloneAssignment().unique_clone_labels is None and CloneAssignment().num_clones is None


def _result_set(res: Any) -> None:
    labels = np.asarray(res["new_assignment"]).copy()
    shallow = res.copy()
    assert shallow.params is res.params, "copy() is shallow: nested parts are shared"
    deep_ = res.copy(deep=True)
    deep_["new_assignment"] = labels[::-1].copy()
    assert np.array_equal(res["new_assignment"], labels), "a deep copy's writes do not reach the original"
    with pytest.raises(ValueError):
        deep_["new_assignment"] = labels + 1  # NB labels must be 0..M-1: validate() refuses
    deep_.lock()
    with pytest.raises(RuntimeError):
        deep_.llf = 0.0
    with pytest.raises(RuntimeError):
        deep_.params.new_log_mu = None


INVARIANT: list[Row] = table(
    "invariant",
    ("hmm_nophasing:hmm_nophasing.pack_params", "synthetic: 3 states, 8 flag settings", lambda c: None, _pack_roundtrip, "unpack(pack(x)) == x; bounds match the packed length"),
    ("hmm_nophasing:hmm_nophasing.unpack_params", "synthetic: 3 states, 8 flag settings", lambda c: None, _pack_roundtrip, "unpack(pack(x)) == x for every params string and sharing"),
    ("hmm_nophasing:hmm_nophasing.get_bounds", "synthetic", lambda c: None, _bounds, "one bound per packed parameter, dispersions in log space at the stated limits"),
    ("hmm_nophasing:hmm_nophasing.get_initial_params", "synthetic", lambda c: None, _initial, "defaults: linspace mu and p, normalized start and transitions; given values pass through"),
    ("hmm_nophasing:hmm_nophasing.__init__", "synthetic", lambda c: None, _init, "stores params and t"),
    ("hmm_nophasing:hmm_nophasing.run_baum_welch_nb_bb", "synthetic: planted BB chain, 600 bins", lambda c: planted_hmm("sp"), _em_is_optimize, "optimize is EM mode, bitwise; posteriors normalized, pred_cnv their argmax"),
    ("hmm_nophasing:hmm_nophasing._run_optimization_pipeline", "synthetic: planted BB chain, 600 bins", lambda c: planted_hmm("sp"), _em_objective, "the fit's log-likelihood is at least the start's"),
    ("hmm:pipeline_baum_welch", "synthetic: planted BB chain, 600 bins", lambda c: planted_hmm("sp"), _baum_welch, "no input mutation; normalized posteriors; llf the forward total; states split at the planted change"),
    ("hmm_initialize:gmm_init", "synthetic: planted BB chain, 600 bins", lambda c: planted_hmm("sp"), _gmm_minor, "only_minor: p in [0, 1/2], near the planted (0.2, 0.5) to 0.06; no input mutation"),
    ("hmm_initialize:gmm_init", "synthetic: planted NB+BB chain, 600 bins", lambda c: planted_hmm("smp"), _gmm_rdr, "with RDR: mu near the planted (0.5, 1.5)"),
    ("hmm:compute_copy_state_posterior", STACK, lambda c: final_estep(c, RDR), _estep_normalized, "the final E step's posteriors are finite and normalized per bin"),
    ("count_encoder:CountEncoder.__init__", STACK, stack_slice(BAF, 1000), _encoder, "unique pairs are np.unique's; decode inverts the mapping; compression rate 1 - unique/total"),
    ("count_encoder:CountEncoder.decode_array", STACK, stack_slice(BAF, 1000), _encoder, "decode maps each unique pair's value to every observation carrying it"),
    ("count_encoder:CountEncoder.get_unique_obs", STACK, stack_slice(BAF, 1000), _encoder, "the first column of the unique pairs"),
    ("count_encoder:CountEncoder.get_unique_total", STACK, stack_slice(BAF, 1000), _encoder, "the second column of the unique pairs"),
    ("count_encoder:CountEncoder.compression_rate", STACK, stack_slice(BAF, 1000), _encoder, "1 - unique pairs / observations"),
    ("count_encoder:CountEncoder.construct_unique_encoding", "synthetic: 4 observations", lambda c: None, _zero_depth, "zero-depth entries collapse to (0, 0) when asked; the mapping counts each pair's observations"),
    ("cna_hmrf_result:CnaHMRFResult.__getitem__", "08_rdr/run_core_inference/out", result_of, _result_access, "a key resolves to its nested field (identity); unknown keys raise; keys/values/items agree; str names them"),
    ("cna_hmrf_result:CnaHMRFResult.keys", "08_rdr/run_core_inference/out", result_of, _result_access, "keys list the top-level and nested fields"),
    ("cna_hmrf_result:CnaHMRFResult.values", "08_rdr/run_core_inference/out", result_of, _result_access, "values follow keys"),
    ("cna_hmrf_result:CnaHMRFResult.items", "08_rdr/run_core_inference/out", result_of, _result_access, "items pair keys with values"),
    ("cna_hmrf_result:CnaHMRFResult.__str__", "08_rdr/run_core_inference/out", result_of, _result_access, "names every parameter, the posteriors and the labels"),
    ("cna_hmrf_result:CloneAssignment.unique_clone_labels", "08_rdr/run_core_inference/out", result_of, _clone_counts, "np.unique of the labels; None without labels"),
    ("cna_hmrf_result:CloneAssignment.num_clones", "08_rdr/run_core_inference/out", result_of, _clone_counts, "the number of distinct labels; None without labels"),
    ("cna_hmrf_result:CnaHMRFResult.__setitem__", "08_rdr/run_core_inference/out", result_of, _result_set, "a set validates: labels must stay 0..M-1"),
    ("cna_hmrf_result:CnaHMRFResult.validate", "08_rdr/run_core_inference/out", result_of, _result_set, "refuses labels not 0..M-1"),
    ("cna_hmrf_result:CnaHMRFResult.__post_init__", "08_rdr/run_core_inference/out", result_of, _result_set, "construction validates (the decode built it through __post_init__)"),
    ("cna_hmrf_result:CnaHMRFResult.copy", "08_rdr/run_core_inference/out", result_of, _result_set, "shallow shares nested parts; deep does not"),
    ("cna_hmrf_result:LockableMixin.lock", "08_rdr/run_core_inference/out", result_of, _result_set, "a locked result refuses writes, nested parts too"),
    ("cna_hmrf_result:LockableMixin.__setattr__", "08_rdr/run_core_inference/out", result_of, _result_set, "a write to a locked instance raises"),
)


# --- captured ---------------------------------------------------------------------


def _estep_is_stored(run_: str) -> Any:
    def build(ctx: Any) -> tuple[np.ndarray, np.ndarray]:
        return final_estep(ctx, run_)["gamma"], stacked_gamma(fitted(ctx, run_)["res"])

    def check(pair: tuple[np.ndarray, np.ndarray]) -> None:
        close(pair[0], pair[1], 1e-9)
    return build, check


def _pred_is_stored(run_: str) -> Any:
    def build(ctx: Any) -> tuple[np.ndarray, np.ndarray]:
        res = fitted(ctx, run_)["res"]
        pred = np.asarray(res["pred_cnv"])
        return np.argmax(final_estep(ctx, run_)["gamma"], axis=0), pred if pred.ndim == 1 else pred.T.ravel()

    def check(pair: tuple[np.ndarray, np.ndarray]) -> None:
        assert np.array_equal(*pair)
    return build, check


CAPTURED: list[Row] = table(
    "captured",
    ("hmm:pipeline_baum_welch", STACK, *_estep_is_stored(BAF), "BAF fit: the closing E step at the stored parameters reproduces the stored posteriors (1e-9)"),
    ("hmm:pipeline_baum_welch", STACK, *_estep_is_stored(RDR), "RDR fit: the closing E step at the stored parameters reproduces the stored posteriors (1e-9)"),
    ("hmm_nophasing:hmm_nophasing.forward_lattice", STACK, *_pred_is_stored(BAF), "BAF fit: the lattices' MAP states are the stored pred_cnv"),
    ("hmm_nophasing:hmm_nophasing.backward_lattice", STACK, *_pred_is_stored(RDR), "RDR fit: the lattices' MAP states are the stored pred_cnv"),
    ("hmm_nophasing:hmm_nophasing.compute_emission_probability_nb_betabinom", STACK, *_estep_is_stored(RDR), "the dense emission on the final stack is the one the stored posteriors were computed from"),
)


# --- known defects without a test_defects.py row ------------------------------------


def _gmm_weights(_: Any) -> None:
    body = ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(gmm_init))))
    assert "sample_weight" in body or "sqrt(base_nb_mean" in body, "the mixture weighs a shallow bin as a deep one"


def _data_terms(_: Any) -> None:
    assert "data_terms" in inspect.signature(hmm_nophasing.compute_emission_probability_nb_betabinom_coded).parameters


DEFECTS: list[Row] = table(
    "oracle",
    ("hmm_initialize:gmm_init", "source", lambda c: None, _gmm_weights, "the RDR mixture accounts for each bin's exposure",
     "Ticket#236: gmm_init divides the exposure out of the mean and leaves it in the variance, so every bin gets an equal vote"),
    ("hmm_nophasing:hmm_nophasing.compute_emission_probability_nb_betabinom_coded", "signature", lambda c: None, _data_terms, "a flag discards the data-only terms",
     "Ticket#860: Flag to discard data-only emission terms in parameter fitting; name them in the paper"),
)

ORACLE += DEFECTS


@pytest.mark.parametrize("row", ORACLE, ids=[r.id for r in ORACLE])
def test_oracle(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """An independent recomputation: scipy's pmfs, the textbook lattices, path enumeration, planted recovery."""
    run(row, ctx, request)


@pytest.mark.parametrize("row", INVARIANT, ids=[r.id for r in INVARIANT])
def test_invariant(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """Round trips, normalization, monotone objectives, no input mutation."""
    run(row, ctx, request)


@pytest.mark.parametrize("row", CAPTURED, ids=[r.id for r in CAPTURED])
def test_captured(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """The closing E step, recomputed from the stored parameters, is the stored one."""
    run(row, ctx, request)
