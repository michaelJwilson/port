"""Mathematical properties the model has on any data: normalisation, limits, invariance, monotonicity, phase symmetry, invalid values.

Inputs: the fitted parameters stored in the staged run (`run_core_inference`'s
outputs, no replay) with few-element synthetic counts, or a synthetic case.
`audit.fn` defines the row; a row whose property cnamaste breaks is a strict
xfail citing its ticket, so a fix turns it into XPASS and the marker goes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.special
import scipy.stats
from cnamaste import hmrf, icm
from cnamaste.hmm import compute_copy_state_posterior
from cnamaste.hmm_nophasing import (
    _bb_logpmf_1d,
    _dense_bb_logpmf,
    _dense_nb_logpmf,
    _nb_logpmf_1d,
    betabinom_logpmf_numba,
    hmm_nophasing,
    nbinom_logpmf_numba,
)
from cnamaste.hmm_phased import hmm_phased
from cnamaste.hmrf import compute_loglike_spot_assignment
from cnamaste.hmrf_utils import cast_csr
from cnamaste.icm import icm_sweep_deque, merge_assignment, unpack_adjacency
from cnamaste.integer_copy import hill_climbing_integer_copynumber_fixdiploid_milp

from audit.fn import Row, run, table
from test_fn_hmm import close
from test_fn_hmrf import field_case
from test_fn_integer import recorded

FITS = ("05_baf", "08_rdr")
PARAMS = "05_baf, 08_rdr run_core_inference out: fitted p, tau, mu, alpha"
DEPTHS = (1, 10, 100, 1000)
EXPOSURES = (0.5, 10.0, 300.0)
TAIL = 1e-12
TRACE = "the capture holds no fit trace: waits for the fit-trace capture (per-iteration total_llf), re-captured"


def fitted_params(ctx: Any) -> list[dict[str, np.ndarray]]:
    """Each fit's stored parameters: log_mu, alphas, p, taus, start, transmat, (states, 1) each but the last two."""
    out = []
    for f in FITS:
        r = ctx.sim.stored(f"{f}/run_core_inference/out")
        out.append({k: np.asarray(r[f"new_{k}"]) for k in ("log_mu", "alphas", "p_binom", "taus", "log_startprob", "log_transmat")})
    return out


def model_counts(params: dict[str, np.ndarray], n: int = 400, seed: int = 9) -> dict[str, np.ndarray]:
    """`n` bins, one spot, drawn from the fitted HMM's emissions along a random state path."""
    rng = np.random.default_rng(seed)
    k = params["p_binom"].shape[0]
    state = rng.integers(0, k, size=n)
    base, total = rng.uniform(5, 50, size=n), rng.integers(5, 60, size=n)
    alpha, lam = params["alphas"][state, 0], base * np.exp(params["log_mu"][state, 0])
    rdr = rng.negative_binomial(1 / alpha, 1 / (1 + alpha * lam))
    tau, p = params["taus"][state, 0], params["p_binom"][state, 0]
    b = rng.binomial(total, rng.beta(p * tau, (1 - p) * tau))
    return {"X": np.stack([rdr, b], axis=1)[:, :, None].astype(float), "base": base[:, None], "total": total[:, None].astype(float)}


# --- normalisation ---------------------------------------------------------------


def nb_support(mu: float, alpha: float, exposure: float) -> np.ndarray:
    """0..K, K past which the NB tail is below `TAIL`."""
    lam = mu * exposure
    return np.arange(scipy.stats.nbinom.isf(TAIL, 1 / alpha, 1 / (1 + alpha * lam)) + 2.0)


def _nb_sums(kernel: str) -> Any:
    def check(fits: list[dict[str, np.ndarray]]) -> None:
        worst = 0.0
        for f in fits:
            for i in range(f["log_mu"].shape[0]):
                mu, alpha = float(np.exp(f["log_mu"][i, 0])), float(f["alphas"][i, 0])
                for e in EXPOSURES:
                    k = nb_support(mu, alpha, e)
                    if kernel == "scalar":
                        lam = mu * e
                        v = np.array([nbinom_logpmf_numba(x, 1 / alpha, 1 / (1 + alpha * lam)) for x in k])
                    elif kernel == "1d":
                        v = np.empty(k.size)
                        _nb_logpmf_1d(k, np.full(k.size, e), mu, alpha, v)
                    else:
                        v = _dense_nb_logpmf(k[:, None], np.full((k.size, 1), e), f["log_mu"], f["alphas"])[i, :, 0]
                    worst = max(worst, abs(scipy.special.logsumexp(v)))
        assert worst < 1e-10, f"max |log sum pmf| {worst:.2e}"
    return check


def _bb_sums(kernel: str) -> Any:
    def check(fits: list[dict[str, np.ndarray]]) -> None:
        worst = 0.0
        for f in fits:
            for i in range(f["p_binom"].shape[0]):
                p, tau = float(f["p_binom"][i, 0]), float(f["taus"][i, 0])
                for n in DEPTHS:
                    k, total = np.arange(n + 1.0), np.full(n + 1, float(n))
                    if kernel == "scalar":
                        v = np.array([betabinom_logpmf_numba(x, float(n), p * tau, (1 - p) * tau) for x in k])
                    elif kernel == "1d":
                        v = np.empty(n + 1)
                        _bb_logpmf_1d(k, total, p, tau, v)
                    else:
                        v = _dense_bb_logpmf(k[:, None], total[:, None], f["p_binom"], f["taus"])[i, :, 0]
                    worst = max(worst, abs(scipy.special.logsumexp(v)))
        assert worst < 1e-10, f"max |log sum pmf| {worst:.2e}"
    return check


# --- limits ----------------------------------------------------------------------


def _bb_binomial(taus: tuple[float, ...]) -> Any:
    """|log BB(k; n, p tau, (1-p) tau) - log Bin(k; n, p)| <= 1e3 / tau (+1e-9): the O(n^2 / tau) approach, n = 20, p = 0.3."""
    def check(_: Any) -> None:
        n, p = 20, 0.3
        k = np.arange(n + 1.0)
        for tau in taus:
            out = np.empty(n + 1)
            _bb_logpmf_1d(k, np.full(n + 1, float(n)), p, tau, out)
            gap = np.max(np.abs(out - scipy.stats.binom.logpmf(k, n, p)))
            assert gap <= 1e3 / tau + 1e-9, f"tau {tau:.0e}: gap {gap:.2e} > {1e3 / tau + 1e-9:.2e}"
    return check


def _nb_poisson(alphas: tuple[float, ...]) -> Any:
    """|log NB(k; mean 50, dispersion phi) - log Poisson(k; 50)| <= 2e4 phi (+1e-9) over k < 200: the O(phi) approach."""
    def check(_: Any) -> None:
        lam, k = 50.0, np.arange(200.0)
        for alpha in alphas:
            out = np.empty(k.size)
            _nb_logpmf_1d(k, np.full(k.size, lam), 1.0, alpha, out)
            gap = np.max(np.abs(out - scipy.stats.poisson.logpmf(k, lam)))
            assert gap <= 2e4 * alpha + 1e-9, f"phi {alpha:.0e}: gap {gap:.2e} > {2e4 * alpha + 1e-9:.2e}"
    return check


# --- invariance ------------------------------------------------------------------


def score(d: dict[str, np.ndarray], g: dict[str, np.ndarray]) -> tuple[float, np.ndarray]:
    """The forward pass' log-likelihood and the posteriors of counts `d` under parameters `g`."""
    n = d["X"].shape[0]
    rdr, baf = hmm_nophasing.compute_emission_probability_nb_betabinom(d["X"], d["base"], g["log_mu"], g["alphas"], d["total"], g["p_binom"], g["taus"])
    args = (np.array([n]), g["log_transmat"], g["log_startprob"], rdr + baf, np.full(n, np.log(1e-4)))
    alpha = hmm_nophasing.forward_lattice(*args)
    return float(scipy.special.logsumexp(alpha[:, -1])), hmm_nophasing().get_state_posteriors(*args)


def _hmm_relabel(fits: list[dict[str, np.ndarray]]) -> None:
    """Permuting the states (every parameter, transmat rows and columns) leaves the log-likelihood and permutes the posterior rows."""
    for f in fits:
        d = model_counts(f)
        perm = np.random.default_rng(3).permutation(f["p_binom"].shape[0])
        g = {key: v[perm] for key, v in f.items()} | {"log_transmat": f["log_transmat"][np.ix_(perm, perm)]}
        (llf, gamma), (llf_p, gamma_p) = score(d, f), score(d, g)
        assert abs(llf - llf_p) <= 1e-9 * abs(llf), f"log-likelihood {llf} -> {llf_p}"
        close(gamma_p, gamma[perm], 1e-9)


def clone_case(_: Any) -> dict[str, Any]:
    case = field_case()
    return case | {"perm": np.array([2, 0, 1])}


def _potts_relabel(f: dict[str, Any]) -> None:
    """merge_assignment's current cost (the Potts energy) under clone c -> perm[c], the field's columns moved with it."""
    spots, neighbours, weights = unpack_adjacency(cast_csr(f["adjacency"]))
    inverse = np.argsort(f["perm"])
    cost, *_ = merge_assignment(f["llf"], spots, neighbours, weights, f["labels"], f["weight"])
    cost_p, *_ = merge_assignment(f["llf"][:, inverse], spots, neighbours, weights, f["perm"][f["labels"]], f["weight"])
    assert np.isclose(cost, cost_p, rtol=0, atol=1e-12), f"Potts energy {cost} -> {cost_p}"


def _icm_relabel(f: dict[str, Any]) -> None:
    """The ICM sweep from relabelled labels and field ends at the relabelled result (same generator state, no floor)."""
    a, inverse = f["adjacency"], np.argsort(f["perm"])
    labels, labels_p = f["labels"].copy(), f["perm"][f["labels"]]
    np.random.seed(0)  # noqa: NPY002
    _, cost = icm_sweep_deque(f["llf"], a.indptr, a.indices, a.data, labels, f["weight"], None, min_clone_spots=0)
    np.random.seed(0)  # noqa: NPY002
    _, cost_p = icm_sweep_deque(f["llf"][:, inverse], a.indptr, a.indices, a.data, labels_p, f["weight"], None, min_clone_spots=0)
    assert np.array_equal(labels_p, f["perm"][labels]) and np.isclose(cost, cost_p, rtol=0, atol=1e-12)


def _field_relabel(_: Any) -> None:
    """compute_loglike_spot_assignment with the clones' paths reordered returns its columns reordered."""
    rng = np.random.default_rng(12)
    n_states, n_obs, n_spots, n_clones = 4, 6, 9, 3
    rdr, baf = rng.normal(size=(n_states, n_obs, n_spots)), rng.normal(size=(n_states, n_obs, n_spots))
    pred = rng.integers(0, n_states, size=(n_obs, n_clones))
    nb, bb = rng.integers(1, 5, size=n_spots), rng.integers(1, 5, size=n_spots)
    perm = np.array([1, 2, 0])
    smooth = sp.identity(n_spots, format="csr")
    found = compute_loglike_spot_assignment(n_spots, nb, bb, np.empty(0), False, rdr, baf, pred, n_obs, n_clones, smooth.indices, smooth.indptr)
    moved = compute_loglike_spot_assignment(n_spots, nb, bb, np.empty(0), False, rdr, baf, pred[:, perm], n_obs, n_clones, smooth.indices, smooth.indptr)
    assert np.array_equal(moved, found[:, perm])


# --- monotonicity ----------------------------------------------------------------


def _llf_rises(ctx: Any) -> list[np.ndarray]:
    traces = [ctx.sim.internal(f).get("total_llf") for f in FITS]
    if any(t is None for t in traces):
        pytest.skip(TRACE)
    return traces


def _monotone(traces: list[np.ndarray]) -> None:
    for t in traces:
        before = np.asarray(t, dtype=float)
        assert np.all(np.diff(before) >= -1e-6 * np.abs(before[1:])), "total_llf falls between outer iterations"


# --- phase flip ------------------------------------------------------------------


def _phase_emission(fits: list[dict[str, np.ndarray]]) -> None:
    """hmm_phased's state k + K is state k with p -> 1 - p: the BAF emission is the unphased one at 1 - p, the RDR one unchanged."""
    for f in fits:
        d = model_counts(f, n=200)
        k = f["p_binom"].shape[0]
        rdr, baf = hmm_phased.compute_emission_probability_nb_betabinom(d["X"], d["base"], f["log_mu"], f["alphas"], d["total"], f["p_binom"], f["taus"])
        _, flipped = hmm_nophasing.compute_emission_probability_nb_betabinom(d["X"], d["base"], f["log_mu"], f["alphas"], d["total"], 1 - f["p_binom"], f["taus"])
        _, plain = hmm_nophasing.compute_emission_probability_nb_betabinom(d["X"], d["base"], f["log_mu"], f["alphas"], d["total"], f["p_binom"], f["taus"])
        close(rdr[k:], rdr[:k], 0.0)
        close(baf[:k], plain, 1e-10)
        close(baf[k:], flipped, 1e-9)


def _phase_decoded(calls: list[dict[str, Any]]) -> None:
    """The integer decoder at 1 - p returns each state's (A, B) as (B, A), at the same cost and ploidy."""
    for call in calls:
        log_mu, base, p, pred = call["args"]
        copies, cost, ploidy = hill_climbing_integer_copynumber_fixdiploid_milp(log_mu, base, 1 - p, pred, **call["kwargs"])
        want = np.asarray(call["out"][0])
        assert np.array_equal(np.asarray(copies), want[:, ::-1]), "(A, B) not swapped"
        assert np.isclose(cost, call["out"][1], rtol=1e-12) and ploidy == call["out"][2]


# --- invalid values --------------------------------------------------------------


def _logsumexp_inf(module: Any) -> Any:
    def check(_: Any) -> None:
        found = module.logsumexp(np.full(3, -np.inf))
        assert np.isneginf(found), f"{module.__name__}.logsumexp of an all -inf row: {found}, scipy -inf"
    return check


def _posterior_check(kind: str) -> Any:
    """An all -inf column (no state can emit the observation) is refused, not normalised to NaN."""
    def check(_: Any) -> None:
        lattice = -np.random.default_rng(4).uniform(1, 5, size=(3, 4))
        lattice[:, 2] = -np.inf
        with pytest.raises(RuntimeError):
            if kind == "hmm":
                compute_copy_state_posterior(lattice.copy(), np.zeros((3, 4)))
            else:
                hmm_nophasing().get_state_posteriors(np.array([4]), np.log(np.full((3, 3), 1 / 3)), np.log(np.full(3, 1 / 3)), lattice[:, :, None], np.zeros(4))
    return check


def _posterior_finite(_: Any) -> None:
    """A finite column is normalised, whatever its log values sum to: [1, -1, 0] sums to 0 and is a valid unnormalised column."""
    lattice = np.array([[1.0, -2.0], [-1.0, -3.0], [0.0, -4.0]])
    gamma = compute_copy_state_posterior(lattice.copy(), np.zeros((3, 2)))
    close(gamma, lattice - scipy.special.logsumexp(lattice, axis=0), 1e-12)


NO_INPUT = "synthetic"
LIMIT_TAU = "Ticket#561: beta-binomial log-pmf loses precision at large concentration tau: lgamma cancellation leaves 5e-5 at tau 1e10, 0.41 at 1e14"
LIMIT_PHI = ("Ticket#560: nbinom_logpmf_numba scores any count at probability 1 when p rounds to 1: r = 1/max(phi, 1e-10) caps the dispersion "
             "and p = 1/(1 + phi lambda) rounds to 1, a gap of 4e-5 at phi 1e-10 and 867 nats at 1e-12")
NAN_LSE = "Ticket#411: icm.logsumexp and hmrf.logsumexp return NaN on an all -inf row, where scipy returns -inf"
POSTERIOR = ("Ticket#411: the posterior check tests sum(log_gamma) == 0 (hmm.py:28, hmm_nophasing.py:443), a sum of logs; an all -inf column "
             "passes it and normalises to NaN")

ROWS: list[Row] = [
    *table(
        "normalisation",
        *((f"hmm_nophasing:{fn}", PARAMS, fitted_params, _nb_sums(kind), f"NB pmf sums to 1 (1e-10) at exposures {EXPOSURES}, support cut where the tail < 1e-12")
          for fn, kind in (("nbinom_logpmf_numba", "scalar"), ("_nb_logpmf_1d", "1d"), ("_dense_nb_logpmf", "dense"))),
        *((f"hmm_nophasing:{fn}", PARAMS, fitted_params, _bb_sums(kind), f"BB pmf sums to 1 (1e-10) over 0..n, n in {DEPTHS}")
          for fn, kind in (("betabinom_logpmf_numba", "scalar"), ("_bb_logpmf_1d", "1d"), ("_dense_bb_logpmf", "dense"))),
    ),
    *table(
        "limit",
        ("hmm_nophasing:_bb_logpmf_1d", NO_INPUT, lambda c: None, _bb_binomial((1e4, 1e6, 1e8)), "BB -> Binomial as tau -> inf, tau 1e4..1e8"),
        ("hmm_nophasing:_bb_logpmf_1d", NO_INPUT, lambda c: None, _bb_binomial((1e4, 1e6, 1e8, 1e10, 1e12, 1e14)), "BB -> Binomial as tau -> inf, tau to 1e14", LIMIT_TAU),
        ("hmm_nophasing:_nb_logpmf_1d", NO_INPUT, lambda c: None, _nb_poisson((1e-4, 1e-6, 1e-8)), "NB -> Poisson as phi -> 0, phi 1e-4..1e-8"),
        ("hmm_nophasing:_nb_logpmf_1d", NO_INPUT, lambda c: None, _nb_poisson((1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16, 1e-18)), "NB -> Poisson as phi -> 0, phi to 1e-18", LIMIT_PHI),
    ),
    *table(
        "invariance",
        ("hmm_nophasing:hmm_nophasing.forward_lattice", PARAMS + "; 400 bins drawn from them", fitted_params, _hmm_relabel, "relabelling HMM states: log-likelihood to 1e-9 relative, posterior rows permuted to 1e-9"),
        ("icm:merge_assignment", "synthetic: 8x8 grid, 3 clones", clone_case, _potts_relabel, "relabelling clones: the Potts energy to 1e-12"),
        ("icm:icm_sweep_deque", "synthetic: 8x8 grid, 3 clones", clone_case, _icm_relabel, "relabelling clones: the sweep ends at the relabelled labels and energy"),
        ("hmrf:compute_loglike_spot_assignment", "synthetic: 9 spots, 3 clones", lambda c: None, _field_relabel, "relabelling clones: the per-spot field's columns move with them, bitwise"),
    ),
    *table(
        "monotone",
        ("hmrf:run_core_inference", "/internal/<fit>/total_llf", _llf_rises, _monotone, "total_llf non-decreasing over outer iterations, 1e-6 relative (Ticket#30, Ticket#146)"),
    ),
    *table(
        "phase",
        ("hmm_phased:hmm_phased.compute_emission_probability_nb_betabinom", PARAMS + "; 200 bins drawn from them", fitted_params, _phase_emission, "state k + K is state k at 1 - p: BAF to 1e-9, RDR bitwise"),
        ("integer_copy:hill_climbing_integer_copynumber_fixdiploid_milp", "09_outputs/integer_copy_{0..3} in and out", recorded, _phase_decoded, "p -> 1 - p swaps each decoded (A, B), cost and ploidy unchanged"),
    ),
    *table(
        "invalid",
        ("icm:logsumexp", NO_INPUT, lambda c: None, _logsumexp_inf(icm), "an all -inf row gives -inf", NAN_LSE),
        ("hmrf:logsumexp", NO_INPUT, lambda c: None, _logsumexp_inf(hmrf), "an all -inf row gives -inf", NAN_LSE),
        ("hmm:compute_copy_state_posterior", NO_INPUT, lambda c: None, _posterior_check("hmm"), "an all -inf posterior column raises", POSTERIOR),
        ("hmm_nophasing:hmm_nophasing.get_state_posteriors", NO_INPUT, lambda c: None, _posterior_check("nophasing"), "an all -inf posterior column raises", POSTERIOR),
        ("hmm:compute_copy_state_posterior", NO_INPUT, lambda c: None, _posterior_finite, "a finite column whose logs sum to 0 is normalised, not refused",
         "Ticket#411: the same check raises RuntimeError on a valid column whose log values sum to 0", RuntimeError),
    ),
]


@pytest.mark.parametrize("row", ROWS, ids=[f"{r.kind}: {r.id}" for r in ROWS])
def test_property(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """A property the model has on any data, checked on the fitted parameters or a synthetic case."""
    run(row, ctx, request)
