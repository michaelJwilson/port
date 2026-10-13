"""Per-function rows for the integer copy-number decoder.

Inputs: the four recorded decoder calls (`09_outputs/integer_copy_{0..3}`, one per
final clone); else a few-state synthetic profile. The MILP's objective is
separable over states but for one ploidy row, so the oracle is the per-state
argmin, checked against that row. The hill climber (`oneclone`, run only for a
stated ploidy) is checked against the brute force over every copy-state tuple
of a 3- or 4-state profile.
"""

from __future__ import annotations

import ast
import inspect
import itertools
import textwrap
from typing import Any

import numpy as np
import pytest
from cnamaste.integer_copy import (
    find_diploid_balanced_state,
    hill_climbing_integer_copynumber_fixdiploid_milp,
    hill_climbing_integer_copynumber_oneclone,
)
from cnamaste.scripts import run_cnamaste as script

from fn import Row, run, table, unchanged

CALLS = range(4)


def recorded(ctx: Any) -> list[dict[str, Any]]:
    out = []
    for k in CALLS:
        given = ctx.sim.stored(f"09_outputs/integer_copy_{k}/in")
        out.append({"args": [np.asarray(a) for a in given["args"]], "kwargs": given["kwargs"], "out": ctx.sim.stored(f"09_outputs/integer_copy_{k}/out")})
    return out


def separable(log_mu: np.ndarray, p: np.ndarray, pred: np.ndarray, weight: float = 0.3, max_allele: int = 5, max_total: int = 6) -> tuple[np.ndarray, float, int]:
    """Per-state argmin of points x (weight |mu - (A+B)/s| + |p - A/(A+B)|), the diploid state fixed at (1, 1), and the least ploidy bound it meets."""
    n = len(log_mu)
    points = np.bincount(pred, minlength=n) + 1e-6
    mu = np.exp(log_mu)
    normal = find_diploid_balanced_state(log_mu, p, pred, 0.0, 0.05)
    scale = 2.0 / mu[normal]
    candidates = np.array([(i, j) for i in range(max_allele + 1) for j in range(max_allele + 1) if (i, j) != (0, 0) and i + j <= max_total])
    total = candidates.sum(axis=1)
    copies, cost = np.zeros((n, 2), dtype=int), 0.0
    for k in range(n):
        c = points[k] * (weight * np.abs(mu[k] - total / scale) + np.abs(p[k] - candidates[:, 0] / total))
        best = int(np.argmin(c)) if k != normal else int(np.flatnonzero((candidates == (1, 1)).all(axis=1))[0])
        copies[k], cost = candidates[best], cost + c[best]
    ploidy = copies.sum(axis=1).dot(points) / points.sum()
    return copies, cost, int(np.ceil(ploidy - 0.5))


def climbing_cost(copies: np.ndarray, ploidy: int, log_mu: np.ndarray, p: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """`oneclone`'s objective for copy tuples (..., states, 2), restated: points x (|1 - (A+B)/2/mu| + |1 - A/(A+B)/p|),
    points the state's bins + 0.1, plus n_bins x |1 - derived/ploidy| where the points-weighted mean total exceeds the ploidy."""
    points = np.bincount(pred, minlength=len(log_mu)) + 0.1
    total = copies.sum(axis=-1)
    cost = (np.abs(1 - total / 2 / np.exp(log_mu)) * points).sum(-1) + (np.abs(1 - copies[..., 0] / total / p) * points).sum(-1)
    derived = total @ points / points.sum()
    return cost + np.where(derived > ploidy, np.abs(1 - derived / ploidy) * len(pred), 0.0)


def climbing_brute(log_mu: np.ndarray, p: np.ndarray, pred: np.ndarray, max_allele: int, max_total: int, max_ploidy: int) -> tuple[float, np.ndarray, int]:
    """The global minimum over every copy-state tuple and ploidy 1..max_ploidy, the least ploidy first."""
    candidates = np.array([(i, j) for i in range(max_allele + 1) for j in range(max_allele + 1) if (i, j) != (0, 0) and i + j <= max_total])
    tuples = candidates[np.array(list(itertools.product(range(len(candidates)), repeat=len(log_mu))))]
    best: tuple[float, np.ndarray, int] = (np.inf, tuples[0], 0)
    for ploidy in range(1, max_ploidy + 1):
        cost = climbing_cost(tuples, ploidy, log_mu, p, pred)
        i = int(np.argmin(cost))
        if cost[i] < best[0]:
            best = (float(cost[i]), tuples[i], ploidy)
    return best


PLANTED = ([[1, 1], [2, 1], [1, 0]], [[2, 2], [3, 1], [1, 1]], [[1, 1], [1, 2], [2, 2], [1, 3]])
"""Copy states whose mu = (A+B)/2 and p = A/(A+B) exactly: each the unique zero of the objective."""


def climbing_profiles(n: int = 40) -> list[tuple[np.ndarray, ...]]:
    """`n` random 3-state profiles over 30 bins: mu in [0.3, 2.5], p in [0.05, 0.95]."""
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        log_mu, p, pred = np.log(rng.uniform(0.3, 2.5, 3)), rng.uniform(0.05, 0.95, 3), rng.integers(0, 3, 30)
        out.append((log_mu, rng.uniform(1, 2, 30), p, pred))
    return out


# --- oracle --------------------------------------------------------------------


def _diploid_nearest_unit_mu(_: Any) -> None:
    log_mu = np.log(np.array([0.5, 0.95, 1.2, 1.0]))
    p = np.array([0.5, 0.48, 0.5, 0.2])
    pred = np.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    assert find_diploid_balanced_state(log_mu, p, pred, 0.0, 0.05) == 1, "balanced states 0-2; the closest mu to 1 is 0.95"
    assert find_diploid_balanced_state(log_mu, p, pred, 0.25, 0.05) == 2, "state 1 (2 of 10 bins) fails a 25% floor"
    with pytest.raises(ValueError):
        find_diploid_balanced_state(log_mu, np.full(4, 0.2), pred, 0.0, 0.05)


def _milp_per_state_argmin(calls: list[dict[str, Any]]) -> None:
    """Cost and copies against the separable argmin, on occupied states only.

    A state with no bins enters the objective with weight 1e-6 (`points + 1e-6`), so
    every copy pair costs it about 1e-6, inside the solver's gap: its copies are not
    identifiable, and two optima differing only there are the same solution.
    """
    for k, call in enumerate(calls):
        log_mu, _, p, pred = call["args"]
        copies, cost, _ = call["out"]
        want, want_cost, _ = separable(log_mu, p, pred)
        assert np.isclose(cost, want_cost, rtol=1e-6), f"call {k}: cost {cost} against the separable minimum {want_cost}"
        occupied = np.bincount(pred, minlength=len(log_mu)) > 0
        assert np.array_equal(copies[occupied], want[occupied]), f"call {k}: copies differ from the per-state argmin"


def _milp_least_ploidy(calls: list[dict[str, Any]]) -> None:
    for k, call in enumerate(calls):
        log_mu, _, p, pred = call["args"]
        _, _, want_ploidy = separable(log_mu, p, pred)
        assert call["out"][2] == max(want_ploidy, 1), f"call {k}: the least ploidy bound the solution meets"


def _climb_recovers_planted(_: Any) -> None:
    for planted in map(np.array, PLANTED):
        total = planted.sum(axis=1)
        log_mu, p, pred = np.log(total / 2), planted[:, 0] / total, np.repeat(np.arange(len(planted)), 5)
        copies, cost, ploidy = hill_climbing_integer_copynumber_oneclone(log_mu, np.ones(len(pred)), p, pred, max_allele_copy=3, max_total_copy=4, max_medploidy=4)
        want_cost, want, want_ploidy = climbing_brute(log_mu, p, pred, 3, 4, 4)
        assert np.array_equal(copies, planted) and np.array_equal(want, planted), f"{planted.tolist()}: the planted states"
        assert abs(cost) < 1e-12 and abs(want_cost) < 1e-12 and ploidy == want_ploidy, f"{planted.tolist()}: cost {cost}, ploidy {ploidy} against {want_cost}, {want_ploidy}"


def _climb_above_brute_force(_: Any) -> None:
    reached = 0
    for log_mu, base, p, pred in climbing_profiles():
        copies, cost, ploidy = hill_climbing_integer_copynumber_oneclone(log_mu, base, p, pred, max_allele_copy=3, max_total_copy=4, max_medploidy=3)
        want_cost, _, _ = climbing_brute(log_mu, p, pred, 3, 4, 3)
        assert np.isclose(cost, climbing_cost(copies, ploidy, log_mu, p, pred), rtol=1e-12), "the cost returned is the copies' objective"
        assert cost >= want_cost - 1e-12, "no cost below the brute-force minimum"
        reached += bool(np.isclose(cost, want_cost, rtol=1e-12))
    assert reached == 36, f"the global minimum on {reached} of 40 profiles; the other 4 stop at a local one (_climb_coordinate_minimum)"


ORACLE: list[Row] = table(
    "oracle",
    ("integer_copy:find_diploid_balanced_state", "synthetic: 4 states", lambda c: None, _diploid_nearest_unit_mu, "the balanced, frequent state with mu closest to 1; none raises"),
    ("integer_copy:hill_climbing_integer_copynumber_fixdiploid_milp", "09_outputs/integer_copy_{0..3} in and out", recorded, _milp_per_state_argmin,
     "on each recorded call: the per-state argmin on occupied states, and its cost"),
    ("integer_copy:hill_climbing_integer_copynumber_fixdiploid_milp", "09_outputs/integer_copy_{0..3} in and out", recorded, _milp_least_ploidy,
     "on each recorded call: the least ploidy bound the solution meets",
     "Ticket#894: integer MILP: best_ploidy takes a looser bound by solver noise"),
    ("integer_copy:hill_climbing_integer_copynumber_oneclone", "synthetic: 3 planted profiles, 3 and 4 states", lambda c: None, _climb_recovers_planted,
     "planted copy states: the brute-force minimum, cost 0, and its least ploidy"),
    ("integer_copy:hill_climbing_integer_copynumber_oneclone", "synthetic: 40 random 3-state profiles", lambda c: None, _climb_above_brute_force,
     "the cost is the copies' objective, never below the brute force over 13^3 tuples x 3 ploidies; equal to it on 36 of 40"),
)


# --- invariants ---------------------------------------------------------------------


def _copies_within_caps(calls: list[dict[str, Any]]) -> None:
    for call in calls:
        log_mu, base, p, pred = call["args"]
        copies, _, ploidy = unchanged(hill_climbing_integer_copynumber_fixdiploid_milp, log_mu, base, p, pred, **call["kwargs"])
        normal = find_diploid_balanced_state(log_mu, p, pred, 0.0, 0.05)
        assert np.all(copies >= 0) and np.all(copies <= 5) and np.all(copies.sum(axis=1) <= 6) and np.all(copies.sum(axis=1) > 0)
        assert copies[normal].tolist() == [1, 1] and 1 <= ploidy <= 4


def _climb_coordinate_minimum(_: Any) -> None:
    candidates = np.array([(i, j) for i in range(4) for j in range(4) if (i, j) != (0, 0) and i + j <= 4])
    for log_mu, base, p, pred in climbing_profiles():
        args = (log_mu, base, p, pred)
        copies, cost, ploidy = unchanged(hill_climbing_integer_copynumber_oneclone, *args, max_allele_copy=3, max_total_copy=4, max_medploidy=3)
        assert np.all(copies.sum(axis=1) <= 4) and np.all(copies <= 3) and np.all(copies.sum(axis=1) > 0) and 1 <= ploidy <= 3
        for k in range(len(log_mu)):
            moved = np.repeat(copies[None], len(candidates), axis=0)
            moved[:, k] = candidates
            assert climbing_cost(moved, ploidy, log_mu, p, pred).min() >= cost - 1e-12, f"state {k}: a one-state move lowers the cost"


def _config_caps_passed(_: Any) -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(script.run_cnamaste)))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "hill_climbing_integer_copynumber_fixdiploid_milp"]
    assert calls and all({"max_allele_copy", "max_total_copy"} <= {k.arg for k in c.keywords} for c in calls), "the decoder takes its caps from the configuration"


INVARIANT: list[Row] = table(
    "invariant",
    ("integer_copy:hill_climbing_integer_copynumber_oneclone", "synthetic: 40 random 3-state profiles", lambda c: None, _climb_coordinate_minimum,
     "copies within the caps, ploidy in 1..3, and no one-state move lowers the cost (a coordinate-wise minimum); no input mutation"),
    ("integer_copy:hill_climbing_integer_copynumber_fixdiploid_milp", "09_outputs/integer_copy_{0..3} in", recorded, _copies_within_caps,
     "copies within the allele and total caps, the diploid state (1, 1), ploidy in 1..4; no input mutation"),
    ("integer_copy:hill_climbing_integer_copynumber_fixdiploid_milp", "source of run_cnamaste", lambda c: None, _config_caps_passed, "run_cnamaste passes int_copy_num's caps",
     "Ticket#788: int_copy_num.max_allele_copy is ignored: run_cnamaste passes the decoder no cap from the configuration"),
)


@pytest.mark.parametrize("row", ORACLE, ids=[r.id for r in ORACLE])
def test_oracle_agrees(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """The decoder against the separable per-state minimum."""
    run(row, ctx, request)


@pytest.mark.parametrize("row", INVARIANT, ids=[r.id for r in INVARIANT])
def test_invariants_hold(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """Caps, the fixed diploid state, no input mutation."""
    run(row, ctx, request)
