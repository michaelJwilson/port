"""Per-function rows for the integer copy-number decoder.

Inputs: the four recorded decoder calls (`09_outputs/integer_copy_{0..3}`, one per
final clone); else a few-state synthetic profile. The MILP's objective is
separable over states but for one ploidy row, so the oracle is the per-state
argmin, checked against that row.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any

import numpy as np
import pytest
from cnamaste.integer_copy import find_diploid_balanced_state, hill_climbing_integer_copynumber_fixdiploid_milp
from cnamaste.scripts import run_cnamaste as script

from audit.fn import Row, run, table, unchanged

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


# --- oracle --------------------------------------------------------------------


def _diploid(_: Any) -> None:
    log_mu = np.log(np.array([0.5, 0.95, 1.2, 1.0]))
    p = np.array([0.5, 0.48, 0.5, 0.2])
    pred = np.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    assert find_diploid_balanced_state(log_mu, p, pred, 0.0, 0.05) == 1, "balanced states 0-2; the closest mu to 1 is 0.95"
    assert find_diploid_balanced_state(log_mu, p, pred, 0.25, 0.05) == 2, "state 1 (2 of 10 bins) fails a 25% floor"
    with pytest.raises(ValueError):
        find_diploid_balanced_state(log_mu, np.full(4, 0.2), pred, 0.0, 0.05)


def _milp(calls: list[dict[str, Any]]) -> None:
    for k, call in enumerate(calls):
        log_mu, _, p, pred = call["args"]
        copies, cost, ploidy = call["out"]
        want, want_cost, want_ploidy = separable(log_mu, p, pred)
        assert np.isclose(cost, want_cost, rtol=1e-6), f"call {k}: cost {cost} against the separable minimum {want_cost}"
        assert np.array_equal(copies, want), f"call {k}: copies differ from the per-state argmin"
        assert ploidy == max(want_ploidy, 1), f"call {k}: the least ploidy bound the solution meets"


ORACLE: list[Row] = table(
    "oracle",
    ("integer_copy:find_diploid_balanced_state", "synthetic: 4 states", lambda c: None, _diploid, "the balanced, frequent state with mu closest to 1; none raises"),
    ("integer_copy:hill_climbing_integer_copynumber_fixdiploid_milp", "09_outputs/integer_copy_{0..3} in and out", recorded, _milp,
     "on each recorded call: the per-state argmin, its cost and its ploidy (the ploidy row does not bind)"),
)


# --- invariants ---------------------------------------------------------------------


def _bounds(calls: list[dict[str, Any]]) -> None:
    for call in calls:
        log_mu, base, p, pred = call["args"]
        copies, _, ploidy = unchanged(hill_climbing_integer_copynumber_fixdiploid_milp, log_mu, base, p, pred, **call["kwargs"])
        normal = find_diploid_balanced_state(log_mu, p, pred, 0.0, 0.05)
        assert np.all(copies >= 0) and np.all(copies <= 5) and np.all(copies.sum(axis=1) <= 6) and np.all(copies.sum(axis=1) > 0)
        assert copies[normal].tolist() == [1, 1] and 1 <= ploidy <= 4


def _config_caps(_: Any) -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(script.run_cnamaste)))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "hill_climbing_integer_copynumber_fixdiploid_milp"]
    assert calls and all({"max_allele_copy", "max_total_copy"} <= {k.arg for k in c.keywords} for c in calls), "the decoder takes its caps from the configuration"


INVARIANT: list[Row] = table(
    "invariant",
    ("integer_copy:hill_climbing_integer_copynumber_fixdiploid_milp", "09_outputs/integer_copy_{0..3} in", recorded, _bounds,
     "copies within the allele and total caps, the diploid state (1, 1), ploidy in 1..4; no input mutation"),
    ("integer_copy:hill_climbing_integer_copynumber_fixdiploid_milp", "source of run_cnamaste", lambda c: None, _config_caps, "run_cnamaste passes int_copy_num's caps",
     "Ticket#788: int_copy_num.max_allele_copy is ignored: run_cnamaste passes the decoder no cap from the configuration"),
)


@pytest.mark.parametrize("row", ORACLE, ids=[r.id for r in ORACLE])
def test_oracle(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """The decoder against the separable per-state minimum."""
    run(row, ctx, request)


@pytest.mark.parametrize("row", INVARIANT, ids=[r.id for r in INVARIANT])
def test_invariant(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """Caps, the fixed diploid state, no input mutation."""
    run(row, ctx, request)
