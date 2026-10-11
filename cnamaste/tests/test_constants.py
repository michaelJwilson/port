"""The hard-coded constants no configuration key sets: each one's value and place, and how often it binds on each fixture.

One row per constant (or per place it acts): the literal, the live function
holding it (`file:line` at the pin), and a count of the items it bound or
clipped in the staged run, pinned per fixture hash. A rise in a count fails
even when nothing else does; a count the capture holds no data for is a skip
with its reason. Tests only: nothing here changes cnamaste.

`hmrf.ari_tolerance` is a configuration key (read at hmrf.py:709), so it is
not in the table; the Neyman-Pearson thresholds are keys too, and their only
call sits in a string literal, which the table pins as never binding.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from cnamaste import hmm_initialize, hmm_nophasing, hmrf, icm, spatial
from cnamaste.scripts import run_cnamaste as script

from audit.fn import Ctx, final_stack, fitted

FITS = ("05_baf", "08_rdr")
TRACE = "the capture holds one ICM input and output per fit, no per-iteration trace: waits for the fit-trace capture"
OPTIMIZER = "the capture holds no optimizer result per M step: waits for the fit-trace capture"


@dataclass(frozen=True)
class Constant:
    name: str
    value: float | str
    """The literal, or the source text of a constant that is not one literal."""
    where: str
    """`file:line` at the pin; the check finds the literal in `function`, so an unrelated edit above it does not fail."""
    function: Callable[..., Any]
    count: Callable[[Ctx], int]
    """Items the constant bound or clipped on the staged run."""
    pinned: dict[str, int] = field(default_factory=dict)
    """Fixture hash -> count."""
    replay: bool = False
    """The count needs a replayed stage: conftest puts the row on the replay's xdist group."""

    @property
    def id(self) -> str:
        value = self.value if isinstance(self.value, str) else f"{self.value:g}"
        return f"{self.name} = {value} ({self.where})"


def literals(function: Callable[..., Any]) -> set[float]:
    """Every numeric literal in `function`'s source (defaults included)."""
    body = getattr(function, "py_func", function)
    tree = ast.parse(textwrap.dedent(inspect.getsource(body)))
    return {float(n.value) for n in ast.walk(tree) if isinstance(n, ast.Constant) and type(n.value) in (int, float)}


def results(ctx: Ctx) -> list[Any]:
    return [ctx.sim.stored(f"{f}/run_core_inference/out") for f in FITS]


# --- counts ----------------------------------------------------------------------


def under_floor(ctx: Ctx) -> int:
    """Clones of 1..199 spots in each fit's recorded ICM input, output and raw labels (the floor dissolves such a clone)."""
    found = 0
    for f in FITS:
        internal = ctx.sim.internal(f)
        for key in ("prev", "icm", "raw"):
            sizes = np.bincount(internal[key].astype(int))
            found += int(np.sum((sizes > 0) & (sizes < 200)))
    return found


def clones_lost(ctx: Ctx) -> int:
    """Clones a fit started with and ended without (lineage `*_init`/`initial_baf` -> `*_raw`): the floor's dissolutions are among them."""
    labels, levels, _ = ctx.sim.clones()
    lost = 0
    for start, end in (("initial_baf", "baf_raw"), ("rdr_init", "rdr_raw")):
        lost += np.unique(labels[:, levels.index(start)]).size - np.unique(labels[:, levels.index(end)]).size
    return int(lost)


def rdr_capped(ctx: Ctx) -> int:
    """Final-stack bins with X / base > 5, whose baseline the M step zeroes (hmm_nophasing.py:825-829)."""
    found = 0
    for f in FITS:
        s = final_stack(ctx, f)
        x, base = s["X"][:, 0, :], s["base_nb_mean"]
        with np.errstate(divide="ignore", invalid="ignore"):
            found += int(np.sum((base > 0) & (x / base > 5.0)))
    return found


def trace(reason: str) -> Callable[[Ctx], int]:
    def count(_: Ctx) -> int:
        pytest.skip(reason)
    return count


def fitted_below(quantity: Callable[[Any], np.ndarray]) -> Callable[[Ctx], int]:
    """Fitted values (both fits, every state) below the 1e-10 floor."""
    return lambda ctx: int(sum(np.sum(quantity(r) < 1e-10) for r in results(ctx)))


def p_clipped(ctx: Ctx) -> int:
    return int(sum(np.sum((np.asarray(r["new_p_binom"]) <= 1e-6) | (np.asarray(r["new_p_binom"]) >= 1 - 1e-6)) for r in results(ctx)))


def rectangles_under(ctx: Ctx) -> int:
    """Initial clones at or under 20% of an equal share (0.2 n / K spots)."""
    clones = ctx.sim.stored("03_phasing/initialize_clones/out")
    n = sum(len(c) for c in clones)
    return int(sum(len(c) <= 0.2 * n / len(clones) for c in clones))


def merged_by_minspots(ctx: Ctx) -> int:
    """Clones merge_by_minspots folded into another (its UMI floor is n_obs x hmrf.min_avgumi_per_clone, run_cnamaste.py:766, 1193)."""
    return int(sum(np.unique(np.asarray(r["new_assignment"])).size - len(ctx.sim.stored(f"{f}/merge_by_minspots/out/0"))
                   for f, r in zip(FITS, results(ctx), strict=True)))


def neyman_pearson_calls(_: Ctx) -> int:
    """Live calls to neyman_pearson_similarity in run_cnamaste (both sit in string literals, run_cnamaste.py:731, 1163)."""
    tree = ast.parse(inspect.getsource(script))
    return sum(isinstance(n, ast.Call) and getattr(n.func, "id", "") == "neyman_pearson_similarity" for n in ast.walk(tree))


EASY = "2d4ce9a9"

CONSTANTS: list[Constant] = [
    Constant("icm clone floor min_clone_spots", 200, "icm.py:821", icm.icm_sweep_deque, under_floor, {EASY: 0}),
    Constant("icm clone floor: clones lost over the fit (upper bound)", 200, "icm.py:821", icm.icm_sweep_deque, clones_lost, {EASY: 4}),
    Constant("max_rdr: X / base above it zeroes the baseline", 5.0, "hmm_nophasing.py:808",
             hmm_nophasing.hmm_nophasing._run_optimization_pipeline, rdr_capped, {EASY: 208}, replay=True),
    Constant("optimizer gtol", 1e-5, "hmm_nophasing.py:1008", hmm_nophasing.hmm_nophasing._run_optimization_pipeline, trace(OPTIMIZER)),
    Constant("optimizer ftol (BFGS does not read it)", 1e-6, "hmm_nophasing.py:1007", hmm_nophasing.hmm_nophasing._run_optimization_pipeline, trace(OPTIMIZER)),
    Constant("NB dispersion floor phi", 1e-10, "hmm_nophasing.py:48", hmm_nophasing._nb_logpmf_1d,
             fitted_below(lambda r: np.asarray(r["new_alphas"])), {EASY: 0}),
    Constant("BB floor p tau", 1e-10, "hmm_nophasing.py:63", hmm_nophasing._bb_logpmf_1d,
             fitted_below(lambda r: np.asarray(r["new_p_binom"]) * np.asarray(r["new_taus"])), {EASY: 0}),
    Constant("BB floor (1 - p) tau", 1e-10, "hmm_nophasing.py:63", hmm_nophasing._bb_logpmf_1d,
             fitted_below(lambda r: (1 - np.asarray(r["new_p_binom"])) * np.asarray(r["new_taus"])), {EASY: 0}),
    Constant("BB floor, dense kernel", 1e-10, "hmm_nophasing.py:91", hmm_nophasing._dense_bb_logpmf,
             fitted_below(lambda r: np.minimum(np.asarray(r["new_p_binom"]), 1 - np.asarray(r["new_p_binom"])) * np.asarray(r["new_taus"])), {EASY: 0}),
    Constant("p clip (use_logit=False only)", 1e-6, "hmm_nophasing.py:651", hmm_nophasing.hmm_nophasing.unpack_params, p_clipped, {EASY: 0}),
    Constant("rectangle test: 20% of an equal share", 0.2, "spatial.py:292, 328", spatial.initialize_rectangular_clones, rectangles_under, {EASY: 0}),
    Constant("rectangle test: rejected draws", 0.2, "spatial.py:328", spatial.initialize_rectangular_clones, trace("the capture holds the accepted partition, not the draws rejected before it")),
    Constant("merge_by_minspots UMI floor: clones merged", "n_obs * config.hmrf.min_avgumi_per_clone", "run_cnamaste.py:766, 1193",
             script.run_cnamaste, merged_by_minspots, {EASY: 0}),
    Constant("merge_by_minspots tumour threshold default", 0.5, "hmrf.py:899", hmrf.merge_by_minspots, merged_by_minspots, {EASY: 0}),
    Constant("Neyman-Pearson merge (thresholds are keys): live calls", "neyman_pearson_similarity(", "run_cnamaste.py:745, 1174",
             script.run_cnamaste, neyman_pearson_calls, {EASY: 0}),
    Constant("gmm_init rdr_ratio clip", 1e-6, "hmm_initialize.py:563", hmm_initialize.gmm_init,
             trace("the capture holds gmm_init's outputs, not the ratios it clips: waits for the fit-trace capture")),
]


@pytest.mark.parametrize("row", CONSTANTS, ids=[c.id for c in CONSTANTS])
def test_constant_value(row: Constant) -> None:
    """The literal is where the table says, at the table's value."""
    if isinstance(row.value, str):
        assert row.value in inspect.getsource(row.function), f"{row.value!r} not in {row.function.__qualname__} ({row.where})"
        return
    assert float(row.value) in literals(row.function), f"{row.value:g} not in {row.function.__qualname__} ({row.where})"


@pytest.mark.parametrize("row", CONSTANTS, ids=[c.id for c in CONSTANTS])
def test_constant_binds(row: Constant, sim_hash: str, ctx: Ctx) -> None:
    """How often the constant bound or clipped on this fixture, against the pinned count."""
    ctx.config  # noqa: B018  NB counts read under the staged run's configuration
    found = row.count(ctx)
    print(f"{sim_hash} {row.id}: {found}")
    if sim_hash not in row.pinned:
        pytest.skip(f"no pinned count for {sim_hash}: measured {found}")
    assert found == row.pinned[sim_hash], f"{row.name}: {found} on {sim_hash}, pinned {row.pinned[sim_hash]}"
