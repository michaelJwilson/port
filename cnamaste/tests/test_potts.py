"""`cnamaste.potts`: the clone-labelling step's Potts energy and its pairwise form (S1).

Plain tests until a solver reaches the module from `run_cnamaste` (S3), when they move into a table module and the
inventory. The referee is `cnaster`'s own `icm.calc_assignment_cost`, the quantity ICM maximizes: w / 2 per
stored edge, so a pair stored both ways counts w and one stored one way w / 2 (Ticket#917).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from cnamaste.hmrf_utils import cast_csr
from cnamaste.icm import calc_assignment_cost, unpack_adjacency
from cnamaste.potts import potts_energy, potts_problem, problem_energy

from test_fn_hmrf import field_case, one_way_case, window


def _cnaster_cost(f: dict[str, Any], labels: np.ndarray, persample: Any = None, samples: Any = None) -> float:
    spots, neighbours, weights = unpack_adjacency(cast_csr(f["adjacency"].tocsr()))
    return calc_assignment_cost(f["llf"], spots, neighbours, weights, labels, f["weight"], persample, samples)


def _easy_graph(ctx: Any) -> dict[str, Any]:
    """Easy's own kNN adjacency over `test_fn_hmrf`'s 300-spot window, a random field over 4 clones, weight 1."""
    w = window(ctx)
    return {"llf": np.random.default_rng(29).normal(scale=3.0, size=(w["idx"].size, 4)), "adjacency": w["adjacency"], "weight": 1.0}


CASES = {"8x8 grid": lambda ctx: field_case(), "8x8 grid, a tenth of directions dropped": lambda ctx: one_way_case(), "easy's kNN window": _easy_graph}


@pytest.mark.parametrize("case", list(CASES))
def test_the_energy_is_cnasters(ctx: Any, case: str) -> None:
    """`potts_energy` is `-calc_assignment_cost` over 20 random labellings.

    Fails where an edge is counted at w rather than w / 2, a direction is dropped or doubled, or a weight is misread."""
    f = CASES[case](ctx)
    rng = np.random.default_rng(17)
    n, k = f["llf"].shape
    for _ in range(20):
        labels = rng.integers(0, k, size=n)
        assert np.isclose(potts_energy(f["llf"], f["adjacency"], labels, f["weight"]), -_cnaster_cost(f, labels), rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("case", list(CASES))
def test_the_pairwise_form_is_the_energy(ctx: Any, case: str) -> None:
    """One coupling per unordered pair, `first < second`, and the energy of every labelling.

    Fails where a pair is posed twice, or a direction's weight is lost in the merge."""
    f = CASES[case](ctx)
    problem = potts_problem(f["llf"], f["adjacency"], f["weight"])
    pairs = problem.first * len(f["llf"]) + problem.second
    assert np.all(problem.first < problem.second) and np.unique(pairs).size == pairs.size
    rng = np.random.default_rng(23)
    for _ in range(20):
        labels = rng.integers(0, f["llf"].shape[1], size=len(f["llf"]))
        assert np.isclose(problem_energy(problem, labels), potts_energy(f["llf"], f["adjacency"], labels, f["weight"]), rtol=1e-12, atol=1e-9)


def test_a_self_loop_is_refused() -> None:
    """A spot stored as its own neighbour is refused by the energy and the pairwise form alike."""
    f = field_case()
    a = f["adjacency"].tolil()
    a[0, 0] = 1.5
    for call in (lambda: potts_energy(f["llf"], a.tocsr(), f["labels"], f["weight"]), lambda: potts_problem(f["llf"], a.tocsr(), f["weight"])):
        with pytest.raises(ValueError, match="self-loop"):
            call()
