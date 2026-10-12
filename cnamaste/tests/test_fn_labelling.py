"""Per-function rows for the clone-labelling solvers: `potts`, `external` and `hmrf`'s OpenGM path (S1-S3).

`hmrf.label_solver` is `icm` on easy, so `run_cnamaste` reaches these functions but easy does not execute them: every
row here is synthetic or poses easy's own labelling problem directly. The referees:

- **`potts`:** `cnaster`'s own `icm.calc_assignment_cost`, the quantity ICM maximizes, w / 2 per stored edge (T- #917).
- **`external`:** enumeration on graphs small enough to list every labelling.
- **`hmrf.opengm_assignment`:** TRW-S's lower bound and `cnaster`'s ICM on easy's BAF labelling problem: the fit's
  field over 3,000 spots and 4 clones, on its kNN adjacency (24,000 stored edges, 1,252 one way).

Every row on OpenGM skips where the library is not built (`cnamaste/infra/build_opengm.sh`).
"""

from __future__ import annotations

import itertools
import os
import subprocess
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp
from cnamaste import external, hmrf
from cnamaste.hmm_nophasing import hmm_nophasing
from cnamaste.hmrf import opengm_assignment, opengm_session, pipeline_clone_assignment
from cnamaste.hmrf_utils import cast_csr
from cnamaste.icm import calc_assignment_cost, icm_sweep_deque, unpack_adjacency
from cnamaste.potts import PottsProblem, potts_energy, potts_problem, problem_energy

from fn import Row, fitted, run, table
from test_fn_hmrf import field_case, one_way_case, window

ALGORITHMS = sorted(external.ALGORITHMS)


def _opengm() -> None:
    if not external.available():
        pytest.skip(f"OpenGM is not built: run {external.BUILD}")


# --- potts ---------------------------------------------------------------------


def _cnaster_cost(f: dict[str, Any], labels: np.ndarray, persample: Any = None, samples: Any = None) -> float:
    spots, neighbours, weights = unpack_adjacency(cast_csr(f["adjacency"].tocsr()))
    return calc_assignment_cost(f["llf"], spots, neighbours, weights, labels, f["weight"], persample, samples)


def _easy_window(ctx: Any) -> dict[str, Any]:
    """Easy's own kNN adjacency over `test_fn_hmrf`'s 300-spot window, a random field over 4 clones, weight 1."""
    w = window(ctx)
    return {"llf": np.random.default_rng(29).normal(scale=3.0, size=(w["idx"].size, 4)), "adjacency": w["adjacency"], "weight": 1.0}


def _energy_is_cnasters(f: dict[str, Any]) -> None:
    """`potts_energy` is `-calc_assignment_cost` over 20 labellings, with and without per-sample weights."""
    rng = np.random.default_rng(17)
    n, k = f["llf"].shape
    persample, samples = rng.normal(size=(k, 2)), rng.integers(0, 2, size=n)
    for _ in range(20):
        labels = rng.integers(0, k, size=n)
        assert np.isclose(potts_energy(f["llf"], f["adjacency"], labels, f["weight"]), -_cnaster_cost(f, labels), rtol=1e-12, atol=1e-9)
        assert np.isclose(potts_energy(f["llf"], f["adjacency"], labels, f["weight"], persample, samples),
                          -_cnaster_cost(f, labels, persample, samples), rtol=1e-12, atol=1e-9)  # fmt: skip


def _problem_is_the_energy(f: dict[str, Any]) -> None:
    """One coupling per unordered pair, `first < second`; a self-loop the constant offset; the energy of every
    labelling."""
    a = f["adjacency"].tolil()
    a[0, 0] = 1.5  # NB a self-loop: aligned in every labelling
    f = f | {"adjacency": a.tocsr()}
    problem = potts_problem(f["llf"], f["adjacency"], f["weight"])
    pairs = problem.first * len(f["llf"]) + problem.second
    assert np.all(problem.first < problem.second) and np.unique(pairs).size == pairs.size
    assert np.isclose(problem.offset, -f["weight"] * 1.5 / 2.0)
    rng = np.random.default_rng(23)
    for _ in range(20):
        labels = rng.integers(0, f["llf"].shape[1], size=len(f["llf"]))
        assert np.isclose(problem_energy(problem, labels), potts_energy(f["llf"], f["adjacency"], labels, f["weight"]), rtol=1e-12, atol=1e-9)


# --- external ------------------------------------------------------------------


def _graph(rows: list[int], cols: list[int], n: int, weights: Any = None) -> sp.csr_matrix:
    data = np.ones(len(rows)) if weights is None else np.asarray(weights, dtype=float)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def small_problems() -> dict[str, PottsProblem]:
    """Three graphs small enough to enumerate, 3 clones: a ring stored both ways, the ring stored one way with a chord
    and a self-loop, as a kNN graph is, and a random graph with uneven weights."""
    rng = np.random.default_rng(31)
    ring = [(i, (i + 1) % 6) for i in range(6)]
    both = _graph([a for a, b in ring] + [b for a, b in ring], [b for a, b in ring] + [a for a, b in ring], 6)
    one_way = _graph([a for a, b in ring] + [0, 2], [b for a, b in ring] + [3, 2], 6)
    pairs = [(a, b) for a in range(7) for b in range(7) if a != b and rng.random() < 0.3]
    random = _graph([a for a, b in pairs], [b for a, b in pairs], 7, rng.uniform(0.2, 2.0, size=len(pairs)))
    return {name: potts_problem(rng.normal(scale=1.5, size=(a.shape[0], 3)), a, 1.3)
            for name, a in (("ring", both), ("one-way ring", one_way), ("random", random))}  # fmt: skip


def _minimum(problem: PottsProblem) -> float:
    n, k = problem.unary.shape
    return min(problem_energy(problem, np.array(labels)) for labels in itertools.product(range(k), repeat=n))


def _against_enumeration(problems: dict[str, PottsProblem]) -> None:
    """Each algorithm's energy is its labels' (1e-9), at or above the minimum; A* attains it; a bound is at or below
    it; ICM descends from its start."""
    _opengm()
    with external.session() as s:
        for name, problem in problems.items():
            exact = _minimum(problem)
            for algorithm in ALGORITHMS:
                found = s.solve(problem, algorithm, max_iterations=200)
                assert np.isclose(found.energy, problem_energy(problem, found.labels), atol=1e-9), (name, algorithm)
                assert found.energy >= exact - 1e-9 and found.bound <= exact + 1e-9, (name, algorithm)
                if algorithm == "astar":
                    assert np.isclose(found.energy, exact, atol=1e-9), f"{name}: A* is exact"
            start = np.random.default_rng(37).integers(0, problem.unary.shape[1], size=problem.unary.shape[0])
            assert s.solve(problem, "icm", start=start).energy <= problem_energy(problem, start) + 1e-12, f"{name}: ICM from its start"


def _session_is_one_shot(problems: dict[str, PottsProblem]) -> None:
    """Every algorithm, twice: labels, energy, bound and trace bitwise in one session and one-shot."""
    _opengm()
    problem = problems["random"]
    with external.session() as s:
        many = [s.solve(problem, algorithm, max_iterations=50) for algorithm in ALGORITHMS for _ in range(2)]
    once = [external.solve(problem, algorithm, max_iterations=50) for algorithm in ALGORITHMS for _ in range(2)]
    for a, b in zip(many, once, strict=True):
        assert np.array_equal(a.labels, b.labels) and a.energy == b.energy and a.bound == b.bound and np.array_equal(a.trace, b.trace)


def _no_files_outlive(problems: dict[str, PottsProblem]) -> None:
    """No block outlives its call, the directory its session, or a killed worker its next call, which raises."""
    _opengm()
    with external.session() as s:
        directory = s.directory
        for algorithm in ALGORITHMS:
            s.solve(problems["ring"], algorithm)
            assert not list(directory.iterdir()), f"{algorithm} left files"
    assert not directory.exists()
    s = external.session()
    directory = s.directory
    s.process.kill()
    s.process.wait()
    with pytest.raises(external.WorkerError):
        s.solve(problems["ring"], "trws")
    assert s.process is None and not directory.exists()
    with pytest.raises(external.WorkerError, match="closed"):
        s.solve(problems["ring"], "trws")


def _absent_refused(problems: dict[str, PottsProblem]) -> None:
    """With no library under `$CNAMASTE_OPENGM_HOME`, `OpenGMUnavailable` names the build script and no process starts."""
    previous, popen = os.environ.get("CNAMASTE_OPENGM_HOME"), subprocess.Popen

    def refuse(*_: Any, **__: Any) -> None:
        raise AssertionError("a subprocess started")

    os.environ["CNAMASTE_OPENGM_HOME"] = os.devnull
    subprocess.Popen = refuse  # type: ignore[misc]
    try:
        assert external.library() is None and not external.available()
        with pytest.raises(external.OpenGMUnavailable, match="build_opengm.sh"):
            external.solve(problems["ring"], "trws")
    finally:
        subprocess.Popen = popen  # type: ignore[misc]
        if previous is None:
            del os.environ["CNAMASTE_OPENGM_HOME"]
        else:
            os.environ["CNAMASTE_OPENGM_HOME"] = previous


def _unknown_refused(problems: dict[str, PottsProblem]) -> None:
    _opengm()
    with external.session() as s, pytest.raises(ValueError, match="not one of"):
        s.solve(problems["ring"], "simplex")


# --- hmrf ----------------------------------------------------------------------


def easy_problem(ctx: Any) -> dict[str, Any]:
    """Easy's BAF labelling problem as the last iteration posed it: the fit's field, its kNN adjacency, its labels."""

    def make() -> dict[str, Any]:
        ctx.config  # noqa: B018  NB the staged configuration, label_solver absent: icm
        f = fitted(ctx, "05_baf")
        res, adjacency = f["res"], f["adjacency_mat"].tocsr()
        labels = np.asarray(res["new_assignment"]).copy()
        pred = np.argmax(np.asarray(res["log_gamma"]), axis=0)
        _, llf, _ = pipeline_clone_assignment(f["single_X"], f["single_base_nb_mean"], f["single_total_bb_RD"], res, pred, adjacency,
                                              labels.copy(), np.zeros(labels.size, dtype=int), 1.0, hmmclass=hmm_nophasing)  # fmt: skip
        return {"llf": llf, "adjacency": adjacency, "labels": labels, "weight": 1.0}

    return ctx.once("labelling/easy", make)


def _solvers_on_easy(e: dict[str, Any]) -> None:
    """Every OpenGM solver lowers the energy by what it returns, keeps every clone over the floor, and ends at or above
    TRW-S's bound; expansion, swap and TRW-S end at or below `cnaster`'s ICM, within 1e-2 of the bound."""
    _opengm()
    llf, a, w = e["llf"], e["adjacency"], e["weight"]
    np.random.seed(0)  # noqa: NPY002
    icm = e["labels"].copy()
    icm_sweep_deque(llf, a.indptr, a.indices, a.data, icm, w, None, min_clone_spots=200)
    cnaster = potts_energy(llf, a, icm, w)
    bound = opengm_session().solve(potts_problem(llf, a, w), "trws").bound
    found = {}
    for algorithm in ("icm", "expansion", "swap", "trws"):
        labels = e["labels"].copy()
        before = potts_energy(llf, a, labels, w)
        _, lowered = opengm_assignment(llf, a, labels, w, algorithm)
        found[algorithm] = potts_energy(llf, a, labels, w)
        assert np.isclose(before - found[algorithm], lowered, atol=1e-6), f"{algorithm}: the energy lowered"
        assert np.bincount(labels).min() >= 200, f"{algorithm}: the floor"
        assert found[algorithm] >= bound - 1e-6, f"{algorithm} below TRW-S's bound"
    print(f"cnaster ICM {cnaster:.3f}; bound {bound:.3f}; " + "; ".join(f"{k} {v:.3f}" for k, v in found.items()))
    for algorithm in ("expansion", "swap", "trws"):
        assert found[algorithm] <= cnaster + 1e-6 and found[algorithm] - bound <= 1e-2, algorithm


def _floor_handed_to_icm(f: dict[str, Any]) -> None:
    """A labelling with a clone under the floor goes to `icm_sweep_deque`, which empties or fills it; the energy lowered
    is still the difference."""
    _opengm()
    llf = f["llf"].copy()
    llf[:, 2] += 50.0  # NB clone 2 wins everywhere, so the floor of 40 leaves clones 0 and 1 under it
    labels = np.zeros(len(llf), dtype=np.int64)
    before = potts_energy(llf, f["adjacency"], labels, f["weight"])
    np.random.seed(0)  # noqa: NPY002
    _, lowered = opengm_assignment(llf, f["adjacency"], labels, f["weight"], "trws", min_clone_spots=40)
    counts = np.bincount(labels, minlength=3)
    assert counts[counts > 0].min() >= 40, f"clone sizes {counts.tolist()}"
    assert np.isclose(before - potts_energy(llf, f["adjacency"], labels, f["weight"]), lowered)


def _one_session_per_run(_: Any) -> None:
    """One worker for every call until it closes; a new one after."""
    _opengm()
    first = opengm_session()
    assert opengm_session() is first and first.process is not None
    first.close()
    second = opengm_session()
    assert second is not first and second.process is not None
    assert hmrf._SESSION == [second]


def _dispatch(ctx: Any) -> None:
    """`hmrf.label_solver` routes `pipeline_clone_assignment` on easy's BAF problem: absent, cnaster's ICM, bitwise;
    `opengm_trws`, `opengm_assignment`'s TRW-S, floor included, which is not ICM's labelling and no worse.

    Fails where the key is not read, or the field or adjacency the solver receives is not ICM's."""
    _opengm()
    e = easy_problem(ctx)
    f = fitted(ctx, "05_baf")
    config = ctx.config
    args = (f["single_X"], f["single_base_nb_mean"], f["single_total_bb_RD"], f["res"], np.argmax(np.asarray(f["res"]["log_gamma"]), axis=0), e["adjacency"])
    samples = np.zeros(e["labels"].size, dtype=int)
    np.random.seed(0)  # noqa: NPY002
    icm, llf, _ = pipeline_clone_assignment(*args, e["labels"].copy(), samples, 1.0, hmmclass=hmm_nophasing)
    labels = e["labels"].copy()
    np.random.seed(0)  # noqa: NPY002
    icm_sweep_deque(llf, e["adjacency"].indptr, e["adjacency"].indices, e["adjacency"].data, labels, 1.0, None, min_clone_spots=200)
    assert np.array_equal(icm, labels), "absent, the key is cnaster's ICM"
    config.hmrf.label_solver = "opengm_trws"
    try:
        np.random.seed(0)  # noqa: NPY002
        trws, llf_trws, _ = pipeline_clone_assignment(*args, e["labels"].copy(), samples, 1.0, hmmclass=hmm_nophasing)
    finally:
        del config.hmrf.label_solver
    assert np.array_equal(llf_trws, llf), "the same field"
    direct = e["labels"].copy()
    np.random.seed(0)  # noqa: NPY002
    opengm_assignment(llf, e["adjacency"], direct, 1.0, "trws")
    assert np.array_equal(trws, direct), "opengm_trws is opengm_assignment's TRW-S, floor included"
    assert not np.array_equal(trws, icm), "TRW-S's labelling is not ICM's: the row tells the two apart"
    assert potts_energy(llf, e["adjacency"], trws, 1.0) < potts_energy(llf, e["adjacency"], icm, 1.0), "lower than cnaster's ICM"


def _unknown_solver_refused(ctx: Any) -> None:
    """A `hmrf.label_solver` outside `config.LABEL_SOLVERS` is refused when the configuration is read."""
    config = ctx.config
    config.hmrf.label_solver = "simplex"
    try:
        with pytest.raises(ValueError, match="label_solver"):
            config.issue_warnings()
    finally:
        del config.hmrf.label_solver


SMALL = "synthetic: a ring, a one-way ring with a chord and a self-loop, a random graph; 3 clones"
EASY = "05_baf run_core_inference in and out: the field over 3,000 spots, 4 clones, the kNN adjacency"

ORACLE: list[Row] = table(
    "oracle",
    ("potts:potts_energy", "synthetic: 8x8 grid, 3 clones", lambda c: field_case(), _energy_is_cnasters, "-calc_assignment_cost, with and without per-sample weights"),
    ("potts:potts_energy", "synthetic: 8x8 grid, a tenth of directions dropped", lambda c: one_way_case(), _energy_is_cnasters,
     "on an asymmetric graph: -calc_assignment_cost (w / 2 per stored edge)"),
    ("potts:_field", "synthetic: 8x8 grid, a tenth of directions dropped", lambda c: one_way_case(), _energy_is_cnasters,
     "the field adds each spot's sample's per-sample weight"),
    ("potts:potts_problem", "synthetic: 8x8 grid, a tenth of directions dropped, a self-loop", lambda c: one_way_case(), _problem_is_the_energy,
     "one coupling per pair, first < second, J = weight (w_ij + w_ji) / 2; a self-loop the offset; the energy of every labelling"),
    ("potts:problem_energy", "synthetic: 8x8 grid, a tenth of directions dropped, a self-loop", lambda c: one_way_case(), _problem_is_the_energy,
     "potts_energy's, offset included"),
    ("external:Session.solve", SMALL, lambda c: small_problems(), _against_enumeration,
     "7 algorithms: energy of the labels, >= the minimum; A* exact; bounds <= the minimum; ICM descends from its start"),
    ("hmrf:opengm_assignment", EASY, easy_problem, _solvers_on_easy,
     "OpenGM's ICM, expansion, swap, TRW-S: >= TRW-S's bound, floor kept; expansion, swap, TRW-S <= cnaster's ICM, within 1e-2 of the bound"),
)

INVARIANT: list[Row] = table(
    "invariant",
    ("potts:potts_energy", "05_baf's kNN adjacency over a 300-spot window, a random field", _easy_window, _energy_is_cnasters,
     "on easy's kNN adjacency: -calc_assignment_cost"),
    ("potts:potts_problem", "05_baf's kNN adjacency over a 300-spot window, a random field", _easy_window, _problem_is_the_energy,
     "on easy's kNN adjacency: the energy of every labelling"),
    ("external:session", SMALL, lambda c: small_problems(), _session_is_one_shot, "a session's calls are the one-shot calls, bitwise"),
    ("external:Session.__init__", SMALL, lambda c: small_problems(), _session_is_one_shot, "each session starts its own worker; one-shot is a session"),
    ("external:Session._block", SMALL, lambda c: small_problems(), _no_files_outlive, "no block outlives its call"),
    ("external:Session.close", SMALL, lambda c: small_problems(), _no_files_outlive, "the directory goes with the session; idempotent"),
    ("external:Session._reply", SMALL, lambda c: small_problems(), _no_files_outlive, "a killed worker is WorkerError, its session closed and its files gone"),
    ("external:home", SMALL, lambda c: small_problems(), _absent_refused, "$CNAMASTE_OPENGM_HOME is where the library is looked for"),
    ("external:library", SMALL, lambda c: small_problems(), _absent_refused, "no library: OpenGMUnavailable naming the build script, no subprocess"),
    ("external:Session.solve", SMALL, lambda c: small_problems(), _unknown_refused, "an unknown algorithm is refused"),
    ("hmrf:opengm_assignment", "synthetic: 8x8 grid, 3 clones, clone 2 favoured by 50", lambda c: field_case(), _floor_handed_to_icm,
     "a clone under the floor is handed to icm_sweep_deque; the energy lowered is the difference"),
    ("hmrf:opengm_session", "synthetic", lambda c: None, _one_session_per_run, "one worker per run until closed; a new one after"),
    ("hmrf:pipeline_clone_assignment", EASY, lambda c: c, _dispatch,
     "hmrf.label_solver: absent, cnaster's ICM bitwise; opengm_trws, TRW-S's labelling of the same field, no worse than ICM"),
    ("config:YAMLConfig.issue_warnings", "the staged configuration", lambda c: c, _unknown_solver_refused, "an unknown hmrf.label_solver is refused"),
)


@pytest.mark.parametrize("row", ORACLE, ids=[r.id for r in ORACLE])
def test_oracle_agrees(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """cnaster's own cost, enumeration, TRW-S's bound."""
    run(row, ctx, request)


@pytest.mark.parametrize("row", INVARIANT, ids=[r.id for r in INVARIANT])
def test_invariants_hold(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """Sessions, files, refusals, the floor."""
    run(row, ctx, request)
