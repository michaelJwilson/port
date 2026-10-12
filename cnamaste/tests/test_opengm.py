"""`cnamaste.external`: OpenGM's Potts solvers through the worker (S2).

Plain tests until a solver reaches the module from `run_cnamaste` (S3). The referee is enumeration: on graphs small
enough to list every labelling, each algorithm's energy is `potts.problem_energy` of its labels, never below the
minimum, and a lower bound never above it. Every row but the refusal skips where the library is not built
(`cnamaste/infra/build_opengm.sh`).
"""

from __future__ import annotations

import itertools
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp
from cnamaste import external
from cnamaste.potts import PottsProblem, potts_problem, problem_energy

built = pytest.mark.skipif(not external.available(), reason=f"OpenGM is not built: run {external.BUILD}")

ALGORITHMS = sorted(external.ALGORITHMS)


def _graph(rows: list[int], cols: list[int], n: int, weights: Any = None) -> sp.csr_matrix:
    data = np.ones(len(rows)) if weights is None else np.asarray(weights, dtype=float)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def _problems() -> dict[str, PottsProblem]:
    """Three graphs small enough to enumerate, 3 clones: a ring stored both ways, the ring stored one way with a chord
    (as a kNN graph is), and a random graph with uneven weights."""
    rng = np.random.default_rng(31)
    ring = [(i, (i + 1) % 6) for i in range(6)]
    both = _graph([a for a, b in ring] + [b for a, b in ring], [b for a, b in ring] + [a for a, b in ring], 6)
    one_way = _graph([a for a, b in ring] + [0], [b for a, b in ring] + [3], 6)
    pairs = [(a, b) for a in range(7) for b in range(7) if a != b and rng.random() < 0.3]
    random = _graph([a for a, b in pairs], [b for a, b in pairs], 7, rng.uniform(0.2, 2.0, size=len(pairs)))
    return {name: potts_problem(rng.normal(scale=1.5, size=(a.shape[0], 3)), a, 1.3)
            for name, a in (("ring", both), ("one-way ring", one_way), ("random", random))}  # fmt: skip


PROBLEMS = _problems()


def _minimum(problem: PottsProblem) -> float:
    n, k = problem.unary.shape
    return min(problem_energy(problem, np.array(labels)) for labels in itertools.product(range(k), repeat=n))


@built
@pytest.mark.parametrize("name", list(PROBLEMS))
def test_every_algorithm_against_enumeration(name: str) -> None:
    """Each algorithm's energy is its labels' energy (1e-9), at or above the minimum; A* attains it; a bound, where
    one is given, is at or below it.

    Fails where the model posed to OpenGM is not the problem's (a coupling's sign, a pair, the unary's sign), or a bound is
    not one."""
    problem = PROBLEMS[name]
    exact = _minimum(problem)
    with external.session() as s:
        for algorithm in ALGORITHMS:
            run = s.solve(problem, algorithm, max_iterations=200)
            assert np.isclose(run.energy, problem_energy(problem, run.labels), atol=1e-9), algorithm
            assert run.energy >= exact - 1e-9, f"{algorithm} below the minimum"
            assert run.bound <= exact + 1e-9, f"{algorithm}'s bound above the minimum"
            if algorithm == "astar":
                assert np.isclose(run.energy, exact, atol=1e-9), "A* is exact"
            if algorithm == "trws":
                print(f"{name}: minimum {exact:.6f}, TRW-S {run.energy:.6f}, bound {run.bound:.6f}, {run.iterations} iterations")


@built
@pytest.mark.parametrize("name", list(PROBLEMS))
def test_icm_descends_from_its_start(name: str) -> None:
    """ICM from a given labelling ends at or below that labelling's energy.

    Fails where the start is not read, or is read as another labelling."""
    problem = PROBLEMS[name]
    rng = np.random.default_rng(37)
    with external.session() as s:
        for _ in range(5):
            start = rng.integers(0, problem.unary.shape[1], size=problem.unary.shape[0])
            assert s.solve(problem, "icm", start=start).energy <= problem_energy(problem, start) + 1e-12


@built
def test_a_session_is_the_one_shot_call() -> None:
    """Every algorithm's labels, energy, bound and trace, bitwise, in one session and one-shot.

    Fails where a session's worker carries state from one call into the next."""
    problem = PROBLEMS["random"]
    with external.session() as s:
        many = [s.solve(problem, algorithm, max_iterations=50) for algorithm in ALGORITHMS for _ in range(2)]
    once = [external.solve(problem, algorithm, max_iterations=50) for algorithm in ALGORITHMS for _ in range(2)]
    for a, b in zip(many, once):
        assert np.array_equal(a.labels, b.labels) and a.energy == b.energy and a.bound == b.bound and np.array_equal(a.trace, b.trace)


@built
def test_a_session_leaves_no_files() -> None:
    """No block outlives its call; the directory goes with the session."""
    with external.session() as s:
        directory = s.directory
        for algorithm in ALGORITHMS:
            s.solve(PROBLEMS["ring"], algorithm)
            assert not list(directory.iterdir()), f"{algorithm} left files"
    assert not directory.exists()


@built
def test_a_dead_worker_is_an_error_and_leaves_no_files() -> None:
    """A worker killed between calls surfaces as `WorkerError` on the next; the session is closed and its files gone.

    Fails where a dead worker hangs the parent or leaves its blocks behind."""
    s = external.session()
    directory = s.directory
    s.solve(PROBLEMS["ring"], "trws")
    s.process.kill()
    s.process.wait()
    with pytest.raises(external.WorkerError):
        s.solve(PROBLEMS["ring"], "trws")
    assert s.process is None and not directory.exists()
    with pytest.raises(external.WorkerError):
        s.solve(PROBLEMS["ring"], "trws")


def test_an_absent_library_is_refused_before_a_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no library built, `OpenGMUnavailable` names the build script, and no process starts."""
    monkeypatch.setenv("CNAMASTE_OPENGM_HOME", str(tmp_path))

    def refuse(*_: Any, **__: Any) -> None:
        raise AssertionError("a subprocess started")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    with pytest.raises(external.OpenGMUnavailable, match="build_opengm.sh"):
        external.solve(PROBLEMS["ring"], "trws")


def test_an_unknown_algorithm_is_refused() -> None:
    if not external.available():
        pytest.skip(f"OpenGM is not built: run {external.BUILD}")
    with external.session() as s, pytest.raises(ValueError, match="not one of"):
        s.solve(PROBLEMS["ring"], "simplex")
