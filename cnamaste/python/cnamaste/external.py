"""OpenGM's Potts solvers, run in a subprocess one session at a time (S2, T- #836).

A trimmed copy of snakes_and_ladders' `sal.external` (MIT, M. J. Wilson; sal #1279, #1282), for its one framework
here. OpenGM is header-only C++ on no package index: `cnamaste/infra/build_opengm.sh` compiles its inference with
`infra/opengm/cnamaste_opengm.cxx` into `libcnamaste_opengm.so`, under `$CNAMASTE_OPENGM_HOME`, else
`${XDG_CACHE_HOME:-~/.cache}/cnamaste/opengm`. No Python package is added.

- **The library runs in a worker process,** `python -m cnamaste.external_worker <library>`, which loads it once with
  `ctypes` and serves one request per line, so its native code stays out of the run's process and a session pays the
  interpreter's start-up once, not per call.
- **Arrays cross as memory-mapped files** in the session's temporary directory. The parent creates every file, the
  outputs too, and unlinks each when its call returns or fails; the worker maps and never unlinks, so a worker that
  dies leaves nothing behind.
- **An absent library is refused before any subprocess,** with `OpenGMUnavailable` naming the build script; it is never
  a fallback to another solver. A worker that dies is a `WorkerError` carrying its standard error, and its session is
  closed from then on.

`Session.solve` takes a `cnamaste.potts.PottsProblem` and returns an `OpenGMRun` whose energy and bound are the
problem's, so they compare with `potts.problem_energy` directly. What a call returns is
reproducible: a block is named within the session's directory, and the inference's seconds, a measurement, are logged
rather than returned or stored.

Unlike `sal.external`, there is one transport (memory-mapped files, sal's default), one framework and no capability
registry: the algorithms are `ALGORITHMS`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

import numpy as np

from cnamaste.config import start_time
from cnamaste.logger import get_logger

logger = get_logger(__name__, start_time=start_time)

ALGORITHMS = {"icm": 0, "lbp": 1, "astar": 2, "trws": 3, "dd": 4, "expansion": 5, "swap": 6}
"""Each algorithm's index in `cnamaste_opengm.cxx`'s `Algorithm`."""

BUILD = "cnamaste/infra/build_opengm.sh"

LIBRARY = "libcnamaste_opengm.so"


class OpenGMUnavailable(RuntimeError):
    """The library is not built; raised before any subprocess."""


class WorkerError(RuntimeError):
    """The worker died, or the library refused the problem; the message is its standard error or the library's."""


class OpenGMRun(NamedTuple):
    """One solve: the labelling, its energy, the algorithm's lower bound (`-inf` where it gives none) and its bound
    after each iteration; the iterations reported."""

    labels: np.ndarray
    energy: float
    bound: float
    trace: np.ndarray
    iterations: int


def home() -> Path:
    """Where `build_opengm.sh` writes the library."""
    if os.environ.get("CNAMASTE_OPENGM_HOME"):
        return Path(os.environ["CNAMASTE_OPENGM_HOME"])
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "cnamaste" / "opengm"


def library() -> Path | None:
    """The built library, or None."""
    path = home() / LIBRARY
    return path if path.is_file() else None


def available() -> bool:
    return library() is not None


class Session:
    """One worker for many solves; a context manager that shuts the worker down and removes its files on exit."""

    def __init__(self) -> None:
        built = library()
        if built is None:
            raise OpenGMUnavailable(f"OpenGM is not built under {home()}: run {BUILD}")
        self.directory = Path(tempfile.mkdtemp(prefix="cnamaste-opengm-"))
        self.process: subprocess.Popen | None = subprocess.Popen(
            [sys.executable, "-m", "cnamaste.external_worker", str(built), str(self.directory)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._calls = 0
        self._reply()

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _reply(self) -> dict:
        assert self.process is not None and self.process.stdout is not None
        line = self.process.stdout.readline()
        if not line:
            stderr = self.process.stderr.read() if self.process.stderr is not None else ""
            self.close()
            raise WorkerError(f"the OpenGM worker died: {stderr.strip()}")
        return json.loads(line)

    def _block(self, name: str, values: np.ndarray) -> list:
        """`values`, at least one element, in a file the worker maps; returns the spec that crosses the pipe, the file
        named within the session's directory."""
        values = np.ascontiguousarray(values)
        if values.size == 0:
            values = np.zeros(1, dtype=values.dtype)
        named = f"{self._calls}-{name}"
        block = np.memmap(self.directory / named, dtype=values.dtype, mode="w+", shape=values.shape)
        block[...] = values
        block.flush()
        del block
        return [named, values.dtype.str, list(values.shape)]

    def solve(self, problem, algorithm: str, max_iterations: int = 1000, tolerance: float = 1e-9, start=None) -> OpenGMRun:
        """Run `algorithm` on `problem` (a `cnamaste.potts.PottsProblem`), from `start` where the algorithm takes one."""
        if self.process is None:
            raise WorkerError("the session is closed")
        if algorithm not in ALGORITHMS:
            raise ValueError(f"algorithm {algorithm!r} is not one of {sorted(ALGORITHMS)}")
        n, k = problem.unary.shape
        steps = max(int(max_iterations), 1)
        self._calls += 1
        blocks = {
            "unary": self._block("unary", np.asarray(problem.unary, dtype=np.float64)),
            "first": self._block("first", np.asarray(problem.first, dtype=np.int64)),
            "second": self._block("second", np.asarray(problem.second, dtype=np.int64)),
            "coupling": self._block("coupling", np.asarray(problem.coupling, dtype=np.float64)),
            "labels": self._block("labels", np.zeros(n, dtype=np.int64)),
            "trace": self._block("trace", np.zeros(steps, dtype=np.float64)),
            "seconds": self._block("seconds", np.zeros(1, dtype=np.float64)),
        }
        if start is not None:
            blocks["start"] = self._block("start", np.asarray(start, dtype=np.int64))
        request = {"algorithm": ALGORITHMS[algorithm], "n_nodes": n, "n_states": k, "n_edges": int(problem.first.size),
                   "max_iterations": steps, "tolerance": float(tolerance), "blocks": blocks}  # fmt: skip
        try:
            assert self.process.stdin is not None
            try:
                self.process.stdin.write(json.dumps(request) + "\n")
                self.process.stdin.flush()
            except BrokenPipeError:
                pass  # NB a dead worker: its empty reply below raises WorkerError with its standard error
            reply = self._reply()
            if reply["status"] != 0:
                raise WorkerError(f"OpenGM refused the problem: {reply['error']}")
            labels = np.array(np.memmap(self.directory / blocks["labels"][0], dtype=np.int64, mode="r", shape=(n,)))
            trace = np.array(np.memmap(self.directory / blocks["trace"][0], dtype=np.float64, mode="r", shape=(steps,))[: reply["taken"]])
            seconds = float(np.memmap(self.directory / blocks["seconds"][0], dtype=np.float64, mode="r", shape=(1,))[0])
        finally:
            for spec in blocks.values():
                (self.directory / spec[0]).unlink(missing_ok=True)
        logger.info(f"OpenGM {algorithm}: {n} nodes, {k} labels, {problem.first.size} pairs; energy {reply['value']:.6e}, "
                    f"bound {reply['bound']:.6e}, {reply['taken']} iterations, {seconds:.3f}s inference.")  # fmt: skip
        return OpenGMRun(labels, reply["value"], reply["bound"], trace, int(reply["taken"]))

    def close(self) -> None:
        """Shut the worker down and remove the session's files; idempotent."""
        if self.process is not None:
            process, self.process = self.process, None
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
        shutil.rmtree(self.directory, ignore_errors=True)


def session() -> Session:
    """A `Session`: `with external.session() as s: s.solve(problem, "trws")`."""
    return Session()


def solve(problem, algorithm: str, max_iterations: int = 1000, tolerance: float = 1e-9, start=None) -> OpenGMRun:
    """One solve in its own session: a one-shot call, which pays the worker's start-up."""
    with Session() as s:
        return s.solve(problem, algorithm, max_iterations, tolerance, start)
