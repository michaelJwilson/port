"""Studies at a stage of `run_cnaster_port --sal`, at the planted clones (#730).

A study measures what the run does. Its problem is the run's problem, built
by the run's code with `--sal`'s swaps installed; an oracle input is
substituted at one named point and said. Two parts:

- `members`: a manifest's realizations in turn, each drawn to disk as the
  run reads a sample (`port.sim.draw.realize`) and loaded as the run's
  fixtures are (`port.sim.fixtures.load_simulated`), with its hash.
- `at_oracle_clones`: `run_cnaster_port --sal` on one member, in this
  process, with its clones set to the planted ones (`--oracle-start`:
  `annotation.clone_label` and `hmrf.fixed_assignment`). At the named
  stage's `run_core_inference` call the clone index is replaced by the
  BAF stage's, which `--oracle-start` sets to the planted labels, so the
  run pools (`merge_pseudobulk_by_index_mix`), stacks (`clone_stack_obs`),
  derives `normal_lambda` and initializes on the oracle clones as it
  would on its own. At that call's first `pipeline_baum_welch` the study
  is handed a `Stage`: the call as the run made it, every argument as the
  run built it, and the planted `(A, B)` of each stacked row. The study
  replays the call with what it varies; every swap is still installed.
  The run then stops: nothing after the stage is the study's.

What a study varies is an argument of the run's call (`init_log_mu` and
`init_p_binom` for a start, `t` for a transition rate), so a difference
from the run is a difference the study names.

- `at_clone_assignment`: the same run, on to the stage's first
  `pipeline_clone_assignment` (#735). The run's Baum-Welch fits at the
  planted clones, from the run's own initial states or, with
  `states="planted"`, from the planted ones (`copy_state_stream.oracle_states`,
  a second oracle input). The installed clone assignment, `--sal`'s, then
  computes the field; `--oracle-start` fixes the assignment, so it solves
  nothing. The study is handed a `Field`: that `(spots, clones)` field, the
  run's adjacency and `spatial_weight`, and each spot's planted clone, the
  Potts problem the run's solver is handed at that stage. The read-depth
  refinement's mask is not in it: the mask is folded in only where the
  solver runs, and at the planted clones of `--oracle-start` it is absent.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, NamedTuple, TypeVar, cast

import numpy as np

__all__ = [
    "Field",
    "Member",
    "Stage",
    "at_clone_assignment",
    "at_oracle_clones",
    "members",
    "missed",
]

_T = TypeVar("_T")

FLAGS = ("--sal", "--no-plots")
"""The run every study mirrors."""


class Member(NamedTuple):
    """One realization of a manifest, on disk as the run reads it."""

    realization: int
    sample: Any
    """`port.sim.fixtures.SimulatedSample`."""
    hash: str
    """`port.sim.fixtures.realization_hash`."""


class Stage(NamedTuple):
    """The run's `pipeline_baum_welch` call at a stage, at the planted clones.

    `X`, `lengths`, `base_nb_mean`, `total_bb_RD` and the rest are the run's
    clone-stacked arrays (`arguments`), clones in the sample's label order;
    `planted` is each stacked row's planted `(A, B)`.
    """

    name: str
    call: Callable[..., Any]
    args: tuple[Any, ...]
    arguments: dict[str, Any]
    planted: np.ndarray
    """`(rows, 2)`: each clone-stacked row's planted `(A, B)`."""
    clone: np.ndarray
    """`(rows,)`: each row's clone, `0` the normal."""
    n_clones: int
    config: str
    """The run's `config.yaml`, which `cnaster`'s helpers read."""
    contig: np.ndarray
    """`(bins,)`: each bin's contig, as the run's segment level names it; `start` and `length` likewise."""
    start: np.ndarray
    length: np.ndarray

    def run(self, **replaced: Any) -> Any:
        """The run's call with `replaced` arguments: its `pipeline_baum_welch` result."""
        return self.call(*self.args, **{**self.arguments, **replaced})

    @property
    def n_states(self) -> int:
        return int(self.args[3])

    @property
    def X(self) -> np.ndarray:
        return np.asarray(self.args[1])

    @property
    def lengths(self) -> np.ndarray:
        return np.asarray(self.args[2])

    @property
    def base_nb_mean(self) -> np.ndarray:
        return np.asarray(self.args[4])

    @property
    def total_bb_RD(self) -> np.ndarray:
        return np.asarray(self.args[5])


class Field(NamedTuple):
    """The run's clone-assignment problem at a stage, at the planted clones.

    Field `k` is clone `k` of the run's clone index, the planted labels'
    order, `0` the normal; `planted[i]` is spot `i`'s clone.
    """

    name: str
    field: np.ndarray
    """`(spots, clones)`: each spot's log-likelihood under each clone, as `pipeline_clone_assignment` computes it."""
    planted: np.ndarray
    """`(spots,)`."""
    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    """The run's `adjacency_mat`, CSR."""
    spatial_weight: float
    states: str
    """`"run"` or `"planted"`: the Baum-Welch's initial states."""
    seconds: float
    """From the run's start to the field."""

    @property
    def n_spots(self) -> int:
        return int(self.field.shape[0])


def members(
    manifest: Path, root: Path, n: int | None = None, first: int = 0
) -> Iterator[Member]:
    """`manifest`'s realizations `first`, `first + 1`, ... (`n` of them if given), drawn under `root`."""
    from dataclasses import replace

    from port.sim import draw as d
    from port.sim.fixtures import load_simulated, realization_hash

    drawn = d.read_manifest(manifest)
    if n is not None:
        drawn = replace(
            drawn,
            tables=d.merged_tables(
                drawn.tables, {"sample": {"realizations": first + n}}
            ),
        )
    out = root / Path(manifest).stem
    for realized in d.realize(drawn, out):
        if realized.index < first:
            continue
        (path,) = out.glob(f"r*{realized.index}")
        yield Member(
            realized.index,
            load_simulated(path.name, path.parent),
            realization_hash(path),
        )


class _Done(Exception):
    def __init__(self, result: Any) -> None:
        self.result = result


def _planted(
    sample: Any, lineage: Any, n_rows: int, n_clones: int
) -> tuple[np.ndarray, Any]:
    """Each stacked row's planted `(A, B)`, clones along the genome, and the segment level the stage pooled."""
    level = next(
        s
        for s in reversed(list(lineage.levels.values()))
        if s.n_segments * n_clones == n_rows
    )
    middle = np.asarray(level.start) + np.asarray(level.length) // 2
    chromosome = np.char.replace(np.asarray(level.contig).astype(str), "chr", "")
    copies = sample.copies_at(chromosome, middle)[:, :n_clones, :]
    return np.asarray(
        copies.transpose(1, 0, 2).reshape(n_rows, 2), dtype=np.int64
    ), level


def _drive(
    sample: Any,
    stage: str,
    overrides: dict[str, Any] | None,
    root: Path | None,
    on_fit: Callable[[Stage], Any] | None,
    on_assignment: Callable[[tuple[Any, ...], dict[str, Any], Callable[..., Any]], Any]
    | None,
) -> Any:
    """`run_cnaster_port --sal --oracle-start` on `sample` to `stage`, its clones the planted ones.

    At the stage's first `pipeline_baum_welch` the `Stage` goes to `on_fit`:
    its result ends the run where `on_assignment` is `None`, and otherwise,
    if a dict, replaces arguments of the run's call. At the stage's first
    `pipeline_clone_assignment`, `on_assignment` takes the call's arguments
    and the installed function, and its result ends the run.
    """
    from cnaster import hmrf

    from port.extensions import segments
    from port.patch.hmrf import core_inference
    from port.qa.audit import audit_sample

    upstream, baum_welch = core_inference.UPSTREAM, hmrf.pipeline_baum_welch
    held: dict[str, Any] = {"oracle": None, "inside": False, "fitted": False}

    def assign(*args: Any, **arguments: Any) -> Any:
        installed = held["assignment"]
        if not held["inside"] or on_assignment is None:
            return installed(*args, **arguments)
        raise _Done(on_assignment(args, arguments, installed))

    def inference(**arguments: Any) -> Any:
        name = "rdrbaf" if "m" in str(arguments.get("params")) else "baf"
        if held["oracle"] is None:
            held["oracle"] = [np.asarray(i) for i in arguments["initial_clone_index"]]
        if name == stage:
            # NB the substitution: the planted clones, as `--oracle-start` set them for the BAF stage
            arguments = {**arguments, "initial_clone_index": held["oracle"]}
            held["inside"] = True
        # NB the installed clone assignment is `--sal`'s, bound when the run installed its swaps
        held["assignment"] = hmrf.pipeline_clone_assignment
        hmrf.pipeline_clone_assignment = assign
        try:
            return upstream(**arguments)
        finally:
            hmrf.pipeline_clone_assignment = held["assignment"]

    def fit(*args: Any, **arguments: Any) -> Any:
        if not held["inside"] or held["fitted"] or on_fit is None:
            return baum_welch(*args, **arguments)
        held["fitted"] = True
        n_clones = len(held["oracle"])
        rows = int(np.asarray(args[1]).shape[0])
        # NB the run writes its configuration at its root, beside `output/`
        config = Path(held["root"]) / "config.yaml"
        planted, level = _planted(sample, lineage, rows, n_clones)
        found = Stage(
            stage, baum_welch, args, arguments, planted, np.repeat(np.arange(n_clones), rows // n_clones),
            n_clones, config.read_text() if config.exists() else "",
            np.asarray(level.contig).astype(str), np.asarray(level.start), np.asarray(level.length),
        )  # fmt: skip
        result = on_fit(found)
        if on_assignment is None:
            raise _Done(result)
        return baum_welch(*args, **{**arguments, **(result or {})})

    root = Path(tempfile.mkdtemp()) if root is None else root
    held["root"] = root
    core_inference.UPSTREAM, hmrf.pipeline_baum_welch = inference, fit
    try:
        with segments.recording() as lineage:
            audit_sample(sample, list(FLAGS), overrides, root, oracle=True)
    except _Done as done:
        return done.result
    finally:
        core_inference.UPSTREAM, hmrf.pipeline_baum_welch = upstream, baum_welch
    msg = f"{stage}: the run finished without reaching the study's call"
    raise RuntimeError(msg)


def at_oracle_clones(
    sample: Any,
    study: Callable[[Stage], _T],
    *,
    stage: str = "rdrbaf",
    overrides: dict[str, Any] | None = None,
    root: Path | None = None,
) -> _T:
    """`study` on `stage`'s `pipeline_baum_welch` call of `run_cnaster_port --sal` on `sample` at its planted clones.

    `stage` is `"baf"` (`params` without `m`) or `"rdrbaf"`. `overrides` sets
    configuration keys, `section.key` to a value, as `run_audit --set` does.
    """
    return cast(_T, _drive(sample, stage, overrides, root, study, None))


def at_clone_assignment(
    sample: Any,
    study: Callable[[Field], _T],
    *,
    stage: str = "rdrbaf",
    states: str = "run",
    overrides: dict[str, Any] | None = None,
    root: Path | None = None,
) -> _T:
    """`study` on the clone-assignment field `run_cnaster_port --sal` builds at `stage`, at `sample`'s planted clones.

    `states` is `"run"`, the run's own initial states for the Baum-Welch
    before it, or `"planted"`, the planted ones (`oracle_states`).
    """
    import time

    if states not in ("run", "planted"):
        msg = f"states is 'run' or 'planted', got {states!r}"
        raise ValueError(msg)
    opened = time.perf_counter()

    def start(found: Stage) -> dict[str, Any]:
        if states == "run":
            return {}
        from port.studies.copy_state_stream import oracle_states

        log_mu, p_binom = oracle_states(found)
        return {"init_log_mu": log_mu, "init_p_binom": p_binom}

    def capture(
        args: tuple[Any, ...], arguments: dict[str, Any], installed: Callable[..., Any]
    ) -> _T:
        import scipy.sparse as sp

        bound = _bind(args, arguments)
        # NB `--oracle-start` fixes the assignment: the installed call builds the field and solves nothing
        _, field, _ = installed(*args, **arguments)
        # NB the run's assignment at the stage's first call: the planted clones, as `run_core_inference`
        #    builds it from the clone index (`hmrf.py:545`)
        planted = np.asarray(bound["prev_assignment"], dtype=np.int64).copy()
        graph = sp.csr_matrix(bound["adjacency_mat"])
        graph.sort_indices()
        return study(
            Field(stage, np.asarray(field, dtype=np.float64), planted, graph.indptr, graph.indices,
                  graph.data.astype(np.float64), float(bound["spatial_weight"]), states,
                  time.perf_counter() - opened)
        )  # fmt: skip

    return cast(_T, _drive(sample, stage, overrides, root, start, capture))


def _bind(args: tuple[Any, ...], arguments: dict[str, Any]) -> dict[str, Any]:
    """`pipeline_clone_assignment`'s arguments by name."""
    import inspect

    # NB `cnaster`'s own, held at import: the module name is the study's wrapper while the run is inside
    from port.patch.hmrf.clone_assignment import UPSTREAM

    signature = inspect.signature(UPSTREAM)
    return dict(signature.bind_partial(*args, **arguments).arguments)


def missed(label: np.ndarray, truth: np.ndarray) -> int:
    """Rows whose state is not the planted one, under the 1-1 matching of fitted to planted states that misses fewest.

    The audits' matcher (`port.qa.scoring.matched` on `overlap`), on states
    rather than clones.
    """
    from port.qa.scoring import matched, overlap

    label, truth = np.asarray(label, dtype=np.int64), np.asarray(truth, dtype=np.int64)
    n = int(max(label.max(), truth.max())) + 1
    counts = overlap(truth, label, n, n)
    kept = sum(int(counts[p, f]) for p, f in matched(counts).items())
    return int(label.size - kept)
