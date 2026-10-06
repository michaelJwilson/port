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
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, NamedTuple, TypeVar

import numpy as np

__all__ = ["Member", "Stage", "at_oracle_clones", "members", "missed"]

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
            tables=d._merge(drawn.tables, {"sample": {"realizations": first + n}}),
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
    from cnaster import hmrf

    from port.extensions import segments
    from port.patch.hmrf import core_inference
    from port.qa.audit import audit_sample

    upstream, baum_welch = core_inference.UPSTREAM, hmrf.pipeline_baum_welch
    held: dict[str, Any] = {"oracle": None, "inside": False}

    def inference(**arguments: Any) -> Any:
        name = "rdrbaf" if "m" in str(arguments.get("params")) else "baf"
        if held["oracle"] is None:
            held["oracle"] = [np.asarray(i) for i in arguments["initial_clone_index"]]
        if name == stage:
            # NB the substitution: the planted clones, as `--oracle-start` set them for the BAF stage
            arguments = {**arguments, "initial_clone_index": held["oracle"]}
            held["inside"] = True
        return upstream(**arguments)

    def fit(*args: Any, **arguments: Any) -> Any:
        if not held["inside"]:
            return baum_welch(*args, **arguments)
        n_clones = len(held["oracle"])
        rows = int(np.asarray(args[1]).shape[0])
        config = Path(held["root"]) / "output" / "config.yaml"
        planted, level = _planted(sample, lineage, rows, n_clones)
        found = Stage(
            stage, baum_welch, args, arguments, planted, np.repeat(np.arange(n_clones), rows // n_clones),
            n_clones, config.read_text() if config.exists() else "",
            np.asarray(level.contig).astype(str), np.asarray(level.start), np.asarray(level.length),
        )  # fmt: skip

        raise _Done(study(found))

    root = Path(tempfile.mkdtemp()) if root is None else root
    held["root"] = root
    core_inference.UPSTREAM, hmrf.pipeline_baum_welch = inference, fit
    try:
        with segments.recording() as lineage:
            audit_sample(sample, list(FLAGS), overrides, root, oracle=True)
    except _Done as done:
        return done.result  # type: ignore[no-any-return]
    finally:
        core_inference.UPSTREAM, hmrf.pipeline_baum_welch = upstream, baum_welch
    msg = f"{stage}: the run finished without reaching its Baum-Welch"
    raise RuntimeError(msg)


def missed(label: np.ndarray, truth: np.ndarray) -> int:
    """Rows whose state is not the planted one, under the 1-1 matching of fitted to planted states that misses fewest."""
    from scipy.optimize import linear_sum_assignment

    label, truth = np.asarray(label, dtype=np.int64), np.asarray(truth, dtype=np.int64)
    n = int(max(label.max(), truth.max())) + 1
    agree = np.zeros((n, n), dtype=np.int64)
    np.add.at(agree, (label, truth), 1)
    rows, cols = linear_sum_assignment(-agree)
    return int(label.size - agree[rows, cols].sum())
