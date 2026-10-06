"""A simulated sample at a named stage of `run_cnaster_port`, run once and cached (#467).

`tests.sim_audit` runs a sample end to end and scores what it wrote. A test of
one stage needs that stage's inputs and output, and re-running the pipeline
per test costs minutes. `stages(sample, ...)` runs the pipeline once per
sample and configuration, records every call the driver
(`cnaster.scripts.run_cnaster`) makes to the names in `CAPTURED`, with its
arguments and result, and caches the record on disk. A test then reads the
stage it is about, or replays that one call with a changed argument.

The names are wrapped **after** `run_cnaster_port` installs its swaps, around
whatever each is bound to when the driver calls it, so the record is of the
run a user gets, `--sal` and the shift included.

**The key** is the sample's content, the flags and overrides, and the code
that can move a stage: `python/port`, the lockfiles, `src/` and these three
harness modules. A change elsewhere in `tests/` leaves the cache valid.

Samples: `r0` is `dev_tree`'s realization 0 at the frozen exponential-length
generation (`sim/manifests/baseline/dev_tree.toml`, #619), drawn on demand and
refused if its content hash is not `R0_HASH` (`port.sim.fixtures`); `easy` and `hard` are CalicoST's committed
samples (`port.sim.fixtures`).
"""

from __future__ import annotations

import gzip
import hashlib
import pickle
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from port.sim.fixtures import R0_HASH, r0, realization_hash

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache" / "sim_stages"

CAPTURED: tuple[str, ...] = (
    "initial_phase_given_partition",
    "run_core_inference",
    "merge_by_minspots",
    "normal_baf_bin_filter",
    "hill_climbing_integer_copynumber_oneclone",
    "hill_climbing_integer_copynumber_fixdiploid_milp",
)
"""The driver's stage calls a record keeps. `load_input_data` is left out:
its AnnData is most of a record's size and a test of the input path reads the
sample's own files."""

DRIVER = "cnaster.scripts.run_cnaster"
ENTRY = "run_cnaster"
"""The driver's entry, looked up on the module by `run_cnaster_port` after its swaps."""

OUTPUTS = ("clone_labels.tsv", "cnv_seglevel.tsv", "cnv_perstate.tsv", "cnv_states.tsv")
"""What a record keeps of the run's output directory: small, and placed."""

KEYED = ("python/port", "src", "uv.lock", "Cargo.lock")
HARNESS = ("tests/sim_stages.py", "tests/sim_audit.py")


@dataclass
class Call:
    """One call the driver made: the name, what it was passed, what came back."""

    name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    result: Any


@dataclass
class Stages:
    """A sample's run, as the calls the driver made, in order."""

    sample: str
    flags: tuple[str, ...]
    oracle: bool
    calls: list[Call] = field(default_factory=list)
    recovery: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    """The run's `OUTPUTS` tables, by file name, as `DataFrame`s."""

    def all(self, name: str) -> list[Call]:
        """Every call to `name`, in call order."""
        return [call for call in self.calls if call.name == name]

    def one(self, name: str, index: int = 0) -> Call:
        """The `index`-th call to `name`; `run_core_inference`'s 0 is the
        BAF-only stage and 1 the RDR+BAF stage."""
        calls = self.all(name)

        if index >= len(calls):
            msg = f"{name} was called {len(calls)} times, not {index + 1}"
            raise LookupError(msg)

        return calls[index]


def _sample(name: str) -> tuple[Any, str]:
    """The `SimulatedSample`, and the content key it is cached under."""
    from port.sim.fixtures import SAMPLES, load_simulated

    if name == "r0":
        r0()
        return load_simulated("generated/dev_tree/r0"), R0_HASH

    sample = load_simulated(SAMPLES.get(name, name))
    return sample, realization_hash(sample.path)


def code_hash() -> str:
    """A digest of the code that can move a stage (`KEYED`, `HARNESS`)."""
    digest = hashlib.sha256()
    files: list[Path] = []

    for name in (*KEYED, *HARNESS):
        path = ROOT / name

        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(
                p
                for p in path.rglob("*")
                if p.is_file()
                and "__pycache__" not in p.parts
                and p.suffix in {".py", ".rs", ".toml"}
            )

    for path in sorted(files):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())

    return digest.hexdigest()[:12]


@contextmanager
def capturing(calls: list[Call], names: tuple[str, ...] = CAPTURED) -> Iterator[None]:
    """Record the driver's calls to `names`, around whatever each is bound to."""
    import importlib

    driver = importlib.import_module(DRIVER)
    undo: list[tuple[str, Any]] = []

    def logged(name: str, current: Any) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            result = current(*args, **kwargs)
            calls.append(Call(name, args, dict(kwargs), result))
            return result

        return call

    try:
        for name in names:
            current = getattr(driver, name)
            undo.append((name, current))
            setattr(driver, name, logged(name, current))

        yield
    finally:
        for name, current in reversed(undo):
            setattr(driver, name, current)


def _record(
    name: str, flags: tuple[str, ...], oracle: bool, overrides: dict[str, Any]
) -> Stages:
    """Run the sample once, with the driver's stage calls recorded."""
    import importlib
    from dataclasses import asdict

    import pandas as pd

    from tests.sim_audit import run_arm

    driver = importlib.import_module(DRIVER)

    sample, _ = _sample(name)
    stages = Stages(name, flags, oracle)
    original = getattr(driver, ENTRY)

    # NB `run_cnaster_port` looks `run_cnaster` up on the driver module after
    #    entering every swap, so wrapping it here records the swapped run.
    def run(*args: Any, **kwargs: Any) -> Any:
        with capturing(stages.calls):
            return original(*args, **kwargs)

    setattr(driver, ENTRY, run)

    try:
        with tempfile.TemporaryDirectory() as scratch:
            recovery, output = run_arm(
                sample, list(flags), overrides, Path(scratch), oracle=oracle
            )
            for table in OUTPUTS:
                found = sorted(output.rglob(table))

                if found:
                    stages.outputs[table] = pd.read_csv(found[0], sep="\t")
    finally:
        setattr(driver, ENTRY, original)

    stages.recovery = asdict(recovery)
    return stages


def stages(
    sample: str,
    flags: tuple[str, ...] = ("--sal",),
    *,
    oracle: bool = False,
    overrides: dict[str, Any] | None = None,
) -> Stages:
    """`sample`'s stages under `flags`, from the cache or from one run."""
    overrides = dict(overrides or {})
    _, content = _sample(sample)
    key = hashlib.sha256(
        repr((content, flags, oracle, sorted(overrides.items()), code_hash())).encode()
    ).hexdigest()[:16]
    path = CACHE / f"{sample}-{key}.pkl.gz"

    if path.is_file():
        with gzip.open(path, "rb") as handle:
            cached: Stages = pickle.load(handle)
        return cached

    recorded = _record(sample, flags, oracle, overrides)
    CACHE.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")

    with gzip.open(partial, "wb", compresslevel=1) as handle:
        pickle.dump(recorded, handle, protocol=pickle.HIGHEST_PROTOCOL)

    partial.replace(path)
    return recorded
