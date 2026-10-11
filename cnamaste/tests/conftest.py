"""Session fixtures over the staged runs: `sim`, `staged`, `replayed`, `lineage`, `clones`, `truth`.

Every test that takes `sim_hash` (directly or through these fixtures) runs
once per selected sample. `SUPPORTED_SIMS` lists the samples a staged file
exists for; the default (`DEFAULT_SIMS`) is CalicoST easy (`2d4ce9a9`) and
`dev_tree_easy_square` r0 (`23acef50`), and easy alone (`FN_DEFAULT_SIMS`)
for the per-function `test_fn_*` files. `--sims=a,b` (or `--sims=all`)
selects fixtures for every file: CalicoST hard (`8797710b`) and
`dev_tree_easy` r0 (`e4bbe03f`) are opt-in.

A pinned value that depends on the fixture (a science floor, a criterion's
exception count, a pass fraction, a defect observed or not) is a table keyed
by hash, each entry set from that fixture's capture.

A sample's planted truth is read from its staged file (`/truth`), so the
science rows need no inputs. Its inputs are the committed sample where there
is one; a drawn sample's are not committed (`e4bbe03f`'s and `23acef50`'s are 21 MB each), and
`--sim-inputs DIR` (or `$CNAMASTE_SIM_INPUTS`) points at regenerated ones,
`DIR/<hash>/` each. A test that needs them (a replay, the input files, a
count recomputed from them) skips, saying so, where they are not found.

`replayed` calls each recorded stage's function on its recorded input once
per session, with the generator states the call found, and checks what it
returns against the recorded sha256. An input not stored in the file (the
counts, the AnnData) is the earlier stage's replayed output, itself checked;
an input file is read from the committed sample, refused unless its sha256
is the one the capture read.
"""

from __future__ import annotations

import copy
import inspect
import json
import os
import random
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml
from numba import _helperlib

from audit.capture import Capture, digest, file_sha256, grch38, input_files, stage_inputs, unordered
from audit.criteria import Units, units_of
from audit.digest import Pins, quiet
from audit.fn import Ctx
from audit.scoring import Truth, planted
from audit.segments import DROPPED, Genes, Segmentation
from audit.sims import DEFAULT_SIMS, FN_DEFAULT_SIMS, INPUTS_ENV, SUPPORTED_SIMS
from cnamaste.config import YAMLConfig, set_global_config

PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parent




REPLAY_SKIPPED = {"run_core_inference"}
"""Not replayed: 100 s (BAF) and 230 s (RDR) on easy; their outputs are stored whole and checked by rows."""


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--sims", default=None, help="fixture hashes from SUPPORTED_SIMS, comma separated, or 'all' "
                     f"(default: {','.join(DEFAULT_SIMS)}, and {','.join(FN_DEFAULT_SIMS)} for the per-function test_fn_* files)")
    parser.addoption("--update-digests", action="store_true", help="rewrite data/digests_<hash>.json from this session's snapshots, deliberately")
    parser.addoption("--sim-inputs", default=os.environ.get(INPUTS_ENV, ""),
                     help=f"a directory of regenerated sample inputs, DIR/<hash>/ each (default ${INPUTS_ENV})")


PINNED: dict[str, dict[str, Any]] = {}
"""`--update-digests`: sample -> row key -> pin, from the reports (`audit.fn.snapshot`'s `user_properties`)."""
ROW_KEYS: dict[str, set[str]] = {}
"""`--update-digests`: sample -> every row's key, where `test_snapshot_pins_are_the_rows` ran: pins of other keys are dropped."""


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    for name, value in report.user_properties:
        if name == "digest":
            sim_hash, key, pin = value
            PINNED.setdefault(sim_hash, {})[key] = pin
        elif name == "digest-keys":
            ROW_KEYS[value[0]] = set(value[1])


def pytest_sessionfinish(session: pytest.Session) -> None:
    """`--update-digests`, on the controller (or the one process): the session's snapshots over the pins on disk."""
    if session.config.getoption("--update-digests") and not hasattr(session.config, "workerinput"):
        for sim_hash in sorted(set(PINNED) | set(ROW_KEYS)):
            Pins.write(sim_hash, PINNED.get(sim_hash, {}), ROW_KEYS.get(sim_hash))
    for (name, _), value in HELD.items():
        if name == "sim":
            value.close()


def selected(config: pytest.Config, module: str = "") -> list[str]:
    """The fixtures `--sims` names; without it, `DEFAULT_SIMS`, and `FN_DEFAULT_SIMS` for a `test_fn_*` module."""
    asked = config.getoption("--sims")
    if asked is None:
        return list(FN_DEFAULT_SIMS if module.startswith("test_fn_") else DEFAULT_SIMS)
    hashes = list(SUPPORTED_SIMS) if asked == "all" else [h for h in asked.split(",") if h]
    unknown = sorted(set(hashes) - set(SUPPORTED_SIMS))
    if unknown:
        raise pytest.UsageError(f"--sims: no staged file for {unknown}")
    return hashes


def stages_of(sim_hash: str) -> list[str]:
    with h5py.File(PROJECT / SUPPORTED_SIMS[sim_hash].staged, "r") as h5:
        return list(json.loads(h5["config"].attrs["stages"]))


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    hashes = selected(metafunc.config, metafunc.module.__name__.rpartition(".")[2])
    if "stage" in metafunc.fixturenames:
        pairs = [(h, s) for h in hashes for s in stages_of(h)]
        metafunc.parametrize(("sim_hash", "stage"), pairs, ids=[f"{h}-{s}" for h, s in pairs], scope="session")
    elif "sim_hash" in metafunc.fixturenames:
        metafunc.parametrize("sim_hash", hashes, scope="session")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Under xdist (`--dist loadgroup`), every test that replays shares one worker per sample.

    Replays chain: a stage's input is an earlier stage's replayed output, cached per session, and a
    session is per worker. Splitting them would replay `load_input_data` and its successors on every
    worker; one group replays each stage once. The rest, which read the file alone, spread.
    """
    if not config.pluginmanager.hasplugin("xdist"):
        return
    for item in items:
        params = item.callspec.params if hasattr(item, "callspec") else {}
        # NB a per-function row (`audit.fn.Row`) that slices a replayed stage, and a defect row whose check takes
        #    `replayed` or `lineage`, ask for `replayed` lazily
        entry = params.get("entry")
        lazy = entry is not None and {"replayed", "lineage"} & set(inspect.signature(entry.check).parameters)
        if "replayed" in getattr(item, "fixturenames", ()) or getattr(params.get("row"), "replay", False) or lazy:
            sim_hash = params.get("sim_hash")
            item.add_marker(pytest.mark.xdist_group(f"replay-{sim_hash}"))


@pytest.fixture(scope="session", autouse=True)
def _working_directory(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """A scratch working directory: `hmm_emission.flush_perf` appends to `cnamaste.perf` in it, whatever `paths.perf_path` says."""
    before = Path.cwd()
    os.chdir(tmp_path_factory.mktemp("cwd"))
    yield
    os.chdir(before)


HELD: dict[tuple[str, str], Any] = {}
"""(fixture, fixture hash) -> its value, for the process. A session fixture parametrized by `sim_hash` is torn down
whenever a worker's next test takes the other hash, and xdist interleaves the hashes: held here, a replay, a lineage or
a staging is built once per worker and hash, not once per alternation."""


def held(name: str, sim_hash: str, make: Callable[[], Any]) -> Any:
    """`HELD[(name, sim_hash)]`, made on first use."""
    key = (name, sim_hash)
    if key not in HELD:
        HELD[key] = make()
    return HELD[key]


def hash_of(sim: Capture) -> str:
    return str(sim.config["fixture_hash"])


@pytest.fixture(scope="session")
def sim(sim_hash: str) -> Capture:
    """The staged file, open for the process (`HELD`)."""

    def opened() -> Capture:
        capture = Capture(PROJECT / SUPPORTED_SIMS[sim_hash].staged)
        assert capture.config["fixture_hash"] == sim_hash
        return capture

    return held("sim", sim_hash, opened)


def located_inputs(config: pytest.Config, sim_hash: str) -> Path | None:
    """`sim_hash`'s inputs: `--sim-inputs DIR/<hash>`, else the committed sample, else None."""
    given = config.getoption("--sim-inputs")
    # NB relative to where pytest was invoked: the session runs from a scratch directory (`_working_directory`)
    found = config.invocation_params.dir / given / sim_hash if given else None
    if found is not None and found.is_dir():
        return found
    committed = SUPPORTED_SIMS[sim_hash].sample
    return REPOSITORY / committed if committed else None


def no_inputs(sim_hash: str) -> str:
    sim = SUPPORTED_SIMS[sim_hash]
    return (f"{sim_hash} ({sim.name}): its inputs are not committed and none were given; regenerate them "
            f"({sim.regenerate}) and pass --sim-inputs DIR with DIR/{sim_hash}/, or set ${INPUTS_ENV}")


@pytest.fixture(scope="session")
def sample(sim_hash: str, request: pytest.FixtureRequest) -> Path:
    """The sample's inputs; a skip where they are not available."""
    found = located_inputs(request.config, sim_hash)
    if found is None:
        pytest.skip(no_inputs(sim_hash))
    return found


@pytest.fixture(scope="session")
def truth(sim: Capture) -> Truth:
    """The planted truth the staged file stores (`/truth`)."""

    def read() -> Truth:
        files = sim.truth()
        assert files, "the staged file stores no /truth: re-capture it"
        return planted(files)

    return held("truth", hash_of(sim), read)


@dataclass
class Staged:
    root: Path
    config_path: Path
    config: Any
    resources: Path


def stage(sim: Capture, sample: Path | None, sim_hash: str, tmp_path_factory: pytest.TempPathFactory) -> Staged:
    """The inputs staged as the capture staged them; refused on a sha256 mismatch, a skip where there are none."""
    if sample is None:
        pytest.skip(no_inputs(sim_hash))
    recorded = json.loads(sim.config["inputs"])
    assert set(input_files(sample)) == set(recorded), f"{sample} does not hold the inputs the capture read"
    for name, path in input_files(sample).items():
        assert file_sha256(path) == recorded[name]["sha256"], f"{path} is not the input the capture read"
    resources = grch38()
    if resources is None:
        pytest.skip("CalicoST's GRCh38_resources not found; set $CNAMASTE_GRCH38")
    root = tmp_path_factory.mktemp("staged")
    # NB under the directory name the capture staged, whatever the given inputs' directory is called
    config_path = stage_inputs(sample, root, yaml.safe_load(sim.config["yaml"]), resources, Path(sim.config["sample"]).name)
    (root / "output").mkdir(exist_ok=True)
    return Staged(root, config_path, YAMLConfig.from_file(str(config_path)), resources)


@pytest.fixture(scope="session")
def staged(sim: Capture, sim_hash: str, request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Staged:
    return held("staged", sim_hash, lambda: stage(sim, located_inputs(request.config, sim_hash), sim_hash, tmp_path_factory))


def _navigate(value: Any, keys: list[str]) -> Any:
    for key in keys:
        if isinstance(value, dict):
            value = value[key] if key in value else value[int(key)]
        elif isinstance(value, (list, tuple)):
            value = value[int(key)]
        else:
            value = getattr(value, key)
    return value


@dataclass
class Replay:
    """Each stage replayed at most once; values decoded through the replays.

    The inputs are staged on the first replay, not before: a test that reads only stored values runs without them,
    and one that needs a replay skips where they are not available (`stage`)."""

    sim: Capture
    stager: Callable[[], Staged]
    outs: dict[str, Any] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)
    values: dict[str, Any] = field(default_factory=dict)
    _staged: Staged | None = None

    @property
    def staged(self) -> Staged:
        if self._staged is None:
            self._staged = self.stager()
        return self._staged

    def resolve(self, path: str) -> Any:
        if path == "config":
            return self.staged.config
        parts = path.split("/")
        stage = "/".join(parts[:2])
        if parts[2] != "out":
            raise LookupError(f"{path}: an input with no stored value, link or recipe")
        return _navigate(self.run(stage), parts[3:])

    def value(self, path: str) -> Any:
        """The value at `path`: stored, linked, rebuilt, or replayed."""
        if path not in self.values:
            self.values[path] = self.sim.decode(path, self.resolve)
        return self.values[path]

    def _rewritten(self, value: Any) -> Any:
        if isinstance(value, str):
            for old, new in ((self.sim.config["run_root"], str(self.staged.root)), (self.sim.config["resources"], str(self.staged.resources))):
                if value.startswith(old):
                    return new + value[len(old):]
            return value
        if isinstance(value, list):
            return [self._rewritten(v) for v in value]
        if isinstance(value, dict):
            return {k: self._rewritten(v) for k, v in value.items()}
        return value

    def run(self, stage: str) -> Any:
        """`stage`'s function on its recorded input, with the recorded generator states."""
        if stage not in self.outs:
            if stage.split("/")[1] in REPLAY_SKIPPED:
                raise LookupError(f"{stage} is not replayed")
            given = self._rewritten(copy.deepcopy(self.value(f"{stage}/in")))
            if "write_tsv" in stage:
                Path(given["args"][0]).parent.mkdir(parents=True, exist_ok=True)
            set_global_config(self.staged.config)
            np_state, numba_state, py_state = self.sim.rng(stage)
            np.random.set_state(np_state)  # noqa: NPY002
            _helperlib.rnd_set_state(_helperlib.rnd_get_np_state_ptr(), numba_state)
            random.setstate(py_state)
            start = time.perf_counter()
            with quiet():  # NB a replay belongs to no per-function row (`audit.digest.QUIET`)
                self.outs[stage] = self.sim.function(stage)(*given["args"], **given["kwargs"])
            self.seconds[stage] = time.perf_counter() - start
            if "write_tsv" in stage:
                self.outs[stage + "#file"] = Path(given["args"][0]).read_bytes()
        return self.outs[stage]

    def matches(self, stage: str) -> bool:
        """The replay's return hashes to the recorded one."""
        out = self.run(stage)
        if digest(out) == self.sim.digest_of(f"{stage}/out"):
            return True
        sets = self.sim.sets_of(f"{stage}/out")
        return sets is not None and digest(unordered(out)) == sets


@pytest.fixture(scope="session")
def replayed(sim: Capture, sim_hash: str, request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Replay:
    sample = located_inputs(request.config, sim_hash)
    return held("replayed", sim_hash, lambda: Replay(sim, lambda: stage(sim, sample, sim_hash, tmp_path_factory)))


@pytest.fixture(scope="session")
def fn_cache(sim: Capture) -> dict[str, Any]:
    """The per-function rows' heavy recomputations, once per session (per xdist worker)."""
    return held("fn_cache", hash_of(sim), dict)


@pytest.fixture
def ctx(sim: Capture, fn_cache: dict[str, Any], request: pytest.FixtureRequest, tmp_path: Path) -> Any:
    """A per-function row's context (`audit.fn.Ctx`): the staged file, the replay on demand, the cache."""
    return Ctx(sim, request, fn_cache, tmp_path)


# --- lineage ---------------------------------------------------------------


@pytest.fixture(scope="session")
def gene_counts(sim: Any, replayed: Any, lineage: Any) -> np.ndarray:
    """(genes, spots) UMI counts in the lineage's gene order, from the loaded AnnData's `count` layer."""
    return held("gene_counts", hash_of(sim), lambda: counts_of(sim, replayed, lineage))


def counts_of(sim: Any, replayed: Any, lineage: Any) -> np.ndarray:
    """`gene_counts`' value."""
    adata = replayed.value("00_inputs/load_input_data/out/2")
    names = np.asarray(sim.stored("02_blocks/assign_initial_blocks/out")["gene"])[lineage.genes.row]
    column = pd.Index(adata.var.index).get_indexer(names)
    assert np.all(column >= 0)
    counts = adata.layers["count"]
    found = np.asarray(counts[:, column].toarray() if hasattr(counts, "toarray") else counts[:, column]).T
    # NB a name the reference carries twice (LINC01505 on easy) is one AnnData column, which the
    #    summaries count once per segment (`np.isin` on names): its second row counts nothing.
    found[pd.Index(names).duplicated()] = 0
    return found



@pytest.fixture(scope="session")
def units(sim: Capture, replayed: Replay, lineage: Lineage, request: pytest.FixtureRequest) -> Units:
    """Each level's units and counts, recomputed from the staged inputs on first use (`audit.criteria`)."""
    return held("units", hash_of(sim), lambda: units_of(sim, replayed, lineage, lambda: counts_of(sim, replayed, lineage)))

@dataclass
class Lineage:
    genes: Genes
    levels: dict[str, Segmentation]
    table: np.ndarray
    names: list[str]


@pytest.fixture(scope="session")
def lineage(sim: Capture) -> Lineage:
    """`/lineage/segments`, each level a `Segmentation.of` the gene rows (which checks contiguity, contigs, order)."""

    def built() -> Lineage:
        frame = sim.stored("02_blocks/assign_initial_blocks/out")
        genes = Genes.from_table(frame)
        table, names = sim.segments()
        assert table.shape == (genes.n_genes, len(names))
        return Lineage(genes, {n: Segmentation.of(genes, table[:, i], name=n) for i, n in enumerate(names)}, table, names)

    return held("lineage", hash_of(sim), built)


@dataclass
class Clones:
    labels: np.ndarray
    levels: list[str]
    parents: dict[str, np.ndarray]

    def __getitem__(self, level: str) -> np.ndarray:
        return self.labels[:, self.levels.index(level)]


@pytest.fixture(scope="session")
def clones(sim: Capture) -> Clones:
    return held("clones", hash_of(sim), lambda: Clones(*sim.clones()))


__all__ = ["DROPPED", "Clones", "Lineage", "Replay", "Staged"]
