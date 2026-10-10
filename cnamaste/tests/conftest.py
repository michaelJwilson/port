"""Session fixtures over the staged runs: `sim`, `staged`, `replayed`, `lineage`, `clones`, `truth`.

Every test that takes `sim_hash` (directly or through these fixtures) runs
once per selected sample. `SUPPORTED_SIMS` lists the samples a staged file
exists for; the default is CalicoST easy (`2d4ce9a9`) alone, and
`--sims=a,b` (or `--sims=all`) selects others.

`replayed` calls each recorded stage's function on its recorded input once
per session, with the generator states the call found, and checks what it
returns against the recorded sha256. An input not stored in the file (the
counts, the AnnData) is the earlier stage's replayed output, itself checked;
an input file is read from the committed sample, refused unless its sha256
is the one the capture read.
"""

from __future__ import annotations

import copy
import json
import os
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml
from numba import _helperlib

from audit.criteria import Units, units_of
from audit.capture import Capture, digest, unordered, file_sha256, grch38, input_files, stage_inputs
from audit.scoring import Truth, planted
from audit.segments import DROPPED, Genes, Segmentation
from cnamaste.config import YAMLConfig, set_global_config

PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parent

SUPPORTED_SIMS: dict[str, tuple[str, str]] = {
    "2d4ce9a9": ("sim/numcnas1.2_cnasize5e7_ploidy2_random0", "tests/data/sim_2d4ce9a9.hdf5"),
    # "8797710b": ("sim/numcnas6.3_cnasize1e7_ploidy2_random0", "tests/data/sim_8797710b.hdf5"),  # hard: no staged file yet
}
"""Fixture hash -> (the committed sample, relative to the repository; its staged file, relative to `cnamaste/`)."""

DEFAULT_SIMS = ("2d4ce9a9",)

REPLAY_SKIPPED = {"run_core_inference"}
"""Not replayed: 100 s (BAF) and 230 s (RDR) on easy; their outputs are stored whole and checked by rows."""


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--sims", default=",".join(DEFAULT_SIMS), help="fixture hashes from SUPPORTED_SIMS, comma separated, or 'all'")


def selected(config: pytest.Config) -> list[str]:
    asked = config.getoption("--sims")
    hashes = list(SUPPORTED_SIMS) if asked == "all" else [h for h in asked.split(",") if h]
    unknown = sorted(set(hashes) - set(SUPPORTED_SIMS))
    if unknown:
        raise pytest.UsageError(f"--sims: no staged file for {unknown}")
    return hashes


def stages_of(sim_hash: str) -> list[str]:
    with h5py.File(PROJECT / SUPPORTED_SIMS[sim_hash][1], "r") as h5:
        return list(json.loads(h5["config"].attrs["stages"]))


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    hashes = selected(metafunc.config)
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
        if "replayed" in getattr(item, "fixturenames", ()):
            sim_hash = getattr(item, "callspec", None) and item.callspec.params.get("sim_hash")
            item.add_marker(pytest.mark.xdist_group(f"replay-{sim_hash}"))


@pytest.fixture(scope="session", autouse=True)
def _working_directory(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """A scratch working directory: `hmm_emission.flush_perf` appends to `cnamaste.perf` in it, whatever `paths.perf_path` says."""
    before = Path.cwd()
    os.chdir(tmp_path_factory.mktemp("cwd"))
    yield
    os.chdir(before)


@pytest.fixture(scope="session")
def sim(sim_hash: str) -> Iterator[Capture]:
    """The staged file, open for the session."""
    capture = Capture(PROJECT / SUPPORTED_SIMS[sim_hash][1])
    assert capture.config["fixture_hash"] == sim_hash
    yield capture
    capture.close()


@pytest.fixture(scope="session")
def sample(sim_hash: str) -> Path:
    return REPOSITORY / SUPPORTED_SIMS[sim_hash][0]


@pytest.fixture(scope="session")
def truth(sample: Path) -> Truth:
    return planted(sample)


@dataclass
class Staged:
    root: Path
    config_path: Path
    config: Any
    resources: Path


@pytest.fixture(scope="session")
def staged(sim: Capture, sample: Path, tmp_path_factory: pytest.TempPathFactory) -> Staged:
    """The committed inputs staged as the capture staged them; refused on a sha256 mismatch."""
    recorded = json.loads(sim.config["inputs"])
    for name, path in input_files(sample).items():
        assert file_sha256(path) == recorded[name]["sha256"], f"{path} is not the input the capture read"
    resources = grch38()
    if resources is None:
        pytest.skip("CalicoST's GRCh38_resources not found; set $CNAMASTE_GRCH38")
    root = tmp_path_factory.mktemp("staged")
    config_path = stage_inputs(sample, root, yaml.safe_load(sim.config["yaml"]), resources)
    (root / "output").mkdir(exist_ok=True)
    return Staged(root, config_path, YAMLConfig.from_file(str(config_path)), resources)


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
    """Each stage replayed at most once; values decoded through the replays."""

    sim: Capture
    staged: Staged
    outs: dict[str, Any] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)
    values: dict[str, Any] = field(default_factory=dict)

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
def replayed(sim: Capture, staged: Staged) -> Replay:
    return Replay(sim, staged)


# --- lineage ---------------------------------------------------------------


@pytest.fixture(scope="session")
def gene_counts(sim: Any, replayed: Any, lineage: Any) -> np.ndarray:
    """(genes, spots) UMI counts in the lineage's gene order, from the loaded AnnData's `count` layer."""
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
def units(sim: Capture, replayed: Replay, lineage: Lineage, gene_counts: np.ndarray) -> Units:
    """Each level's units and counts, recomputed from the staged inputs (`audit.criteria`)."""
    return units_of(sim, replayed, lineage, gene_counts)

@dataclass
class Lineage:
    genes: Genes
    levels: dict[str, Segmentation]
    table: np.ndarray
    names: list[str]


@pytest.fixture(scope="session")
def lineage(sim: Capture) -> Lineage:
    """`/lineage/segments`, each level a `Segmentation.of` the gene rows (which checks contiguity, contigs, order)."""
    frame = sim.stored("02_blocks/assign_initial_blocks/out")
    genes = Genes.from_table(frame)
    table, names = sim.segments()
    assert table.shape == (genes.n_genes, len(names))
    return Lineage(genes, {n: Segmentation.of(genes, table[:, i], name=n) for i, n in enumerate(names)}, table, names)


@dataclass
class Clones:
    labels: np.ndarray
    levels: list[str]
    parents: dict[str, np.ndarray]

    def __getitem__(self, level: str) -> np.ndarray:
        return self.labels[:, self.levels.index(level)]


@pytest.fixture(scope="session")
def clones(sim: Capture) -> Clones:
    return Clones(*sim.clones())


__all__ = ["DROPPED", "Clones", "Lineage", "Replay", "Staged"]
