"""Stage one `run_cnamaste` run into `cnamaste/tests/data/sim_<hash>.hdf5` (T- #836).

    uv run --locked --project cnamaste python scripts/capture_cnamaste.py \
        --work DIR [--sample sim/numcnas1.2_cnasize5e7_ploidy2_random0] [--out FILE] \
        [--from-stage NN_group/stage --base FILE]

port owns the capture; cnamaste holds the result. The script runs in
cnamaste's own environment (`--project cnamaste`), where port is not
installed, so it imports neither port nor `cnaster`: it reads cnamaste's file
codec from `cnamaste/tests/audit/capture.py` and port's
`[tool.port] max_file_bytes` from `pyproject.toml`. CalicoST's
`GRCh38_resources` are found as cnamaste's tests find them
(`$CNAMASTE_GRCH38`, else uv's git checkout); `$PORT_GRCH38` is read as a
fallback.

Stages a committed CalicoST sample as port's `write_sim_inputs` does (its
inputs, the sample sheet, `tests/data/zenodo_sim_config.yaml` pointed at
them and at CalicoST's `GRCh38_resources`), seeds numpy's legacy generator
with port's `ENTRY_POINT_SEED` (0) as `tests/test_cnamaste_end2end.py` does,
and runs `cnamaste.scripts.run_cnamaste.run_cnamaste` with every key stage
wrapped on the module's namespace. Each wrapped call is recorded once, the
call the pipeline made: its arguments, deep-copied on entry, its return,
deep-copied on exit, and the generator states it found. Plots are stubbed
(`plot_*` and `write_fig` return None): they read and draw, and the run's
scores equal the ledger row with them on (`cnamaste/tests/test_stages.py`).

The run is staged under `DIR/staged_run` and run from `DIR`, so every path
the file records is relative to `DIR` (`/config` `run_root`): nothing in the
file names the host's checkout. `/config` `capture_commit` is the checkout's
commit (`-dirty` where `cnamaste/python` has uncommitted edits).

Inside `run_core_inference` it also records, from its last outer iteration,
what `hmrf` computes and discards: the ICM labels, each Potts merge
`merge_assignment` applied, the labels before the empty-clone re-indexing,
and that re-indexing. `/lineage` is built from the recorded frames.

**Resume.** `--from-stage NN_group/stage --base FILE` copies every stage
FILE recorded before the named one, unchanged, and records the rest as a
full capture does; `/config` `resumed_from` and `base_sha256` say so. The
granularity is any recorded stage. `run_cnamaste` cannot be entered
mid-pipeline, so it runs from the start, and each earlier stage's call
returns FILE's value instead of computing it (a value FILE holds only as a
digest is recomputed, at FILE's generator states, and checked against it).
Every call's inputs must hash to FILE's, and the named stage starts from
FILE's generator states. FILE is refused unless its fixture hash, template,
inputs and staged configuration are this run's. Copied stages are only as
current as FILE: resume from the first stage a change reaches.

**numba.** `$NUMBA_CACHE_DIR` defaults to `$XDG_CACHE_HOME/cnamaste-numba`
(else `~/.cache/cnamaste-numba`); set it to override. numba keys its own
cache by the source file's absolute path and mtime, so every fresh worktree
recompiled; here a kernel is cached by module and source sha256, so
checkouts of the same source share one cache. `/config` records the
directory, the seconds spent compiling or loading kernels, and the cache's
hits and misses.

**Threads.** The capture sets no thread limit (`NUMBA_NUM_THREADS`,
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`); it records
them in `/config` `thread_env`, as found. A capture should hold the host
lock alone. Run concurrently, easy and square peak at about 3 GB and 7.6 GB,
which fits in about 14 GB.

On easy (`2d4ce9a9`) the run takes 562 s on 4 threads.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

NUMBA_CACHE_DIR = os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    str(
        Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        / "cnamaste-numba"
    ),
)
"""numba's cache, shared by every checkout: set before numba is imported, which reads it once."""

from numba.core import caching, dispatcher  # noqa: E402


class SharedCacheLocator(caching._SourceFileBackedLocatorMixin, caching._CacheLocator):  # type: ignore[misc,name-defined]
    """`$NUMBA_CACHE_DIR/<module>`, stamped by the source file's sha256.

    numba's own locators key the cache by the source file's absolute path and
    its mtime, so a fresh worktree misses on every kernel. Keyed by module and
    content instead, one cache serves every checkout of the same source.
    """

    def __init__(self, py_func: Any, py_file: str) -> None:
        self._py_file = py_file
        self._lineno = py_func.__code__.co_firstlineno
        self._cache_path = str(Path(NUMBA_CACHE_DIR) / py_func.__module__)

    def get_cache_path(self) -> str:
        return self._cache_path

    def get_source_stamp(self) -> str:
        return hashlib.sha256(Path(self._py_file).read_bytes()).hexdigest()

    @classmethod
    def from_function(cls, py_func: Any, py_file: str) -> Any:
        if not getattr(py_func, "__module__", None):
            return None
        return super().from_function(py_func, py_file)


caching.CacheImpl._locator_classes.insert(0, SharedCacheLocator)

COMPILE: dict[str, Any] = {"seconds": 0.0, "depth": 0, "dispatchers": set()}
"""Wall time inside numba's `Dispatcher.compile`, outermost calls only: compiling, or loading from the cache."""
_compile = dispatcher.Dispatcher.compile


def timed_compile(self: Any, sig: Any) -> Any:
    COMPILE["dispatchers"].add(self)
    COMPILE["depth"] += 1
    start = time.perf_counter()
    try:
        return _compile(self, sig)
    finally:
        COMPILE["depth"] -= 1
        if COMPILE["depth"] == 0:
            COMPILE["seconds"] += time.perf_counter() - start


dispatcher.Dispatcher.compile = timed_compile

import h5py  # noqa: E402
import numba  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.optimize  # noqa: E402
import yaml  # noqa: E402
from numba import _helperlib  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

import cnamaste.hmrf as hmrf  # type: ignore[import-not-found]  # noqa: E402, PLR0402
import cnamaste.scripts.run_cnamaste as rc  # type: ignore[import-not-found]  # noqa: E402

REPOSITORY = Path(__file__).resolve().parents[1]
PROJECT = REPOSITORY / "cnamaste"
sys.path.insert(0, str(PROJECT / "tests"))

from audit.capture import (  # type: ignore[import-not-found]  # noqa: E402
    Capture,
    Writer,
    digest,
    file_sha256,
    grch38,
    input_files,
    stage_inputs,
    unordered,
)

ENTRY_POINT_SEED = 0
"""port's `port.sim.run_config.ENTRY_POINT_SEED`: numpy's legacy generator, seeded before the entry point."""

MAX_BYTES: int = tomllib.loads((REPOSITORY / "pyproject.toml").read_text())["tool"][
    "port"
]["max_file_bytes"]
"""port's `[tool.port] max_file_bytes`."""

RUN_ROOT = Path("staged_run")
"""The staged run, relative to `--work`, which the capture runs from."""

SCHEDULE: dict[str, list[str]] = {
    "load_input_data": ["00_inputs/load_input_data"],
    "get_sample_list": ["00_inputs/get_sample_list"],
    "read_tumor_prop": ["00_inputs/read_tumor_prop"],
    "form_gene_snp_table": ["01_genes/form_gene_snp_table"],
    "assign_initial_blocks": ["02_blocks/assign_initial_blocks"],
    "summarize_counts_for_blocks": ["02_blocks/summarize_counts_for_blocks"],
    "get_sitewise_transmat": [
        "02_blocks/get_sitewise_transmat",
        "04_bins/get_sitewise_transmat",
        "06_normal/get_sitewise_transmat",
        "07_rebin/get_sitewise_transmat",
    ],
    "initialize_clones": ["03_phasing/initialize_clones"],
    "initial_phase_given_partition": ["03_phasing/initial_phase_given_partition"],
    "create_bin_ranges": ["04_bins/create_bin_ranges", "07_rebin/create_bin_ranges"],
    "summarize_counts_for_bins": [
        "04_bins/summarize_counts_for_bins",
        "07_rebin/summarize_counts_for_bins",
    ],
    "construct_multislice_lattice_adjacency": [
        "04_bins/construct_multislice_lattice_adjacency"
    ],
    "run_core_inference": ["05_baf/run_core_inference", "08_rdr/run_core_inference"],
    "merge_by_minspots": ["05_baf/merge_by_minspots", "08_rdr/merge_by_minspots"],
    "construct_df_clone_label": [
        "05_baf/construct_df_clone_label",
        "09_outputs/construct_df_clone_label",
    ],
    "determine_normal_candidates": ["06_normal/determine_normal_candidates"],
    "normal_baf_bin_filter": ["06_normal/normal_baf_bin_filter"],
    "binned_gene_snp": ["06_normal/binned_gene_snp", "07_rebin/binned_gene_snp"],
    "filter_normal_diffexp": ["06_normal/filter_normal_diffexp"],
    "determine_normal_baseline": ["07_rebin/determine_normal_baseline"],
    "initialize_rdr_clone_refininement": ["08_rdr/initialize_rdr_clone_refininement"],
    "reindex_clones": ["08_rdr/reindex_clones"],
    "hill_climbing_integer_copynumber_fixdiploid_milp": [
        f"09_outputs/integer_copy_{c}" for c in range(16)
    ],
    "write_tsv": [f"09_outputs/write_tsv_{k}" for k in range(16)],
}
"""Each wrapped function -> the stage each of its calls records, in call order."""

NOT_REPLAYED = {"run_core_inference"}
"""Stages whose outputs are stored whole: replaying one is 100 to 230 s, so the tests read them instead."""

DERIVED = {"binned_gene_snp"}
"""Replayed stages whose every output is left to the replay, however small: tables of joined ids."""

BINS = "04_bins/summarize_counts_for_bins/out"
REBIN = "07_rebin/summarize_counts_for_bins/out"
RECIPES: dict[str, list[Any]] = {
    "05_baf/run_core_inference/in/args/0": ["zero_rdr", f"{BINS}/X"],
    "05_baf/run_core_inference/in/args/2": ["zeros_like", f"{BINS}/base_nb_mean"],
    "06_normal/determine_normal_candidates/in/args/3": ["zero_rdr", f"{BINS}/X"],
    "06_normal/determine_normal_candidates/in/args/4": ["rdr", f"{BINS}/X"],
    "07_rebin/determine_normal_baseline/in/args/0": ["rdr", f"{REBIN}/X"],
    "07_rebin/determine_normal_baseline/out/1": [
        "zero_rows",
        "07_rebin/determine_normal_baseline/in/args/0",
        "07_rebin/determine_normal_baseline/out/0",
    ],
    "07_rebin/determine_normal_baseline/out/2": [
        "outer_coverage",
        "07_rebin/determine_normal_baseline/out/0",
        "07_rebin/determine_normal_baseline/out/1",
    ],
    "08_rdr/run_core_inference/in/args/0": [
        "with_rdr",
        f"{REBIN}/X",
        "07_rebin/determine_normal_baseline/out/1",
    ],
}
"""A large input that equals no earlier node, rebuilt from the nodes `run_cnamaste`'s glue built it from."""

PLOTS = (
    "plot_clones_genomic",
    "plot_clones_spatial",
    "plot_copy_number_profile",
    "plot_he",
    "write_fig",
)


def rng_state() -> tuple[Any, Any, Any]:
    return (
        np.random.get_state(),  # noqa: NPY002
        _helperlib.rnd_get_state(_helperlib.rnd_get_np_state_ptr()),  # type: ignore[attr-defined]
        random.getstate(),
    )


def write_rng(g: h5py.Group, state: tuple[Any, Any, Any]) -> None:
    numpy_state, numba_state, py_state = state
    r = g.create_group("rng")
    r.create_dataset(
        "numpy", data=np.asarray(numpy_state[1], dtype=np.uint32), compression="gzip"
    )
    r.attrs.update(
        {
            "numpy_pos": numpy_state[2],
            "has_gauss": numpy_state[3],
            "gauss": numpy_state[4],
        }
    )
    r.create_dataset(
        "numba", data=np.asarray(numba_state[1], dtype=np.uint32), compression="gzip"
    )
    r.attrs["numba_pos"] = numba_state[0]
    r.create_dataset(
        "random", data=np.asarray(py_state[1], dtype=np.uint64), compression="gzip"
    )


class Recorder:
    """The wrapped stages, writing into one open file."""

    def __init__(self, h5: h5py.File) -> None:
        self.h5 = h5
        self.writer = Writer(h5)
        self.writer.recipes = RECIPES
        self.writer.digest_only = self.digest_only
        self.calls: dict[str, int] = {}
        self.order: list[str] = []
        self.current = ""
        self.internal: dict[str, dict[str, Any]] = {}
        self.files: dict[str, str] = {}
        self.base: Capture | None = None
        self.resume = ""
        self.copied: set[str] = set()
        """Resuming: the base's stages before `resume`, each copied from it unchanged."""
        self.scratch: h5py.File | None = None

    def resume_from(self, base: Capture, stage: str) -> None:
        """Copy every stage `base` recorded before `stage`, and restart the run at `stage`.

        A copied stage's call still happens, so `run_cnamaste`'s glue runs as it did:
        its inputs must hash to the base's, or the base is not this run's. Its return
        is the base's stored value, decoded; where the base stores only a digest (the
        counts, the AnnData), the function runs, at the base's generator states, and
        must return that digest. Its group is copied from the base unchanged, and its
        values are encoded into a scratch file so that later stages link to them and
        `/lineage` reads them as a full capture does.
        """
        self.base, self.resume = base, stage
        self.copied = set(base.stages[: base.stages.index(stage)])

    def based(self) -> Capture:
        if self.base is None:
            message = "not resuming"
            raise RuntimeError(message)
        return self.base

    def set_rng(self, label: str) -> None:
        np_state, numba_state, py_state = self.based().rng(label)
        np.random.set_state(np_state)  # noqa: NPY002
        _helperlib.rnd_set_state(_helperlib.rnd_get_np_state_ptr(), numba_state)  # type: ignore[attr-defined]
        random.setstate(py_state)

    def check_inputs(self, label: str, sha: str) -> None:
        recorded = self.based().digest_of(f"{label}/in")
        if sha != recorded:
            message = (
                f"--base: {label} was called with inputs other than the base recorded "
                f"(sha256 {sha[:12]} against {recorded[:12]}): "
                "the base is not this run's, or the code before --from-stage changed"
            )
            raise SystemExit(message)

    def copy_stage(self, f: Any, name: str, label: str, args: Any, kwargs: Any) -> Any:
        """A stage before `resume`: the base's record, copied; its return, the base's."""
        base = self.based()
        if self.scratch is None:
            self.scratch = h5py.File(
                f"scratch-{os.getpid()}.h5", "w", driver="core", backing_store=False
            )
        self.set_rng(label)
        given = copy.deepcopy((args, kwargs))
        items = {"args": list(given[0]), "kwargs": dict(given[1])}
        sha = digest(items)
        self.check_inputs(label, sha)
        if name == "run_core_inference":
            self.current = label.split("/")[0]
            self.internal[self.current] = base.internal(self.current)
        try:
            if name == "write_tsv":
                raise LookupError(label)  # NB the file, written where the run reads it
            out = base.stored(f"{label}/out")
        except LookupError:
            out = f(*args, **kwargs)
            found, sets = digest(out), base.sets_of(f"{label}/out")
            if found != base.digest_of(f"{label}/out") and not (
                sets is not None and digest(unordered(out)) == sets
            ):
                message = f"--base: {label} returned other than the base recorded"
                raise SystemExit(message) from None

        g = self.scratch.create_group(label)
        self.writer._items(g, "in", "call", f"{label}/in", sha, items, {})
        self.writer.put(g, "out", copy.deepcopy(out), f"{label}/out")
        parent, _, child = label.rpartition("/")
        base.h5.copy(base.h5[label], self.h5.require_group(parent), name=child)
        if name == "write_tsv":
            written = base.h5[label].attrs["file"]
            self.files[written] = label
            base.h5.copy(base.h5[f"files/{written}"], self.h5.require_group("files"))
        self.order.append(label)
        print(f"{label} copied", flush=True)
        return out

    def digest_only(self, path: str, value: Any) -> bool:
        """A large output of a replayed stage: the replay re-derives it."""
        parts = path.split("/")
        if len(parts) < 3 or parts[2] != "out" or parts[1] in NOT_REPLAYED:
            return False
        if parts[1] in DERIVED:
            return True
        wide = hasattr(value, "columns") and value.shape[1] > 100
        return bool(
            (isinstance(value, np.ndarray) and value.nbytes > 256_000)
            or hasattr(value, "nnz")
            or wide
        )

    def label(self, name: str) -> str:
        k = self.calls.get(name, 0)
        self.calls[name] = k + 1
        return SCHEDULE[name][k]

    def wrap(self, module: Any, name: str) -> None:
        f = getattr(module, name)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            label = self.label(name)
            if label in self.copied:
                return self.copy_stage(f, name, label, args, kwargs)
            if label == self.resume:
                self.set_rng(label)
            state = rng_state()
            given = copy.deepcopy((args, kwargs))
            if name == "run_core_inference":
                self.current = label.split("/")[0]
                self.internal[self.current] = {}
            start = time.perf_counter()
            out = f(*args, **kwargs)
            seconds = time.perf_counter() - start
            returned = copy.deepcopy(out)

            g = self.h5.create_group(label)
            g.attrs.update(
                {"function": f"{f.__module__}:{f.__qualname__}", "seconds": seconds}
            )
            write_rng(g, state)
            items = {"args": list(given[0]), "kwargs": dict(given[1])}
            sha = digest(items)
            if label == self.resume:
                self.check_inputs(label, sha)
            self.writer._items(g, "in", "call", f"{label}/in", sha, items, {})
            entry = self.writer.put(g, "out", returned, f"{label}/out")
            g.attrs["out"] = json.dumps(entry)
            g.attrs["meta"] = json.dumps({} if entry is None else {"out": entry})
            if name == "write_tsv":
                written = Path(given[0][0])
                self.files[written.name] = label
                g.attrs["file"] = written.name
                g.attrs["file_sha256"] = file_sha256(written)
                data = np.frombuffer(
                    gzip.compress(written.read_bytes(), mtime=0), dtype=np.uint8
                )
                self.h5.require_group("files").create_dataset(written.name, data=data)
            self.order.append(label)
            print(f"{label} {seconds:.1f}s", flush=True)
            return out

        setattr(module, name, wrapped)

    def wrap_internal(self, hmrf: Any) -> None:
        """`run_core_inference`'s last outer iteration: ICM labels, merges, labels before re-indexing."""
        icm, merge, assign = (
            hmrf.icm_sweep_deque,
            hmrf.merge_assignment,
            hmrf.pipeline_clone_assignment,
        )

        def icm_wrapped(*args: Any, **kwargs: Any) -> Any:
            out = icm(*args, **kwargs)
            self.internal[self.current]["icm"] = np.array(
                kwargs["new_assignment"], copy=True
            )
            return out

        def merge_wrapped(*args: Any, **kwargs: Any) -> Any:
            out = merge(*args, **kwargs)
            new_cost, best_cost, pair = out
            if best_cost > new_cost:
                self.internal[self.current]["merges"].append(
                    [int(pair[0]), int(pair[1])]
                )
            return out

        def assign_wrapped(*args: Any, **kwargs: Any) -> Any:
            record = self.internal[self.current]
            record["merges"] = []
            record["iterations"] = record.get("iterations", 0) + 1
            record["prev"] = np.array(args[6], copy=True)
            out = assign(*args, **kwargs)
            record["raw"] = np.array(out[0], copy=True)
            # NB per outer iteration: the ARI the stop rule reads (hmrf.py:706-720) and total_llf
            record.setdefault("ari", []).append(
                float(adjusted_rand_score(args[6], out[0]))
            )
            record.setdefault("total_llf", []).append(float(out[2]))
            return out

        minimize = scipy.optimize.minimize

        def minimize_wrapped(fun: Any, x0: Any, *args: Any, **kwargs: Any) -> Any:
            """The M step's optimizer (hmm_nophasing.py:1028-1035) inside a fit: iterations, success, objective trace."""
            caller = sys._getframe(1).f_code.co_name
            if (
                caller != "_run_optimization_pipeline"
                or self.current not in self.internal
            ):
                return minimize(fun, x0, *args, **kwargs)
            trace: list[float] = []
            given = kwargs.get("callback")

            def callback(intermediate_result: Any = None) -> Any:
                trace.append(float(intermediate_result.fun))
                return given(intermediate_result) if given else None

            kwargs["callback"] = callback
            out = minimize(fun, x0, *args, **kwargs)
            last = (
                abs(trace[-1] - trace[-2]) / max(abs(trace[-1]), abs(trace[-2]), 1.0)
                if len(trace) > 1
                else None
            )
            self.internal[self.current].setdefault("optimizer", []).append(
                {
                    "nit": int(out.nit),
                    "success": bool(out.success),
                    "fun": float(out.fun),
                    "message": str(out.message),
                    "maxiter": (kwargs.get("options") or {}).get("maxiter"),
                    "last_relative_change": last,
                }
            )
            return out

        scipy.optimize.minimize = minimize_wrapped

        hmrf.icm_sweep_deque, hmrf.merge_assignment, hmrf.pipeline_clone_assignment = (
            icm_wrapped,
            merge_wrapped,
            assign_wrapped,
        )


def value(rec: Recorder, path: str) -> Any:
    return rec.writer.memo[path]


def labels_of(index: list[np.ndarray], n: int) -> np.ndarray:
    out = np.full(n, -1, dtype=np.int64)
    for c, idx in enumerate(index):
        out[np.asarray(idx)] = c
    return out


def lineage(rec: Recorder, h5: h5py.File) -> None:
    """`/lineage/segments`, `/lineage/clones` and `/internal`, from the recorded values."""
    blocks = value(rec, "02_blocks/assign_initial_blocks/out")
    genes = np.flatnonzero(blocks["is_interval"].to_numpy(dtype=bool))
    rows = blocks.index[genes]

    def at_genes(frame: pd.DataFrame, column: str) -> np.ndarray:
        found = frame[column].reindex(rows)
        return np.where(found.isna(), -1, found.fillna(-1).astype(np.int64)).astype(
            np.int64
        )

    block = at_genes(blocks, "block_id")
    refined = np.asarray(value(rec, "03_phasing/initial_phase_given_partition/out")[2])
    phase_segment = np.searchsorted(np.cumsum(refined), block, side="right")
    bins = at_genes(value(rec, "04_bins/create_bin_ranges/out"), "bin_id")
    kept = at_genes(value(rec, "06_normal/normal_baf_bin_filter/out")[0], "bin_id")
    rebinned = at_genes(value(rec, "07_rebin/create_bin_ranges/out"), "bin_id")

    final = value(rec, "08_rdr/reindex_clones/out")[0]
    pred = np.asarray(final["pred_cnv"])
    contig = blocks["CHR"].to_numpy()[genes]
    columns = {
        "genes": np.arange(genes.size),
        "blocks": block,
        "phase_segments": phase_segment,
        "bins": bins,
        "kept_bins": kept,
        "rebinned": rebinned,
    }
    n_bins = pred.shape[0]
    bin_contig = np.full(n_bins, -1)
    bin_contig[rebinned[rebinned >= 0]] = contig[rebinned >= 0]
    for c in range(pred.shape[1]):
        change = np.ones(n_bins, dtype=bool)
        change[1:] = (pred[1:, c] != pred[:-1, c]) | (bin_contig[1:] != bin_contig[:-1])
        run = np.cumsum(change) - 1
        columns[f"state_runs_clone{c}"] = np.where(
            rebinned >= 0, run[np.maximum(rebinned, 0)], -1
        )
    columns["seglevel"] = rebinned
    gene_rows = blocks.iloc[genes]
    called = rebinned >= 0
    columns["genelevel"] = np.where(called, np.cumsum(called) - 1, -1)
    del gene_rows
    table = np.stack(list(columns.values()), axis=1).astype(np.int32)
    ds = h5.create_dataset(
        "lineage/segments", data=table, compression="gzip", shuffle=True
    )
    ds.attrs["levels"] = json.dumps(list(columns))
    ds.attrs["rows"] = (
        "the gene rows (is_interval) of 02_blocks/assign_initial_blocks/out, in its order"
    )

    n = len(value(rec, "00_inputs/load_input_data/out")[1])
    baf, rdr = rec.internal["05_baf"], rec.internal["08_rdr"]
    baf_fit = np.asarray(value(rec, "05_baf/run_core_inference/out")["new_assignment"])
    baf_groups, baf_merged_res = value(rec, "05_baf/merge_by_minspots/out")
    rdr_init = np.asarray(value(rec, "08_rdr/initialize_rdr_clone_refininement/out")[0])
    rdr_fit = np.asarray(value(rec, "08_rdr/run_core_inference/out")["new_assignment"])
    rdr_groups, rdr_merged_res = value(rec, "08_rdr/merge_by_minspots/out")
    levels = {
        "initial_baf": labels_of(value(rec, "03_phasing/initialize_clones/out"), n),
        "baf_icm": baf["icm"],
        "baf_raw": baf["raw"],
        "baf_fit": baf_fit,
        "baf_merged": np.asarray(baf_merged_res["new_assignment"]),
        "rdr_init": rdr_init,
        "rdr_icm": rdr["icm"],
        "rdr_raw": rdr["raw"],
        "rdr_fit": rdr_fit,
        "rdr_merged": np.asarray(rdr_merged_res["new_assignment"]),
        "final": np.asarray(final["new_assignment"]),
    }
    parent: dict[str, np.ndarray] = {}

    def merged_map(record: dict[str, Any], before: np.ndarray) -> np.ndarray:
        m = np.arange(int(before.max()) + 1)
        for u, v in record["merges"]:
            m[m == u] = v
        return m

    def reindex(raw: np.ndarray) -> np.ndarray:
        m = np.full(int(raw.max()) + 1, -1)
        kept_labels = np.unique(raw)
        m[kept_labels] = np.arange(kept_labels.size)
        return m

    def groups_map(groups: list[list[int]]) -> np.ndarray:
        m = np.full(max(max(g) for g in groups) + 1, -1)
        for i, g in enumerate(groups):
            m[g] = i
        return m

    parent["baf_raw"] = merged_map(baf, baf["icm"])
    parent["baf_fit"] = reindex(baf["raw"])
    parent["baf_merged"] = groups_map(baf_groups)
    parent["rdr_init"] = np.array(
        [
            np.bincount(levels["baf_merged"][rdr_init == c]).argmax()
            for c in range(rdr_init.max() + 1)
        ]
    )
    parent["rdr_raw"] = merged_map(rdr, rdr["icm"])
    parent["rdr_fit"] = reindex(rdr["raw"])
    parent["rdr_merged"] = groups_map(rdr_groups)
    perm = np.full(int(levels["rdr_merged"].max()) + 1, -1)
    for old, new in zip(levels["rdr_merged"], levels["final"], strict=True):
        perm[old] = new
    parent["final"] = perm
    labels = h5.create_dataset(
        "lineage/clones/labels",
        data=np.stack(list(levels.values()), axis=1).astype(np.int16),
        compression="gzip",
        shuffle=True,
    )
    labels.attrs["levels"] = json.dumps(list(levels))
    for k, v in parent.items():
        ds = h5.create_dataset(f"lineage/clones/parent/{k}", data=v.astype(np.int16))
        ds.attrs["direction"] = (
            "child to parent" if k == "rdr_init" else "previous level to this one"
        )

    for inference, record in rec.internal.items():
        if rec.base is not None and f"{inference}/run_core_inference" in rec.copied:
            rec.base.h5.copy(
                rec.base.h5[f"internal/{inference}"], h5.require_group("internal")
            )
            continue
        g = h5.create_group(f"internal/{inference}")
        for k in ("icm", "raw", "prev"):
            g.create_dataset(
                k, data=np.asarray(record[k]).astype(np.int16), compression="gzip"
            )
        g.attrs["merges"] = json.dumps(record["merges"])
        g.attrs["iterations"] = json.dumps(record["iterations"])
        for key in ("ari", "total_llf", "optimizer"):
            g.attrs[key] = json.dumps(record.get(key, []))
        g.attrs["re_indexing"] = json.dumps(
            {int(k): int(v) for k, v in enumerate(reindex(record["raw"])) if v >= 0}
        )


def commit() -> str:
    found = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        check=False,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "cnamaste/python"],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        check=False,
    )
    return found.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def verify(path: Path) -> None:
    """Every stored node decodes to the value it hashes: the codec's round trip, on this file."""
    capture = Capture(path)
    for stage in capture.stages:
        for side in ("in", "out"):
            try:
                capture.stored(f"{stage}/{side}")
            except LookupError:
                print(f"verify: {stage}/{side} needs a replay")
            except Exception as error:  # noqa: BLE001
                print(f"verify FAILED: {stage}/{side}: {error!r}")
    capture.close()


def report(path: Path) -> None:
    """Stored bytes per top-level group and per stage."""
    sizes: dict[str, int] = {}
    with h5py.File(path, "r") as h5:

        def add(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                key = "/".join(name.split("/")[:2])
                sizes[key] = sizes.get(key, 0) + obj.id.get_storage_size()

        h5.visititems(add)
    for key, size in sorted(sizes.items(), key=lambda kv: -kv[1]):
        print(f"{size:>10_} {key}")


def refusals(base: Capture, arguments: argparse.Namespace) -> list[str]:
    """Why `base` is not a run of this sample and template, at this stage; none where it is."""
    found = []
    if base.config["fixture_hash"] != arguments.hash:
        found.append(
            f"fixture hash {base.config['fixture_hash']}, not {arguments.hash}"
        )
    if base.config["template_sha256"] != file_sha256(arguments.template):
        found.append(f"template sha256 differs from {arguments.template.name}'s")
    recorded = json.loads(base.config["inputs"])
    for name, path in input_files(arguments.sample).items():
        if name not in recorded or recorded[name]["sha256"] != file_sha256(path):
            found.append(f"input {name} differs from {path}")
    if arguments.from_stage not in base.stages:
        found.append(f"{arguments.from_stage} is not a stage it recorded")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--sample",
        type=Path,
        default=REPOSITORY / "sim" / "numcnas1.2_cnasize5e7_ploidy2_random0",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=REPOSITORY / "tests" / "data" / "zenodo_sim_config.yaml",
    )
    parser.add_argument(
        "--hash",
        default="2d4ce9a9",
        help="the sample's fixture hash, as port's realization_hash names it",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--work",
        type=Path,
        required=True,
        help="a scratch directory for the staged run",
    )
    parser.add_argument(
        "--from-stage",
        default=None,
        metavar="NN_group/stage",
        help="resume: copy the stages --base recorded before this one, and run from it",
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=None,
        help="resume: the staged-run file to copy from",
    )
    arguments = parser.parse_args(argv)
    if (arguments.from_stage is None) != (arguments.base is None):
        parser.error("--from-stage and --base go together")
    arguments.work, arguments.sample, arguments.template = (
        p.resolve() for p in (arguments.work, arguments.sample, arguments.template)
    )
    base = None
    if arguments.base is not None:
        base = Capture(arguments.base.resolve())
        refused = refusals(base, arguments)
        if refused:
            base.close()
            raise SystemExit("--base refused: " + "; ".join(refused))
    out = (
        arguments.out.resolve()
        if arguments.out
        else PROJECT / "tests" / "data" / f"sim_{arguments.hash}.hdf5"
    )

    if not os.environ.get("CNAMASTE_GRCH38") and os.environ.get("PORT_GRCH38"):
        os.environ["CNAMASTE_GRCH38"] = os.environ["PORT_GRCH38"]
    resources = grch38()
    if resources is None:
        missing = "CalicoST's GRCh38_resources not found; set $CNAMASTE_GRCH38"
        raise SystemExit(missing)
    document = yaml.safe_load(arguments.template.read_text())
    # NB a scratch working directory, for anything the run writes relative to it
    arguments.work.mkdir(parents=True, exist_ok=True)
    os.chdir(arguments.work)
    config = stage_inputs(arguments.sample, RUN_ROOT, document, resources)
    if base is not None and config.read_text() != base.config["yaml"]:
        message = "--base refused: its configuration is not the one staged"
        raise SystemExit(message)

    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".partial")
    h5 = h5py.File(partial, "w", libver="latest")
    rec = Recorder(h5)
    for name in SCHEDULE:
        rec.wrap(rc, name)
    rec.wrap_internal(hmrf)
    if base is not None:
        rec.resume_from(base, arguments.from_stage)
    for name in PLOTS:
        setattr(rc, name, lambda *_, **__: None)

    np.random.seed(ENTRY_POINT_SEED)  # noqa: NPY002
    start = time.perf_counter()
    rc.run_cnamaste(str(config))
    wall = time.perf_counter() - start

    lineage(rec, h5)
    inputs = {
        name: {"path": str(p.relative_to(REPOSITORY)), "sha256": file_sha256(p)}
        for name, p in input_files(arguments.sample).items()
    }
    h5.create_group("config").attrs.update(
        {
            "yaml": config.read_text(),
            "template": arguments.template.name,
            "template_sha256": file_sha256(arguments.template),
            "run_root": str(RUN_ROOT),
            "resources": str(resources),
            "sample": str(arguments.sample.relative_to(REPOSITORY)),
            "fixture_hash": arguments.hash,
            "inputs": json.dumps(inputs),
            "capture_commit": commit(),
            "entry_point_seed": ENTRY_POINT_SEED,
            "random_state": int(document["hmrf"]["random_state"]),
            "gmm_random_state": int(document["hmm"]["gmm_random_state"]),
            "threads": numba.get_num_threads(),  # type: ignore[no-untyped-call]
            "thread_env": json.dumps(
                {
                    k: os.environ.get(k)
                    for k in (
                        "NUMBA_NUM_THREADS",
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                    )
                }
            ),
            "plots": "stubbed",
            "wall_seconds": wall,
            "resumed_from": arguments.from_stage or "",
            "base_sha256": "" if base is None else file_sha256(base.path),
            "numba_cache_dir": numba.config.CACHE_DIR,
            "numba_compile_seconds": COMPILE["seconds"],
            "numba_cache": json.dumps(
                {
                    "hits": sum(
                        sum(d.stats.cache_hits.values()) for d in COMPILE["dispatchers"]
                    ),
                    "misses": sum(
                        sum(d.stats.cache_misses.values())
                        for d in COMPILE["dispatchers"]
                    ),
                }
            ),
            "stages": json.dumps(rec.order),
            "files": json.dumps(rec.files),
        }
    )
    h5.close()
    if base is not None:
        base.close()
    partial.replace(out)
    size = out.stat().st_size
    report(out)
    verify(out)
    print(f"wall {wall:.1f}s, {out} {size:_} bytes")
    if size > MAX_BYTES:
        print(f"over the repository's {MAX_BYTES:_} bytes", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
