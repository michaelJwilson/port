"""Stage one `run_cnamaste` run into `cnamaste/tests/data/sim_<hash>.hdf5` (T- #836).

    uv run --locked --project cnamaste python scripts/capture_cnamaste.py \
        --work DIR [--sample sim/numcnas1.2_cnasize5e7_ploidy2_random0] [--out FILE]

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

On easy (`2d4ce9a9`) the run takes 562 s on 4 threads.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import os
import random
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

import cnamaste.hmrf as hmrf  # type: ignore[import-not-found]  # noqa: PLR0402
import cnamaste.scripts.run_cnamaste as rc  # type: ignore[import-not-found]
import h5py
import numba
import numpy as np
import pandas as pd
import scipy.optimize
import yaml
from numba import _helperlib
from sklearn.metrics import adjusted_rand_score

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
            self.writer._items(g, "in", "call", f"{label}/in", digest(items), items, {})
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
    arguments = parser.parse_args(argv)
    arguments.work, arguments.sample, arguments.template = (
        p.resolve() for p in (arguments.work, arguments.sample, arguments.template)
    )
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
    # NB `hmm_emission.flush_perf` appends to `cnamaste.perf` in the working directory
    arguments.work.mkdir(parents=True, exist_ok=True)
    os.chdir(arguments.work)
    config = stage_inputs(arguments.sample, RUN_ROOT, document, resources)

    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".partial")
    h5 = h5py.File(partial, "w", libver="latest")
    rec = Recorder(h5)
    for name in SCHEDULE:
        rec.wrap(rc, name)
    rec.wrap_internal(hmrf)
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
            "stages": json.dumps(rec.order),
            "files": json.dumps(rec.files),
        }
    )
    h5.close()
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
