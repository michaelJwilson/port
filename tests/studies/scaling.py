"""The scaling ladder: `run_cnaster_port --sal` stage by stage, by spot count (#569).

Run as `python -m tests.studies.scaling <command> <target> [options] [-- flags]`,
one command per fresh process, one `SCALING` JSON line per measurement on
stdout. Numbers are in `docs/study-scaling.md`.

- `e2e REALIZATION`: `tests.sim_audit`'s run with every call the driver
  (`cnaster.scripts.run_cnaster`) makes, and the substages in `LEVELS`, timed
  and their RSS high-water sampled, outermost call per level; then scored
  against the planted truth. `--dump DIR` pickles every Potts problem.
- `load REALIZATION`: the patched loader and the installed adjacency,
  `--repeat` times in one process, for the stage alone and for leaks.
- `potts CALL.pkl`: each solver row on one Potts problem, floorless, against
  TRW-S's lower bound; `--repeat` for leaks.
- `resample TARGET --source CALL.pkl --source-truth R --out OUT.pkl`: a
  Potts problem at a larger rung, from a captured one (`TARGET` a
  realization or a manifest, laid out without drawing counts).
- `figure OUT.png --records LOG ...`: wall and peak against spots, stamped.

**Peak** is a sampler's: RSS read every `PERIOD` seconds within a window
opened at the stage's call, so a stage's peak includes what it inherited
(`entry_gb`), and `rise_gb` is the largest of its calls' peak less that
call's own entry. `ru_maxrss` bounds every row.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import resource
import sys
import threading
import time
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PERIOD = 0.05
"""Seconds between RSS reads: below the shortest stage worth a row."""

DRIVER = "cnaster.scripts.run_cnaster"


def rss() -> int:
    """This process's resident set, in bytes, from `/proc/self/statm`."""
    import os

    pages = int(Path("/proc/self/statm").read_text().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE")


class HighWater:
    """The largest RSS within each open window, sampled on a daemon thread.

    Windows nest: a stage inside another has its own, and the sampler raises
    every open one, so an outer stage's peak includes its inner stages'.
    """

    def __init__(self, period: float = PERIOD) -> None:
        self.period = period
        self.open: dict[int, int] = {}
        self._next = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _raise(self) -> None:
        now = rss()
        for key, peak in list(self.open.items()):
            if now > peak:
                self.open[key] = now

    def _run(self) -> None:
        while not self._stop.wait(self.period):
            self._raise()

    def enter(self) -> tuple[int, int]:
        """A new window: its key and the RSS it opened at."""
        self._next += 1
        now = rss()
        self.open[self._next] = now
        return self._next, now

    def leave(self, key: int) -> int:
        """Close a window; its peak."""
        self._raise()
        return self.open.pop(key)

    def close(self) -> None:
        self._stop.set()
        self._thread.join()


@dataclass
class Row:
    """One stage's calls: count, wall, and the largest of their peaks."""

    calls: int = 0
    wall: float = 0.0
    peak_gb: float = 0.0
    entry_gb: float = 0.0
    """RSS when its first call opened: what it inherited."""
    rise_gb: float = 0.0
    """The largest of its calls' peak less that call's own entry RSS."""
    target: str = ""
    """What the name was bound to when called: `module.qualname`."""
    arrays_mb: dict[str, float] = field(default_factory=dict)
    """Its first call's array arguments over 10 MB: `name shape dtype` to MB."""


@dataclass
class Profile:
    rows: dict[str, Row] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)


def _gb(value: int) -> float:
    return round(value / 1e9, 3)


LEVELS: tuple[tuple[str, str, tuple[str, ...] | None], ...] = (
    ("", DRIVER, None),
    (
        "hmrf.",
        "cnaster.hmrf",
        (
            "merge_pseudobulk_by_index_mix",
            "pipeline_baum_welch",
            "pipeline_clone_assignment",
            "clone_stack_obs",
        ),
    ),
    ("field.", "port.patch.hmrf.tabulated_field", ("spot_clone_field", "field_kernel")),
    (
        "potts.",
        "port.extensions.label_solver",
        (
            "fusion_then_merge",
            "expansion_then_merge",
            "expansion_then_floor",
            "sal_icm_sweep",
        ),
    ),
    ("potts.", "port.patch.icm.interface", ("icm_sweep",)),
)
"""`(prefix, module, names)`: the driver's every callable, then the
substages inside `run_core_inference` and `pipeline_clone_assignment`,
each counted at its own level's outermost call. `None` is every public
callable the module binds."""


def _callables(module: Any, names: tuple[str, ...] | None) -> list[tuple[str, Any]]:
    if names is not None:
        return [(n, getattr(module, n)) for n in names if hasattr(module, n)]
    return [
        (name, value)
        for name, value in vars(module).items()
        if not name.startswith("_")
        and name != "run_cnaster"
        and not isinstance(value, type)
        and callable(value)
        and getattr(value, "__module__", "") not in {"builtins", "typing"}
    ]


def _dump(
    path: Path,
    solver: str,
    field: Any,
    graph: Any,
    assignment: Any,
    spatial_weight: float,
    **knobs: Any,
) -> None:
    """One Potts problem in `tests.studies.potts_solvers.capture`'s record format."""
    import pickle

    record = {
        "solver": solver,
        "field": np.array(field),
        "indptr": np.array(graph.indptr),
        "indices": np.array(graph.indices),
        "weights": np.array(graph.weights),
        "start": np.array(assignment),
        "spatial_weight": float(spatial_weight),
        "knobs": {k: np.array(v) for k, v in knobs.items()},
    }
    with path.open("wb") as handle:
        pickle.dump(record, handle, protocol=5)


def _arrays(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, float]:
    """`position-or-name shape dtype` to MB, for each array argument over 10 MB."""
    out = {}
    named = [(str(i), a) for i, a in enumerate(args)] + list(kwargs.items())
    for name, value in named:
        if isinstance(value, np.ndarray) and value.nbytes > 10e6:
            key = f"{name} {value.shape} {value.dtype}"
            out[key] = round(value.nbytes / 1e6, 1)
    return out


@contextmanager
def profiled(
    profile: Profile, water: HighWater, dump: Path | None = None
) -> Iterator[None]:
    """Time and peak the stages `LEVELS` names, outermost call per level.

    With `dump`, every Potts solve's problem -- the folded field, the graph,
    the start, the coupling and the knobs -- is pickled there as
    `call<k>.pkl`, the isolated labelling study's input.
    """
    driver = importlib.import_module(DRIVER)
    original = getattr(driver, "run_cnaster")  # noqa: B009 -- untyped module
    depth: dict[str, int] = {}
    dumped = [0]

    def wrap(prefix: str, name: str, current: Callable[..., Any]) -> Callable[..., Any]:
        label = prefix + name

        def call(*args: Any, **kwargs: Any) -> Any:
            if depth.get(prefix, 0):
                return current(*args, **kwargs)
            if dump is not None and prefix == "potts.":
                _dump(dump / f"call{dumped[0]:03d}.pkl", name, *args, **kwargs)
                dumped[0] += 1
            depth[prefix] = depth.get(prefix, 0) + 1
            census = _arrays(args, kwargs) if label not in profile.rows else {}
            key, entry = water.enter()
            started = time.perf_counter()
            try:
                return current(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - started
                peak = water.leave(key)
                depth[prefix] -= 1
                row = profile.rows.setdefault(
                    label,
                    Row(
                        entry_gb=_gb(entry),
                        target=f"{getattr(current, '__module__', '?')}."
                        f"{getattr(current, '__qualname__', '?')}",
                    ),
                )
                if label not in profile.order:
                    profile.order.append(label)
                    row.arrays_mb = census
                row.calls += 1
                row.wall += elapsed
                row.peak_gb = max(row.peak_gb, _gb(peak))
                row.rise_gb = max(row.rise_gb, _gb(peak - entry))

        return call

    def run(*args: Any, **kwargs: Any) -> Any:
        undo: list[tuple[Any, str, Any]] = []
        for prefix, path, names in LEVELS:
            module = importlib.import_module(path)
            for name, value in _callables(module, names):
                undo.append((module, name, value))
                setattr(module, name, wrap(prefix, name, value))
        try:
            return original(*args, **kwargs)
        finally:
            for module, name, value in reversed(undo):
                setattr(module, name, value)

    setattr(driver, "run_cnaster", run)  # noqa: B010 -- untyped module
    try:
        yield
    finally:
        setattr(driver, "run_cnaster", original)  # noqa: B010 -- untyped module


def _sample(path: str) -> Any:
    from tests.sim_fixtures import load_simulated

    return load_simulated(str(Path(path).resolve()))


def code_stamp() -> str:
    """The commit, and `+<8 hex>` of the working tree's diff where it differs."""
    import hashlib
    import subprocess

    here = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=here,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--", "python", "src"],
        cwd=here,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    dirty = f"+{hashlib.sha256(diff.encode()).hexdigest()[:8]}" if diff else ""
    return f"{commit or 'unknown'}{dirty}"  # fmt: skip


CODE = code_stamp()
"""Read at import, before the run: the code a record measured."""


def _emit(record: dict[str, Any]) -> None:
    record["code"] = CODE
    record["ru_maxrss_gb"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3
    )
    print("SCALING " + json.dumps(record), flush=True)


def e2e(arguments: argparse.Namespace) -> None:
    """`run_cnaster_port` on the realization, profiled per driver call, scored."""
    import matplotlib as mpl

    mpl.use("Agg")
    from dataclasses import asdict

    from tests.sim_audit import run_arm

    sample = _sample(arguments.realization)
    profile = Profile()
    water = HighWater()
    flags = list(arguments.flags) or ["--sal", "--no-plots"]

    dump = Path(arguments.dump) if arguments.dump else None
    if dump is not None:
        dump.mkdir(parents=True, exist_ok=True)

    with profiled(profile, water, dump):
        root = Path(arguments.root) if arguments.root else None
        started = time.perf_counter()
        if arguments.cprofile:
            import cProfile

            profiler = cProfile.Profile()
            profiler.enable()
        try:
            recovery, output = run_arm(sample, flags, {}, root)
        finally:
            if arguments.cprofile:
                profiler.disable()
                profiler.dump_stats(arguments.cprofile)
        wall = time.perf_counter() - started

    water.close()
    _emit(
        {
            "command": "e2e",
            "realization": arguments.realization,
            "spots": int(sample.barcodes.size),
            "flags": flags,
            "wall_s": round(wall, 2),
            "recovery": {
                k: v for k, v in asdict(recovery).items() if k not in {"clone_of"}
            },
            "stages": {
                name: vars(profile.rows[name])
                | {"wall": round(profile.rows[name].wall, 3)}
                for name in profile.order
            },
            "output": str(output),
        }
    )


POTTS_ROWS = ("icm", "alpha-rust-fuse-merge", "alpha-rust", "icm-numba")
"""`cnaster`'s ICM (`icm_sweep_deque`, through port's interface), `--sal`'s
row, sal's Rust alpha expansion alone, and sal's compiled single-site ICM."""


def potts(arguments: argparse.Namespace) -> None:
    """Each solver on one dumped problem, floorless, against TRW-S's bound.

    Every row starts from the captured assignment and runs once per process
    call, so its peak is its own; `bound` is TRW-S's lower bound and `gap`
    the row's energy above it, both in nats. `--rows` restricts the rows;
    `trws` is a row too, and the costliest.
    """
    import torch
    from port.extensions.label_solver import sweep_for
    from port.patch.icm.interface import icm_sweep
    from sal.search.potts_starts import LabellingEnergy

    from tests.studies.potts_solvers import _load

    rung, call = _load(arguments.realization)
    energy = LabellingEnergy(rung)
    rows = arguments.rows.split(",") if arguments.rows else [*POTTS_ROWS, "trws"]
    water = HighWater()
    np.random.seed(0)  # noqa: NPY002 -- cnaster's ICM reads the global stream
    start_energy = float(energy(torch.as_tensor(np.asarray(call["start"]))))

    for row, repetition in itertools.product(rows, range(arguments.repeat)):
        key, entry = water.enter()
        opened = time.perf_counter()
        bound = None
        if row == "trws":
            from sal.search.trws import trws

            result = trws(rung.graph, rung.field)
            labels = np.asarray(result.labelling)
            bound = float(result.bound)
        else:
            sweep = icm_sweep if row == "icm" else sweep_for(row)
            labels = np.array(call["start"], dtype=np.int64)
            knobs = {**call["knobs"], "min_clone_spots": 0}
            sweep(call["field"], call["graph"], labels, call["spatial_weight"], **knobs)
        seconds = time.perf_counter() - opened
        peak = water.leave(key)
        _emit(
            {
                "command": "potts",
                "problem": arguments.realization,
                "spots": int(rung.n_nodes),
                "clones": int(rung.n_states),
                "row": row,
                "seconds": round(seconds, 4),
                "energy": float(energy(torch.as_tensor(labels))),
                "start_energy": start_energy,
                "bound": bound,
                "entry_gb": _gb(entry),
                "peak_gb": _gb(peak),
                "repetition": repetition,
                "after_gb": _gb(rss()),
            }
        )
    water.close()


def load(arguments: argparse.Namespace) -> None:
    """The patched loader and the installed adjacency, `--repeat` times in one process.

    Each repetition is one row: wall and peak per step, and the RSS after it
    with the result released. Growth across repetitions beyond `LEAK` per
    repetition is a leak. `--sparse` loads under `sparse_counts`. The
    adjacency is `port.patch.spatial.lattice_multislice_adjacency`, the
    function `run_cnaster_port` binds over `cnaster`'s (#190, #417).
    """
    import gc

    from cnaster.config import YAMLConfig, set_global_config
    from port.patch.io import load_input_data
    from port.patch.spatial import lattice_multislice_adjacency

    from tests.sim_audit import _drawn_config

    sample = _sample(arguments.realization)
    root = Path(arguments.root or "/tmp/scaling-load")
    config_path = _drawn_config(sample, root, {})
    config = YAMLConfig.from_file(str(config_path))
    set_global_config(config)
    water = HighWater()

    for repetition in range(arguments.repeat):
        key, entry = water.enter()
        opened = time.perf_counter()
        loaded = load_input_data(
            config,
            filter_gene_file=config.references.filtergenelist_file,
            filter_range_file=config.references.filterregion_file,
            min_snp_umis=config.quality.spot_min_snp_umis,
            min_percent_expressed_spots=config.quality.min_percent_expressed_spots,
            sparse_counts=arguments.sparse,
        )
        load_seconds = time.perf_counter() - opened
        load_peak = water.leave(key)
        sizes = {
            name: _nbytes(getattr(loaded, name))
            for name in ("exp_counts", "cell_snp_Aallele", "cell_snp_Ballele", "coords")
        }
        sizes["count_layer"] = _nbytes(loaded.adata.layers["count"])

        from cnaster.io import get_sample_list

        sample_list, sample_ids = get_sample_list(loaded.adata)
        key, _ = water.enter()
        opened = time.perf_counter()
        adjacency = lattice_multislice_adjacency(
            sample_ids, sample_list, loaded.coords, None, maxspots_pooling=1
        )
        adjacency_seconds = time.perf_counter() - opened
        adjacency_peak = water.leave(key)
        nnz = int(adjacency.adjacency_mat.nnz)
        del loaded, adjacency
        gc.collect()
        _emit(
            {
                "command": "load",
                "realization": arguments.realization,
                "spots": int(sample.barcodes.size),
                "sparse": bool(arguments.sparse),
                "repetition": repetition,
                "load_s": round(load_seconds, 3),
                "load_entry_gb": _gb(entry),
                "load_peak_gb": _gb(load_peak),
                "adjacency_s": round(adjacency_seconds, 3),
                "adjacency_peak_gb": _gb(adjacency_peak),
                "adjacency_nnz": nnz,
                "resident_gb": {k: round(v / 1e9, 4) for k, v in sizes.items()},
                "after_gb": _gb(rss()),
            }
        )
    water.close()


def _nbytes(value: Any) -> int:
    """Bytes an array, a sparse matrix, a `DataFrame` or a `NamedCounts` holds."""
    import scipy.sparse as sp

    if hasattr(value, "matrix"):
        value = value.matrix
    if sp.issparse(value):
        return int(value.data.nbytes + value.indices.nbytes + value.indptr.nbytes)
    if hasattr(value, "memory_usage"):
        return int(value.memory_usage(deep=True).sum())
    return int(np.asarray(value).nbytes)


def _planted(target: str) -> tuple[np.ndarray, np.ndarray]:
    """Planted labels and integer coordinates of a realization, or of a manifest's layout.

    A `.toml` is laid out without drawing counts -- `port.sim.draw.layout`
    on its array, `normal` 0 -- so a rung too large to draw still has its
    clones; a directory is read as a realization.
    """
    if target.endswith(".toml"):
        from port.sim.draw import ARRAYS, layout, read_manifest

        manifest = read_manifest(target)
        rows, cols, points = ARRAYS[manifest.array["kind"]](
            int(manifest.array["rows"]), int(manifest.array["columns"])
        )
        layout_rng = np.random.default_rng(
            np.random.SeedSequence(manifest.seed).spawn(3)[1]
        )
        (labels,), _ = layout(manifest, points, layout_rng)
        return labels + 1, np.column_stack([rows, cols]).astype(np.int64)

    sample = _sample(target)
    return sample.labels, np.rint(sample.coords).astype(np.int64)


def resample(arguments: argparse.Namespace) -> None:
    """A Potts problem at a larger rung, from a captured one and both truths.

    `--source CALL.pkl --source-truth R` (the realization the call came
    from) to `realization` (the target rung): each target spot takes the
    field row of a source spot of the same planted clone, drawn uniformly
    with seed 0, so the field's per-clone law is the captured one at any
    size. The graph is the target's own `knn` Moore adjacency, the one the
    run installs; the start is the field's argmax; the coupling and knobs
    are the source call's. Written to `--out` in the capture's format.
    """
    import pickle

    from port.extensions.adjacency import knn_adjacency

    with Path(arguments.source).open("rb") as handle:
        call: dict[str, Any] = pickle.load(handle)
    source = _sample(arguments.source_truth)
    labels, coords = _planted(arguments.realization)
    rng = np.random.default_rng(0)
    field = np.empty((labels.size, call["field"].shape[1]))

    for clone in np.unique(labels):
        spots = np.flatnonzero(labels == clone)
        pool = np.flatnonzero(source.labels == clone)
        field[spots] = call["field"][rng.choice(pool, spots.size)]

    graph = knn_adjacency(coords, "moore")
    record = {
        "solver": call["solver"],
        "field": field,
        "indptr": graph.indptr,
        "indices": graph.indices,
        "weights": graph.data,
        "start": np.argmax(field, axis=1).astype(np.int64),
        "spatial_weight": call["spatial_weight"],
        "knobs": {},
    }
    with Path(arguments.out).open("wb") as handle:
        pickle.dump(record, handle, protocol=5)
    _emit({"command": "resample", "out": arguments.out, "spots": int(field.shape[0])})


SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
"""The categorical slots, in fixed order: one per stage, never cycled."""

FIGURE_STAGES = (
    "run_core_inference",
    "summarize_counts_for_bins",
    "filter_normal_diffexp",
    "hill_climbing_integer_copynumber_fixdiploid_milp",
    "load_input_data",
)
"""The stages drawn beside the whole run: the largest by wall or peak."""


def records(paths: list[str]) -> list[dict[str, Any]]:
    """Every `SCALING` e2e record in `paths`, in order."""
    out = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.startswith("SCALING "):
                record = json.loads(line.removeprefix("SCALING "))
                if record["command"] == "e2e":
                    out.append(record)
    return out


def stamp(fig: Any, record: Any) -> str:
    """The figure's reference, drawn on it: a hash of the records it plots and the code.

    `tests/studies/potts_plot.py`'s stamp: `data` is the first 8 hex of
    SHA-256 over the records' JSON, `code` the commit and diff (`CODE`).
    """
    import hashlib

    data = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    text = f"data {data[:8]} · code {CODE}"
    fig.text(0.995, 0.005, text, ha="right", va="bottom", fontsize=7, color="#6b6b6b")
    return text


def figure(arguments: argparse.Namespace) -> None:
    """Wall and peak against spots, log-log, the run and its largest stages.

    Two panels, one quantity each. The run is the dark line; each stage keeps
    its slot across both panels. `realization` names the output PNG; the
    records are `--records`.
    """
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(records(arguments.records), key=lambda r: r["spots"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
    spots = np.array([r["spots"] for r in rows], dtype=np.float64)

    for ax, key, total, label in (
        (axes[0], "wall", "wall_s", "wall (s)"),
        (axes[1], "peak_gb", "ru_maxrss_gb", "peak RSS (GB)"),
    ):
        ax.plot(spots, [r[total] for r in rows], color="#1a1a19", lw=2, marker="o",
                ms=5, label="whole run")  # fmt: skip
        for colour, stage in zip(SERIES, FIGURE_STAGES, strict=False):
            values = [r["stages"].get(stage, {}).get(key, np.nan) for r in rows]
            ax.plot(spots, values, color=colour, lw=2, marker="o", ms=4, label=stage)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("spots")
        ax.set_ylabel(label)
        ax.grid(True, which="major", color="#e4e4e0", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    if key == "peak_gb":
        axes[1].axhline(12.0, color="#6b6b6b", lw=1, ls="--")
        axes[1].text(spots[0], 12.0, " 12 GB cap", va="bottom", fontsize=8,
                     color="#6b6b6b")  # fmt: skip
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, frameon=False, loc="outside lower center",
               ncol=3)  # fmt: skip
    stamp(fig, rows)
    fig.savefig(arguments.realization, dpi=150)
    plt.close(fig)
    _emit({"command": "figure", "out": arguments.realization, "rows": len(rows)})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command", choices=["e2e", "potts", "resample", "load", "figure"]
    )
    parser.add_argument("realization")
    parser.add_argument("--root", default=None)
    parser.add_argument("--dump", default=None, help="pickle each Potts problem here")
    parser.add_argument("--cprofile", default=None, help="e2e: cProfile stats here")
    parser.add_argument("--source", default=None, help="resample: a captured call")
    parser.add_argument(
        "--source-truth", default=None, help="resample: its realization"
    )
    parser.add_argument("--out", default=None, help="resample: where to write")
    parser.add_argument("--repeat", type=int, default=1, help="load: repetitions")
    parser.add_argument("--sparse", action="store_true", help="load: sparse_counts")
    parser.add_argument("--records", nargs="*", default=[], help="figure: logs")
    parser.add_argument("--rows", default="", help="potts: comma-separated rows")
    argv = list(argv or [])
    cut = argv.index("--") if "--" in argv else len(argv)
    arguments = parser.parse_args(argv[:cut])
    arguments.flags = argv[cut + 1 :]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        {
            "e2e": e2e,
            "potts": potts,
            "resample": resample,
            "load": load,
            "figure": figure,
        }[arguments.command](arguments)


if __name__ == "__main__":
    main(sys.argv[1:])
