"""Population study: clone detection against UMIs, CNA detection against length (#544).

One member is one seed of `sim/manifests/population.toml`: its tree, events,
layout (each clone's size drawn from `[layout.size]`) and counts all come
from that seed, so clone sizes and event lengths vary across members. Each
member is run once per analysis coupling `J` (`hmrf.spatial_weight`, read by
`cnaster.hmrf`; the generator has none) and scored against its own truth.
J starts just above the lattice's critical coupling (`J_CRITICAL`): below it
the prior orders nothing, so the study reads J where it does.

    python -m tests.studies.population run --seeds 0:40 --J 0.8 --out DIR
    python -m tests.studies.population report --out DIR --study2-J 0.8

`run` is resumable: a `(seed, J)` with a record in `DIR/records/` is skipped.
Each worker runs one thread, one worker per core, and a member's draw and
run directories are deleted once its records are written, so the disk a
study holds is its records. `report` is `tests.studies.population_report`.

**Scored per tumour clone:** its spots, the sum of its spots' drawn UMIs,
and its completeness -- the share of its spots in the fitted clone matched
to it by overlap (`tests.scoring.matched`), on `clone_labels.tsv`, which
carries #518's merge. Detected at completeness >= `DETECTED`.

**Scored per event of a detected clone** (every event on its path from the
root): the run's bins whose midpoint lies in the event and where the clone's
planted `(A, B)` is the event's, so an event a later one overwrote is not
scored as this one. Recovered where at least `RECOVERED` of those bins
decode, in the matched fitted clone, to the planted pair up to phase. An
event covering no bin midpoint is kept, as not recovered: the run cannot
report it. An event overwritten on more than half its bins is dropped.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "sim" / "manifests" / "population.toml"

J_CRITICAL = float(np.log(2.0))
"""The q = 4 Potts model's critical coupling on the triangular lattice.

`hmrf.spatial_weight` multiplies unit-weight edges between a spot and its six
neighbours (`port.extensions.adjacency`: `(z / d_i + z / d_j) / 2`, 1 inside
the array), so it is J in `exp(J sum delta)`. The population's
configuration fits `hmrf.n_clones = 4`. On the triangular lattice the
critical point solves `v^3 + 3 v^2 = q`, `v = e^J - 1` (Baxter); at q = 4
that is v = 1, J_c = ln 2 = 0.693. The configured default, 1.0, is 1.44 J_c.
"""

J_VALUES = (0.8,)
"""Where the study starts: just above J_c (1.15 J_c), then tailored (#544)."""

DETECTED = 0.90
"""A clone is detected when this share of its spots is in its matched clone."""

RECOVERED = 0.90
"""An event is recovered when this share of its bins decodes to its pair."""

FLAGS = ("--sal", "--no-plots")

CLASSES = {
    "LOH": {(1, 0), (0, 1), (2, 0), (0, 2)},
    "balanced gain": {(2, 2)},
    "imbalanced gain": {(2, 1), (1, 2), (3, 1), (1, 3)},
}
"""Copy-state classes of Study 2; `all` pools every event."""


def copy_class(a: int, b: int) -> str:
    """`(A, B)`'s class in `CLASSES`, or `other`."""
    return next((k for k, pairs in CLASSES.items() if (a, b) in pairs), "other")


def draw_member(seed: int, into: Path) -> Path:
    """Seed `seed` of the population manifest, written under `into`; its sample path."""
    from dataclasses import replace

    from port.sim.draw import _merge, draw, read_manifest

    manifest = read_manifest(MANIFEST)
    manifest = replace(
        manifest, tables=_merge(manifest.tables, {"sample": {"seed": seed}})
    )
    return draw(manifest, into).realizations[0]


def spot_umis(path: Path, barcodes: np.ndarray) -> np.ndarray:
    """Each spot's total gene UMI, in `barcodes`' order."""
    import anndata

    totals: dict[str, float] = {}
    for h5ad in sorted(path.glob("*/filtered_feature_bc_matrix.h5ad")):
        counts = anndata.read_h5ad(h5ad)
        matrix: Any = counts.X
        summed = np.asarray(matrix.sum(axis=1)).ravel()
        totals.update(zip(counts.obs_names.astype(str), summed, strict=True))
    return np.array([totals[str(b)] for b in barcodes], dtype=np.float64)


def clone_events(path: Path) -> dict[str, list[tuple[str, int, int, int, int]]]:
    """Each clone's events, root first, from `truth_tree.tsv`."""
    tree = pd.read_csv(path / "truth_tree.tsv", sep="\t", dtype={"chr": str})
    parent = {
        str(n): (None if pd.isna(p) else str(p))
        for n, p in tree.drop_duplicates("node")[["node", "parent"]].itertuples(
            index=False
        )
    }
    edges = tree.dropna(subset=["chr"])
    on_edge: dict[str, list[tuple[str, int, int, int, int]]] = {}
    for row in edges.itertuples(index=False):
        on_edge.setdefault(str(row.node), []).append(
            (str(row.chr), int(row.start), int(row.end), int(row.A), int(row.B))
        )

    out: dict[str, list[tuple[str, int, int, int, int]]] = {}
    for node in parent:
        chain: list[str] = []
        up: str | None = node
        while up is not None:
            chain.append(up)
            up = parent[up]
        out[node] = [e for n in reversed(chain) for e in on_edge.get(n, [])]
    return out


def score_member(sample: Any, output: Path) -> dict[str, Any]:
    """Per tumour clone its size, UMIs and completeness; per event its recovery."""
    from tests.scoring import matched, overlap
    from tests.sim_audit import read_run

    run = read_run(sample, output)
    fitted = np.asarray(run["labels"])
    scored = fitted >= 0
    planted = np.asarray(sample.labels)
    n_fitted = int(fitted.max()) + 1
    counts = overlap(planted[scored], fitted[scored], len(sample.clones), n_fitted)
    match = matched(counts)
    umis = spot_umis(sample.path, sample.barcodes)
    events = clone_events(sample.path)

    seglevel = run["seglevel"]
    chrom = seglevel["CHR"].astype(str).str.removeprefix("chr").to_numpy()
    middle = ((seglevel["START"] + seglevel["END"]) // 2).to_numpy()
    truth = sample.copies_at(chrom, middle)

    clones, scored_events = [], []
    for c, name in enumerate(sample.clones):
        if name == "normal":
            continue
        spots = planted == c
        m = match.get(c)
        hit = 0 if m is None else int(counts[c, m])
        completeness = hit / int(spots.sum())
        precision = 0.0 if m is None else hit / max(int(counts[:, m].sum()), 1)
        detected = completeness >= DETECTED
        clones.append(
            {
                "clone": name,
                "spots": int(spots.sum()),
                "umis": float(umis[spots].sum()),
                "completeness": round(completeness, 4),
                "precision": round(precision, 4),
                "detected": bool(detected),
            }
        )
        if not detected or m is None:
            continue

        a, b = run["a"][:, m], run["b"][:, m]
        for chromosome, start, end, pa, pb in events.get(name, []):
            inside = (chrom == chromosome) & (middle >= start) & (middle < end)
            visible = inside & (truth[:, c, 0] == pa) & (truth[:, c, 1] == pb)
            if inside.sum() and visible.sum() < 0.5 * inside.sum():
                continue
            right = np.minimum(a, b) == min(pa, pb)
            right &= np.maximum(a, b) == max(pa, pb)
            share = float(right[visible].mean()) if visible.any() else 0.0
            scored_events.append(
                {
                    "clone": name,
                    "chr": chromosome,
                    "length": int(end - start),
                    "a": pa,
                    "b": pb,
                    "class": copy_class(pa, pb),
                    "bins": int(visible.sum()),
                    "correct": round(share, 4),
                    "recovered": bool(visible.any() and share >= RECOVERED),
                }
            )

    return {"n_fitted": int(np.unique(fitted[scored]).size), "clones": clones,
            "events": scored_events}  # fmt: skip


def _record(out: Path, seed: int, j: float) -> Path:
    return out / "records" / f"s{seed:04d}-J{j:g}.json"


def run_member(seed: int, js: tuple[float, ...], out: Path) -> None:
    """Draw seed `seed`, run and score it at each `J`, keep only the records."""
    from tests.sim_audit import run_arm
    from tests.sim_fixtures import load_simulated

    todo = [j for j in js if not _record(out, seed, j).exists()]
    if not todo:
        return

    draws = out / "draws" / f"s{seed:04d}"
    path = draw_member(seed, draws)
    sample = load_simulated(str(path))

    for j in todo:
        runs = out / "runs" / f"s{seed:04d}-J{j:g}"
        started = time.perf_counter()
        _, output = run_arm(sample, list(FLAGS), {"hmrf.spatial_weight": j}, root=runs)
        wall = time.perf_counter() - started
        record = {"seed": seed, "J": j, "wall": round(wall, 1), "flags": list(FLAGS),
                  **score_member(sample, output)}  # fmt: skip
        target = _record(out, seed, j)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record) + "\n")
        shutil.rmtree(runs, ignore_errors=True)

    shutil.rmtree(draws, ignore_errors=True)


def _worker(task: tuple[int, tuple[float, ...], str]) -> str:
    seed, js, out = task
    log = Path(out) / "logs" / f"s{seed:04d}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        sys.stdout = sys.stderr = handle
        try:
            run_member(seed, js, Path(out))
        except Exception as error:  # noqa: BLE001 -- recorded, the study goes on
            print(f"FAILED seed {seed}: {error!r}", flush=True)
            return f"s{seed}: failed ({error!r})"
    return f"s{seed}: done"


def _seeds(text: str) -> list[int]:
    start, _, stop = text.partition(":")
    return list(range(int(start), int(stop))) if stop else [int(start)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", default="0:60", help="START:STOP")
    parser.add_argument("--J", default=",".join(f"{j:g}" for j in J_VALUES))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--study2-J", type=float, default=J_VALUES[0],
                        help="the J Study 2's length curves are read at")  # fmt: skip
    arguments = parser.parse_args(argv)

    if arguments.command == "report":
        from tests.studies.population_report import report

        report(arguments.out, arguments.study2_J)
        return 0

    import multiprocessing

    import matplotlib as mpl

    mpl.use("Agg")
    for name in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):  # fmt: skip
        os.environ[name] = "1"

    js = tuple(float(j) for j in arguments.J.split(","))
    cores = min(arguments.workers, os.cpu_count() or 1)
    tasks = [(s, js, str(arguments.out)) for s in _seeds(arguments.seeds)]
    context = multiprocessing.get_context("spawn")
    with context.Pool(cores, maxtasksperchild=1) as pool:
        for line in pool.imap_unordered(_worker, tasks):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
