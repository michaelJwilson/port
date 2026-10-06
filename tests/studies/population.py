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
study holds is its records and each run's `KEPT` outputs in `DIR/outputs/`. `report` is `tests.studies.population_report`.

**Scored per tumour clone:** its spots, the sum of its spots' drawn UMIs,
and its completeness -- the share of its spots in the fitted clone matched
to it by overlap (`port.qa.scoring.matched`), on `clone_labels.tsv`, which
carries #518's merge. Detected at completeness >= `DETECTED`.

**Scored per event of a detected clone** (every event on its path from the
root): the run's bins whose midpoint lies in the event and where the clone's
planted `(A, B)` is the event's, so an event a later one overwrote is not
scored as this one. Recovered where at least `RECOVERED` of those bins
decode, in the matched fitted clone, to the planted pair up to phase. An
event covering no bin midpoint is kept, as not recovered: the run cannot
report it. An event overwritten on more than half its bins is dropped.

**Scored per `(1, 1)` segment of a detected clone** (specificity): each
`cnv_seglevel.tsv` row whose midpoint the clone's truth holds at `(1, 1)`.
Specific where the matched fitted clone decodes it to `(1, 1)`. Its
covariate is the SNP-covering UMI it holds: `A + B` summed over the SNPs in
`[START, END)` and over the planted clone's spots. Stored per clone as two
columns, `snp_umis` and `specific`, one entry per segment.

**A run that raises is a result**: its record carries the error and no
clones, and the report counts such runs per J rather than dropping the member
silently.
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

KEPT = ("clone_labels.tsv", "cnv_seglevel.tsv")
"""A run's outputs kept beside its record, so a new score needs no rerun."""

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


def draw_member(seed: int, into: Path, manifest_path: Path = MANIFEST) -> Path:
    """Seed `seed` of a population manifest, written under `into`; its sample path."""
    from dataclasses import replace

    from port.sim.draw import _merge, draw, read_manifest

    manifest = read_manifest(manifest_path)
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


def snp_umis(path: Path, barcodes: np.ndarray) -> tuple[np.ndarray, np.ndarray, Any]:
    """Each SNP's chromosome and position, and `A + B` per spot in `barcodes`' order."""
    import scipy.sparse

    ids = np.load(path / "snp" / "unique_snp_ids.npy", allow_pickle=True).astype(str)
    fields = np.char.split(ids, "_")
    chrom = np.array([f[0] for f in fields])
    pos = np.array([int(f[1]) for f in fields], dtype=np.int64)
    order = {
        b: i for i, b in enumerate((path / "snp" / "barcodes.txt").read_text().split())
    }
    rows = np.array([order[str(b)] for b in barcodes])
    total = scipy.sparse.load_npz(path / "snp" / "cell_snp_Aallele.npz").tocsr()
    total = total + scipy.sparse.load_npz(path / "snp" / "cell_snp_Ballele.npz").tocsr()
    return chrom, pos, total[rows]


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
    from port.qa.scoring import matched, overlap

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
            "events": scored_events,
            "neutral_segments": _neutral(sample, run, planted, match, clones)}  # fmt: skip


def _neutral(
    sample: Any,
    run: dict[str, Any],
    planted: np.ndarray,
    match: dict[int, int],
    clones: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Each detected clone's `(1, 1)` segments: SNP UMIs held, and decoded `(1, 1)`."""
    seglevel = run["seglevel"]
    chrom = seglevel["CHR"].astype(str).str.removeprefix("chr").to_numpy()
    start, end = seglevel["START"].to_numpy(), seglevel["END"].to_numpy()
    truth = sample.copies_at(chrom, (start + end) // 2)
    snp_chrom, snp_pos, total = snp_umis(sample.path, sample.barcodes)
    detected = {c["clone"] for c in clones if c["detected"]}

    out = []
    for c, name in enumerate(sample.clones):
        if name not in detected or c not in match:
            continue
        m = match[c]
        per_snp = np.asarray(total[planted == c].sum(axis=0)).ravel()
        called = (run["a"][:, m] == 1) & (run["b"][:, m] == 1)
        neutral = np.flatnonzero((truth[:, c, 0] == 1) & (truth[:, c, 1] == 1))
        held = [
            int(per_snp[(snp_chrom == chrom[i]) & (snp_pos >= start[i])
                        & (snp_pos < end[i])].sum())
            for i in neutral
        ]  # fmt: skip
        out.append(
            {
                "clone": name,
                "snp_umis": held,
                "specific": called[neutral].astype(int).tolist(),
            }
        )
    return out


def _kept_run(sample: Any, kept: Path) -> dict[str, Any]:
    """`read_run`'s labels, seglevel, `a` and `b`, from a run's `KEPT` outputs."""
    from tests.sim_audit import _barcode

    table = pd.read_csv(kept / "clone_labels.tsv", sep="\t", comment="#")
    barcodes = table["barcode"] if "barcode" in table else table.iloc[:, 0]
    by_barcode = dict(
        zip(_barcode(barcodes), table["clone_label"].to_numpy(), strict=True)
    )
    labels = np.array([by_barcode.get(b, -1) for b in sample.barcodes])
    seglevel = pd.read_csv(kept / "cnv_seglevel.tsv", sep="\t")
    missing = np.full(len(seglevel), -1)
    n_fitted = int(labels.max()) + 1
    a, b = (
        np.stack([seglevel.get(f"clone{c} {k}", missing) for c in range(n_fitted)], 1)
        for k in ("A", "B")
    )
    return {"labels": labels, "seglevel": seglevel, "a": a, "b": b}


def rescore(out: Path) -> int:
    """Add `neutral_segments` to each record scored before it, from its outputs."""
    from port.qa.scoring import matched, overlap

    from tests.sim_fixtures import load_simulated

    done = 0
    for path in sorted((out / "records").glob("*.json")):
        record = json.loads(path.read_text())
        if "error" in record or "neutral_segments" in record:
            continue
        seed, j = int(record["seed"]), float(record["J"])
        manifest = ROOT / "sim" / "manifests" / f"{record['manifest']}.toml"
        draws = out / "draws" / f"rescore-s{seed:04d}"
        sample = load_simulated(str(draw_member(seed, draws, manifest)))
        run = _kept_run(sample, out / "outputs" / f"s{seed:04d}-J{j:g}")
        planted, fitted = np.asarray(sample.labels), run["labels"]
        scored = fitted >= 0
        counts = overlap(planted[scored], fitted[scored], len(sample.clones),
                         int(fitted.max()) + 1)  # fmt: skip
        record["neutral_segments"] = _neutral(
            sample, run, planted, matched(counts), record["clones"]
        )
        record.pop("neutral", None)
        partial = path.with_suffix(".partial")
        partial.write_text(json.dumps(record) + "\n")
        partial.replace(path)
        shutil.rmtree(draws, ignore_errors=True)
        done += 1
    return done


def _record(out: Path, seed: int, j: float) -> Path:
    return out / "records" / f"s{seed:04d}-J{j:g}.json"


def run_member(
    seed: int, js: tuple[float, ...], out: Path, manifest: Path = MANIFEST
) -> None:
    """Draw seed `seed`, run and score it at each `J`, keep only the records."""
    from tests.sim_audit import run_arm
    from tests.sim_fixtures import load_simulated

    todo = [j for j in js if not _record(out, seed, j).exists()]
    if not todo:
        return

    draws = out / "draws" / f"s{seed:04d}"
    path = draw_member(seed, draws, manifest)
    sample = load_simulated(str(path))

    for j in todo:
        runs = out / "runs" / f"s{seed:04d}-J{j:g}"
        started = time.perf_counter()
        base = {"seed": seed, "J": j, "flags": list(FLAGS), "manifest": manifest.stem}
        try:
            _, output = run_arm(
                sample, list(FLAGS), {"hmrf.spatial_weight": j}, root=runs
            )
        except Exception as error:  # noqa: BLE001 -- a failed run is a result
            import traceback

            traceback.print_exc()
            record = base | {"error": repr(error), "clones": [], "events": []}
        else:
            wall = time.perf_counter() - started
            record = base | {"wall": round(wall, 1), **score_member(sample, output)}
            kept = out / "outputs" / f"s{seed:04d}-J{j:g}"
            kept.mkdir(parents=True, exist_ok=True)
            for name in KEPT:
                shutil.copy(next(output.rglob(name)), kept / name)
        target = _record(out, seed, j)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record) + "\n")
        shutil.rmtree(runs, ignore_errors=True)

    shutil.rmtree(draws, ignore_errors=True)


def _worker(task: tuple[int, tuple[float, ...], str, str]) -> str:
    seed, js, out, manifest = task
    log = Path(out) / "logs" / f"s{seed:04d}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        sys.stdout = sys.stderr = handle
        try:
            run_member(seed, js, Path(out), Path(manifest))
        except Exception as error:  # noqa: BLE001 -- recorded, the study goes on
            import traceback

            traceback.print_exc()
            print(f"FAILED seed {seed}: {error!r}", flush=True)
            return f"s{seed}: failed ({error!r})"
    return f"s{seed}: done"


def _seeds(text: str) -> list[int]:
    start, _, stop = text.partition(":")
    return list(range(int(start), int(stop))) if stop else [int(start)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("run", "report", "rescore"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", default="0:60", help="START:STOP")
    parser.add_argument("--J", default=",".join(f"{j:g}" for j in J_VALUES))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--study2-J", type=float, default=J_VALUES[0],
                        help="the J Study 2's length curves are read at")  # fmt: skip
    arguments = parser.parse_args(argv)

    if arguments.command == "rescore":
        print(f"rescored {rescore(arguments.out)}")
        return 0

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
    tasks = [(s, js, str(arguments.out), str(arguments.manifest))
             for s in _seeds(arguments.seeds)]  # fmt: skip
    context = multiprocessing.get_context("spawn")
    with context.Pool(cores, maxtasksperchild=1) as pool:
        for line in pool.imap_unordered(_worker, tasks):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
