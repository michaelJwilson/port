"""Population study: clone detection against UMIs, CNA detection against length (#544).

One member is one seed of `sim/manifests/population.toml`: its tree, events,
layout (each clone's size drawn from `[layout.size]`) and counts all come
from that seed, so clone sizes and event lengths vary across members. Each
member is run once per analysis coupling `J` (`hmrf.spatial_weight`, read by
`cnaster.hmrf`; the generator has none) and scored against its own truth.
J starts just above the lattice's critical coupling (`J_CRITICAL`): below it
the prior orders nothing, so the study reads J where it does.

    run_study --population run --seeds 0:40 --J 0.8 --out DIR
    run_study --population stay --seeds 0:15 --out DIR   # #729's t arm, at the planted clones
    run_study --population report --out DIR --study2-J 0.8

`run` is resumable: a `(seed, J)` with a record in `DIR/records/` is skipped.
Each worker runs one thread, one worker per core, and a member's draw and
run directories are deleted once its records are written, so the disk a
study holds is its records and each run's `KEPT` outputs in `DIR/outputs/`. `report` is `port.studies.population_report`.

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

**Scored against the credible sets** (#705), under an arm that runs
`--copy-errors`: for each event, the share of its visible bins whose
continuous state's set in its clone (`cnv_copy_sets.tsv`, folded `A >= B`,
taken at the point decode's shift and tumour fraction) holds the planted pair
(`covered`), the share where it holds `(1, 1)` too (`ambiguous`), the share
whose set is empty (`empty`), the mean set size, and each missed bin by the
#705 explanation it falls under; once per level of `LEVELS` (`sets_2sigma`,
`sets_3sigma`). The arms (`ARMS`) differ only in the run's flags; each writes
its own `--out`, and its records name it.

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
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
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

KEPT_IF_WRITTEN = ("cnv_copy_sets.tsv", "cnv_segment_sets.tsv", "cnv_bin_loglik.npz")
"""Kept as `KEPT` is, where the arm's flags write them."""

FLAGS = ("--sal", "--no-plots")

LEVELS = {"2sigma": 0.9545, "3sigma": 0.9973}
"""The credible levels each event is scored at (#705); the arms write the
widest, with each pair's distance, so the narrower is read from the same run."""

SETS = ("--copy-errors",)
"""The per-state credible sets (#353), at the entry point's 95 per cent; the
segment sets come from `port.sandbox.extensions.segment_sets`, written at
the widest of `LEVELS` so each level is read from one run."""

ARMS: dict[str, tuple[str, ...]] = {
    "sal": FLAGS,
    "errors": (*FLAGS, *SETS),
    "flat": (*FLAGS, *SETS, "--no-parsimony-decode"),
    "shared": (*FLAGS, *SETS, "--copy-decode", "shared"),
}
"""The decode arms of #705: `sal` is the study as #544 ran it; the others add
the credible sets and change only the point decode."""

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

    from port.sim.draw import draw, merged_tables, read_manifest

    manifest = read_manifest(manifest_path)
    manifest = replace(
        manifest, tables=merged_tables(manifest.tables, {"sample": {"seed": seed}})
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


Sets = dict[tuple[int, int], set[tuple[int, int]]]
"""Credible pairs by `(clone, state)`; clone `-1` where the sets are per state."""


def credible_sets(table: pd.DataFrame, level: float | None = None) -> Sets:
    """Each `(clone, state)`'s credible pairs from `cnv_copy_sets.tsv`.

    With `level`, a pair is kept where its `distance` is within that level's
    threshold (`chi2`, 1 degree of freedom for the neutral state, else 2),
    so it must be no wider than the table's own; without, or for a table
    with no `distance`, every written pair. An empty set has none.
    """
    from scipy.stats import chi2

    table = table.copy()
    if "clone" not in table:
        table["clone"] = -1
    pairs = table.dropna(subset=["A", "B"])
    if level is not None and "distance" in pairs:
        if level > float(table["level"].max()) + 1e-9:
            msg = f"level {level} is wider than the table's {table['level'].max()}"
            raise ValueError(msg)
        dof = np.where(pairs["neutral"].astype(bool), 1, 2)
        pairs = pairs[pairs["distance"].to_numpy(float) <= chi2.ppf(level, dof)]
    sets: Sets = {
        (int(c), int(k)): set()
        for c, k in zip(table["clone"], table["state"], strict=True)
    }
    for c, k, a, b in zip(
        pairs["clone"], pairs["state"], pairs["A"], pairs["B"], strict=True
    ):
        sets[int(c), int(k)].add((int(a), int(b)))
    return sets


def clone_sets(sets: Sets, clone: int) -> dict[int, set[tuple[int, int]]]:
    """`clone`'s sets by state, or the per-state sets where the table has no clones."""
    own = {k: pairs for (c, k), pairs in sets.items() if c == clone}
    return own or {k: pairs for (c, k), pairs in sets.items() if c == -1}


def set_scores(
    held: list[set[tuple[int, int]]],
    right: np.ndarray,
    planted: tuple[int, int],
) -> dict[str, float]:
    """Each bin's set (folded, `A >= B`) against the folded `planted`, and why a missed bin was.

    Shares over the bins: `covered`, the set holds the planted pair;
    `ambiguous`, it holds `(1, 1)` too, so the counts cannot tell the event
    from none; `empty`, it holds no pair; and `set_size`, its mean size.

    Counts over the bins the point decode missed (`right` false), one per
    #705 explanation: `miss_ambiguous` (1, the truth and `(1, 1)` both in
    the set), `miss_decoder` (2, the truth in and `(1, 1)` out),
    `miss_empty` (3a, no pair in the set) and `miss_excluded` (3b, a set
    without the truth).
    """
    folded = (max(planted), min(planted))
    has = np.array([folded in h for h in held], dtype=bool)
    neutral = np.array([(1, 1) in h for h in held], dtype=bool)
    empty = np.array([not h for h in held], dtype=bool)
    size = np.array([len(h) for h in held], dtype=np.float64)
    missed = ~np.asarray(right, dtype=bool)

    def share(mask: np.ndarray) -> float:
        return round(float(mask.mean()), 4) if mask.size else 0.0

    return {
        "covered": share(has),
        "ambiguous": share(has & neutral),
        "empty": share(empty),
        "set_size": round(float(size.mean()), 3) if size.size else 0.0,
        "miss_ambiguous": int((missed & has & neutral).sum()),
        "miss_decoder": int((missed & has & ~neutral).sum()),
        "miss_empty": int((missed & empty).sum()),
        "miss_excluded": int((missed & ~empty & ~has).sum()),
    }


SegmentSets = dict[int, list[set[tuple[int, int]]]]
"""Per clone of the decode, each bin's segment's folded pairs."""


def segment_bins(table: pd.DataFrame, n_obs: int, level: float) -> SegmentSets:
    """Each clone's per-bin sets from `cnv_segment_sets.tsv`, at `level`.

    A pair is kept where its `deviance` is within `chi2(level, 2)`; `level`
    must be no wider than the table's own. Pairs are folded, `A >= B`.
    """
    from scipy.stats import chi2

    if level > float(table["level"].max()) + 1e-9:
        msg = f"level {level} is wider than the table's {table['level'].max()}"
        raise ValueError(msg)
    kept = table[table["deviance"].to_numpy(float) <= chi2.ppf(level, 2)]
    out: SegmentSets = {}
    for (clone, start, end), rows in kept.groupby(["clone", "start_bin", "end_bin"]):
        pairs = {
            (max(int(a), int(b)), min(int(a), int(b)))
            for a, b in zip(rows["A"], rows["B"], strict=True)
        }
        bins = out.setdefault(int(clone), [set() for _ in range(n_obs)])
        for i in range(int(start), int(end)):
            bins[i] = pairs
    return out


def _decoded_agree(table: pd.DataFrame, run: dict[str, Any]) -> float:
    """The share of clone-bins whose segment's decoded pair is `cnv_seglevel`'s, folded."""
    agree, total = 0, 0
    segments = table.drop_duplicates(["clone", "start_bin", "end_bin"])
    for clone, start, end, a, b in zip(
        segments["clone"], segments["start_bin"], segments["end_bin"],
        segments["decoded_A"], segments["decoded_B"], strict=True,
    ):  # fmt: skip
        if int(clone) >= run["a"].shape[1]:
            continue
        span = slice(int(start), int(end))
        major = np.maximum(run["a"][span, clone], run["b"][span, clone])
        minor = np.minimum(run["a"][span, clone], run["b"][span, clone])
        agree += int(((major == max(a, b)) & (minor == min(a, b))).sum())
        total += int(end) - int(start)
    return round(agree / total, 4) if total else 0.0


def known_set(
    pairs: np.ndarray,
    loglik: np.ndarray,
    visible: np.ndarray,
    planted: tuple[int, int],
    level: float,
) -> dict[str, Any]:
    """The set over a planted event's own bins, from `cnv_bin_loglik.npz`.

    `loglik` is one clone's `(n_obs, n_pairs)`. Each pair is folded per bin,
    the better of its two phases, since phasing switches within an event;
    the event's bins are held at one folded pair and the rest as decoded,
    so `l(A, B)` differs from pair to pair by the event's sum alone. The set
    is every folded pair within `chi2(level, 2)` of the best. Shift and
    fraction stay at the decode's.
    """
    from scipy.stats import chi2

    folded = sorted({(max(int(a), int(b)), min(int(a), int(b))) for a, b in pairs})
    position = {(int(a), int(b)): k for k, (a, b) in enumerate(pairs)}
    rows = loglik[np.asarray(visible, dtype=bool)].astype(np.float64)
    totals = np.array(
        [
            np.maximum(
                rows[:, position[a, b]], rows[:, position.get((b, a), position[a, b])]
            ).sum()
            for a, b in folded
        ]
    )
    deviance = 2.0 * (totals.max() - totals)
    threshold = float(chi2.ppf(level, 2))
    found = {pair for pair, d in zip(folded, deviance, strict=True) if d <= threshold}
    index = {pair: k for k, pair in enumerate(folded)}
    truth = (max(planted), min(planted))
    return {
        "covered": truth in found,
        "ambiguous": truth in found and (1, 1) in found,
        "set_size": len(found),
        "best": list(folded[int(np.argmax(totals))]),
        "deviance_truth": (
            round(float(deviance[index[truth]]), 3) if truth in index else None
        ),
        "deviance_neutral": round(float(deviance[index[1, 1]]), 3),
    }


def _read_sets(directory: Path) -> dict[str, Any]:
    """What `--copy-errors` wrote under `directory`, by kind."""
    found: dict[str, Any] = {}
    for kind, name in (
        ("state", "cnv_copy_sets.tsv"),
        ("segment", "cnv_segment_sets.tsv"),
    ):
        written = next(directory.rglob(name), None)
        if written is not None:
            found[kind] = pd.read_csv(written, sep="\t", comment="#")
    written = next(directory.rglob("cnv_bin_loglik.npz"), None)
    if written is not None:
        with np.load(written) as table:
            found["bins"] = {key: table[key] for key in table.files}
    return found


def _bin_sets(
    tables: dict[str, Any], run: dict[str, Any]
) -> dict[str, Callable[[int, np.ndarray], list[set[tuple[int, int]]]]]:
    """Per `<kind>_<level>`, each fitted clone's visible bins' sets.

    `state`: the bin's continuous state's set (`cnv_copy_sets.tsv`, read at
    the levels its distances allow, as written otherwise); `segment`: its
    decoded segment's (`cnv_segment_sets.tsv`).
    """
    seglevel = run["seglevel"]
    lookups: dict[str, Callable[[int, np.ndarray], list[set[tuple[int, int]]]]] = {}

    if "state" in tables:
        table = tables["state"]
        levels = (
            {"written": None}
            if "distance" not in table
            else {k: v for k, v in LEVELS.items() if v <= table["level"].max() + 1e-9}
        )
        for name, level in levels.items():
            found = credible_sets(table, level)

            def by_state(
                m: int, visible: np.ndarray, found: Sets = found
            ) -> list[set[tuple[int, int]]]:
                states = seglevel[f"clone{m} Z"].to_numpy()[visible].astype(np.int64)
                own = clone_sets(found, m)
                return [own.get(int(z), set()) for z in states]

            lookups[f"state_{name}"] = by_state

    if "segment" in tables:
        table = tables["segment"]
        for name, level in LEVELS.items():
            if level > table["level"].max() + 1e-9:
                continue
            bins = segment_bins(table, len(seglevel), level)

            def by_segment(
                m: int, visible: np.ndarray, bins: SegmentSets = bins
            ) -> list[set[tuple[int, int]]]:
                own = bins.get(m)
                return [own[i] if own else set() for i in np.flatnonzero(visible)]

            lookups[f"segment_{name}"] = by_segment

    return lookups


def score_member(sample: Any, output: Path) -> dict[str, Any]:
    """Per tumour clone its size, UMIs and completeness; per event its recovery."""
    from port.qa.audit import read_run

    return score_run(sample, read_run(sample, output), _read_sets(output) or None)


def score_run(
    sample: Any,
    run: dict[str, Any],
    sets: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """:func:`score_member` on `read_run`'s fields, scored on each of `sets` where given."""
    from port.qa.scoring import matched, overlap

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
    lookups = _bin_sets(sets or {}, run)

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
            credible: dict[str, Any] = {
                f"sets_{name}": set_scores(lookup(m, visible), right[visible], (pa, pb))
                for name, lookup in lookups.items()
            }
            bins = (sets or {}).get("bins")
            if bins is not None and m < bins["loglik"].shape[0] and visible.any():
                for level_name, level in LEVELS.items():
                    credible[f"known_{level_name}"] = known_set(
                        bins["pairs"], bins["loglik"][m], visible, (pa, pb), level
                    )
            scored_events.append(
                credible
                | {
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

    agree = (
        {}
        if "segment" not in (sets or {})
        else {"segment_agree": _decoded_agree((sets or {})["segment"], run)}
    )
    return agree | {"n_fitted": int(np.unique(fitted[scored]).size), "clones": clones,
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
    from port.qa.audit import barcode_strings

    table = pd.read_csv(kept / "clone_labels.tsv", sep="\t", comment="#")
    barcodes = table["barcode"] if "barcode" in table else table.iloc[:, 0]
    by_barcode = dict(
        zip(barcode_strings(barcodes), table["clone_label"].to_numpy(), strict=True)
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
    from port.sim.fixtures import load_simulated

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


def rescore_sets(out: Path) -> int:
    """Rescore each record whose run kept `cnv_copy_sets.tsv`, from its outputs.

    The whole record's clones and events are rebuilt by :func:`score_run`, so
    records scored by an earlier :func:`set_scores` carry the current fields.
    """
    from port.sim.fixtures import load_simulated

    done = 0
    for path in sorted((out / "records").glob("*.json")):
        record = json.loads(path.read_text())
        seed, j = int(record["seed"]), float(record["J"])
        kept = out / "outputs" / f"s{seed:04d}-J{j:g}"
        sets = {} if "error" in record else _read_sets(kept)
        if not sets:
            continue
        manifest = ROOT / "sim" / "manifests" / f"{record['manifest']}.toml"
        draws = out / "draws" / f"rescore-s{seed:04d}"
        sample = load_simulated(str(draw_member(seed, draws, manifest)))
        record |= score_run(sample, _kept_run(sample, kept), sets)
        partial = path.with_suffix(".partial")
        partial.write_text(json.dumps(record) + "\n")
        partial.replace(path)
        shutil.rmtree(draws, ignore_errors=True)
        done += 1
    return done


def _record(out: Path, seed: int, j: float) -> Path:
    return out / "records" / f"s{seed:04d}-J{j:g}.json"


def _segment_sets(arm: str, output: Path) -> Any:
    """`port.sandbox.extensions.segment_sets`' writer around a run, for an arm
    with the credible sets; nothing for `sal`. Set aside for version 2 (#705)."""
    import contextlib

    if "--copy-errors" not in ARMS[arm]:
        return contextlib.nullcontext()

    from port.sandbox.extensions.segment_sets import writing_segment_sets

    return writing_segment_sets(output, level=max(LEVELS.values()))


def run_member(
    seed: int,
    js: tuple[float, ...],
    out: Path,
    manifest: Path = MANIFEST,
    arm: str = "sal",
) -> None:
    """Draw seed `seed`, run and score it at each `J`, keep only the records."""
    from port.qa.audit import audit_sample
    from port.sim.fixtures import load_simulated

    todo = [j for j in js if not _record(out, seed, j).exists()]
    if not todo:
        return

    draws = out / "draws" / f"s{seed:04d}"
    path = draw_member(seed, draws, manifest)
    sample = load_simulated(str(path))

    for j in todo:
        runs = out / "runs" / f"s{seed:04d}-J{j:g}"
        started = time.perf_counter()
        flags = list(ARMS[arm])
        base = {"seed": seed, "J": j, "flags": flags, "arm": arm,
                "manifest": manifest.stem}  # fmt: skip
        try:
            with _segment_sets(arm, runs / "output"):
                _, output = audit_sample(
                    sample, flags, {"hmrf.spatial_weight": j}, root=runs
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
            for name in KEPT_IF_WRITTEN:
                found = next(output.rglob(name), None)
                if found is not None:
                    shutil.copy(found, kept / name)
        target = _record(out, seed, j)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record) + "\n")
        shutil.rmtree(runs, ignore_errors=True)

    shutil.rmtree(draws, ignore_errors=True)


STAY = tuple(float(v) for v in np.logspace(-9.0, -2.0, 8))
"""`1 - t`, the HMM's switch probability, for the stay arm (#729): 1e-9 to 1e-2,
the configuration's 1e-7 among them."""


def _planted_events(found: Any) -> list[tuple[np.ndarray, tuple[int, int], int]]:
    """Each clone's runs of one planted non-`(1, 1)` pair along a contig, as
    `(rows, pair, length)`: the events the stage's rows hold."""
    n_bins = len(found.contig)
    events = []
    for clone in range(found.n_clones):
        rows = np.arange(clone * n_bins, (clone + 1) * n_bins)
        pairs = [(int(found.planted[r][0]), int(found.planted[r][1])) for r in rows]
        start = 0
        for k in range(1, n_bins + 1):
            ends = (
                k == n_bins
                or pairs[k] != pairs[start]
                or found.contig[k] != found.contig[start]
            )
            if not ends:
                continue
            pair = pairs[start]
            if pair != (1, 1):
                span = np.arange(start, k)
                events.append((rows[span], pair, int(np.sum(found.length[span]))))
            start = k
    return events


def _phase_free(pair: tuple[int, int]) -> tuple[int, int]:
    return (max(pair), min(pair))


def stay_scores(found: Any) -> list[dict[str, Any]]:
    """Each planted event, recovered or not, by the run's Baum-Welch at each `1 - t` in `STAY`.

    The call is the run's own at its RDR + BAF stage at the planted clones
    (`port.studies.stage.at_oracle_clones`), `t` alone replaced. A fitted
    state is read as the planted pair most of its rows hold, up to phase, and
    an event is recovered where `RECOVERED` of its rows' states read as its
    pair. A stated difference from the pipeline arms: those score the run's
    integer decode; this scores the HMM's states, before integer copy.
    """
    events = _planted_events(found)
    planted = [_phase_free((int(a), int(b))) for a, b in found.planted]
    out = []
    for omt in STAY:
        result = found.run(t=1.0 - omt)
        states = np.argmax(np.asarray(result["log_gamma"]), axis=0)
        reads: dict[int, tuple[int, int]] = {}
        for state in np.unique(states):
            held = pd.Series([planted[r] for r in np.flatnonzero(states == state)])
            reads[int(state)] = held.value_counts().index[0]
        for rows, pair, length in events:
            share = float(
                np.mean([reads[int(states[r])] == _phase_free(pair) for r in rows])
            )
            out.append({"omt": omt, "class": copy_class(*pair), "length": length,
                        "recovered": share >= RECOVERED})  # fmt: skip
    return out


def stay_member(seed: int, out: Path, manifest: Path = MANIFEST) -> None:
    """Seed `seed`'s stay-arm record (#729), from the run's own stage at its planted clones."""
    from port.sim.fixtures import load_simulated
    from port.studies.stage import at_oracle_clones

    target = out / "stay" / f"s{seed:04d}.json"
    if target.exists():
        return
    draws = out / "draws" / f"stay{seed:04d}"
    path = draw_member(seed, draws, manifest)
    sample = load_simulated(str(path))
    started = time.perf_counter()
    try:
        rows = at_oracle_clones(
            sample, stay_scores, root=out / "runs" / f"stay{seed:04d}"
        )
        record: dict[str, Any] = {"seed": seed, "manifest": manifest.stem, "rows": rows,
                                  "wall": round(time.perf_counter() - started, 1)}  # fmt: skip
    except Exception as error:  # noqa: BLE001 -- a failed member is a result
        record = {
            "seed": seed,
            "manifest": manifest.stem,
            "rows": [],
            "error": repr(error),
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record) + "\n")
    shutil.rmtree(out / "runs" / f"stay{seed:04d}", ignore_errors=True)
    shutil.rmtree(draws, ignore_errors=True)


def _stay_worker(task: tuple[int, str, str]) -> str:
    seed, out, manifest = task
    log = Path(out) / "logs" / f"stay{seed:04d}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        sys.stdout = sys.stderr = handle
        stay_member(seed, Path(out), Path(manifest))
    return f"stay s{seed}: done"


def _worker(task: tuple[int, tuple[float, ...], str, str, str]) -> str:
    seed, js, out, manifest, arm = task
    log = Path(out) / "logs" / f"s{seed:04d}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        sys.stdout = sys.stderr = handle
        try:
            run_member(seed, js, Path(out), Path(manifest), arm)
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
    parser.add_argument(
        "command", choices=("run", "stay", "report", "rescore", "rescore-sets")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", default="0:60", help="START:STOP")
    parser.add_argument("--J", default=",".join(f"{j:g}" for j in J_VALUES))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--arm", choices=list(ARMS), default="sal",
                        help="the decode arm (#705); one --out per arm")  # fmt: skip
    parser.add_argument("--study2-J", type=float, default=J_VALUES[0],
                        help="the J Study 2's length curves are read at")  # fmt: skip
    arguments = parser.parse_args(argv)

    if arguments.command == "rescore-sets":
        print(f"rescored {rescore_sets(arguments.out)}")
        return 0
    if arguments.command == "rescore":
        print(f"rescored {rescore(arguments.out)}")
        return 0

    if arguments.command == "report":
        from port.studies.population_report import report

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
    tasks = [(s, js, str(arguments.out), str(arguments.manifest), arguments.arm)
             for s in _seeds(arguments.seeds)]  # fmt: skip
    context = multiprocessing.get_context("spawn")
    with context.Pool(cores, maxtasksperchild=1) as pool:
        if arguments.command == "stay":
            stays = [(s, str(arguments.out), str(arguments.manifest))
                     for s in _seeds(arguments.seeds)]  # fmt: skip
            for line in pool.imap_unordered(_stay_worker, stays):
                print(line, flush=True)
            return 0
        for line in pool.imap_unordered(_worker, tasks):
            print(line, flush=True)
    return 0
