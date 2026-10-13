"""Set aside (T- #831): the population study scored against the credible sets (#705).

Ticket: #705 -- the decode arms `errors`, `flat` and `shared`, each adding
  `--copy-errors`' credible sets to the `sal` arm; T- #831 set them aside with
  `--copy-errors`, which no `--sal` run takes.
Measurement: #705's population records under each arm. The arms ran
  `run_cnaster_port --copy-errors` (with `--no-parsimony-decode` or
  `--copy-decode shared`), flags T- #831 removed from the run.
Exit: graduate with `port.sandbox.extensions.copy_errors`; else retire with it.

For each event, the share of its visible bins whose continuous state's set in
its clone (`cnv_copy_sets.tsv`, folded `A >= B`, taken at the point decode's
shift and tumour fraction) holds the planted pair (`covered`), the share where
it holds `(1, 1)` too (`ambiguous`), the share whose set is empty (`empty`),
the mean set size, and each missed bin by the #705 explanation it falls under;
once per level of `LEVELS` (`sets_2sigma`, `sets_3sigma`).
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from port.extensions.repository import ROOT
from port.studies.population import clone_events, draw_member, score_run

KEPT_IF_WRITTEN = ("cnv_copy_sets.tsv", "cnv_segment_sets.tsv", "cnv_bin_loglik.npz")
"""Kept as `KEPT` is, where the arm's flags write them."""


LEVELS = {"2sigma": 0.9545, "3sigma": 0.9973}
"""The credible levels each event is scored at (#705); the arms write the
widest, with each pair's distance, so the narrower is read from the same run."""

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


def rescore_sets(out: Path) -> int:
    """Rescore each record whose run kept `cnv_copy_sets.tsv`, from its outputs.

    The whole record's clones and events are rebuilt by :func:`score_sets`, so
    records scored by an earlier :func:`set_scores` carry the current fields.
    """
    from port.qa.audit import read_tables
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
        record |= score_sets(sample, read_tables(sample, kept), sets)
        partial = path.with_suffix(".partial")
        partial.write_text(json.dumps(record) + "\n")
        partial.replace(path)
        shutil.rmtree(draws, ignore_errors=True)
        done += 1
    return done


def segment_sets(output: Path) -> Any:
    """`port.sandbox.extensions.segment_sets`' writer around a run, at the widest of `LEVELS`."""
    from port.sandbox.extensions.segment_sets import writing_segment_sets

    return writing_segment_sets(output, level=max(LEVELS.values()))


def score_sets(
    sample: Any, run: dict[str, Any], sets: dict[str, Any]
) -> dict[str, Any]:
    """`score_run`, each event with its credible columns from `sets` (`_read_sets`)."""
    from port.qa.scoring import matched, overlap

    scored = score_run(sample, run)
    fitted, planted = np.asarray(run["labels"]), np.asarray(sample.labels)
    kept = fitted >= 0
    match = matched(
        overlap(planted[kept], fitted[kept], len(sample.clones), int(fitted.max()) + 1)
    )
    seglevel = run["seglevel"]
    chrom = seglevel["CHR"].astype(str).str.removeprefix("chr").to_numpy()
    middle = ((seglevel["START"] + seglevel["END"]) // 2).to_numpy()
    truth = sample.copies_at(chrom, middle)
    lookups, bins = _bin_sets(sets, run), sets.get("bins")
    by_event = {
        (e["clone"], e["chr"], e["a"], e["b"], e["length"]): e for e in scored["events"]
    }
    for c, name in enumerate(sample.clones):
        m = match.get(c)
        if m is None:
            continue
        a, b = run["a"][:, m], run["b"][:, m]
        for chromosome, start, end, pa, pb in clone_events(sample.path).get(name, []):
            event = by_event.get((name, chromosome, pa, pb, int(end - start)))
            if event is None:
                continue
            inside = (chrom == chromosome) & (middle >= start) & (middle < end)
            visible = inside & (truth[:, c, 0] == pa) & (truth[:, c, 1] == pb)
            right = (np.minimum(a, b) == min(pa, pb)) & (
                np.maximum(a, b) == max(pa, pb)
            )
            for level_name, lookup in lookups.items():
                event[f"sets_{level_name}"] = set_scores(
                    lookup(m, visible), right[visible], (pa, pb)
                )
            if bins is not None and m < bins["loglik"].shape[0] and visible.any():
                for level_name, level in LEVELS.items():
                    event[f"known_{level_name}"] = known_set(
                        bins["pairs"], bins["loglik"][m], visible, (pa, pb), level
                    )
    agree = (
        {}
        if "segment" not in sets
        else {"segment_agree": _decoded_agree(sets["segment"], run)}
    )
    return agree | scored
