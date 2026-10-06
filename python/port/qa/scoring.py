"""One scorer for every audit and study: a fit against its planted truth (T- #673 G2).

Labels are compared after relabelling: each planted label is paired with the
fitted label it shares the most spots (or bins) with, one to one, by
`linear_sum_assignment` on the overlap counts (#517 step 7). Copy states are
compared as pairs coded `A * 1_000 + B`, by planted class (#511).

`tests.scoring`, `tests.sim_audit`, `tests.recovery_audit.integer_clones`
and `tests.studies.paper_figures.exact_by_class` each held a part of this.
`port.extensions.outputs.integer_clones` merges a run's written table under
an agreement threshold; `integer_clones` here is its exact rule (1.0) on
decoded arrays, which every scorer reads.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = [
    "CLASSES",
    "NEUTRAL",
    "OTHER",
    "class_ari",
    "confusion_table",
    "copy_confusion",
    "copy_states",
    "exact_by_class",
    "integer_clones",
    "matched",
    "overlap",
    "phase_free",
    "planted_classes",
    "swapped",
]

NEUTRAL = 1_001
"""The planted pair `(1, 1)`, as `A * 1_000 + B`."""

CLASSES = ("loh", "balanced_gain", "unbalanced_gain", "neutral")
"""`planted_classes`' keys, in the order the audits and figure 17 report them."""

OTHER = "other"
"""The decoded column for a pair outside `copy_states`, or a bin left unfit."""


def overlap(
    planted: np.ndarray, fitted: np.ndarray, n_planted: int, n_fitted: int
) -> np.ndarray:
    """`(n_planted, n_fitted)` counts of each planted label under each fitted one."""
    counts = np.zeros((n_planted, n_fitted), dtype=np.int64)
    np.add.at(counts, (planted, fitted), 1)
    return counts


def matched(counts: np.ndarray) -> dict[int, int]:
    """Each planted label's fitted label, one to one, maximizing total overlap."""
    rows, columns = linear_sum_assignment(-counts)
    return dict(zip(rows.tolist(), columns.tolist(), strict=True))


def integer_clones(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Each fitted clone's label after merging clones of one `(A, B)` profile.

    Index `c` holds the smallest clone whose decoded `(A, B)` equals clone
    `c`'s at every bin (#344), so the normal clone keeps `0`.
    """
    merged = np.arange(a.shape[1])
    seen: dict[bytes, int] = {}

    for clone in range(a.shape[1]):
        profile = np.stack([a[:, clone], b[:, clone]]).astype(np.int64)
        merged[clone] = seen.setdefault(profile.tobytes(), clone)

    return merged


def planted_classes(t: np.ndarray) -> dict[str, np.ndarray]:
    """Clone-bins by planted class, from pairs coded `A * 1_000 + B`.

    `loh` one haplotype at 0; `balanced_gain` `A = B > 1`; `unbalanced_gain`
    both present, `A + B > 2`, `A != B`; `neutral` `(1, 1)`. One partition
    for every per-class metric.
    """
    major, minor = t // 1_000, t % 1_000
    loh = np.minimum(major, minor) == 0
    gain = (major + minor > 2) & ~loh
    return {
        "loh": loh,
        "balanced_gain": gain & (major == minor),
        "unbalanced_gain": gain & (major != minor),
        "neutral": t == NEUTRAL,
    }


def phase_free(codes: np.ndarray) -> np.ndarray:
    """`A * 1000 + B` codes as `(minor, major)`: `(A, B)` and `(B, A)` coded alike."""
    major, minor = codes // 1_000, codes % 1_000
    return np.asarray(np.minimum(major, minor) * 1_000 + np.maximum(major, minor))


def swapped(codes: np.ndarray) -> np.ndarray:
    """`A * 1000 + B` codes with the haplotypes exchanged: `(B, A)`."""
    return np.asarray((codes % 1_000) * 1_000 + codes // 1_000)


def class_ari(t: np.ndarray, ab: np.ndarray, where: np.ndarray) -> float:
    """Copy-state ARI over the bins in `where`; NaN where the planted pairs
    there take fewer than two values, ARI being undefined.
    """
    from sklearn.metrics import adjusted_rand_score

    if np.unique(t[where]).size < 2:
        return float("nan")
    return round(float(adjusted_rand_score(t[where], ab[where])), 4)


def exact_by_class(
    t: np.ndarray, ab: np.ndarray
) -> dict[str, tuple[float, float, int]]:
    """Per `CLASSES`: the exact share, the share up to phase, and the clone-bins.

    Shares are unrounded, and NaN where the class is not planted.
    """
    exact, either = t == ab, (t == ab) | (t == swapped(ab))
    found = {}
    for name, where in planted_classes(t).items():
        if not where.any():
            found[name] = (float("nan"), float("nan"), 0)
            continue
        found[name] = (
            float(np.mean(exact[where])),
            float(np.mean(either[where])),
            int(where.sum()),
        )
    return found


def copy_states(max_total: int) -> list[tuple[int, int]]:
    """Every `(A, B)` with `A + B <= max_total`, by total, then by `A`."""
    return [(a, n - a) for n in range(max_total + 1) for a in range(n + 1)]


def _pair(code: int) -> str:
    return f"{code // 1_000},{code % 1_000}"


def _order(code: int) -> tuple[int, int]:
    return (code // 1_000 + code % 1_000, code // 1_000)


def copy_confusion(
    t: np.ndarray, ab: np.ndarray, max_total: int
) -> dict[str, dict[str, float]]:
    """Planted against decoded `(A, B)`, as fractions of each planted row.

    Rows are the planted pairs within `max_total`, columns every pair within
    it and `OTHER`; zero entries are left out, so a row's values sum to 1.
    """
    states = {a * 1_000 + b for a, b in copy_states(max_total)}
    confusion: dict[str, dict[str, float]] = {}

    for planted in sorted(states & set(np.unique(t).tolist()), key=_order):
        decoded = ab[t == planted]
        named = [_pair(int(d)) if int(d) in states else OTHER for d in decoded]
        values, counts = np.unique(named, return_counts=True)
        confusion[_pair(planted)] = {
            str(v): round(float(c) / decoded.size, 4)
            for v, c in zip(values, counts, strict=True)
        }
    return confusion


def confusion_table(
    confusion: dict[str, dict[str, float]], max_total: int, *, sampled: bool = False
) -> str:
    """`confusion` as a markdown table, rows planted and columns decoded, in
    `copy_states` order with `OTHER` last.

    Every pair within `max_total` is a row and a column; `sampled` drops the
    rows of pairs never planted and the columns of pairs never decoded.
    """
    decoded = {d for row in confusion.values() for d in row}
    pairs = [f"{a},{b}" for a, b in copy_states(max_total)]
    rows = [p for p in pairs if p in confusion] if sampled else pairs
    columns = [p for p in pairs if p in decoded or not sampled]
    columns += [OTHER] if OTHER in decoded or not sampled else []
    lines = [
        "| planted \\ decoded | " + " | ".join(columns) + " |",
        "| --- |" + " --- |" * len(columns),
    ]
    for planted in rows:
        row = confusion.get(planted, {})
        cells = [f"{row[c]:.4f}" if c in row else "" for c in columns]
        lines.append(f"| {planted} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
