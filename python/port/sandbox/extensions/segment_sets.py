"""Set aside for version 2 (#705): each decoded segment's credible pairs, by the point decode's own likelihood.

Ticket: #705 -- why planted (1,2) and (2,2) gains decode (1,1); the
  population study that would measure it is blocked by the owner, and the
  sets wait for version 2.
Measurement: one long-arm member (s1074, `population_long.toml`, J = 1):
  96 of 97 segment sets hold one pair; with an event's boundaries given,
  (1,1) sits at deviance 35-141 on every missed (2,2) event and 389 on a
  (1,2) event decoded right on 21 per cent of its bins (3 sigma: 11.8).
Exit: graduate to `extensions/` and `run_cnaster_port --copy-errors` when
  #705's paired study (`errors`, `flat`, `shared`) reports them across the
  population; else it stays the study's tool.

`copy_errors` takes a set per continuous state from the curvature of the
continuous fit: on the #544 population its errors are `sigma_p` of about
0.002, below the fit's distance from any integer pair, and 69-74 per cent of
event bins fell in an empty set. Here the set is taken on the counts, in the
terms `copy_likelihood.lattice_decode` chose by.

A **segment** is a run of bins in one clone that the decode gave one pair,
within one contig. For each segment and each candidate `(A, B)`, the segment
is held at `(A, B)`, every other segment of the clone at its decoded pair,
and the clone's shift and tumour fraction refitted:

    l(A, B) = max over (shift, rho) of the clone's NB + BB log-likelihood.

The set is every pair with `2 (l_best - l(A, B)) <= chi2(level, 2)`. It is
never empty: `l_best` is a member.

What is held, and why:

- **the prior.** The likelihood is flat: the decode's
  `-parsimony |A + B - 2|` decides the point call, and a set holding the
  planted pair and `(1, 1)` both shows where it does;
- **the transitions.** The segment's boundaries are the decode's, and a
  candidate pays no switching cost;
- **the dispersions** (`alpha`, `tau`), at the decode's;
- **the normal clone** at shift 0 and fraction 1, as the decode holds it.

Beside the sets, :func:`bin_loglik` keeps each bin's log-likelihood under
every pair at the decode's shift and fraction, so a set over any other run
of bins -- a planted event's, where the decode drew no boundary -- is a sum
away, with no rerun.

The shift and fraction are the clone's, shared by all its segments, so the
others anchor them; refitted per segment alone, two parameters would absorb
a segment's two observables and admit nearly every pair. The refit is on a
grid about the decode's optimum (:data:`SHIFT_STEPS`, :data:`FRACTION_STEPS`),
which every candidate shares, so the differences are on equal terms.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

__all__ = [
    "SegmentSet",
    "bin_loglik",
    "segment_sets",
    "segments",
    "write_segment_sets",
    "writing_segment_sets",
]

SHIFT_STEPS = np.round(np.arange(-0.1, 0.1001, 0.01), 3)
"""Offsets from the decode's shift the refit searches."""

FRACTION_STEPS = np.round(np.arange(-0.1, 0.1001, 0.02), 3)
"""Offsets from the decode's tumour fraction the refit searches, within (0, 1]."""


class SegmentSet(NamedTuple):
    """One segment's decoded pair, its likeliest, and every pair within `level`."""

    clone: int
    start: int
    end: int
    decoded: tuple[int, int]
    best: tuple[int, int]
    threshold: float
    level: float
    consistent: tuple[tuple[int, int, float], ...]
    """`(A, B, 2 (l_best - l(A, B)))`, nearest first."""


def segments(path: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """`(n_segments, 2)` half-open bin ranges: runs of one state, split at contigs."""
    path = np.asarray(path)
    stops = np.cumsum(np.asarray(lengths, dtype=np.int64))
    if stops[-1] != path.size:
        msg = f"lengths sum to {stops[-1]}, not the path's {path.size} bins"
        raise ValueError(msg)
    breaks = set(np.flatnonzero(path[1:] != path[:-1]) + 1) | set(stops[:-1].tolist())
    edges = np.array(sorted({0, path.size} | breaks), dtype=np.int64)
    return np.column_stack([edges[:-1], edges[1:]])


def _grid(shift: float, fraction: float, held: bool) -> list[tuple[float, float]]:
    if held:
        return [(shift, fraction)]
    fractions = np.unique(np.clip(fraction + FRACTION_STEPS, 0.05, 1.0))
    return [(shift + d, float(f)) for d in SHIFT_STEPS for f in fractions]


def segment_sets(
    decode: Any,
    clones: list[tuple[np.ndarray, Any, float]],
    normal_clone: int,
    lengths: np.ndarray,
    *,
    level: float = 0.95,
) -> list[SegmentSet]:
    """Every segment's set, for each clone of `decode` (module docstring).

    `decode` is `copy_likelihood.lattice_decode`'s `CopyFit`; `clones` the
    `(path, pseudobulk, shift)` it was given (`copy_likelihood.clones_of`),
    in the same order.
    """
    from scipy.stats import chi2

    from port.extensions.copy_likelihood import (
        pair_rate_and_share,
        pseudobulk_log_pmf,
        with_dispersions,
    )

    threshold = float(chi2.ppf(level, 2))
    # NB the configured lattice and every pair the decode used: the lattice
    #    decode's states are the lattice, the shared decode's one per state.
    decoded_states = np.asarray(decode.states, dtype=np.int64)
    states = _lattice(decode)
    position = {(int(a), int(b)): k for k, (a, b) in enumerate(states)}
    out = []

    for clone, (_, bulk, _) in enumerate(clones):
        fitted = with_dispersions(bulk, decode.dispersion, decode.taus)
        pairs = decoded_states[np.asarray(decode.paths[clone], dtype=np.int64)]
        path = np.array([position[int(a), int(b)] for a, b in pairs], dtype=np.int64)
        ranges = segments(path, lengths)
        assigned = path[ranges[:, 0]]
        bins = np.arange(path.size)
        best_total = np.full((ranges.shape[0], states.shape[0]), -np.inf)

        for shift, fraction in _grid(
            float(decode.shifts[clone]),
            float(decode.purity[clone]),
            clone == normal_clone,
        ):
            log_mu, p = pair_rate_and_share(states, fraction)
            table = pseudobulk_log_pmf(
                (log_mu - shift)[:, None], p[:, None], fitted, bins
            )
            table = np.where(np.isfinite(table), table, -1e10)
            summed = np.add.reduceat(table, ranges[:, 0], axis=1).T
            held = summed[np.arange(ranges.shape[0]), assigned]
            total = held.sum()
            np.maximum(best_total, total - held[:, None] + summed, out=best_total)

        for index, (start, end) in enumerate(ranges.tolist()):
            row = best_total[index]
            gap = 2.0 * (row.max() - row)
            order = np.argsort(gap, kind="stable")
            out.append(
                SegmentSet(
                    clone=clone,
                    start=int(start),
                    end=int(end),
                    decoded=(
                        int(states[assigned[index], 0]),
                        int(states[assigned[index], 1]),
                    ),
                    best=(int(states[order[0], 0]), int(states[order[0], 1])),
                    threshold=threshold,
                    level=level,
                    consistent=tuple(
                        (int(states[k, 0]), int(states[k, 1]), float(gap[k]))
                        for k in order
                        if gap[k] <= threshold
                    ),
                )
            )

    return out


def _lattice(decode: Any) -> np.ndarray:
    """The configured lattice and every pair the decode used, sorted."""
    from port.extensions.copy_likelihood import candidates
    from port.patch.integer_copy import decode_caps

    allele, total_cap = decode_caps()
    decoded_states = np.asarray(decode.states, dtype=np.int64)
    return np.unique(np.vstack([candidates(total_cap, allele), decoded_states]), axis=0)


def bin_loglik(
    decode: Any, clones: list[tuple[np.ndarray, Any, float]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(pairs, loglik, decoded)`: each clone's per-bin NB + BB log-likelihood
    under every pair, `(n_clones, n_obs, n_pairs)`, at the decode's shift,
    fraction and dispersions, and each bin's decoded pair's index."""
    from port.extensions.copy_likelihood import (
        pair_rate_and_share,
        pseudobulk_log_pmf,
        with_dispersions,
    )

    pairs = _lattice(decode)
    position = {(int(a), int(b)): k for k, (a, b) in enumerate(pairs)}
    decoded_states = np.asarray(decode.states, dtype=np.int64)
    tables, decoded = [], []

    for clone, (_, bulk, _) in enumerate(clones):
        fitted = with_dispersions(bulk, decode.dispersion, decode.taus)
        assigned = decoded_states[np.asarray(decode.paths[clone], dtype=np.int64)]
        log_mu, p = pair_rate_and_share(pairs, float(decode.purity[clone]))
        bins = np.arange(assigned.shape[0])
        table = pseudobulk_log_pmf(
            (log_mu - float(decode.shifts[clone]))[:, None], p[:, None], fitted, bins
        )
        tables.append(np.where(np.isfinite(table), table, -1e10).T)
        decoded.append([position[int(a), int(b)] for a, b in assigned])

    return pairs, np.stack(tables), np.asarray(decoded, dtype=np.int64)


def segment_set_table(found: list[SegmentSet], decode: Any) -> pd.DataFrame:
    """One row per `(clone, segment, A, B)` in a set, phased as decoded."""
    rows = []
    for entry in found:
        base = {
            "clone": entry.clone,
            "start_bin": entry.start,
            "end_bin": entry.end,
            "shift": float(decode.shifts[entry.clone]),
            "tumour_fraction": float(decode.purity[entry.clone]),
            "decoded_A": entry.decoded[0],
            "decoded_B": entry.decoded[1],
            "best_A": entry.best[0],
            "best_B": entry.best[1],
            "threshold": entry.threshold,
            "level": entry.level,
            "set_size": len(entry.consistent),
        }
        for a, b, gap in entry.consistent:
            rows.append(base | {"A": a, "B": b, "deviance": gap})
    return pd.DataFrame(rows)


def write_segment_sets(
    run: Path,
    decode: Any,
    captured: Any,
    *,
    level: float = 0.95,
) -> Path:
    """Write `cnv_segment_sets.tsv` and `cnv_bin_loglik.npz` (:func:`bin_loglik`)
    into `run` for `decode` of `captured`, and return the first's path."""
    from port.extensions.copy_likelihood import clones_of, normal_of

    clones = clones_of(captured)
    found = segment_sets(
        decode, clones, normal_of(captured), captured.lengths, level=level
    )
    pairs, loglik, decoded = bin_loglik(decode, clones)
    np.savez_compressed(
        Path(run) / "cnv_bin_loglik.npz",
        pairs=pairs,
        loglik=loglik.astype(np.float32),
        decoded=decoded,
    )
    path = Path(run) / "cnv_segment_sets.tsv"
    with path.open("w") as handle:
        handle.write(
            f"# every (A, B) within {level:.2%} of each decoded segment's best, by "
            "the point decode's likelihood, shift and fraction refitted (#705); "
            "bins are the decode's, end exclusive\n"
        )
        segment_set_table(found, decode).to_csv(handle, sep="\t", index=False)
    return path


@contextlib.contextmanager
def writing_segment_sets(output: Path, *, level: float = 0.95) -> Iterator[None]:
    """Around a whole `run_cnaster_port`: afterwards, the segment sets beside its final fit.

    Collects the run's `params="smp"` fits (`copy_errors.captured_fits`) and
    its point decodes (`port.patch.integer_copy.recorded`), and writes
    :func:`write_segment_sets` into the directory of the newest
    `rdrbaf_final_nstates*_smp.npz` under `output`, from the last decode
    whose clones and bins are the last fit's. Without either, nothing is
    written. How `port.studies.population` reaches these sets without the
    entry point installing them.
    """
    from port.extensions.copy_likelihood import captured_fits
    from port.patch.integer_copy import recorded

    with captured_fits() as kept, recorded() as decodes:
        yield

    fits = sorted(
        Path(output).rglob("rdrbaf_final_nstates*_smp.npz"),
        key=lambda f: f.stat().st_mtime,
    )
    if not kept or not fits:
        return
    captured = kept[-1]
    path = np.asarray(captured.res["pred_cnv"])
    path = path.reshape(path.shape[0], -1)
    for decode in reversed(decodes):
        if len(decode.shifts) == path.shape[1] and all(
            np.asarray(pairs).shape[0] == path.shape[0] for pairs in decode.pairs
        ):
            write_segment_sets(fits[-1].parent, decode, captured, level=level)
            return
