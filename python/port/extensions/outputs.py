"""What a run fitted and what it decoded, written side by side (#331).

`run_cnaster` records the continuous fit and the integer copies only through
the fitted state index `Z` of `cnv_seglevel.tsv`: several of `K` fitted states
decode to one integer `(A, B)`, and the map between the two views is left
implicit. This writes the seam explicitly, beside `cnaster`'s own files and
without touching them, in each run directory that holds a
`cnv_seglevel.tsv` and its `rdrbaf_final_nstates{K}_smp.npz`:

- `cnv_states.tsv`: one row per fitted state and clone -- the state's
  `logmu` and `p`, the `(A, B)` it decodes to in that clone, and the
  share of the clone's bins it holds. The map from the oversampled states
  to the deduplicated integer copies.
- `cnv_segments.tsv`: per clone, the runs of equal `(A, B)` within a
  chromosome, their first and last bin, the fitted states they span, and
  the posterior-mean `mu` and `p` over the run. The deduplicated view.
- `cnv_binlevel.tsv`: per bin and clone, the fitted state and the
  posterior-mean `mu` and `p` under `log_gamma`. The continuous view.
- `manifest.json`: the run's shape and provenance -- states, clones,
  likelihoods, the shift, the configuration's copy caps and ploidy, and
  what `run_cnaster_port` was asked for.

**A clone's columns are matched by content, not position.** The table's
`clone{c}` columns carry `cnaster`'s clone id, and `pred_cnv` and `log_gamma`
are indexed by the clone's position among the final clones; each column is
matched to the position whose decoded path is its `Z`.

What the run does not keep is not reconstructed: the observed pseudobulk RDR
and BAF per bin, per-state errors and the decoder's own loss are not in
either file (#331 lists them).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "binlevel",
    "clone_columns",
    "config_keys",
    "run_directories",
    "segments",
    "states",
    "write_outputs",
]


def run_directories(output_dir: Path) -> Iterator[Path]:
    """Each directory under `output_dir` holding a finished run's tables."""
    for table in sorted(Path(output_dir).rglob("cnv_seglevel.tsv")):
        if any(table.parent.glob("rdrbaf_final_nstates*_smp.npz")):
            yield table.parent


def _load(run: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    seglevel = pd.read_csv(run / "cnv_seglevel.tsv", sep="\t", comment="#")
    perstate = pd.read_csv(run / "cnv_perstate.tsv", sep="\t", comment="#")
    (npz,) = sorted(run.glob("rdrbaf_final_nstates*_smp.npz"))

    with np.load(npz, allow_pickle=True) as fit:
        return seglevel, perstate, {key: fit[key] for key in fit.files}


def clone_columns(seglevel: pd.DataFrame, pred_cnv: np.ndarray) -> dict[str, int]:
    """`cnaster`'s clone id -> its position in `pred_cnv`, by decoded path.

    Two clones with one path are told apart by order, which is `cnaster`'s
    own: its columns are written in the order of the final clones.
    """
    ids = [c.split()[0][len("clone") :] for c in seglevel.columns if c.endswith(" Z")]
    free = list(range(pred_cnv.shape[1]))
    found: dict[str, int] = {}

    for clone in ids:
        path = seglevel[f"clone{clone} Z"].to_numpy(dtype=int)
        position = next((s for s in free if np.array_equal(pred_cnv[:, s], path)), None)

        if position is None:
            msg = f"clone {clone}'s Z matches no column of pred_cnv"
            raise ValueError(msg)

        free.remove(position)
        found[clone] = position

    return found


def _posterior_means(fit: dict[str, Any], position: int) -> tuple[np.ndarray, ...]:
    """The posterior-mean `mu` and `p` per bin of one clone."""
    gamma = np.exp(fit["log_gamma"][:, :, position])
    gamma = gamma / gamma.sum(axis=0, keepdims=True)
    mu = np.exp(fit["new_log_mu"][:, 0]) @ gamma
    p = fit["new_p_binom"][:, 0] @ gamma
    return mu, p


def states(
    seglevel: pd.DataFrame, perstate: pd.DataFrame, fit: dict[str, Any]
) -> pd.DataFrame:
    """One row per fitted state and clone: its fit, the `(A, B)` the clone's
    decoder gave it (`cnv_perstate.tsv`), and the share of the clone's bins
    it holds -- zero for a state the clone decodes but never visits."""
    rows = []

    for clone in clone_columns(seglevel, fit["pred_cnv"]):
        path = seglevel[f"clone{clone} Z"].to_numpy(dtype=int)

        for state in range(int(fit["n_states"])):
            rows.append(
                {
                    "clone": clone,
                    "state": state,
                    "logmu": float(fit["new_log_mu"][state, 0]),
                    "p": float(fit["new_p_binom"][state, 0]),
                    "A": int(perstate[f"clone{clone} A"].iloc[state]),
                    "B": int(perstate[f"clone{clone} B"].iloc[state]),
                    "share": float((path == state).mean()),
                }
            )

    return pd.DataFrame(rows)


def binlevel(seglevel: pd.DataFrame, fit: dict[str, Any]) -> pd.DataFrame:
    """Per bin and clone: the fitted state and the posterior-mean `mu`, `p`."""
    frame = seglevel[["CHR", "START", "END"]].copy()

    for clone, position in clone_columns(seglevel, fit["pred_cnv"]).items():
        mu, p = _posterior_means(fit, position)
        frame[f"clone{clone} Z"] = seglevel[f"clone{clone} Z"].to_numpy(dtype=int)
        frame[f"clone{clone} mu"] = mu
        frame[f"clone{clone} p"] = p

    return frame


def segments(seglevel: pd.DataFrame, fit: dict[str, Any]) -> pd.DataFrame:
    """Per clone, the runs of equal `(A, B)` within a chromosome."""
    rows = []
    chromosome = seglevel["CHR"].to_numpy()

    for clone, position in clone_columns(seglevel, fit["pred_cnv"]).items():
        mu, p = _posterior_means(fit, position)
        pairs = seglevel[[f"clone{clone} A", f"clone{clone} B"]].to_numpy(dtype=int)
        path = seglevel[f"clone{clone} Z"].to_numpy(dtype=int)
        # NB a run breaks where the pair or the chromosome changes.
        breaks = np.flatnonzero(
            np.any(pairs[1:] != pairs[:-1], axis=1)
            | (chromosome[1:] != chromosome[:-1])
        )
        starts = np.concatenate([[0], breaks + 1])
        ends = np.concatenate([breaks + 1, [len(seglevel)]])

        for first, stop in zip(starts, ends, strict=True):
            rows.append(
                {
                    "clone": clone,
                    "CHR": chromosome[first],
                    "START": seglevel["START"].iloc[first],
                    "END": seglevel["END"].iloc[stop - 1],
                    "first_bin": int(first),
                    "last_bin": int(stop - 1),
                    "n_bins": int(stop - first),
                    "A": int(pairs[first, 0]),
                    "B": int(pairs[first, 1]),
                    "states": ",".join(str(s) for s in np.unique(path[first:stop])),
                    "mu": float(mu[first:stop].mean()),
                    "p": float(p[first:stop].mean()),
                }
            )

    return pd.DataFrame(rows)


def config_keys(config: Path | None) -> dict[str, Any]:
    """The configuration's copy caps, ploidy and state count, where set."""
    if config is None:
        return {}

    import yaml

    wanted = {"n_states", "max_total_copy", "ploidy", "output_dir"}
    found: dict[str, Any] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in wanted and not isinstance(value, dict):
                    found[key] = value
                walk(value)

    walk(yaml.safe_load(Path(config).read_text()))
    return found


def write_outputs(
    run: Path, config: Path | None = None, flags: dict[str, Any] | None = None
) -> list[Path]:
    """Write the four files into `run`; return their paths."""
    from importlib.metadata import PackageNotFoundError, version

    def installed(name: str) -> str:
        try:
            return version(name)
        except PackageNotFoundError:
            return "not installed"

    def finite(value: Any) -> float | None:
        # NB JSON has no NaN; a likelihood `cnaster` left unset is null.
        number = float(value)
        return number if np.isfinite(number) else None

    run = Path(run)
    seglevel, perstate, fit = _load(run)
    written = []

    for name, table in (
        ("cnv_states.tsv", states(seglevel, perstate, fit)),
        ("cnv_segments.tsv", segments(seglevel, fit)),
        ("cnv_binlevel.tsv", binlevel(seglevel, fit)),
    ):
        table.to_csv(run / name, sep="\t", index=False)
        written.append(run / name)

    shift = fit.get("new_log_mu_shift")
    shift_value = None if shift is None else shift.item()
    manifest = {
        "n_states": int(fit["n_states"]),
        "n_bins": len(seglevel),
        "clones": clone_columns(seglevel, fit["pred_cnv"]),
        "llf": finite(fit["llf"]),
        "total_llf": finite(fit["total_llf"]),
        "log_mu_shift": None
        if shift_value is None
        else np.asarray(shift_value, dtype=float).tolist(),
        "integer_decoder": "cnaster MILP, first ploidy pass (max_medploidy=None)",
        "config": config_keys(config),
        "run_cnaster_port": flags or {},
        "versions": {name: installed(name) for name in ("cnaster", "port")},
    }
    (run / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    written.append(run / "manifest.json")
    return written
