"""A version-3 manifest from a finished `run_cnaster` run (#460), set aside.

Ticket: #460 -- a version-3 manifest from a finished run, a tool kept here
  at request: nothing reads its output, and its offsets are commented.
Measurement: none recorded on a run; on a draw's truth written as a run's
  outputs it recovers `dev_shared_unique`'s 3 clones, 1 shared and 2 unique
  events (`tests/test_sandbox_sim_from_run.py`).
Exit: graduate to `port.sim` once its unmeasured sections (slice offsets,
  the tree, normal fractions) can be filled from a run.

In the sandbox: nothing reads what it writes yet, and a manifest from it
does not draw until its commented offsets are stated.

A run's outputs say what it found: which spots it put in which clone, where
the spots sit, and each clone's integer copies along the genome. This writes
that as a manifest `port.sim.draw` reads, so the problem a run found can be
planted and drawn again with known truth. It replaces the YAML record of a
run's inputs (#116), which described the data but could not be drawn.

    python -m port.sandbox.sim_from_run <run dir> > sim/manifests/<name>.toml

From `clone_labels.tsv` and `cnv_segments.tsv` of the run directory:

- `[cna]`: the clones with any segment other than `(1, 1)` are the tumour
  clones, the rest are `normal`; `shared` counts the first tumour clone's
  events that every other carries (same `(A, B)`, overlapping), `unique` the
  mean of the rest; `states` the `(A, B)` seen; event lengths exponential at
  their mean, floored at their minimum;
- `[array]`: the hex array's rows and columns from the spots' positions;
- `[[slice]]`: one per `sample_id`, listing the clones on it, each tumour
  clone's region on the slice holding most of its spots, centred on its
  spots and sized to their count;
- `[config.hmrf] n_clones`: the clones the run found.

What a run does not measure is written commented `#`, naming why, and the
manifest `extends` CalicoST's defaults for the rest: a slice's offset in the
shared frame (a run keeps each slice's own positions), the tree the clones
share, each clone's normal fraction, and the count laws, which are fitted on
CalicoST's normal spots rather than a run's. A manifest with its offsets
commented is refused by `port.sim.draw` until they are stated.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

OVERLAP = 0.5
"""Reciprocal overlap at which two clones' events of one `(A, B)` are one event."""


@dataclass(frozen=True)
class Run:
    """What the manifest is written from: spots, their clones, and the copies."""

    name: str
    labels: pd.DataFrame
    """`barcode`, `sample_id`, `x` (array row), `y` (array column), `clone_label`."""
    segments: pd.DataFrame
    """`clone`, `CHR`, `START`, `END`, `A`, `B`, one row per run of equal copies."""


def read_run(path: Path) -> Run:
    """The run in `path`, a directory holding `clone_labels.tsv` and `cnv_segments.tsv`."""
    return Run(
        name=path.name,
        labels=pd.read_csv(path / "clone_labels.tsv", sep="\t"),
        segments=pd.read_csv(path / "cnv_segments.tsv", sep="\t", comment="#"),
    )


def events(run: Run) -> dict[int, pd.DataFrame]:
    """Each tumour clone's segments other than `(1, 1)`, keyed by the run's label."""
    altered = run.segments[(run.segments["A"] != 1) | (run.segments["B"] != 1)]
    return {
        int(clone): table.reset_index(drop=True)
        for clone, table in altered.groupby("clone", sort=True)
    }


def _carried(event: pd.Series, others: pd.DataFrame) -> bool:
    """Whether `others` holds an event of `event`'s `(A, B)` overlapping it."""
    same = others[
        (others["CHR"] == event["CHR"])
        & (others["A"] == event["A"])
        & (others["B"] == event["B"])
    ]
    shared = np.minimum(same["END"], event["END"]) - np.maximum(
        same["START"], event["START"]
    )
    longer = np.maximum(same["END"] - same["START"], event["END"] - event["START"])
    return bool(np.any(shared >= OVERLAP * longer))


def shared_unique(by_clone: dict[int, pd.DataFrame]) -> tuple[int, int]:
    """`shared`: the first clone's events every other carries; `unique`: the mean rest."""
    tables = list(by_clone.values())
    first, others = tables[0], tables[1:]
    shared = sum(
        all(_carried(row, other) for other in others) for _, row in first.iterrows()
    )
    rest = [len(table) - shared for table in tables]
    return int(shared), int(round(float(np.mean(rest))))


def hex_size(labels: pd.DataFrame) -> tuple[int, int]:
    """`(rows, columns)` of the hex array the positions fill (`array_col = 2 col + row % 2`)."""
    return int(labels["x"].max()) + 1, (int(labels["y"].max()) + 2) // 2


def regions(labels: pd.DataFrame, names: dict[int, str]) -> dict[str, list[str]]:
    """Per slice, TOML region lines for the tumour clones whose majority lies on it.

    A centre is the clone's spots' mean in fractions of its slice's extent,
    and a radius that of the disc of the clone's spot count, in the shorter
    extent, as `port.sim.draw.polygon` reads them.
    """
    points = np.column_stack(
        [labels["y"].to_numpy() / 2.0, labels["x"].to_numpy() * np.sqrt(3.0) / 2.0]
    )
    lines: dict[str, list[str]] = {sid: [] for sid in labels["sample_id"].unique()}

    for label, name in names.items():
        spots = labels["clone_label"].to_numpy() == label
        home = labels.loc[spots, "sample_id"].value_counts().index[0]
        on = labels["sample_id"].to_numpy() == home
        lower, upper = points[on].min(axis=0), points[on].max(axis=0)
        extent = upper - lower
        centre = (points[spots & on].mean(axis=0) - lower) / extent
        radius = np.sqrt(np.sum(spots & on) * np.sqrt(3.0) / 2.0 / np.pi)
        lines[home].append(
            f'{{ clone = "{name}", center = [{centre[0]:.3f}, {centre[1]:.3f}], '
            f"radius = {radius / extent.min():.3f} }}"
        )
    return lines


def to_toml(run: Run) -> str:
    """The version-3 manifest of `run`, what it cannot state commented."""
    by_clone = events(run)
    if not by_clone:
        msg = f"{run.name}: no clone carries a segment other than (1, 1)"
        raise ValueError(msg)

    names = {label: f"clone_{k}" for k, label in enumerate(by_clone)}
    shared, unique = shared_unique(by_clone)
    everything = pd.concat(by_clone.values())
    lengths = (everything["END"] - everything["START"]).to_numpy()
    states = sorted({(int(a), int(b)) for a, b in zip(everything["A"], everything["B"], strict=True)})  # fmt: skip
    rows, columns = hex_size(run.labels)
    by_slice = regions(run.labels, names)
    lines = [
        f"# Generated by `python -m port.sandbox.sim_from_run` from the run `{run.name}` (#460).",
        "# A run measures its clones, copies and spots; what it does not is",
        "#    commented, and the rest is `calicost_grch38.toml`'s.",
        "version = 3",
        'extends = "calicost_grch38.toml"',
        "",
        "[sample]",
        f'name = "{run.name}"',
        "seed = 0",
        'output = "../generated"',
        "realizations = 1",
        "",
        "[array]",
        'kind = "hex"',
        f"rows = {rows}",
        f"columns = {columns}",
        "",
        "# NB [model] normal_frac: a run does not estimate a clone's normal",
        "#    fraction; calicost_grch38.toml's 0 holds.",
        "# NB [model] dirichlet_concentration, snp_dispersion and [reference]",
        "#    coverage: CalicoST's normal-spot fits, not refitted on this run.",
        "",
        "[cna]",
        'mode = "shared.unique"',
        f"n_clones = {len(by_clone)}",
        f"shared = {shared}",
        f"unique = {unique}",
        f"states = {[list(s) for s in states]}",
        "# NB the tree the clones share: a run's copies do not order their events,",
        '#    so mode = "tree" and its trunk, per_leaf and per_internal are not stated.',
        "",
        "[cna.length]",
        'law = "exponential"',
        f"mean = {float(lengths.mean()):.4g}",
        f"minimum = {float(lengths.min()):.4g}",
        "",
        "[config.hmrf]",
        f"n_clones = {run.labels['clone_label'].nunique()}",
    ]
    for sid in run.labels["sample_id"].unique():
        on = run.labels["sample_id"] == sid
        present = [
            names[label]
            for label in sorted(run.labels.loc[on, "clone_label"].unique())
            if label in names
        ]
        lines += ["", "[[slice]]", f"# NB sample_id {sid}."]
        lines += [
            "# offset = [?, ?]  # NB where the slice sits in the shared frame:",
            "#    a run keeps each slice's own positions, so this is not measured.",
            f"clones = {present}".replace("'", '"'),
        ]
        if by_slice[sid]:
            lines += ["regions = [", *(f"    {r}," for r in by_slice[sid]), "]"]

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="a run directory holding clone_labels.tsv")
    arguments = parser.parse_args(argv)
    print(to_toml(read_run(Path(arguments.run))), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
