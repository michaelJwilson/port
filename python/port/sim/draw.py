"""Draw new samples from a version-3 TOML manifest (#445).

Its clones, events, layout and counts are drawn from what the manifest states:

    version = 3
    [sample]      name, seed, output, realizations; r0_hash, which the draw does
                  not read: r0's `realization_hash`, naming the generation (#583)
    [reference]   baseline, coverage, snps; GRCh38 resources via $PORT_GRCH38
    [array]       kind = "hex" or "square", rows, columns
    [model]       admixture, normal_frac, dirichlet_concentration, bb_overdispersion,
                  snp_dispersion, snp_depth_follows_copies, counts_sampler
    [cna]         mode = "shared.unique" (shared, unique), "tree"
                  (trunk, per_leaf, per_internal) or "felsenstein"
                  (expected_cnas); n_clones;
                  states = [[A, B], ...]. Defaults are CalicoST's easy
                  sample, `numcnas1.2`, state them
    [cna.length]  law = "fixed" (size), "exponential" (mean, minimum) or
                  "lognormal" (sigma, minimum, and exactly one of mean or
                  median): `max(L, minimum)`, `log L ~ Normal(log median,
                  sigma^2)`; a stated mean keys median = mean exp(-sigma^2 / 2),
                  so the stated quantity is the law's. `sigma` is
                  `port.sim.laws.lognormal_sigma`'s (#619)
    [phasing]     switch_errors, nu, unit
    [layout]      overlap, radius, vertices, jitter
    [layout.size] optional: law = "loguniform" (minimum, maximum) or
                  "lognormal" (median, sigma), in spots (#544)
    [[slice]]     clones, offset = [x, y], regions = [{clone, center, ...}]

**The generative model**, per slice and spot `s` of clone `c`, from the
per-entry laws `port.sim.normal_fit` fits on CalicoST's normal spots (#455):

- gene UMI: `N_s ~ [coverage] spot_umi`, times the clone's library factor
  `sum_g lambda_g d_g(c)`, so a gain grows the library; shares
  `p_s ~ Dirichlet(kappa q_c)`, `q_c` proportional to `lambda_g d_g(c)`; and
  `Multinomial(N_s, p_s)`, drawn as `[model] counts_sampler` names
  (`port.sim.entries.COUNT_SAMPLERS`: normalized gammas, or a Pólya urn). A
  gene's expected UMI is `N_s lambda_g d_g(c)`, so its read-depth ratio to
  normal is `d_g(c) = (1 - f) (A + B) / 2 + f`, the admixture law's depth
  factor at the gene's `(A, B)`, `f = normal_frac`. `lambda` is
  `normal_baseline.txt.gz`'s and `kappa = [model] dirichlet_concentration`,
  both fitted on CalicoST's normal spots (#455);
- SNP reads: every (SNP, spot) entry drawn independently from
  `[coverage] snp_spot_umi` (`snp_law`, `b = [model] snp_dispersion`),
  unordered and independent of copies. Per realization: the counts and the
  phase. Both samplers draw each nonzero value exactly, with no tail cut;
- the haplotype-A count `BetaBinomial(n, share, rho)`, `share` the admixture
  law's (`port.sim.laws.allele_share`).

**Clones.** `shared.unique` is CalicoST's `numcnas{shared}.{unique}`:
`shared` events on every tumour clone and `unique` on each. `tree` draws a
topology on `normal` and the tumour clones with `snakes_and_ladders`'
`random_topology` -- uniform over unrooted binary trees -- and roots it at
`normal`: the edge out of `normal` is the trunk and carries `trunk` events,
each edge into a clone `per_leaf` and each edge into an unobserved ancestor
`per_internal`. `felsenstein` draws the same topology -- uniform over the
`(2n - 3)!!` rooted binary trees on `n = n_clones` labelled leaves
(Felsenstein 1978) -- and Poisson event counts per edge whose means sum to
`expected_cnas` (`_felsenstein_counts`). A clone carries every event on its path from
the root, later ones overriding earlier ones where they overlap. `(1, 1)`
elsewhere.

**Phase.** One phasing for all slices, as the SNPs are phased on the
pseudobulk: between consecutive SNPs of a chromosome the reported phase
switches with probability `(1 - exp(-2 nu d)) / 2`, `d` the map distance in
`unit`: Haldane's recombination law in Morgans, or -- in cM at `nu = 1` --
Numbat's phase-switch law, which `cnaster.recomb` assumes. `cnaster` then
multiplies by `exp(-logphase_shift)` and floors at `min_prob` per bin, which
does not compose over SNPs and is its inference's, so it is not drawn. The
map is `genetic_map_GRCh38_merged.tab.gz`. A switched SNP's `A`
and `B` are exchanged in the written matrices; `truth_phase.npy` is that
binary vector, one entry per SNP.

**Barcodes.** One whitelist of Visium-like 16-mers, `-1`, shared by every
slice at the same array position, as Visium's is; each spot is written as
`{barcode}-1_{sample_id}` with a hexadecimal `sample_id` per slice, the
form `cnaster.io.get_aggregated_barcodes` splits.
"""

from __future__ import annotations

import argparse
import functools
import itertools
import os
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from port.sim.entries import COUNT_SAMPLERS, independent, nodes, snp_law
from port.sim.files import load_ids
from port.sim.laws import ADMIXTURE_LAWS, Event, Law, allele_share

MANIFEST_VERSION = 3

REQUIRED: dict[str, tuple[str, ...]] = {
    "sample": ("name", "seed", "output", "realizations"),
    "reference": (
        "baseline", "coverage", "snps", "resources", "gene_table",
        "genetic_map", "filter_genes", "filter_regions",
    ),
    "genome": ("chromosome_lengths",),
    "array": ("kind", "rows", "columns"),
    "barcodes": ("length", "suffix", "sample_id_bytes"),
    "model": (
        "admixture", "normal_frac", "dirichlet_concentration", "bb_overdispersion",
        "snp_dispersion", "snp_depth_follows_copies", "counts_sampler",
    ),
    "cna": ("mode", "n_clones", "states", "length"),
    "phasing": ("switch_errors", "nu", "unit"),
    "layout": ("overlap", "radius", "vertices", "jitter", "max_placements"),
    "config": ("base",),
}  # fmt: skip
"""Every table and key a manifest must state. Nothing the draw assumes has a
default in code: a manifest that leaves one out is refused, naming it."""

BY_MODE = {
    "shared.unique": ("shared", "unique"),
    "tree": ("trunk", "per_leaf", "per_internal"),
    "felsenstein": ("expected_cnas",),
}
BY_LAW = {
    "fixed": ("size",),
    "exponential": ("mean", "minimum"),
    "lognormal": ("sigma", "minimum"),
}
LOGNORMAL_KEYS = ("mean", "median")
"""`[cna.length] law = "lognormal"` states exactly one of these (#619)."""
SIZE_LAWS = {"loguniform": ("minimum", "maximum"), "lognormal": ("median", "sigma")}
"""`[layout.size]`: each drawn clone's size in spots, per seed (#544). Optional:
without it every drawn clone takes `[layout] radius`, as before."""


def references(directory: Path, names: tuple[str, ...]) -> Path:
    """`directory` if it holds `names`, else CalicoST's uv git checkout's.

    `[reference] resources` is usually `$PORT_GRCH38`; where that is unset
    the checkout `uv` made of CalicoST for `cnaster` is found instead.
    """
    candidates = [str(directory)] if str(directory) not in {"", "."} else []
    candidates += sorted(
        str(p)
        for p in (Path.home() / ".cache/uv/git-v0/checkouts").glob(
            "*/*/GRCh38_resources"
        )
    )

    for candidate in candidates:
        if all((Path(candidate) / f).exists() for f in names):
            return Path(candidate)

    msg = f"none of {candidates} holds {list(names)}; set [reference] resources"
    raise FileNotFoundError(msg)


# --- the manifest --------------------------------------------------------------


class Region(NamedTuple):
    """One clone's polygon on one slice, in fractions of the array's extent."""

    clone: str
    center: tuple[float, float] | None
    """`None` draws it, uniform over the array; a polygon past an edge is clipped."""
    radius: float
    """Of the shorter extent."""
    vertices: int
    jitter: float
    """Each vertex's radius is `radius (1 + jitter U(-1, 1))`."""


class Slice(NamedTuple):
    clones: tuple[str, ...]
    regions: tuple[Region, ...]
    offset: tuple[float, float]
    """Where the slice sits in the shared frame, in fractions of the array's extent."""


@dataclass(frozen=True)
class DrawManifest:
    """A parsed version-3 manifest: the TOML's tables, checked."""

    tables: dict[str, dict[str, Any]]
    slices: tuple[Slice, ...]
    root: Path

    def __getattr__(self, table: str) -> dict[str, Any]:
        tables = self.__dict__["tables"]
        if table in tables:
            return dict(tables[table])
        raise AttributeError(table)

    @property
    def name(self) -> str:
        return str(self.tables["sample"]["name"])

    @property
    def seed(self) -> int:
        return int(self.tables["sample"]["seed"])

    def resolve(self, value: str) -> Path:
        return self.root / os.path.expandvars(value)

    @property
    def tumour(self) -> tuple[str, ...]:
        return tuple(f"clone_{k}" for k in range(int(self.tables["cna"]["n_clones"])))

    def normal_frac(self, clone: str) -> float:
        value = self.tables["model"]["normal_frac"]
        if clone == "normal":
            return 0.0
        if isinstance(value, dict):
            return float(value[clone])
        return float(value)

    def resources(self) -> Path:
        reference = self.tables["reference"]
        names = tuple(reference[k] for k in ("gene_table", "genetic_map"))
        return references(Path(os.path.expandvars(reference["resources"])), names)


def read_manifest(path: str | Path) -> DrawManifest:
    """Parse and check a version-3 manifest."""
    source = Path(path)
    document = extended(source)
    version = int(document.get("version", 0))

    if version != MANIFEST_VERSION:
        msg = f"manifest version {version}, this reads {MANIFEST_VERSION}"
        raise ValueError(msg)

    return from_document(document, source.parent)


def extended(path: Path) -> dict[str, Any]:
    """`path`'s document laid over the one its `extends` names, recursively.

    Tables merge key by key and the extending file wins; `[[slice]]` and
    every other value it states replace the base's. A `[cna.length]` that
    states its `law` replaces the base's whole, so a law never inherits
    another's parameters (#619: a `median` over a base's `mean`).
    """
    document = _anchored(tomllib.loads(path.read_text()), path.parent)
    parent = document.pop("extends", None)
    if parent is None:
        return document
    base = extended(path.parent / parent)
    if "law" in document.get("cna", {}).get("length", {}) and "cna" in base:
        base["cna"] = {k: v for k, v in base["cna"].items() if k != "length"}
    return _merge(base, document)


PATH_KEYS = (
    ("sample", "output"),
    ("reference", "baseline"),
    ("reference", "coverage"),
    ("reference", "snps"),
    ("config", "base"),
)
"""Keys holding a path relative to the file that states them."""


def _anchored(document: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Each relative path in `PATH_KEYS` made absolute against `directory`.

    So a key keeps its meaning when the file it is in is extended from
    another directory; a `$VARIABLE` path is left for `resolve` to expand.
    """
    for table, key in PATH_KEYS:
        value = document.get(table, {}).get(key)
        if (
            isinstance(value, str)
            and not value.startswith("$")
            and not Path(value).is_absolute()
        ):
            document[table][key] = str((directory / value).resolve())
    return document


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _missing(document: dict[str, Any]) -> list[str]:
    """`table.key` for every required key the document does not state."""
    missing = [
        f"[{t}] {k}" for t, keys in REQUIRED.items() for k in keys
        if k not in document.get(t, {})
    ]  # fmt: skip
    cna = document.get("cna", {})
    missing += [f"[cna] {k}" for k in BY_MODE.get(cna.get("mode"), ()) if k not in cna]
    length = cna.get("length", {})
    missing += [
        f"[cna.length] {k}" for k in ("law", *BY_LAW.get(length.get("law"), ()))
        if k not in length
    ]  # fmt: skip
    for index, piece in enumerate(document.get("slice", [])):
        missing += [
            f"[[slice]] {index}: {k}" for k in ("clones", "offset") if k not in piece
        ]
    if not document.get("slice"):
        missing.append("[[slice]]")
    return missing


def from_document(document: dict[str, Any], root: Path = Path()) -> DrawManifest:
    missing = _missing(document)
    if missing:
        msg = f"the manifest does not state {', '.join(missing)}"
        raise ValueError(msg)

    layout = document["layout"]
    slices = tuple(
        Slice(
            clones=tuple(s["clones"]),
            offset=(float(s["offset"][0]), float(s["offset"][1])),
            regions=tuple(
                Region(
                    clone=r["clone"],
                    center=None if "center" not in r else tuple(r["center"]),
                    radius=float(r.get("radius", layout["radius"])),
                    vertices=int(r.get("vertices", layout["vertices"])),
                    jitter=float(r.get("jitter", layout["jitter"])),
                )
                for r in s.get("regions", [])
            ),
        )
        for s in document["slice"]
    )
    tables = {k: v for k, v in document.items() if isinstance(v, dict)}
    manifest = DrawManifest(tables, slices, root)
    _check(manifest)
    return manifest


def _check(manifest: DrawManifest) -> None:
    problems = []
    mode = manifest.cna["mode"]

    if mode not in BY_MODE:
        problems.append(f"[cna] mode {mode!r}: one of {sorted(BY_MODE)}")
    elif mode == "felsenstein":
        leaves = int(manifest.cna["n_clones"])
        if leaves < 2 or float(manifest.cna["expected_cnas"]) < leaves:
            problems.append(
                f"[cna] felsenstein: n_clones >= 2 and expected_cnas >= n_clones, "
                f"not {leaves} and {manifest.cna['expected_cnas']}"
            )
    if manifest.model["admixture"] not in ADMIXTURE_LAWS:
        problems.append(f"[model] admixture {manifest.model['admixture']!r}")
    if manifest.model["counts_sampler"] not in COUNT_SAMPLERS:
        problems.append(f"[model] counts_sampler: one of {sorted(COUNT_SAMPLERS)}")
    length = manifest.cna["length"]
    if length["law"] not in BY_LAW:
        problems.append(f"[cna.length] law: one of {sorted(BY_LAW)}")
    elif length["law"] == "lognormal":
        keyed = [k for k in LOGNORMAL_KEYS if k in length]
        if len(keyed) != 1:
            problems.append(
                f"[cna.length] lognormal: exactly one of {LOGNORMAL_KEYS}, not {keyed}"
            )
    size = manifest.layout.get("size")
    if size is not None:
        if size.get("law") not in SIZE_LAWS:
            problems.append(f"[layout.size] law: one of {sorted(SIZE_LAWS)}")
        else:
            absent = [k for k in SIZE_LAWS[size["law"]] if k not in size]
            problems += [f"[layout.size] {k}" for k in absent]
    if manifest.phasing["unit"] not in UNITS:
        problems.append(f"[phasing] unit: one of {sorted(UNITS)}")
    if manifest.array["kind"] not in ARRAYS:
        kind = manifest.array["kind"]
        problems.append(f"[array] kind {kind!r}: one of {sorted(ARRAYS)}")

    known = set(manifest.tumour)
    for index, piece in enumerate(manifest.slices):
        unknown = (set(piece.clones) | {r.clone for r in piece.regions}) - known
        if unknown:
            problems.append(f"slice {index} names unknown clones {sorted(unknown)}")

    if problems:
        raise ValueError("; ".join(problems))


# --- clones --------------------------------------------------------------------


@dataclass(frozen=True)
class CloneTree:
    """`parent[node]`, the events on the edge into `node`, and the leaves."""

    parent: dict[str, str | None]
    edge_events: dict[str, tuple[Event, ...]]
    leaves: tuple[str, ...]

    def path(self, node: str) -> list[str]:
        """Root to `node`, inclusive."""
        out = [node]
        while (up := self.parent[out[-1]]) is not None:
            out.append(up)
        return out[::-1]

    def events(self, node: str) -> tuple[Event, ...]:
        """Every event on the path to `node`, in the order they arose."""
        return tuple(e for n in self.path(node) for e in self.edge_events[n])


def _topology(
    manifest: DrawManifest, rng: np.random.Generator
) -> dict[str, str | None]:
    """`parent` of each node of the tree, rooted at `normal`."""
    tumour = manifest.tumour

    if manifest.cna["mode"] == "shared.unique" or len(tumour) < 2:
        return {"normal": None, "founder": "normal"} | dict.fromkeys(tumour, "founder")

    from sal.sim.topology import random_topology
    from sal.sim.tree import edges

    unrooted = random_topology(["normal", *tumour], rng)
    neighbours: dict[str, list[str]] = {}
    for up, down in edges(unrooted):
        neighbours.setdefault(up.name, []).append(down.name)
        neighbours.setdefault(down.name, []).append(up.name)

    parent: dict[str, str | None] = {"normal": None}
    frontier = ["normal"]
    while frontier:
        node = frontier.pop()
        for other in neighbours[node]:
            if other not in parent:
                parent[other] = node
                frontier.append(other)

    names = {n: n for n in parent if n in {"normal", *tumour}}
    internal = sorted((n for n in parent if n not in names), key=str)
    names |= {n: f"ancestor_{k}" for k, n in enumerate(internal)}
    trunk = next(n for n, p in parent.items() if p == "normal")
    names[trunk] = "founder" if trunk not in tumour else trunk
    return {names[n]: None if p is None else names[p] for n, p in parent.items()}


def _event_length(manifest: DrawManifest, rng: np.random.Generator) -> int:
    law = manifest.cna["length"]
    if law["law"] == "fixed":
        return int(law["size"])
    if law["law"] == "lognormal":
        median = lognormal_median(law)
        drawn = rng.lognormal(np.log(median), float(law["sigma"]))
        return int(max(drawn, float(law["minimum"])))
    return int(max(rng.exponential(float(law["mean"])), float(law["minimum"])))


def lognormal_median(law: dict[str, Any]) -> float:
    """`[cna.length]`'s lognormal median: stated, or `mean exp(-sigma^2 / 2)` (#619)."""
    if "median" in law:
        return float(law["median"])
    return float(law["mean"]) * float(np.exp(-(float(law["sigma"]) ** 2) / 2.0))


def _events(
    manifest: DrawManifest, count: int, rng: np.random.Generator
) -> tuple[Event, ...]:
    """`count` events: a chromosome by length, a start uniform on it, a state."""
    lengths = np.asarray(manifest.genome["chromosome_lengths"], dtype=np.float64)
    states = [tuple(s) for s in manifest.cna["states"]]
    out = []

    for _ in range(count):
        index = int(rng.choice(lengths.size, p=lengths / lengths.sum()))
        span = min(_event_length(manifest, rng), int(lengths[index]))
        start = int(rng.integers(0, int(lengths[index]) - span + 1))
        a, b = states[int(rng.integers(len(states)))]
        out.append(Event(str(index + 1), start, start + span, int(a), int(b)))

    return tuple(out)


def _felsenstein_counts(
    parent: dict[str, str | None], leaves: set[str], expected: float,
    rng: np.random.Generator,
) -> dict[str, int]:  # fmt: skip
    """Events per edge of a `felsenstein` tree: each leaf edge `1 + Poisson(m)`, every other edge `Poisson(m)`.

    Choices, not consequences of the model: one event is held on each leaf
    edge so no clone's profile equals its parent's, and so no two sibling
    clones are drawn identical; the remaining `expected - leaves` is spread
    equally over every edge, the trunk included, `m = (expected - leaves) /
    edges`. The expected total is `expected` exactly. Edges are visited in
    sorted order, one Poisson draw each, before any event is placed.
    """
    edges = sorted(n for n in parent if parent[n] is not None)
    rate = (expected - len(leaves)) / len(edges)
    return {
        node: int(rng.poisson(rate)) + (1 if node in leaves else 0) for node in edges
    }


def draw_tree(manifest: DrawManifest, rng: np.random.Generator) -> CloneTree:
    """The clones' tree and the events on each of its edges."""
    parent = _topology(manifest, rng)
    mode = manifest.cna["mode"]
    leaves = set(manifest.tumour)
    if mode == "felsenstein":
        counts = _felsenstein_counts(
            parent, leaves, float(manifest.cna["expected_cnas"]), rng
        )
        edge_events: dict[str, tuple[Event, ...]] = {"normal": ()}
        for node in sorted(counts):
            edge_events[node] = _events(manifest, counts[node], rng)
        return CloneTree(parent, edge_events, manifest.tumour)

    shared_unique = mode == "shared.unique"
    table = manifest.cna
    trunk = int(table["shared" if shared_unique else "trunk"])
    leaf = int(table["unique" if shared_unique else "per_leaf"])
    internal = 0 if shared_unique else int(table["per_internal"])

    edge_events = {"normal": ()}
    for node in sorted(n for n in parent if n != "normal"):
        if parent[node] == "normal":
            count = trunk
        else:
            count = leaf if node in leaves else internal
        edge_events[node] = _events(manifest, count, rng)

    return CloneTree(parent, edge_events, manifest.tumour)


def clone_copies(
    tree: CloneTree,
    clones: tuple[str, ...],
    chromosome: np.ndarray,
    position: np.ndarray,
) -> np.ndarray:
    """`(n_positions, n_clones, 2)` `(A, B)`; `(1, 1)` where no event covers."""
    copies = np.ones((position.size, len(clones), 2), dtype=np.int64)
    query = np.asarray(chromosome).astype(str)

    for label, clone in enumerate(clones):
        for event in tree.events(clone) if clone != "normal" else ():
            covered = (query == event.chromosome) & (position >= event.start)
            covered &= position < event.end
            copies[covered, label] = (event.a, event.b)

    return copies


def truth_profile(
    tree: CloneTree, clones: tuple[str, ...], chromosome_lengths: list[int]
) -> pd.DataFrame:
    """`truth_acn_profile.tsv`: `chr start end`, `A`, `B` per clone, CalicoST's order."""
    rows = []
    for index, length in enumerate(chromosome_lengths, start=1):
        cuts = {0, int(length)}
        for clone in clones:
            for event in tree.events(clone) if clone != "normal" else ():
                if event.chromosome == str(index):
                    cuts |= {event.start, event.end}
        rows += [(index, s, e) for s, e in itertools.pairwise(sorted(cuts))]

    frame = pd.DataFrame(rows, columns=["chr", "start", "end"])
    middle = ((frame["start"] + frame["end"]) // 2).to_numpy()
    copies = clone_copies(tree, clones, frame["chr"].astype(str).to_numpy(), middle)
    for label in sorted(range(len(clones)), key=lambda i: clones[i] == "normal"):
        frame[f"{clones[label]}_A_copy"] = copies[:, label, 0]
        frame[f"{clones[label]}_B_copy"] = copies[:, label, 1]
    return frame


def tree_table(tree: CloneTree) -> pd.DataFrame:
    """`truth_tree.tsv`: one row per event, with the edge it arose on."""
    rows = [
        {"node": node, "parent": tree.parent[node], "chr": e.chromosome,
         "start": e.start, "end": e.end, "A": e.a, "B": e.b}
        for node in tree.parent
        for e in tree.edge_events[node]
    ]  # fmt: skip
    edges = [{"node": n, "parent": p} for n, p in tree.parent.items()]
    columns = ["node", "parent", "chr", "start", "end", "A", "B"]
    return pd.concat(
        [
            pd.DataFrame(edges, columns=["node", "parent"]),
            pd.DataFrame(rows, columns=columns),
        ],
        ignore_index=True,
    )[columns]


# --- the array and the layout --------------------------------------------------


def hex_array(rows: int, columns: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Visium's packing: `array_row`, `array_col = 2 col + row % 2`, and points.

    Points are in spot spacings: `x = array_col / 2`, `y = row sqrt(3) / 2`,
    so distances are isotropic and a regular polygon stays regular.
    """
    index = np.arange(rows * columns)
    row = index // columns
    col = 2 * (index % columns) + row % 2
    points = np.column_stack([col / 2.0, row * np.sqrt(3.0) / 2.0])
    return row, col, points


def square_array(rows: int, columns: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A square grid at unit spacing: `array_row`, `array_col`, and points `(col, row)`.

    Integer coordinates, axis neighbours at distance 1, as Visium HD's bins
    and `port.extensions.adjacency.lattice_kind`'s `square` (#417, #569).
    """
    index = np.arange(rows * columns)
    row = index // columns
    col = index % columns
    points = np.column_stack([col, row]).astype(np.float64)
    return row, col, points


ARRAYS = {"hex": hex_array, "square": square_array}
"""`[array] kind` to its packing; `hex` is Visium's, `square` Visium HD's."""


def polygon(
    region: Region, center: np.ndarray, scale: float, rng: np.random.Generator
) -> np.ndarray:
    """`(vertices, 2)`: evenly spaced angles from a random phase, radii jittered."""
    angles = 2 * np.pi * np.arange(region.vertices) / region.vertices
    angles = angles + rng.uniform(0, 2 * np.pi)
    radii = (
        region.radius * scale * (1 + region.jitter * rng.uniform(-1, 1, angles.size))
    )
    return center + np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def clone_size(law: dict[str, Any], rng: np.random.Generator) -> int:
    """A clone's size in spots from `[layout.size]` (#544), at least one."""
    if law["law"] == "loguniform":
        low, high = np.log(float(law["minimum"])), np.log(float(law["maximum"]))
        return max(1, round(float(np.exp(rng.uniform(low, high)))))
    median, sigma = float(law["median"]), float(law["sigma"])
    return max(1, round(float(median * np.exp(sigma * rng.standard_normal()))))


def sized_polygon(
    region: Region,
    center: np.ndarray,
    scale: float,
    points: np.ndarray,
    target: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """`(vertices, claimed)`: the shape `polygon` draws, scaled to claim `target` spots.

    The phase and jitter are drawn once, then the radius is bisected: scaling
    a star-shaped polygon about its centre nests the spots it claims, so the
    count is monotone in the radius. The smallest radius claiming at least
    `target` is kept; a polygon clipped by the array's edge claims fewer, and
    the caller reads the realized count, not the target.
    """
    unit = polygon(region._replace(radius=1.0), np.zeros(2), 1.0, rng)
    low, high = 0.0, float(np.ptp(points, axis=0).max()) / scale

    def claimed(radius: float) -> np.ndarray:
        return _inside(center + unit * radius * scale, points)

    for _ in range(40):
        middle = 0.5 * (low + high)
        if int(claimed(middle).sum()) >= target:
            high = middle
        else:
            low = middle

    return center + unit * high * scale, claimed(high)


def _inside(vertices: np.ndarray, points: np.ndarray) -> np.ndarray:
    from matplotlib.path import Path as MplPath

    return np.asarray(MplPath(vertices).contains_points(points))


def layout(
    manifest: DrawManifest, points: np.ndarray, rng: np.random.Generator
) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    """Clone per spot of every slice, `-1` for normal, and each clone's polygon.

    The slices lie in one frame, each shifted by its `offset` in fractions of
    the array's extent, so slices that overlap image a common piece of tissue.
    Each clone is placed once in that frame: a region a slice states is
    placed where it says, relative to that slice; any other clone is drawn
    with its centre uniform over the part of the frame every slice listing it
    covers -- so a clone on two slices sits where they overlap and is imaged
    by both. A polygon that reaches onto another slice is imaged there too.

    With `overlap = false` a stated overlap -- a spot inside two polygons -- is
    refused, and a drawn polygon is redrawn until it claims no taken spot.
    With `overlap = true` a later region takes the spot.
    """
    lower, upper = points.min(axis=0), points.max(axis=0)
    extent = upper - lower
    scale = float(extent.min())
    origins = [lower + np.asarray(piece.offset) * extent for piece in manifest.slices]
    frames = [points - lower + origin for origin in origins]
    everything = np.concatenate(frames)

    stated: dict[str, tuple[Region, np.ndarray]] = {}
    for piece, origin in zip(manifest.slices, origins, strict=True):
        for region in piece.regions:
            if region.clone in stated:
                msg = f"{region.clone} has a region on more than one slice"
                raise ValueError(msg)
            stated[region.clone] = (region, origin)

    listed = [c for c in manifest.tumour if any(c in p.clones for p in manifest.slices)]
    listed += [c for c in stated if c not in listed]
    overlap = bool(manifest.layout["overlap"])
    size = manifest.layout.get("size")
    # NB drawn before any placement, in clone order, so a clone's size does
    #    not depend on how many placements another took.
    targets = {} if size is None else {c: clone_size(size, rng) for c in sorted(listed)}
    labels = np.full(everything.shape[0], -1, dtype=np.int64)
    shapes: dict[str, np.ndarray] = {}

    for clone in sorted(listed, key=lambda c: c not in stated):
        if clone in stated:
            region, origin = stated[clone]
            box = None
        else:
            region = Region(
                clone=clone,
                center=None,
                radius=float(manifest.layout["radius"]),
                vertices=int(manifest.layout["vertices"]),
                jitter=float(manifest.layout["jitter"]),
            )
            covering = [o for p, o in zip(manifest.slices, origins, strict=True)
                        if clone in p.clones]  # fmt: skip
            box = (np.max(covering, axis=0), np.min(covering, axis=0) + extent)
            if np.any(box[0] > box[1]):
                msg = f"{clone} is listed on slices whose arrays do not overlap"
                raise ValueError(msg)

        for _ in range(int(manifest.layout["max_placements"])):
            if box is None:
                center = origin + np.asarray(region.center, dtype=np.float64) * extent
            else:
                # NB anywhere the listing slices share: a polygon past an edge
                #    is clipped, so a clone may be smaller than its radius says.
                center = rng.uniform(box[0], box[1])
            if box is not None and size is not None:
                vertices, claimed = sized_polygon(
                    region, center, scale, everything, targets[clone], rng
                )
            else:
                vertices = polygon(region, center, scale, rng)
                claimed = _inside(vertices, everything)
            if overlap or not np.any(labels[claimed] >= 0):
                break
            if box is None:
                msg = f"region of {clone} overlaps another and [layout] overlap = false"
                raise ValueError(msg)
        else:
            msg = (
                f"no placement of {clone} in {manifest.layout['max_placements']} "
                "clears the others"
            )
            raise ValueError(msg)
        labels[claimed] = manifest.tumour.index(clone)
        shapes[clone] = vertices

    return list(np.split(labels, len(frames))), shapes


def barcodes(
    n_spots: int, length: int, suffix: str, rng: np.random.Generator
) -> np.ndarray:
    """`n_spots` distinct `length`-mers over `ACGT`, then `suffix` (Visium: 16, `-1`)."""
    out: dict[str, None] = {}
    while len(out) < n_spots:
        draws = rng.integers(0, 4, (n_spots, length))
        for row in draws:
            out.setdefault("".join("ACGT"[i] for i in row) + suffix)
            if len(out) == n_spots:
                break
    return np.array(list(out))


def _numeric(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def sample_ids(n_slices: int, n_bytes: int, rng: np.random.Generator) -> list[str]:
    """Distinct hexadecimal ids of `2 n_bytes` digits, none of them a number.

    `cnaster` reads the sample sheet without a string dtype, so an id such as
    `823468e7` arrives as the float `8.23e74` and `12345678` as an integer,
    and joining the sample names then raises (#544). Such an id is redrawn;
    an id that parses as no number is drawn as before.
    """
    out: list[str] = []
    while len(out) < n_slices:
        candidate = rng.bytes(n_bytes).hex()
        if candidate not in out and not _numeric(candidate):
            out.append(candidate)
    return out


# --- phase ---------------------------------------------------------------------


MAP_CACHE = Path(os.environ.get("PORT_CACHE", Path.home() / ".cache" / "port"))
"""Where `genetic_map` keeps its parsed maps, one Parquet file per map (#549)."""


def _map_digest(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def _parse_map(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    table = pd.read_csv(
        path, sep="\t", usecols=["chrom", "pos", "pos_cm"], engine="pyarrow"
    )
    out = {}
    for contig, group in table.groupby("chrom", sort=False):
        pos, cm = group["pos"].to_numpy(), group["pos_cm"].to_numpy()
        if np.any(np.diff(pos) < 0):  # NB chrX, once; the autosomes are sorted.
            order = np.argsort(pos, kind="stable")
            pos, cm = pos[order], cm[order]
        out[str(contig).removeprefix("chr")] = (pos, cm)
    return out


def _write_map(out: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    """One uncompressed row group per contig; the contigs' order in the schema."""
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    pos, cm = next(iter(out.values()))
    schema = pa.schema(
        [
            ("pos", pa.from_numpy_dtype(pos.dtype)),
            ("cm", pa.from_numpy_dtype(cm.dtype)),
        ],
        metadata={"contigs": json.dumps(list(out))},
    )
    with pq.ParquetWriter(path, schema, compression="none") as writer:
        for pos, cm in out.values():
            writer.write_table(pa.table({"pos": pos, "cm": cm}, schema=schema))


def _read_map(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    import json

    import pyarrow.parquet as pq

    handle = pq.ParquetFile(path, memory_map=True)
    contigs = json.loads(handle.schema_arrow.metadata[b"contigs"])
    out = {}
    for group, contig in enumerate(contigs):
        table = handle.read_row_group(group)
        out[contig] = (table["pos"].to_numpy(), table["cm"].to_numpy())
    return out


@functools.lru_cache(maxsize=2)
def genetic_map(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """`chrom` (no `chr`) to its sorted positions and cM.

    Parsed once per map file and kept in `MAP_CACHE` as Parquet under the
    file's SHA-256, so a changed map is parsed again: the text parse costs
    1.2-2.8 s and +0.62 GB peak RSS to hold 54 MB (#549). The cache is a copy
    of the parse, never a source: deleting it changes nothing but the time.
    """
    cached = MAP_CACHE / f"genetic_map-{_map_digest(path)}.parquet"
    if cached.exists():
        return _read_map(cached)

    out = _parse_map(path)
    cached.parent.mkdir(parents=True, exist_ok=True)
    partial = cached.with_suffix(f".{os.getpid()}.partial")
    _write_map(out, partial)
    partial.replace(cached)
    return out


UNITS = {"morgan": 100.0, "centimorgan": 1.0}
"""`d = cM / UNITS[unit]`: Haldane's map in Morgans, or Numbat's and `cnaster`'s in cM."""


def switch_probabilities(
    chromosome: np.ndarray,
    position: np.ndarray,
    gmap: dict[str, tuple[np.ndarray, np.ndarray]],
    nu: float,
    unit: str,
) -> np.ndarray:
    """`p[j]`: the phase switches between SNP `j - 1` and `j`; 0 at each chromosome's first.

    `p = (1 - exp(-2 nu d)) / 2`, `d` the map distance in `unit`: the
    two-state Markov chain whose `1 - 2p` multiplies along a chromosome, so
    the SNP-level draw composes exactly to the same law over any bin.
    """
    p = np.zeros(position.size)
    for name in np.unique(chromosome):
        at = np.flatnonzero(chromosome == name)
        pos, cm = gmap[str(name)]
        distance = np.diff(np.interp(position[at], pos, cm)) / UNITS[unit]
        p[at[1:]] = 0.5 * (1 - np.exp(-2 * nu * distance))
    return p


# --- the draw ------------------------------------------------------------------


@dataclass
class Drawn:
    """What `draw` wrote, and what a test reads back."""

    root: Path
    realizations: list[Path]
    """One complete sample directory per `[sample] realizations`, `r<k>`."""
    sample_ids: list[str]
    clones: tuple[str, ...]
    labels: list[np.ndarray]
    tree: CloneTree
    switched: list[np.ndarray]
    """Each realization's phase: True where `A` and `B` are exchanged."""
    switch_p: np.ndarray

    @property
    def path(self) -> Path:
        """The first realization."""
        return self.realizations[0]


def _laws(manifest: DrawManifest) -> dict[str, Law]:
    from port.sim.normal_fit import read_coverage

    return read_coverage(manifest.resolve(manifest.reference["coverage"]))


def _lognormal(law: Law, size: int, rng: np.random.Generator) -> np.ndarray:
    if law.family != "lognormal":
        msg = f"a spot law is lognormal here, not {law.family}"
        raise ValueError(msg)
    drawn = rng.lognormal(law.parameters["mu"], law.parameters["sigma"], size)
    return np.maximum(np.rint(drawn), 1).astype(np.int64)


def _depth(
    manifest: DrawManifest, clones: tuple[str, ...], copies: np.ndarray
) -> np.ndarray:
    """`(n_loci, n_clones)` the admixture law's depth factor; `normal` 1."""
    total = copies.sum(axis=2).astype(np.float64)
    factor = np.ones_like(total)
    for label, clone in enumerate(clones):
        if clone != "normal":
            tumour = 1.0 - manifest.normal_frac(clone)
            factor[:, label] = tumour * total[:, label] / 2.0 + (1.0 - tumour)
    return factor


def _snps(manifest: DrawManifest) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = load_ids(manifest.resolve(manifest.reference["snps"]))
    chromosome = np.array([s.split("_")[0] for s in ids])
    position = np.array([int(s.split("_")[1]) for s in ids])
    return ids, chromosome, position


def draw(
    manifest: DrawManifest, into: Path | None = None, *, resources: Path | None = None
) -> Drawn:
    """Draw `manifest` and write each realization as `<into>/<name>/r<k>/`.

    `realize` with somewhere to write, run to the end.
    """
    root = (
        into if into is not None else manifest.resolve(manifest.sample["output"])
    ) / manifest.name
    realized = list(realize(manifest, root, resources=resources))
    truth = realized[0].truth
    return Drawn(
        root,
        [r.path for r in realized if r.path is not None],
        truth.sample_ids,
        truth.clones,
        truth.labels,
        truth.tree,
        [r.phase for r in realized],
        truth.p_switch,
    )


class Truth(NamedTuple):
    """What every realization of a manifest shares: the clones and where they are."""

    clones: tuple[str, ...]
    tree: CloneTree
    profile: pd.DataFrame
    """`truth_acn_profile.tsv`: `chr start end`, then `A`, `B` per clone."""
    labels: list[np.ndarray]
    """Per slice, each spot's clone, indexing `clones` (`normal` is 0)."""
    sample_ids: list[str]
    barcodes: list[np.ndarray]
    """Per slice, `{barcode}_{sample_id}`."""
    rows: np.ndarray
    cols: np.ndarray
    genes: np.ndarray
    snp_ids: np.ndarray
    p_switch: np.ndarray
    """Per SNP, the probability of a phase switch from the one before."""


@dataclass
class Realized:
    """One realization: the truth it shares, and what was drawn for it."""

    index: int
    truth: Truth
    counts: list[Any]
    """Per slice, spots x genes CSR UMI."""
    a: Any
    """Spots (every slice, in order) x SNPs CSR: the written `A` reads."""
    b: Any
    phase: np.ndarray
    """Per SNP: True where the written `A` and `B` are exchanged."""
    path: Path | None = None
    """Where it was written, if it was."""


def realize(
    manifest: DrawManifest,
    into: Path | None = None,
    *,
    resources: Path | None = None,
) -> Iterator[Realized]:
    """Each realization of `manifest` in turn, written under `into/r<k>/` if given.

    The clones, their layout and the barcodes are drawn once. Each of
    `[sample] realizations` then draws, from its own stream, the counts and
    the phase; a
    realization does not change when more are asked for. Nothing is held
    between realizations, so a population of them can be streamed through
    an analysis without writing one (`into=None`).
    """
    tree_rng, layout_rng, id_rng = (
        np.random.default_rng(s) for s in np.random.SeedSequence(manifest.seed).spawn(3)
    )
    realization_seeds = np.random.SeedSequence([manifest.seed, 1]).spawn(
        int(manifest.sample["realizations"])
    )
    resources = resources or manifest.resources()

    tree = draw_tree(manifest, tree_rng)
    clones = ("normal", *manifest.tumour)
    laws = _laws(manifest)

    baseline = pd.read_csv(
        manifest.resolve(manifest.reference["baseline"]), sep="\t", comment="#"
    )
    lam = baseline["lambda"].to_numpy()
    gene_chrom = baseline["chrom"].astype(str).str.removeprefix("chr").to_numpy()
    gene_pos = ((baseline["cdsStart"] + baseline["cdsEnd"]) // 2).to_numpy()
    depth_factor = _depth(
        manifest, clones, clone_copies(tree, clones, gene_chrom, gene_pos)
    )
    gene_weights = lam[:, None] * depth_factor
    library = lam @ depth_factor
    kappa = float(manifest.model["dirichlet_concentration"])
    sample_counts = COUNT_SAMPLERS[manifest.model["counts_sampler"]]

    snp_ids, snp_chrom, snp_pos = _snps(manifest)
    snp_copies = clone_copies(tree, clones, snp_chrom, snp_pos)
    share = np.stack(
        [
            allele_share(
                snp_copies[:, label, 0].astype(np.float64),
                snp_copies[:, label, 1].astype(np.float64),
                manifest.normal_frac(clone),
                manifest.model["admixture"],
            )
            for label, clone in enumerate(clones)
        ]
    )
    if manifest.model["snp_depth_follows_copies"]:
        msg = "SNP entries are drawn independently of copies (#455)"
        raise ValueError(msg)
    snp_entries = snp_law(
        nodes(laws["spot_snp_umi"]),
        laws["snp_total"].parameters["dispersion"],
        float(manifest.model["snp_dispersion"]),
        snp_ids.size,
    )
    p_switch = switch_probabilities(
        snp_chrom,
        snp_pos,
        genetic_map(resources / manifest.reference["genetic_map"]),
        float(manifest.phasing["nu"]),
        manifest.phasing["unit"],
    )
    if not manifest.phasing["switch_errors"]:
        p_switch = np.zeros_like(p_switch)

    rows, cols, points = ARRAYS[manifest.array["kind"]](
        int(manifest.array["rows"]), int(manifest.array["columns"])
    )
    codes = manifest.barcodes
    whitelist = barcodes(rows.size, int(codes["length"]), codes["suffix"], id_rng)
    ids = sample_ids(len(manifest.slices), int(codes["sample_id_bytes"]), id_rng)
    labels = [
        lab + 1 for lab in layout(manifest, points, layout_rng)[0]
    ]  # NB `normal` is 0.
    combined = [np.array([f"{b}_{sid}" for b in whitelist]) for sid in ids]
    profile = truth_profile(tree, clones, manifest.genome["chromosome_lengths"])
    shared = Truth(clones, tree, profile, labels, ids, combined, rows, cols,
                   baseline["gene"].to_numpy(), snp_ids, p_switch)  # fmt: skip

    width = len(str(len(realization_seeds) - 1))
    for k, seed in enumerate(realization_seeds):
        count_seed, phase_seed = seed.spawn(2)
        rng = np.random.default_rng(count_seed)
        switched = phased(p_switch, snp_chrom, np.random.default_rng(phase_seed))
        counts, a_blocks, b_blocks = [], [], []

        for lab in labels:
            totals = np.rint(_lognormal(laws["spot_umi"], lab.size, rng) * library[lab])
            counts.append(sample_counts(totals, gene_weights, lab, kappa, rng))
            trials = independent(snp_entries, (lab.size, snp_ids.size), rng)
            a, b = _alleles(manifest, trials, share, switched, lab, rng)
            a_blocks.append(a)
            b_blocks.append(b)

        import scipy.sparse

        realized = Realized(
            k, shared, counts,
            scipy.sparse.vstack(a_blocks, format="csr").astype(np.int64),
            scipy.sparse.vstack(b_blocks, format="csr").astype(np.int64),
            switched,
        )  # fmt: skip
        if into is not None:
            realized.path = write(
                realized, into / f"r{k:0{width}d}", manifest, resources
            )
        yield realized


def write(
    realized: Realized, out: Path, manifest: DrawManifest, resources: Path
) -> Path:
    """A realization as a complete sample directory for `run_cnaster(_port)`."""
    t = realized.truth
    (out / "snp").mkdir(parents=True, exist_ok=True)
    for counts, lab, sid, names in zip(realized.counts, t.labels, t.sample_ids,
                                       t.barcodes, strict=True):  # fmt: skip
        _write_slice(out / sid, counts, lab, names, t.rows, t.cols, t.genes, t.clones)
    _write_snps(out / "snp", t.barcodes, t.snp_ids, realized.a, realized.b)

    pd.concat(
        pd.DataFrame(
            {
                "labels": np.asarray(t.clones)[lab],
                "x": t.rows,
                "y": t.cols,
                "sample_id": sid,
            },
            index=pd.Index(names, name="barcode"),
        )
        for lab, sid, names in zip(t.labels, t.sample_ids, t.barcodes, strict=True)
    ).to_csv(out / "truth_clone_labels.tsv", sep="\t")
    t.profile.to_csv(out / "truth_acn_profile.tsv", sep="\t", index=False)
    tree_table(t.tree).to_csv(out / "truth_tree.tsv", sep="\t", index=False)
    # NB the realized phase: True where the written `A` and `B` are exchanged,
    #    in `unique_snp_ids.npy`'s order; a switch is a change within a chromosome.
    np.save(out / "truth_phase.npy", realized.phase)
    write_inputs(manifest, out, t.sample_ids, resources)
    write_manifest(manifest, out, resources, realized.index)
    return out


def phased(
    p_switch: np.ndarray, chromosome: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """One draw of the phase: switches at `p_switch`, their parity per chromosome."""
    switches = rng.random(p_switch.size) < p_switch
    switched = np.zeros(p_switch.size, dtype=bool)
    for name in np.unique(chromosome):
        at = np.flatnonzero(chromosome == name)
        switched[at] = np.cumsum(switches[at]) % 2 == 1
    return switched


COMPRESSION_LEVEL = 1
"""zlib level for the written counts: 2.8 s of 17.7 at level 6 on 2 x 3,000 spots."""


def save_npz(path: Path, matrix: Any) -> None:
    """`scipy.sparse.save_npz`'s format, deflated at `COMPRESSION_LEVEL`.

    The same members `load_npz` reads -- `indices`, `indptr`, `format`,
    `shape`, `data` -- written with `zipfile` at the level set, which
    `numpy.savez_compressed` does not expose.
    """
    import zipfile

    members = {
        "indices": matrix.indices,
        "indptr": matrix.indptr,
        "format": np.array(matrix.format.encode("ascii")),
        "shape": np.array(matrix.shape),
        "data": matrix.data,
    }
    with zipfile.ZipFile(
        path, "w", zipfile.ZIP_DEFLATED, compresslevel=COMPRESSION_LEVEL
    ) as archive:
        for name, array in members.items():
            with archive.open(f"{name}.npy", "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, np.asanyarray(array))


def _normalized(weights: np.ndarray) -> np.ndarray:
    """Each clone's column summing to 1: the library size is held per spot."""
    return np.asarray(weights / weights.sum(axis=0, keepdims=True))


def _write_slice(
    out: Path,
    counts: Any,
    labels: np.ndarray,
    names: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    genes: np.ndarray,
    clones: tuple[str, ...],
) -> None:
    """One slice's `filtered_feature_bc_matrix.h5ad` and tissue positions."""
    import anndata

    (out / "spatial").mkdir(parents=True, exist_ok=True)
    anndata.AnnData(
        X=counts,
        obs=pd.DataFrame({"labels": np.asarray(clones)[labels]}, index=names),
        var=pd.DataFrame(index=genes),
    ).write_h5ad(
        out / "filtered_feature_bc_matrix.h5ad",
        compression="gzip",
        compression_opts=COMPRESSION_LEVEL,
    )
    pd.DataFrame({0: names, 1: 1, 2: rows, 3: cols, 4: rows, 5: cols}).to_csv(
        out / "spatial" / "tissue_positions_list.csv", header=False, index=False
    )


def _alleles(
    manifest: DrawManifest,
    trials: Any,
    share: np.ndarray,
    switched: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> tuple[Any, Any]:
    """One slice's written `A` and `B` reads: the planted haplotype's of `trials`, then phase."""
    import scipy.sparse

    trials = trials.tocoo()

    p = share[labels[trials.row], trials.col]
    rho = float(manifest.model["bb_overdispersion"])
    if rho > 0:
        inner = (p > 0) & (p < 1)
        scale = 1 / rho - 1
        p = p.copy()
        p[inner] = rng.beta(p[inner] * scale, (1 - p[inner]) * scale)
    first = rng.binomial(trials.data, p)
    second = trials.data - first
    flip = switched[trials.col]
    shape = trials.shape
    at = (trials.row, trials.col)
    return (
        scipy.sparse.csr_matrix((np.where(flip, second, first), at), shape=shape),
        scipy.sparse.csr_matrix((np.where(flip, first, second), at), shape=shape),
    )


def _write_snps(
    out: Path, combined: list[np.ndarray], snp_ids: np.ndarray, a: Any, b: Any
) -> None:
    (out / "barcodes.txt").write_text("\n".join(np.concatenate(combined)) + "\n")
    np.save(out / "unique_snp_ids.npy", snp_ids.astype(object))
    for name, matrix in (("A", a), ("B", b)):
        save_npz(out / f"cell_snp_{name}allele.npz", matrix)


def write_manifest(
    manifest: DrawManifest, out: Path, resources: Path, realization: int
) -> Path:
    """`manifest.json`: the tables this realization was drawn from, paths resolved.

    What `port.sim.analysis` reads the baseline, the laws and the genome from,
    so a realization directory is self-describing (#452).
    """
    import json

    tables: dict[str, Any] = {
        name: dict(values) for name, values in manifest.tables.items()
    }
    reference = tables["reference"]
    for key in ("baseline", "coverage", "snps"):
        reference[key] = str(manifest.resolve(reference[key]).resolve())
    reference["resources"] = str(resources.resolve())
    tables["sample"]["realization"] = realization
    tables["slice"] = [
        {"clones": list(p.clones), "offset": list(p.offset)} for p in manifest.slices
    ]
    path = out / "manifest.json"
    path.write_text(json.dumps(tables, indent=1, default=str))
    return path


def write_inputs(
    manifest: DrawManifest, out: Path, ids: list[str], resources: Path
) -> Path:
    """`sample_sheet.tsv` and `config.yaml` for `run_cnaster` / `run_cnaster_port`.

    The configuration is `[config] base`, a `run_cnaster` YAML, with each
    `[config.<section>]` table of the manifest laid over its section; then its
    paths pointed here and its references at `[reference] resources`. Paths
    are absolute, so either runs from any directory.
    """
    import yaml

    pd.DataFrame(
        {
            "bam": ["unused.bam"] * len(ids),
            "sample_id": ids,
            "spaceranger_dir": [str((out / sid).resolve()) for sid in ids],
            "snp_dir": [str((out / "snp").resolve())] * len(ids),
        }
    ).to_csv(out / "sample_sheet.tsv", sep="\t", index=False)

    config = manifest.config
    document: dict[str, Any] = yaml.safe_load(
        manifest.resolve(config.pop("base")).read_text()
    )
    for section, values in config.items():
        document.setdefault(section, {}).update(values)

    reference = manifest.reference
    document["paths"] = {
        "sample_sheet": str((out / "sample_sheet.tsv").resolve()),
        "output_dir": str((out / "output").resolve()),
        "perf_path": str((out / "cnaster.perf").resolve()),
    }
    document["references"] |= {
        "geneticmap_file": str(resources / reference["genetic_map"]),
        "hgtable_file": str(resources / reference["gene_table"]),
        "filtergenelist_file": str(resources / reference["filter_genes"]),
        "filterregion_file": str(resources / reference["filter_regions"]),
    }
    path = out / "config.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="a version-3 `.toml` manifest")
    parser.add_argument("--into", default=None, help="default: the manifest's `output`")
    parser.add_argument("--seed", type=int, default=None, help="override the seed")
    arguments = parser.parse_args(argv)

    manifest = read_manifest(arguments.manifest)
    if arguments.seed is not None:
        from dataclasses import replace

        seed = {"sample": {"seed": arguments.seed}}
        manifest = replace(manifest, tables=_merge(manifest.tables, seed))
    drawn = draw(manifest, None if arguments.into is None else Path(arguments.into))
    print(
        f"wrote {len(drawn.realizations)} realizations of {len(drawn.sample_ids)} "
        f"slices to {drawn.root}; configs {drawn.root}/r*/config.yaml"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
