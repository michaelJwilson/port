"""A generative simulation manifest, as TOML (#382).

Where `port.sim.manifest` records what a run's inputs *were* (#116), this
states what a simulated sample *is to be*: its sizes, its layout, its clones
and their planted `(A, B)` events, the generative model and the coverage laws
it draws from. `port.sim.run_sim_gen` turns one into a sample directory in
the format of CalicoST's `sim/<name>/`.

The file reads like `pyproject.toml`, and stdlib `tomllib` parses it:

    version = 2
    [sample]    name, seed, output
    [size]      n_spots, lattice rows x columns, n_genes, n_snps,
                n_chromosomes, n_segments, chromosome_lengths
    [layout]    genes, snps, labels, gene_table
    [model]     admixture, normal_frac per clone, dispersions
    [model.fitted]  what the source sample measured, per clone
    [coverage.<law>]  family, parameters, and the fit's KS statistic
    [[clone]]   name, spots, center, events = [{chr, start, end, A, B}]

Paths in `[layout]` and `[sample] output` are relative to the manifest's
directory, so a manifest travels with the samples it names.

**The admixture law.** `admixture = "read"` mixes reads: a tumour spot of
clone `c` with normal fraction `f` has haplotype-A share
`(1 - f) A / (A + B) + f / 2` and depth factor `(1 - f)(A + B) / 2 + f`.
`"cell"` mixes cells: share `((1 - f) A + f) / ((1 - f)(A + B) + 2 f)`.
Measured on CalicoST's `easy` and `hard` samples, the read law fits better
in 5 of 6 clones (`fit_normal_frac`), which is why it is the default.
"""

from __future__ import annotations

import itertools
import math
import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

MANIFEST_VERSION = 2
"""The TOML schema. The YAML record of #116 is version 1 and a different job."""

ADMIXTURE_LAWS = ("read", "cell")

SNP_ID_SUFFIX = "N_N"
"""CalicoST's simulated SNP ids carry no alleles: `1_826354_N_N`."""


@dataclass(frozen=True)
class Event:
    """One planted `(A, B)` over `[start, end)` of a chromosome, for one clone."""

    chromosome: str
    start: int
    end: int
    a: int
    b: int


@dataclass(frozen=True)
class Clone:
    """A clone: how many spots it has, where, and what it carries.

    `normal` carries no events and is never admixed; every other clone is
    `(1, 1)` wherever no event covers it.
    """

    name: str
    spots: int
    events: tuple[Event, ...] = ()
    center: tuple[float, float] | None = None


@dataclass(frozen=True)
class Law:
    """A fitted distribution: its family, its parameters and how well it fit.

    `lognormal` carries `mu` and `sigma` of the log; `negative_binomial`
    carries `mean` and `dispersion` (`var = mean + dispersion mean^2`).
    `ks` is the Kolmogorov-Smirnov statistic of the fit on the values it was
    fitted to, and `ks_alternative` the other family's, so the choice is
    evidence rather than assumption. `expressed` is the share of nonzero
    values, for a law fitted to the positive part only.
    """

    family: str
    parameters: dict[str, float]
    ks: float | None = None
    ks_alternative: float | None = None
    n: int | None = None
    expressed: float | None = None


@dataclass(frozen=True)
class Model:
    """The generative model a sample is drawn from, and what a fit measured."""

    admixture: str = "read"
    normal_frac: dict[str, float] = field(default_factory=dict)
    """Per tumour clone; a clone absent here is pure (0)."""
    nb_dispersion: float = 0.0
    """Gene-level Gamma modulation of each spot's profile; 0 is multinomial."""
    bb_overdispersion: float = 0.0
    """Beta-binomial intra-class correlation `rho`; 0 is binomial."""
    snp_dispersion: float = 0.0
    """Gamma modulation of each spot-SNP entry's read rate; 0 is Poisson."""
    snp_depth_follows_copies: bool = False
    """Whether a SNP's reads scale with its total copies. CalicoST's do not."""
    fitted: dict[str, dict[str, float]] = field(default_factory=dict)
    """`[model.fitted]`: e.g. `normal_frac` per clone, measured on the source."""


@dataclass(frozen=True)
class SimManifest:
    """Everything `run_sim_gen` needs to write one sample."""

    name: str
    seed: int
    output: str
    size: dict[str, Any]
    layout: dict[str, str]
    model: Model
    clones: tuple[Clone, ...]
    coverage: dict[str, Law]
    source: str = ""
    root: Path = Path()
    """The manifest's directory; relative paths resolve against it."""

    def resolve(self, value: str) -> Path:
        """A `[layout]` or output path, with `$VARS` expanded, from `root`."""
        return self.root / os.path.expandvars(value)

    def normal_frac(self, clone: str) -> float:
        return float(self.model.normal_frac.get(clone, 0.0))

    @property
    def tumour(self) -> tuple[Clone, ...]:
        return tuple(c for c in self.clones if c.name != "normal")


def segments(manifest: SimManifest) -> list[tuple[str, int, int]]:
    """`(chr, start, end)` rows of the truth profile: every event breakpoint."""
    lengths = manifest.size["chromosome_lengths"]
    rows: list[tuple[str, int, int]] = []

    for index, length in enumerate(lengths, start=1):
        chromosome = str(index)
        cuts = {0, int(length)}
        for clone in manifest.clones:
            for event in clone.events:
                if event.chromosome == chromosome:
                    cuts |= {event.start, event.end}
        ordered = sorted(cuts)
        rows += [(chromosome, s, e) for s, e in itertools.pairwise(ordered)]

    return rows


def copies_at(
    manifest: SimManifest, chromosome: np.ndarray, position: np.ndarray
) -> np.ndarray:
    """`(n_positions, n_clones, 2)` planted `(A, B)`; `(1, 1)` where no event."""
    copies = np.ones((position.size, len(manifest.clones), 2), dtype=np.int64)
    query = np.asarray(chromosome).astype(str)

    for label, clone in enumerate(manifest.clones):
        for event in clone.events:
            covered = (query == event.chromosome) & (position >= event.start)
            covered &= position < event.end
            copies[covered, label] = (event.a, event.b)

    return copies


# --- reading and writing ----------------------------------------------------


def read_manifest(path: str | Path) -> SimManifest:
    """Parse a TOML manifest, refusing a version or a size it contradicts."""
    source = Path(path)
    document = tomllib.loads(source.read_text())
    version = int(document.get("version", 0))

    if version != MANIFEST_VERSION:
        msg = f"manifest version {version}, this reads {MANIFEST_VERSION}"
        raise ValueError(msg)

    return from_document(document, source.parent)


def from_document(document: dict[str, Any], root: Path = Path()) -> SimManifest:
    """A manifest from an already parsed TOML document."""
    sample = document["sample"]
    model = dict(document.get("model", {}))
    fitted = model.pop("fitted", {})

    if model.get("admixture", "read") not in ADMIXTURE_LAWS:
        msg = f"admixture {model['admixture']!r}; one of {ADMIXTURE_LAWS}"
        raise ValueError(msg)

    clones = tuple(
        Clone(
            name=str(c["name"]),
            spots=int(c["spots"]),
            events=tuple(
                Event(str(e["chr"]), int(e["start"]), int(e["end"]), e["A"], e["B"])
                for e in c.get("events", [])
            ),
            center=None if "center" not in c else (c["center"][0], c["center"][1]),
        )
        for c in document["clone"]
    )
    coverage = {
        name: Law(
            family=str(table["family"]),
            parameters={
                k: float(v)
                for k, v in table.items()
                if k not in {"family", "ks", "ks_alternative", "n", "expressed"}
            },
            ks=table.get("ks"),
            ks_alternative=table.get("ks_alternative"),
            n=table.get("n"),
            expressed=table.get("expressed"),
        )
        for name, table in document.get("coverage", {}).items()
    }
    manifest = SimManifest(
        name=str(sample["name"]),
        seed=int(sample.get("seed", 0)),
        output=str(sample.get("output", ".")),
        source=str(sample.get("source", "")),
        size=dict(document["size"]),
        layout={k: str(v) for k, v in document.get("layout", {}).items()},
        model=Model(**model, fitted=fitted),
        clones=clones,
        coverage=coverage,
        root=root,
    )
    _check(manifest)
    return manifest


def _check(manifest: SimManifest) -> None:
    """Sizes the manifest states must be the sizes its parts imply."""
    size = manifest.size
    stated = int(size["n_spots"])
    counted = sum(c.spots for c in manifest.clones)
    lattice = int(size["rows"]) * int(size["columns"])
    problems = []

    if counted != stated:
        problems.append(f"clones hold {counted} spots, [size] states {stated}")
    if lattice != stated:
        problems.append(f"lattice holds {lattice} spots, [size] states {stated}")
    if len(size["chromosome_lengths"]) != int(size["n_chromosomes"]):
        problems.append("chromosome_lengths disagrees with n_chromosomes")
    if "n_segments" in size and len(segments(manifest)) != int(size["n_segments"]):
        problems.append(
            f"events imply {len(segments(manifest))} segments, "
            f"[size] states {size['n_segments']}"
        )
    unknown = set(manifest.model.normal_frac) - {c.name for c in manifest.tumour}
    if unknown:
        problems.append(f"normal_frac for unknown clones {sorted(unknown)}")

    if problems:
        raise ValueError("; ".join(problems))


def with_normal_frac(manifest: SimManifest, value: float | str) -> SimManifest:
    """The same manifest with every tumour clone at `value`, or at `fitted`."""
    if value == "fitted":
        chosen = dict(manifest.model.fitted["normal_frac"])
    else:
        chosen = {c.name: float(value) for c in manifest.tumour}
    return replace(manifest, model=replace(manifest.model, normal_frac=chosen))


def to_document(manifest: SimManifest) -> dict[str, Any]:
    """The manifest as the nested dict its TOML parses to."""
    model: dict[str, Any] = {
        "admixture": manifest.model.admixture,
        "nb_dispersion": manifest.model.nb_dispersion,
        "bb_overdispersion": manifest.model.bb_overdispersion,
        "snp_dispersion": manifest.model.snp_dispersion,
        "snp_depth_follows_copies": manifest.model.snp_depth_follows_copies,
        "normal_frac": dict(manifest.model.normal_frac),
    }
    if manifest.model.fitted:
        model["fitted"] = {k: dict(v) for k, v in manifest.model.fitted.items()}

    coverage: dict[str, Any] = {}
    for name, law in manifest.coverage.items():
        table: dict[str, Any] = {"family": law.family, **law.parameters}
        for key in ("ks", "ks_alternative", "n", "expressed"):
            if getattr(law, key) is not None:
                table[key] = getattr(law, key)
        coverage[name] = table

    clones = []
    for clone in manifest.clones:
        entry: dict[str, Any] = {"name": clone.name, "spots": clone.spots}
        if clone.center is not None:
            entry["center"] = list(clone.center)
        entry["events"] = [
            {"chr": e.chromosome, "start": e.start, "end": e.end, "A": e.a, "B": e.b}
            for e in clone.events
        ]
        clones.append(entry)

    sample: dict[str, Any] = {
        "name": manifest.name,
        "seed": manifest.seed,
        "output": manifest.output,
    }
    if manifest.source:
        sample["source"] = manifest.source

    return {
        "version": MANIFEST_VERSION,
        "sample": sample,
        "size": dict(manifest.size),
        "layout": dict(manifest.layout),
        "model": model,
        "coverage": coverage,
        "clone": clones,
    }


def to_toml(manifest: SimManifest, header: str = "") -> str:
    """Write the manifest as TOML; `tomllib` reads it back to the same dict.

    stdlib reads TOML and does not write it, and the schema is small enough
    that a writer for it is shorter than a dependency.
    """
    document = to_document(manifest)
    lines = [f"# {line}".rstrip() for line in header.splitlines()]
    lines += [f"version = {document.pop('version')}", ""]
    clones = document.pop("clone")

    for name, table in document.items():
        _table(lines, name, table)

    for clone in clones:
        lines.append("[[clone]]")
        for key, value in clone.items():
            if key == "events" and value:
                lines.append("events = [")
                lines += [f"    {_value(event)}," for event in value]
                lines.append("]")
            else:
                lines.append(f"{key} = {_value(value)}")
        lines.append("")

    return "\n".join(lines)


def _table(lines: list[str], name: str, table: dict[str, Any]) -> None:
    """`[name]` with its scalars, then each subtable as `[name.sub]`."""
    nested = {k: v for k, v in table.items() if _is_subtable(v, name)}
    scalars = [
        f"{key} = {_value(value)}" for key, value in table.items() if key not in nested
    ]
    if scalars or not nested:
        lines += [f"[{name}]", *scalars, ""]
    for key, value in nested.items():
        _table(lines, f"{name}.{key}", value)


def _is_subtable(value: Any, parent: str) -> bool:
    """A dict of dicts is a table; a flat dict stays inline (`normal_frac`)."""
    if not isinstance(value, dict):
        return False
    return parent == "coverage" or any(isinstance(v, dict) for v in value.values())


def _value(value: Any) -> str:
    """One TOML value: string, bool, int, float, list or inline table."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        number = float(value)
        if math.isnan(number):
            return "nan"
        if math.isinf(number):
            return "inf" if number > 0 else "-inf"
        return repr(number)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_value(v) for v in value) + "]"
    if isinstance(value, dict):
        inner = ", ".join(f"{_key(k)} = {_value(v)}" for k, v in value.items())
        return "{ " + inner + " }" if inner else "{}"
    msg = f"no TOML form for {type(value).__name__}"
    raise TypeError(msg)


def _key(key: str) -> str:
    bare = key.replace("_", "").replace("-", "").isalnum()
    return key if bare else _value(key)


# --- fitting a manifest to a CalicoST sample -----------------------------------


def fit_lognormal(values: np.ndarray) -> tuple[float, float, float]:
    """`mu`, `sigma` of `log(values > 0)`, and the KS statistic of the fit."""
    from scipy.stats import kstest

    logs = np.log(np.asarray(values, dtype=np.float64)[np.asarray(values) > 0])
    mu, sigma = float(logs.mean()), float(logs.std(ddof=1))
    return mu, sigma, float(kstest((logs - mu) / sigma, "norm").statistic)


def fit_negative_binomial(values: np.ndarray) -> tuple[float, float, float]:
    """`mean`, `dispersion` by moments, and the KS statistic on the counts.

    The KS statistic of a discrete law is the largest gap between the
    empirical and fitted CDFs at the observed support; a Poisson where the
    moments are underdispersed.
    """
    from scipy.stats import nbinom, poisson

    counts = np.sort(np.asarray(values, dtype=np.float64))
    mean = float(counts.mean())
    dispersion = max((float(counts.var(ddof=1)) - mean) / mean**2, 0.0)
    support = np.unique(counts)
    empirical = np.searchsorted(counts, support, side="right") / counts.size

    if dispersion > 0:
        number = 1.0 / dispersion
        fitted = nbinom.cdf(support, number, number / (number + mean))
    else:
        fitted = poisson.cdf(support, mean)

    return mean, dispersion, float(np.max(np.abs(empirical - fitted)))


def _counted(values: np.ndarray, prefer: str) -> Law:
    """Both families fitted; `prefer` is the one the manifest draws from."""
    mu, sigma, ks_log = fit_lognormal(values)
    mean, dispersion, ks_nb = fit_negative_binomial(values)
    lognormal = Law("lognormal", {"mu": mu, "sigma": sigma}, ks_log, ks_nb)
    negative = Law(
        "negative_binomial", {"mean": mean, "dispersion": dispersion}, ks_nb, ks_log
    )
    chosen = lognormal if prefer == "lognormal" else negative
    return replace(chosen, n=int(np.asarray(values).size))


def fit_coverage(path: Path) -> dict[str, Law]:
    """The `[coverage]` laws of a sample directory in `sim/<name>/`'s format.

    - `spot_umi`: per-spot total UMI;
    - `spot_snp_umi`: per-spot total SNP reads, `A + B` over SNPs;
    - `snp_total`: per-SNP total reads over spots;
    - `gene_profile`: each gene's share of all UMI, over expressed genes.

    Each carries the family with the smaller KS statistic on CalicoST's
    samples, and the other family's statistic beside it.
    """
    import anndata
    import scipy.sparse

    assay = anndata.read_h5ad(path / "filtered_feature_bc_matrix.h5ad")
    counts = scipy.sparse.csr_matrix(assay.X)
    trials = scipy.sparse.load_npz(path / "cell_snp_Aallele.npz").tocsr()
    trials = trials + scipy.sparse.load_npz(path / "cell_snp_Ballele.npz").tocsr()

    per_gene = np.asarray(counts.sum(axis=0)).ravel().astype(np.float64)
    profile = _counted(per_gene / per_gene.sum(), "lognormal")

    return {
        "spot_umi": _counted(np.asarray(counts.sum(axis=1)).ravel(), "lognormal"),
        "spot_snp_umi": _counted(np.asarray(trials.sum(axis=1)).ravel(), "lognormal"),
        "snp_total": _counted(
            np.asarray(trials.sum(axis=0)).ravel(), "negative_binomial"
        ),
        "gene_profile": replace(
            profile,
            parameters={k: profile.parameters[k] for k in ("mu", "sigma")},
            ks_alternative=None,
            expressed=float(np.mean(per_gene > 0)),
        ),
    }


def allele_share(
    a: np.ndarray, b: np.ndarray, normal_frac: float, admixture: str
) -> np.ndarray:
    """Expected haplotype-A share of reads at planted `(a, b)`, admixed."""
    total = a + b
    if admixture == "cell":
        tumour = 1.0 - normal_frac
        denominator = tumour * total + 2.0 * normal_frac
        return np.asarray(
            (tumour * a + normal_frac) / np.where(denominator > 0, denominator, 1.0)
        )
    share = np.where(total > 0, a / np.maximum(total, 1), 0.5)
    return np.asarray((1.0 - normal_frac) * share + normal_frac / 2.0)


def fit_normal_frac(
    a_reads: np.ndarray, reads: np.ndarray, copies: np.ndarray, admixture: str
) -> float:
    """Maximum-likelihood normal fraction from one clone's pooled SNP reads.

    `a_reads`, `reads`: `(n_snps,)` pooled over the clone's spots; `copies`:
    `(n_snps, 2)` planted `(A, B)`. Only SNPs whose pure share is not 0.5
    carry information, so only they enter.
    """
    from scipy.optimize import minimize_scalar

    a, b = copies[:, 0].astype(np.float64), copies[:, 1].astype(np.float64)
    informative = (a != b) & (reads > 0)

    def loss(normal_frac: float) -> float:
        q = allele_share(a, b, normal_frac, admixture)[informative]
        q = np.clip(q, 1e-12, 1 - 1e-12)
        hits, trials = a_reads[informative], reads[informative]
        return -float(np.sum(hits * np.log(q) + (trials - hits) * np.log(1 - q)))

    found = minimize_scalar(loss, bounds=(0.0, 0.95), method="bounded")
    return float(found.x)


def fit_dispersions(path: Path, labels: np.ndarray) -> dict[str, float]:
    """The model's dispersions, by moments, and one matched to the zeros.

    - `nb_dispersion`: normal spots, the 2,000 most expressed genes, each
      spot's counts against its total times the pooled normal profile,
      `alpha = sum((x-mu)^2-mu)/sum(mu^2)`;
    - `nb_dispersion_zero_matched`: the `alpha` at which a gamma-Poisson over
      every spot and the pooled profile has the observed count of zeros;
    - `snp_dispersion`: the same moment estimate over every spot-SNP entry,
      against `M_s` times the pooled SNP profile;
    - `bb_overdispersion`: normal-spot SNP entries with at least 2 reads,
      where the share is 0.5, `rho` from the variance's excess over the
      binomial's.
    """
    import anndata
    import scipy.sparse
    from scipy.optimize import brentq

    normal = labels == 0
    counts = scipy.sparse.csr_matrix(
        anndata.read_h5ad(path / "filtered_feature_bc_matrix.h5ad").X
    )[normal].astype(np.float64)
    depth = np.asarray(counts.sum(axis=1)).ravel()
    profile = np.asarray(counts.sum(axis=0)).ravel() / depth.sum()
    top = np.argsort(profile)[-2000:]
    observed = counts[:, top].toarray()
    expected = depth[:, None] * profile[top][None, :]
    nb = float(((observed - expected) ** 2 - expected).sum() / (expected**2).sum())

    first = scipy.sparse.load_npz(path / "cell_snp_Aallele.npz").tocsr()[normal]
    second = scipy.sparse.load_npz(path / "cell_snp_Ballele.npz").tocsr()[normal]
    trials = (first + second).tocoo()
    hits = np.asarray(first[trials.row, trials.col]).ravel().astype(np.float64)
    n = trials.data.astype(np.float64)
    kept = n >= 2
    hits, n = hits[kept], n[kept]
    binomial = float((n * 0.25).sum())
    excess = float(((hits - n / 2) ** 2).sum()) - binomial
    bb = excess / float((n * (n - 1) * 0.25).sum())

    everything = scipy.sparse.csr_matrix(
        anndata.read_h5ad(path / "filtered_feature_bc_matrix.h5ad").X
    ).astype(np.float64)
    totals = np.asarray(everything.sum(axis=1)).ravel()
    pooled = np.asarray(everything.sum(axis=0)).ravel() / totals.sum()
    shares = pooled[pooled > 0]
    zeros = everything.shape[0] * everything.shape[1] - everything.nnz
    zeros -= (pooled.size - shares.size) * everything.shape[0]

    def excess_zeros(alpha: float) -> float:
        found = 0.0
        for start in range(0, totals.size, 256):
            mean = totals[start : start + 256, None] * shares[None, :]
            found += float(((1.0 + alpha * mean) ** (-1.0 / alpha)).sum())
        return found - float(zeros)

    zero_matched: float = brentq(excess_zeros, 1e-3, 1e4, xtol=1e-3)

    reads = (
        scipy.sparse.load_npz(path / "cell_snp_Aallele.npz").tocsr()
        + scipy.sparse.load_npz(path / "cell_snp_Ballele.npz").tocsr()
    ).tocoo()
    per_spot = np.asarray(reads.sum(axis=1)).ravel().astype(np.float64)
    per_snp = np.asarray(reads.sum(axis=0)).ravel().astype(np.float64)
    per_snp /= per_snp.sum()
    data = reads.data.astype(np.float64)
    squared = float((per_spot**2).sum() * (per_snp**2).sum())
    cross = float((data * per_spot[reads.row] * per_snp[reads.col]).sum())
    snp = ((data**2).sum() - 2 * cross + squared - per_spot.sum()) / squared

    return {
        "nb_dispersion": max(nb, 0.0),
        "nb_dispersion_zero_matched": zero_matched,
        "snp_dispersion": max(float(snp), 0.0),
        "bb_overdispersion": max(bb, 0.0),
    }


def manifest_from_sample(
    path: Path,
    *,
    name: str,
    root: Path,
    seed: int = 0,
    gene_table: str = "$PORT_GRCH38/hgTables_hg38_gencode.txt",
) -> SimManifest:
    """The manifest that replicates a CalicoST sample, drawn pure.

    Sizes, layout and events are the sample's; the coverage laws, the
    dispersions and each clone's fitted normal fraction are measured on it.
    `normal_frac` is 0 for every clone, so the manifest draws the pure
    sample; `with_normal_frac(manifest, "fitted")` draws the admixed one.
    """
    import pandas as pd

    labels_table = pd.read_csv(path / "truth_clone_labels.tsv", sep="\t", index_col=0)
    profile = pd.read_csv(path / "truth_acn_profile.tsv", sep="\t")
    names = [
        "normal",
        *sorted(
            (n for n in labels_table["labels"].unique() if n != "normal"),
            key=lambda n: int(n.removeprefix("clone_")),
        ),
    ]
    label = labels_table["labels"].map({n: i for i, n in enumerate(names)}).to_numpy()
    spot_counts = labels_table["labels"].value_counts()

    clones = []
    for clone in names:
        a_col, b_col = f"{clone}_A_copy", f"{clone}_B_copy"
        altered = profile[(profile[a_col] != 1) | (profile[b_col] != 1)]
        events = tuple(
            Event(
                str(row["chr"]),
                int(row["start"]),
                int(row["end"]),
                int(row[a_col]),
                int(row[b_col]),
            )
            for _, row in altered.iterrows()
        )
        clones.append(Clone(clone, int(spot_counts[clone]), events))

    snp_ids = np.load(path / "unique_snp_ids.npy", allow_pickle=True).astype(str)
    import anndata
    import scipy.sparse

    n_genes = anndata.read_h5ad(path / "filtered_feature_bc_matrix.h5ad").n_vars
    rows = int(labels_table["x"].max()) + 1
    columns = len(labels_table) // rows
    lengths = profile.groupby("chr", sort=False)["end"].max()

    relative = os.path.relpath(path, root)
    layout = {
        "genes": f"{relative}/filtered_feature_bc_matrix.h5ad",
        "snps": f"{relative}/unique_snp_ids.npy",
        "labels": f"{relative}/truth_clone_labels.tsv",
        "gene_table": gene_table,
    }
    size = {
        "n_spots": len(labels_table),
        "lattice": "hex",
        "rows": rows,
        "columns": columns,
        "n_genes": int(n_genes),
        "n_snps": int(snp_ids.size),
        "n_chromosomes": int(lengths.size),
        "n_segments": len(profile),
        "chromosome_lengths": [int(x) for x in lengths],
    }

    first = scipy.sparse.load_npz(path / "cell_snp_Aallele.npz").tocsr()
    trials = first + scipy.sparse.load_npz(path / "cell_snp_Ballele.npz").tocsr()
    chromosome = np.array([s.split("_")[0] for s in snp_ids])
    position = np.array([int(s.split("_")[1]) for s in snp_ids])
    draft = SimManifest(
        name, seed, "..", size, layout, Model(), tuple(clones), {}, root=root
    )
    copies = copies_at(draft, chromosome, position)

    fitted: dict[str, dict[str, float]] = {"normal_frac": {}, "normal_frac_cell": {}}
    for index, clone in enumerate(clones):
        if clone.name == "normal":
            continue
        rows_of = label == index
        hits = np.asarray(first[rows_of].sum(axis=0)).ravel().astype(np.float64)
        reads = np.asarray(trials[rows_of].sum(axis=0)).ravel().astype(np.float64)
        for key, law in (("normal_frac", "read"), ("normal_frac_cell", "cell")):
            value = fit_normal_frac(hits, reads, copies[:, index], law)
            fitted[key][clone.name] = round(value, 4)

    dispersions = fit_dispersions(path, label)
    fitted["dispersion"] = {
        "nb_dispersion_zero_matched": round(
            dispersions["nb_dispersion_zero_matched"], 2
        )
    }
    model = Model(
        admixture="read",
        normal_frac={c.name: 0.0 for c in clones if c.name != "normal"},
        nb_dispersion=round(dispersions["nb_dispersion"], 4),
        bb_overdispersion=round(dispersions["bb_overdispersion"], 5),
        snp_dispersion=round(dispersions["snp_dispersion"], 4),
        snp_depth_follows_copies=False,
        fitted=fitted,
    )
    coverage = {
        key: replace(
            law,
            parameters={k: round(v, 6) for k, v in law.parameters.items()},
            ks=None if law.ks is None else round(law.ks, 5),
            ks_alternative=None
            if law.ks_alternative is None
            else round(law.ks_alternative, 5),
            expressed=None if law.expressed is None else round(law.expressed, 5),
        )
        for key, law in fit_coverage(path).items()
    }
    return replace(draft, model=model, coverage=coverage, source=path.name)


HEADER = """Generated by `python -m port.sim.toml_manifest {source} {path}` (#382).
Replicates CalicoST's `{source}` drawn pure: every `normal_frac` is 0.
`run_sim_gen --normal-frac fitted` draws it at `[model.fitted] normal_frac`."""


def main(argv: list[str] | None = None) -> int:
    """Fit a manifest to a CalicoST sample directory and write it as TOML."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("sample", help="a sample directory, `sim/<name>`")
    parser.add_argument("manifest", help="the `.toml` to write")
    parser.add_argument("--name", default=None, help="default `<sample>_pure`")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args(argv)

    sample, path = Path(arguments.sample), Path(arguments.manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = manifest_from_sample(
        sample,
        name=arguments.name or f"{sample.name}_pure",
        root=path.parent,
        seed=arguments.seed,
    )
    header = HEADER.format(source=sample, path=path)
    path.write_text(to_toml(manifest, header))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
