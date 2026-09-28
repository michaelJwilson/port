"""The normal baseline and coverage laws, fitted on CalicoST's normal spots (#445).

Laws fitted over every spot of a sample move with a tumour spot's copies, so
these are fitted on the spots `truth_clone_labels.tsv` calls `normal`,
pooled over the samples given, and written where `port.sim.draw` reads them:

- `normal_baseline.txt`: one row per gene of the reference that the assay
  measures -- `name2` of CalicoST's `hgTables_hg38_gencode.txt` present in
  the AnnData's `var_names` -- with `chrom`, `cdsStart`, `cdsEnd` and
  `lambda`, that gene's share of all normal-spot UMI, summing to 1. A name the
  reference repeats keeps its first row, as CalicoST's simulator places
  genes; a name the assay repeats sums its columns.
- `normal_coverage.toml`: `[coverage.<law>]` tables, each with both families
  fitted and the one with the smaller KS statistic drawn from:

  - `spot_umi`: a normal spot's total UMI;
  - `spot_snp_umi`: a normal spot's total SNP reads, `A + B` over SNPs;
  - `snp_a`, `snp_b`, `snp_total`: a SNP's `A`, `B` and `A + B` reads summed
    over normal spots.

  and `ENTRY_LAWS`, `port.sim.entries`' laws of one entry (#455), each with
  the observed nonzero share as `expressed`:

  - `gene_spot_umi`: one gene's UMI in one normal spot, Dirichlet-multinomial
    about `lambda` at `concentration`;
  - `snp_spot_umi`: one SNP's `A + B` reads in one normal spot, NB at
    `dispersion` about `snp_total`'s Gamma weights.

    python -m port.sim.normal_fit sim/<easy> sim/<hard> --into sim \\
        --gene-table $PORT_GRCH38/hgTables_hg38_gencode.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from port.sim.laws import Law, counted

LAWS = ("spot_umi", "spot_snp_umi", "snp_a", "snp_b", "snp_total")
"""The laws `normal_coverage.toml` carries, in the order it writes them."""

ENTRY_LAWS = ("gene_spot_umi", "snp_spot_umi")
"""The per-entry laws, fitted to the histograms `plot_coverage` draws."""

BASELINE_COLUMNS = ("gene", "chrom", "cdsStart", "cdsEnd", "lambda")


def normal_spots(path: Path) -> np.ndarray:
    """`(n_spots,)` bool, in `barcodes.txt`'s order: the truth calls it `normal`."""
    barcodes = (path / "barcodes.txt").read_text().split()
    truth = pd.read_csv(path / "truth_clone_labels.tsv", sep="\t", index_col=0)
    labels = truth.iloc[:, 0].reindex(barcodes)

    if labels.isna().any():
        msg = f"{path}: {int(labels.isna().sum())} barcodes carry no truth label"
        raise ValueError(msg)

    return np.asarray(labels.to_numpy() == "normal")


def _read(path: Path) -> dict[str, Any]:
    """One sample's normal spots: gene counts, allele counts, names."""
    import anndata
    import scipy.sparse

    normal = normal_spots(path)
    assay = anndata.read_h5ad(path / "filtered_feature_bc_matrix.h5ad")
    barcodes = (path / "barcodes.txt").read_text().split()

    if list(assay.obs_names) != barcodes:
        msg = f"{path}: the AnnData's spots are not `barcodes.txt`'s"
        raise ValueError(msg)

    alleles = [
        scipy.sparse.load_npz(path / f"cell_snp_{x}allele.npz").tocsr()[normal]
        for x in ("A", "B")
    ]
    return {
        "counts": scipy.sparse.csr_matrix(assay.X)[normal],
        "genes": np.asarray(assay.var_names).astype(str),
        "a": alleles[0],
        "b": alleles[1],
        "snps": np.load(path / "unique_snp_ids.npy", allow_pickle=True).astype(str),
    }


def _pooled(paths: list[Path]) -> dict[str, Any]:
    """The samples' normal spots, stacked; they must share genes and SNPs."""
    import scipy.sparse

    samples = [_read(p) for p in paths]

    for path, sample in zip(paths[1:], samples[1:], strict=True):
        for key in ("genes", "snps"):
            if not np.array_equal(sample[key], samples[0][key]):
                msg = f"{path} does not share {key} with {paths[0]}"
                raise ValueError(msg)

    pooled = {k: scipy.sparse.vstack([s[k] for s in samples]).tocsr() for k in "ab"}
    pooled["counts"] = scipy.sparse.vstack([s["counts"] for s in samples]).tocsr()
    return pooled | {"genes": samples[0]["genes"], "snps": samples[0]["snps"]}


def fit_baseline(paths: list[Path], gene_table: Path) -> pd.DataFrame:
    """`BASELINE_COLUMNS`, one row per reference gene the assay measures.

    `lambda` is the gene's summed normal-spot UMI over the pooled samples,
    divided by the sum over the genes kept, so it sums to 1 over them. A gene
    the assay measures but no normal spot expresses is kept at 0.
    """
    pooled = _pooled(paths)
    per_column = np.asarray(pooled["counts"].sum(axis=0)).ravel().astype(np.float64)
    per_gene = pd.Series(per_column).groupby(pooled["genes"], sort=False).sum()

    table = pd.read_csv(gene_table, sep="\t", index_col=0)
    table = table.drop_duplicates("name2").set_index("name2")
    measured = table.loc[table.index.isin(per_gene.index)]

    total = per_gene.loc[measured.index].to_numpy()
    return pd.DataFrame(
        {
            "gene": measured.index.to_numpy(),
            "chrom": measured["chrom"].to_numpy(),
            "cdsStart": measured["cdsStart"].to_numpy(),
            "cdsEnd": measured["cdsEnd"].to_numpy(),
            "lambda": total / total.sum(),
        }
    )


def fit_normal_coverage(paths: list[Path]) -> dict[str, Law]:
    """`LAWS`, fitted on the pooled normal spots."""
    pooled = _pooled(paths)
    a, b = pooled["a"], pooled["b"]

    def per_spot(matrix: Any) -> np.ndarray:
        return np.asarray(matrix.sum(axis=1)).ravel()

    def per_snp(matrix: Any) -> np.ndarray:
        return np.asarray(matrix.sum(axis=0)).ravel()

    return {
        "spot_umi": best(per_spot(pooled["counts"])),
        "spot_snp_umi": best(per_spot(a + b)),
        "snp_a": best(per_snp(a)),
        "snp_b": best(per_snp(b)),
        "snp_total": best(per_snp(a + b)),
    }


def by_gene(pooled: dict[str, Any], genes: np.ndarray) -> Any:
    """Pooled normal-spot UMI as spots x `genes`, a name's columns summed."""
    import scipy.sparse

    index = {name: i for i, name in enumerate(genes)}
    columns = np.array([index.get(name, -1) for name in pooled["genes"]])
    kept = np.nonzero(columns >= 0)[0]
    gather = scipy.sparse.csr_matrix(
        (np.ones(kept.size), (kept, columns[kept])), shape=(columns.size, genes.size)
    )
    matrix = (pooled["counts"] @ gather).tocsr()
    matrix.data = np.rint(matrix.data).astype(np.int64)
    return matrix


def fit_entries(
    paths: list[Path], baseline: pd.DataFrame, laws: dict[str, Law]
) -> dict[str, Law]:
    """`ENTRY_LAWS` on the pooled normal spots.

    The gene law at the spots' own totals and `baseline`'s `lambda`; the SNP
    law at `laws`' `spot_snp_umi` and `snp_total` dispersion, as
    `port.sim.draw` draws them.
    """
    from port.sim.entries import fit_genes, fit_snps

    pooled = _pooled(paths)
    genes = by_gene(pooled, baseline["gene"].to_numpy())
    trials = (pooled["a"] + pooled["b"]).tocsr()
    weight = laws["snp_total"].parameters.get("dispersion", 0.0)

    gene_fit = fit_genes(genes, baseline["lambda"].to_numpy())
    snp_fit = fit_snps(trials, laws["spot_snp_umi"], weight)
    return {
        name: Law(
            fit.family,
            {key: fit.value},
            n=int(matrix.shape[0] * matrix.shape[1]),
            expressed=fit.observed,
        )
        for name, key, fit, matrix in (
            ("gene_spot_umi", "concentration", gene_fit, genes),
            ("snp_spot_umi", "dispersion", snp_fit, trials),
        )
    }


def best(values: np.ndarray) -> Law:
    """Both families fitted to `values`; the one with the smaller KS statistic."""
    lognormal = counted(values, "lognormal")
    negative = counted(values, "negative_binomial")
    return lognormal if (lognormal.ks or 0.0) <= (negative.ks or 0.0) else negative


def to_toml(laws: dict[str, Law], header: str) -> str:
    """`[coverage.<law>]` tables, in `LAWS`' order."""
    lines = [f"# {line}" for line in header.splitlines()]

    for name in LAWS + ENTRY_LAWS:
        law = laws[name]
        lines += ["", f"[coverage.{name}]", f'family = "{law.family}"']
        lines += [f"{k} = {v:.6g}" for k, v in law.parameters.items()]
        lines += [f"{k} = {v:.4g}" for k, v in _scores(law).items()]
        lines += [f"n = {law.n}"]

    return "\n".join(lines) + "\n"


def _scores(law: Law) -> dict[str, float]:
    """The fit statistics `law` carries, by the key `to_toml` writes."""
    scores = {
        "ks": law.ks,
        "ks_alternative": law.ks_alternative,
        "expressed": law.expressed,
    }
    return {k: v for k, v in scores.items() if v is not None}


def read_coverage(path: Path) -> dict[str, Law]:
    """The laws of a `normal_coverage.toml`."""
    import tomllib

    tables = tomllib.loads(path.read_text())["coverage"]
    return {
        name: Law(
            family=table["family"],
            parameters={
                k: float(v)
                for k, v in table.items()
                if k not in {"family", "ks", "ks_alternative", "expressed", "n"}
            },
            ks=table.get("ks"),
            ks_alternative=table.get("ks_alternative"),
            n=table.get("n"),
            expressed=table.get("expressed"),
        )
        for name, table in tables.items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", nargs="+", help="CalicoST sample directories")
    parser.add_argument("--into", required=True, help="where to write both files")
    parser.add_argument(
        "--gene-table", required=True, help="CalicoST's hgTables_hg38_gencode.txt"
    )
    arguments = parser.parse_args(argv)

    paths = [Path(p) for p in arguments.samples]
    gene_table = Path(arguments.gene_table)
    into = Path(arguments.into)
    names = ", ".join(p.name for p in paths)

    laws = fit_normal_coverage(paths)
    baseline = fit_baseline(paths, gene_table)
    laws |= fit_entries(paths, baseline, laws)
    with (into / "normal_baseline.txt").open("w") as stream:
        stream.write(
            f"# Generated by `python -m port.sim.normal_fit` (#445) from the normal "
            f"spots of {names}.\n# {len(baseline)} genes of {gene_table.name} the "
            f"assay measures; {int((baseline['lambda'] == 0).sum())} at lambda 0; "
            "lambda sums to 1.\n"
        )
        baseline.to_csv(stream, sep="\t", index=False, float_format="%.9e")

    header = (
        "Generated by `python -m port.sim.normal_fit` (#445).\n"
        f"Fitted on the normal spots of {names}."
    )
    (into / "normal_coverage.toml").write_text(to_toml(laws, header))
    print(f"wrote {len(baseline)} genes and {len(laws)} laws to {into}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
