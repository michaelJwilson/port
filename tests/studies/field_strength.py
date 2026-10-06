"""#556: clone-field strength, dev_tree against CalicoST easy/hard, and what sets it.

Subcommands of `python -m tests.studies.field_strength`:

`pipeline CAPTURE.npz ...`
    The pipeline's field at the planted clones: `--sal --oracle-start`'s first
    BAF + RDR inference, captured by `tests.studies.clone_labels capture`,
    rebuilt at the planted labels (`port.sandbox.clone_starts.problem.build`).
`known MANIFEST.toml ...`
    The known-law field (`port.sandbox.known_field`) of each manifest's first
    3 realizations, RDR and BAF apart.
`calicost`
    CalicoST easy and hard under port's laws, from their planted profiles:
    the known-law field, RDR and BAF apart (BAF at the fitted tumour share,
    their written phase); the expected BAF margin to the nearest clone; the
    per-spot tumour fraction and the Gaussian random field fitted to it; the
    tumour expression beyond copy number.

Strength is read three ways per field: the median over spots of the planted
clone's field less the best other clone's (the margin), the fraction of spots
whose argmax is not the planted clone, and that argmax's ARI.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TUMOUR_SHARE = 0.84
"""CalicoST's tumour share under the cell law: its LOH BAF, 0.915-0.928 at (0, 2) and 0.085 at (1, 0)."""

GRID = np.linspace(0.0, 1.0, 201)
"""The per-spot tumour fraction's grid."""


def strength(field: np.ndarray, planted: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import adjusted_rand_score

    own = field[np.arange(planted.size), planted]
    other = field.copy()
    other[np.arange(planted.size), planted] = -np.inf
    margin = own - other.max(axis=1)
    return {"margin": float(np.median(margin)), "q10": float(np.quantile(margin, 0.1)),
            "wrong": float(np.mean(margin < 0)), "ari": float(adjusted_rand_score(planted, field.argmax(1)))}  # fmt: skip


def _line(name: str, s: dict[str, float]) -> str:
    return f"{name:28s} median margin {s['margin']:7.1f}  10% {s['q10']:7.1f}  argmax wrong {s['wrong']:.4f}  ARI {s['ari']:.4f}"


def pipeline(paths: list[Path]) -> None:
    from port.sandbox.clone_starts.problem import build, load

    for path in paths:
        capture = load(path)
        problem = build(capture, capture.planted, np.random.default_rng([0, 540]))
        print(
            _line(path.stem, strength(problem.field, problem.labels)),
            f"states {problem.states_by}",
            flush=True,
        )


def known(paths: list[Path], n: int = 3) -> None:
    import port.sandbox.known_field as kf
    from port.sim import draw as d
    from port.sim.laws import allele_share

    for path in paths:
        # NB the problem's two terms apart: `known_field.problems` sums them
        manifest = d.read_manifest(path)
        tree = d.draw_tree(
            manifest,
            np.random.default_rng(np.random.SeedSequence(manifest.seed).spawn(3)[0]),
        )
        clones = ("normal", *manifest.tumour)
        base = pd.read_csv(
            manifest.resolve(manifest.reference["baseline"]), sep="\t", comment="#"
        )
        loci = (
            base["chrom"].astype(str).str.removeprefix("chr").to_numpy(),
            ((base.cdsStart + base.cdsEnd) // 2).to_numpy(),
        )
        weights = base["lambda"].to_numpy()[:, None] * d._depth(
            manifest, clones, d.clone_copies(tree, clones, *loci)
        )
        kappa = float(manifest.model["dirichlet_concentration"])
        alpha = np.maximum(kappa * weights / weights.sum(0, keepdims=True), 1e-300)
        _, chrom, pos = d._snps(manifest)
        cp = d.clone_copies(tree, clones, chrom, pos)
        share = np.stack([allele_share(cp[:, k, 0].astype(float), cp[:, k, 1].astype(float), manifest.normal_frac(c),
                                       manifest.model["admixture"]) for k, c in enumerate(clones)])  # fmt: skip
        rows = []
        for realized in itertools.islice(d.realize(manifest, None), n):
            planted = np.asarray(realized.truth.labels[0], dtype=np.int64)
            rdr = kf.rdr_field(realized.counts[0], alpha)
            baf = kf.baf_field(
                realized.a,
                realized.b,
                realized.phase,
                share,
                float(manifest.model["bb_overdispersion"]),
            )
            rows.append(
                {
                    k: strength(f, planted)
                    for k, f in (("RDR", rdr), ("BAF", baf), ("RDR + BAF", rdr + baf))
                }
            )
        for term in ("RDR", "BAF", "RDR + BAF"):
            mean = {
                key: float(np.mean([r[term][key] for r in rows]))
                for key in rows[0][term]
            }
            print(_line(f"{path.stem} {term} (mean of {n})", mean), flush=True)


def _profile_states(
    profile: pd.DataFrame, clones: list[str], chrom: np.ndarray, pos: np.ndarray
) -> np.ndarray:
    """`(loci, clones, 2)` planted (A, B), (1, 1) where no segment differs."""
    out = np.ones((pos.size, len(clones), 2))
    for _, seg in profile.iterrows():
        at = (
            (chrom == str(seg["chr"]).removeprefix("chr"))
            & (pos >= seg["start"])
            & (pos < seg["end"])
        )
        for k, clone in enumerate(clones):
            out[at, k] = seg[f"{clone}_A_copy"], seg[f"{clone}_B_copy"]
    return out


def _cell_share(a: np.ndarray, b: np.ndarray, tumour: float | np.ndarray) -> np.ndarray:
    """Haplotype-A share at planted (a, b) under the cell law at tumour share `tumour`."""
    return np.asarray(
        (tumour * a + (1 - tumour)) / (tumour * (a + b) + 2 * (1 - tumour))
    )


def _entries(
    total: Any, b_reads: Any, rows: np.ndarray, p_snp: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`rows`' nonzero (SNP, spot) entries: B reads, total reads, and each entry's share."""
    n_rows = total[rows].tocoo()
    b_at = np.asarray(b_reads[rows][n_rows.row, n_rows.col]).ravel().astype(float)
    return b_at, n_rows.data.astype(float), p_snp[n_rows.col]


def events(profile: pd.DataFrame, clones: list[str]) -> pd.DataFrame:
    """Each clone's CNA events: runs of adjacent segments at one non-(1, 1) state on one chromosome, and their kind."""
    rows = []
    for clone in clones:
        run: dict[str, Any] | None = None
        for _, seg in profile.iterrows():
            state = (int(seg[f"{clone}_A_copy"]), int(seg[f"{clone}_B_copy"]))
            if state == (1, 1):
                run = None
                continue
            if (
                run is not None
                and run["state"] == state
                and run["chr"] == seg["chr"]
                and run["end"] == seg["start"]
            ):
                run["end"] = seg["end"]
                continue
            run = {
                "clone": clone,
                "chr": seg["chr"],
                "start": seg["start"],
                "end": seg["end"],
                "state": state,
            }
            rows.append(run)
    table = pd.DataFrame(rows)
    a, b = np.array([s[0] for s in table.state]), np.array([s[1] for s in table.state])
    table["kind"] = np.where(
        np.minimum(a, b) == 0, "LOH", np.where(a == b, "balanced", "imbalanced")
    )
    table["mb"] = (table.end - table.start) / 1e6
    return table


def calicost() -> None:
    import anndata
    import port.sandbox.known_field as kf
    import scipy.sparse as sp
    from port.sim.fixtures import SIM_ROOT
    from scipy.optimize import curve_fit
    from scipy.spatial import cKDTree

    from tests.studies.calicost_figures import SAMPLES, baseline, baseline_counts

    base = baseline()
    gene_loci = (
        base["chrom"].astype(str).str.removeprefix("chr").to_numpy(),
        ((base.cdsStart + base.cdsEnd) // 2).to_numpy(),
    )
    for key, name in SAMPLES.items():
        src = SIM_ROOT / name
        truth = pd.read_csv(src / "truth_clone_labels.tsv", sep="\t", index_col=0)
        clones = ["normal", *sorted(set(truth.labels) - {"normal"})]
        planted = truth.labels.map({c: k for k, c in enumerate(clones)}).to_numpy()
        profile = pd.read_csv(src / "truth_acn_profile.tsv", sep="\t")
        ids = np.load(src / "unique_snp_ids.npy", allow_pickle=True).astype(str)
        chrom = np.array([s.split("_")[0] for s in ids])
        pos = np.array([int(s.split("_")[1]) for s in ids])
        cp = _profile_states(profile, clones, chrom, pos)
        a_reads, b_reads = (
            sp.load_npz(src / "cell_snp_Aallele.npz").tocsr(),
            sp.load_npz(src / "cell_snp_Ballele.npz").tocsr(),
        )

        # --- the event mix: `[cna] imbalanced` and `[cna.length]`
        mix = events(profile, clones[1:])
        rates = (mix.kind.value_counts() / len(mix)).round(2).to_dict()
        print(f"{key}: {len(mix)} clone-events (a shared event once per clone); rates {rates}; length Mb median "
              f"{mix.mb.median():.1f}, mean {mix.mb.mean():.1f}", flush=True)  # fmt: skip

        # --- the known-law field under port's laws
        tumour = np.array([1.0 if c == "normal" else TUMOUR_SHARE for c in clones])
        share = np.stack(
            [
                _cell_share(cp[:, k, 0], cp[:, k, 1], tumour[k])
                for k in range(len(clones))
            ]
        )
        baf = kf.baf_field(a_reads, b_reads, np.zeros(ids.size, dtype=bool), share)
        counts = baseline_counts(
            anndata.read_h5ad(src / "filtered_feature_bc_matrix.h5ad"), base["gene"]
        )
        gc = _profile_states(profile, clones, *gene_loci)
        depth = np.stack(
            [
                tumour[k] * gc[:, k].sum(1) / 2 + (1 - tumour[k])
                for k in range(len(clones))
            ],
            axis=1,
        )
        w = base["lambda"].to_numpy()[:, None] * depth
        rdr = kf.rdr_field(
            counts, np.maximum(100.0 * w / w.sum(0, keepdims=True), 1e-300)
        )
        for term, f in (("RDR", rdr), ("BAF", baf), ("RDR + BAF", rdr + baf)):
            print(_line(f"{key} {term}", strength(f, planted)), flush=True)

        # --- expected BAF margin to the nearest clone: mean reads x KL, summed over SNPs
        reads = np.asarray((a_reads + b_reads).mean(0)).ravel()
        p = np.clip(share, 1e-4, 1 - 1e-4)

        def kl(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return np.asarray(x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y)))

        nearest = {clones[i]: min(float((reads * kl(p[i], p[j])).sum()) for j in range(len(clones)) if j != i)
                   for i in range(len(clones))}  # fmt: skip
        print(
            f"   expected BAF margin to the nearest clone [nats]: { {c: round(v, 1) for c, v in nearest.items()} }"
        )

        # --- BAF overdispersion by moments: normal spots at p = 1/2, tumour spots at their clone's share
        #     under its pooled tumour fraction (below); the estimator's null sd from binomial redraws
        total = (a_reads + b_reads).tocsr()

        normal_rows = np.flatnonzero(planted == 0)
        b_n, n_n, p_n = _entries(total, b_reads, normal_rows, np.full(ids.size, 0.5))
        null = np.std([kf.overdispersion(np.random.default_rng(seed).binomial(n_n.astype(np.int64), 0.5).astype(float),
                                         n_n, p_n) for seed in range(100)])  # fmt: skip
        rhos = {"normal": kf.overdispersion(b_n, n_n, p_n)}

        # --- per-spot tumour fraction given the truth, and its field
        fraction, pooled = np.full(len(truth), np.nan), {}
        for k, clone in enumerate(clones[1:], start=1):
            rows = np.flatnonzero(planted == k)
            s = np.clip(
                _cell_share(cp[None, :, k, 1], cp[None, :, k, 0], GRID[:, None]),
                1e-6,
                1 - 1e-6,
            )
            loglik = np.asarray(
                b_reads[rows] @ np.log(s).T + a_reads[rows] @ np.log1p(-s).T
            )
            fraction[rows] = GRID[np.argmax(loglik, axis=1)]
            pooled[clone] = float(GRID[np.argmax(loglik.sum(0))])
        for k, clone in enumerate(clones[1:], start=1):
            b_share = _cell_share(cp[:, k, 1], cp[:, k, 0], pooled[clone])
            rhos[clone] = kf.overdispersion(
                *_entries(
                    total,
                    b_reads,
                    np.flatnonzero(planted == k),
                    np.clip(b_share, 1e-6, 1 - 1e-6),
                )
            )
        print(
            f"   BAF overdispersion rho by moments: { {c: round(r, 4) for c, r in rhos.items()} }; null sd {null:.4f}"
        )
        spots = ~np.isnan(fraction)
        xy, e = truth[["x", "y"]].to_numpy(float)[spots], fraction[spots]
        pairs = cKDTree(xy).query_pairs(20, output_type="ndarray")
        dist = np.linalg.norm(xy[pairs[:, 0]] - xy[pairs[:, 1]], axis=1)
        prod = (e[pairs[:, 0]] - e.mean()) * (e[pairs[:, 1]] - e.mean())
        edges = np.arange(0, 21, 2.0)
        mid = (edges[:-1] + edges[1:]) / 2
        cov = np.array(
            [
                prod[(dist > lo) & (dist <= hi)].mean()
                for lo, hi in itertools.pairwise(edges)
            ]
        )
        (sill, length), _ = curve_fit(
            lambda x, sill, length: sill * np.exp(-x / length),
            mid,
            cov,
            p0=[1e-3, 3.0],
            bounds=([0, 0.1], [1, 200]),
        )
        print(f"   tumour fraction pooled per clone {pooled}; per-spot MLE sd {e.std():.3f}; covariance by distance "
              f"{dict(zip(mid.tolist(), cov.round(5).tolist(), strict=True))}; exponential fit sd {np.sqrt(sill):.3f}, "
              f"length {length:.2f} (spacing 1.41)")  # fmt: skip

        # --- tumour expression beyond copy number, on copy-neutral genes
        neutral = (gc == 1).all(axis=(1, 2))
        tumour_umi = np.asarray(counts[planted > 0].sum(0)).ravel()
        normal_umi = np.asarray(counts[planted == 0].sum(0)).ravel()
        ok = neutral & (tumour_umi >= 50) & (normal_umi >= 50)
        ratio = np.log(tumour_umi[ok] / tumour_umi[ok].sum()) - np.log(
            normal_umi[ok] / normal_umi[ok].sum()
        )
        noise = float(np.mean(1 / tumour_umi[ok] + 1 / normal_umi[ok]))
        print(f"   expression: {ok.sum()} copy-neutral genes; log(tumour / normal) sd {ratio.std():.3f} (Poisson "
              f"{np.sqrt(noise):.3f}); 5/50/95% {np.quantile(ratio, [0.05, 0.5, 0.95]).round(2).tolist()}; "
              f"corr(log tumour, log normal) {np.corrcoef(np.log(tumour_umi[ok]), np.log(normal_umi[ok]))[0, 1]:.3f}", flush=True)  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("pipeline").add_argument("captures", nargs="+", type=Path)
    sub.add_parser("known").add_argument("manifests", nargs="+", type=Path)
    sub.add_parser("calicost")
    arguments: Any = parser.parse_args(argv)
    if arguments.command == "pipeline":
        pipeline(arguments.captures)
    elif arguments.command == "known":
        known(arguments.manifests)
    else:
        calicost()


if __name__ == "__main__":
    main()
