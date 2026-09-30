"""A drawn realization's copy-state problem with its clones known (#540, #556).

The clones are the planted ones, so what remains is the copy states: each
clone's pseudobulk over genomic bins, and the states an HMM decodes along it.

- **Bins.** 1 Mb bins by gene midpoint and SNP position, merged within a
  chromosome until the normal clone's pseudobulk holds `FLOOR` UMIs -- #551's
  300 normal-UMI floor, which removes the short low-exposure bins whose read
  depth ratio scatters. A chromosome's unclosed remainder joins its last bin.
- **Read depth.** Each clone's UMIs per bin; its exposure is the normal
  clone's share of UMIs in the bin times the clone's total, as `cnaster`
  derives `base_nb_mean` from normal spots.
- **Allele reads.** Haplotype-B reads under the truth phase (the written B,
  or the written A where the phase switched), and the SNP reads.
- **Truth.** Each (clone, bin) row's planted `(A, B)`: the state covering
  most of the bin's SNPs. The planted states' parameters are the draw's:
  `log_mu` the admixed depth factor, `p_binom` the admixed B share.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

__all__ = ["FLOOR", "KnownCopyProblem", "floored_bins", "problems"]

FLOOR = 300.0
"""#551's floor: a bin's normal-clone UMIs, merged until at least this."""

BIN = 1_000_000
"""The bins' starting width, `port.sim.analysis.BIN`'s."""


class KnownCopyProblem(NamedTuple):
    """One realization's copy-state problem, one row per (clone, bin), clones stacked genome after genome."""

    realization: int
    total: np.ndarray
    b: np.ndarray
    exposure: np.ndarray
    trials: np.ndarray
    clone: np.ndarray
    contig: np.ndarray
    start: np.ndarray
    length: np.ndarray
    lengths: np.ndarray
    """Bins per chromosome, repeated per clone: the HMM's sequences."""
    planted: np.ndarray
    """`(rows, 2)` each row's planted `(A, B)`."""
    states: np.ndarray
    """`(n_states, 2)` the distinct planted `(A, B)`."""
    truth_log_mu: np.ndarray
    truth_p_binom: np.ndarray
    """The planted states' parameters, row-aligned with `states`."""
    draw_seconds: float
    build_seconds: float

    @property
    def n_states(self) -> int:
        return int(self.states.shape[0])

    @property
    def truth_label(self) -> np.ndarray:
        """Each row's index into `states`."""
        index = {tuple(s): k for k, s in enumerate(self.states.tolist())}
        return np.array(
            [index[tuple(p)] for p in self.planted.tolist()], dtype=np.int64
        )


def floored_bins(chrom: np.ndarray, weight: np.ndarray, floor: float) -> np.ndarray:
    """Each 1 Mb bin's merged bin: a chromosome's bins joined in order until their `weight` reaches `floor`.

    `chrom` and `weight` are per 1 Mb bin in genome order. A chromosome's
    unclosed remainder joins its last closed bin, or is its only bin when
    none closed. Returns a contiguous, increasing id per 1 Mb bin.
    """
    merged = np.empty(chrom.size, dtype=np.int64)
    current = 0
    for name in dict.fromkeys(chrom.tolist()):
        groups: list[list[int]] = []
        run: list[int] = []
        held = 0.0
        for k in np.flatnonzero(chrom == name):
            run.append(int(k))
            held += float(weight[k])
            if held >= floor:
                groups.append(run)
                run, held = [], 0.0
        if run:
            if groups:
                groups[-1].extend(run)
            else:
                groups.append(run)
        for group in groups:
            merged[group] = current
            current += 1
    return merged


def problems(
    manifest_path: Path, n: int | None = None, realizations: int | None = None
) -> Iterator[KnownCopyProblem]:
    """Each realization's problem in turn, `n` of them if given; `realizations` overrides the manifest's count."""
    import pandas as pd
    import scipy.sparse as sp

    from port.sim import draw as d
    from port.sim.laws import allele_share

    manifest = d.read_manifest(manifest_path)
    if realizations is not None:
        from dataclasses import replace

        manifest = replace(
            manifest,
            tables=d._merge(
                manifest.tables, {"sample": {"realizations": int(realizations)}}
            ),
        )
    tree = d.draw_tree(
        manifest,
        np.random.default_rng(np.random.SeedSequence(manifest.seed).spawn(3)[0]),
    )
    clones = ("normal", *manifest.tumour)
    lengths = np.asarray(manifest.genome["chromosome_lengths"], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum((lengths + BIN - 1) // BIN)[:-1]])
    n_bins = int(((lengths + BIN - 1) // BIN).sum())
    bin_chrom = np.repeat(np.arange(1, lengths.size + 1), (lengths + BIN - 1) // BIN)
    bin_start = np.concatenate([np.arange(0, length, BIN) for length in lengths])

    def bins(chrom: np.ndarray, pos: np.ndarray) -> np.ndarray:
        index = np.asarray(chrom).astype(int) - 1
        return np.asarray(offsets[index] + np.asarray(pos) // BIN, dtype=np.int64)

    baseline = pd.read_csv(
        manifest.resolve(manifest.reference["baseline"]), sep="\t", comment="#"
    )
    gene_chrom = baseline["chrom"].astype(str).str.removeprefix("chr").to_numpy()
    keep = np.isin(gene_chrom, [str(c) for c in range(1, lengths.size + 1)])
    gene_bin = np.full(gene_chrom.size, -1)
    gene_bin[keep] = bins(
        gene_chrom[keep], ((baseline.cdsStart + baseline.cdsEnd) // 2).to_numpy()[keep]
    )
    _, snp_chrom, snp_pos = d._snps(manifest)
    snp_ok = np.isin(snp_chrom, [str(c) for c in range(1, lengths.size + 1)])
    snp_bin = np.full(snp_chrom.size, -1)
    snp_bin[snp_ok] = bins(snp_chrom[snp_ok], snp_pos[snp_ok])
    snp_copies = d.clone_copies(tree, clones, snp_chrom, snp_pos)

    def gather(columns: np.ndarray, width: int) -> Any:
        rows = np.flatnonzero(columns >= 0)
        return sp.csr_matrix(
            (np.ones(rows.size), (rows, columns[rows])), shape=(columns.size, width)
        )

    opened = time.perf_counter()
    for realized in d.realize(manifest, None):
        if n is not None and realized.index >= n:
            return
        drawn = time.perf_counter()
        (labels,) = realized.truth.labels
        (counts,) = realized.counts
        member = sp.csr_matrix(
            (np.ones(labels.size), (labels, np.arange(labels.size))),
            shape=(len(clones), labels.size),
        )
        umi_1mb = np.asarray(
            (member @ counts @ gather(gene_bin, n_bins)).todense()
        )  # (clones, 1 Mb bins)
        merged = floored_bins(bin_chrom, umi_1mb[0], FLOOR)
        n_merged = int(merged.max()) + 1
        to_merged = sp.csr_matrix(
            (np.ones(n_bins), (np.arange(n_bins), merged)), shape=(n_bins, n_merged)
        )
        umi = umi_1mb @ to_merged  # (clones, bins)
        hap_b = np.where(
            realized.phase[None, :], realized.a.toarray(), realized.b.toarray()
        )
        snp_to_merged = gather(snp_bin, n_bins) @ to_merged
        b = member @ sp.csr_matrix(hap_b) @ snp_to_merged
        trials = member @ (realized.a + realized.b) @ snp_to_merged
        b, trials = np.asarray(b.todense()), np.asarray(trials.todense())
        share = umi[0] / umi[0].sum()
        exposure = share[None, :] * umi.sum(axis=1, keepdims=True)

        # NB each bin's planted state: the state covering most of its SNPs, per clone
        planted = np.zeros((len(clones), n_merged, 2), dtype=np.int64)
        snp_merged = np.where(snp_bin >= 0, merged[np.maximum(snp_bin, 0)], -1)
        for k in range(len(clones)):
            codes = snp_copies[:, k, 0] * 100 + snp_copies[:, k, 1]
            table = pd.crosstab(snp_merged[snp_merged >= 0], codes[snp_merged >= 0])
            mode = (
                table.idxmax(axis=1)
                .reindex(range(n_merged))
                .fillna(101)
                .astype(int)
                .to_numpy()
            )
            planted[k, :, 0], planted[k, :, 1] = mode // 100, mode % 100

        chrom_of = bin_chrom[np.searchsorted(merged, np.arange(n_merged))]
        start_of = bin_start[np.searchsorted(merged, np.arange(n_merged))]
        end_of = np.append(start_of[1:], 0)
        end_of = np.where(
            np.append(chrom_of[1:], -1) == chrom_of, end_of, lengths[chrom_of - 1]
        )
        per_chrom = np.bincount(chrom_of - 1, minlength=lengths.size)
        states = np.unique(planted.reshape(-1, 2), axis=0)
        truth_log_mu, truth_p = [], []
        for a, bb in states.astype(float):
            clone = clones[1]
            f = manifest.normal_frac(clone)
            depth = (
                (1 - f) * (a + bb) / 2 + f
                if manifest.model["admixture"] == "cell"
                else (a + bb) / 2
            )
            truth_log_mu.append(np.log(max(depth, 1e-3)))
            truth_p.append(
                1
                - float(
                    allele_share(
                        np.array([a]), np.array([bb]), f, manifest.model["admixture"]
                    )[0]
                )
            )
        yield KnownCopyProblem(
            realized.index,
            umi.ravel().astype(np.float64), b.ravel(), exposure.ravel(), trials.ravel(),
            np.repeat(np.arange(len(clones)), n_merged), np.tile(chrom_of.astype(str), len(clones)),
            np.tile(start_of, len(clones)).astype(np.float64), np.tile(end_of - start_of, len(clones)).astype(np.float64),
            np.tile(per_chrom[per_chrom > 0], len(clones)), planted.reshape(-1, 2), states,
            np.asarray(truth_log_mu), np.asarray(truth_p), drawn - opened, time.perf_counter() - drawn,
        )  # fmt: skip
        opened = time.perf_counter()
