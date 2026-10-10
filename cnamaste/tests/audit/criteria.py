"""The staged run's units and counts, recomputed from its inputs, for the construction rules and the config criteria.

`Units` holds, per row of `df_gene_snp` and per gene, what the inputs say: the
SNP-covering UMI (A+B over all spots of the row's SNP), the phased B and A+B
counts over the normal candidates, and each gene's UMI. A level's count is
that, summed over the level's labels: `aggregate` over the gene rows, or a
sum over the SNP rows of the stage frame that labels them. Nothing here
reads a stage's own counts back.

`Bins` classifies each bin of a greedy binning (`omics.greedy_binning_nobreak`,
omics.py:16-105, inside the runs `create_bin_ranges` cuts, omics.py:867-904)
by the mechanism that let it miss a floor or pass `max_binlength`.
"""

from __future__ import annotations

from ast import literal_eval
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import betabinom

from audit.segments import DROPPED
from cnamaste.config import set_global_config
from cnamaste.hmm_emission import Weighted_BetaBinom
from cnamaste.hmm_utils import get_em_solver_params

FRAMES = {
    "blocks": ("02_blocks/assign_initial_blocks/out", "block_id"),
    "bins": ("04_bins/create_bin_ranges/out", "bin_id"),
    "kept_bins": ("06_normal/normal_baf_bin_filter/out/0", "bin_id"),
    "rebinned": ("07_rebin/create_bin_ranges/out", "bin_id"),
}
"""Level -> the stage frame whose column labels every row of `df_gene_snp`, genes and SNPs."""


class RuleBroken(AssertionError):
    """The code departs from a rule as stated, by a mechanism the check names: what a strict xfail expects."""


@dataclass
class Units:
    sim: Any
    replayed: Any
    lineage: Any
    config: dict[str, Any]
    gene_counts: np.ndarray
    normal: np.ndarray
    rows: dict[str, np.ndarray]
    """Per row of `df_gene_snp`, 0 on a gene row: `snp_umi` (A+B over all spots), and over the normal candidates
    `normal_b` (the phased B count) and `normal_total` (A+B)."""

    def setting(self, key: str) -> Any:
        """`/config`'s value at a dotted key."""
        value: Any = self.config
        for part in key.split("."):
            value = value[part]
        return value

    def label(self, level: str) -> np.ndarray:
        path, key = FRAMES[level]
        return self.sim.stored(path)[key].fillna(DROPPED).to_numpy(dtype=np.int64)

    def per_unit(self, level: str, row: str) -> np.ndarray:
        """`rows[row]` summed over each unit's rows."""
        label = self.label(level)
        kept = label != DROPPED
        return np.bincount(label[kept], weights=self.rows[row][kept], minlength=self.lineage.levels[level].n_segments)

    def umi(self, level: str, *, normal: bool = False) -> np.ndarray:
        """Each unit's UMI over all spots, or over the normal candidates: `aggregate` of the gene counts."""
        counts = self.gene_counts[:, self.normal] if normal else self.gene_counts
        return self.lineage.levels[level].aggregate(counts.sum(axis=1))

    def span(self, level: str) -> np.ndarray:
        """Each unit's `END` of its last row less `START` of its first, as `create_bin_ranges` measures it (omics.py:821-827)."""
        path, key = FRAMES[level]
        grouped = self.sim.stored(path).groupby(key).agg({"START": "first", "END": "last"})
        return (grouped["END"] - grouped["START"]).to_numpy()


def units_of(sim: Any, replayed: Any, lineage: Any, gene_counts: np.ndarray) -> Units:
    load = "00_inputs/load_input_data/out"
    a, b = (np.asarray(replayed.value(f"{load}/{k}")) for k in (4, 5))
    snp_ids = pd.Index(np.asarray(replayed.value(f"{load}/6")).astype(str))
    frame = sim.stored(FRAMES["bins"][0])
    snp = frame["snp_id"].notna().to_numpy()
    column = snp_ids.get_indexer(frame["snp_id"][snp].astype(str))
    assert np.all(column >= 0)
    normal = np.asarray(sim.stored("06_normal/determine_normal_candidates/out"), dtype=bool)
    a_normal, b_normal = a[normal][:, column].sum(axis=0), b[normal][:, column].sum(axis=0)
    phase = frame["phase"][snp].astype(bool).to_numpy()
    rows = {name: np.zeros(len(frame)) for name in ("snp_umi", "normal_b", "normal_total")}
    rows["snp_umi"][snp] = a[:, column].sum(axis=0) + b[:, column].sum(axis=0)
    rows["normal_b"][snp] = np.where(phase, a_normal, b_normal)
    rows["normal_total"][snp] = a_normal + b_normal
    return Units(sim, replayed, lineage, yaml.safe_load(sim.config["yaml"]), gene_counts, normal, rows)


@dataclass
class Bins:
    """Each bin of `coarse` over the units of `fine`: its floors, its span, and why it may miss either."""

    first: np.ndarray
    stop: np.ndarray
    floors: dict[str, tuple[np.ndarray, float]]
    """Floor key -> (each bin's count, the floor)."""
    span: np.ndarray
    alone: np.ndarray
    """Its run's only bin: a short last bin is merged into the one before unless there is none (omics.py:85-98)."""
    backed: np.ndarray
    """Under a floor because `greedy_binning_nobreak` backed off one unit at max_binlength (omics.py:66-73)."""
    head_short: np.ndarray
    """Without its last unit, it misses a floor: the greedy extension that made it long was needed."""
    ends_run: np.ndarray
    """It ends its run: where a short tail is merged in (omics.py:85-98)."""
    run_edges_kept: bool


def bins_of(u: Units, coarse: str, fine: str, runs: np.ndarray, *, normal: bool) -> Bins:
    quality = u.config["quality"]
    longest = quality["max_binlength"]
    span = u.span(fine)
    unit_floors = {
        "quality.secondary_min_umi": (u.umi(fine), quality["secondary_min_umi"]),
        "quality.secondary_min_snp_umi": (u.per_unit(fine, "snp_umi"), quality["secondary_min_snp_umi"]),
    }
    if normal:
        unit_floors["quality.secondary_min_normal_umi"] = (u.umi(fine, normal=True), quality["secondary_min_normal_umi"])
    over = np.flatnonzero(span > longest)
    edges = np.unique(np.concatenate([[0], np.cumsum(runs), over, over + 1]))
    run = np.searchsorted(edges, np.arange(span.size), side="right") - 1
    parent = u.lineage.levels[coarse].label[u.lineage.levels[fine].first]
    assert np.all(np.diff(parent) >= 0)
    first = np.searchsorted(parent, np.arange(parent.max() + 1), "left")
    stop = np.searchsorted(parent, np.arange(parent.max() + 1), "right")

    def meets(lo: int, hi: int) -> bool:
        return all(values[lo:hi].sum() >= floor for values, floor in unit_floors.values())

    n = first.size
    bins = Bins(
        first, stop,
        {key: (np.add.reduceat(values, first), floor) for key, (values, floor) in unit_floors.items()},
        np.add.reduceat(span, first),
        np.bincount(run[first], minlength=edges.size)[run[first]] == 1,
        np.zeros(n, dtype=bool), np.zeros(n, dtype=bool), stop == edges[run[first] + 1],
        bool(np.all(run[first] == run[stop - 1])),
    )
    for k, (lo, hi) in enumerate(zip(first, stop, strict=True)):
        bins.backed[k] = (not meets(lo, hi) and hi < span.size and run[hi] == run[lo] and meets(lo, hi + 1)
                          and span[lo:hi + 1].sum() >= longest)
        bins.head_short[k] = hi - lo > 1 and not meets(lo, hi - 1)
    return bins


def minor_baf(res: Any, n_states: int, n_blocks: int) -> np.ndarray:
    """(clones, blocks) minor BAF of the phasing fit's MAP states, as `initial_phase_given_partition` forms it (phasing.py:140-155)."""
    pred = np.argmax(np.asarray(res["log_gamma"]), axis=0).reshape(-1, n_blocks)
    p = np.asarray(res["new_p_binom"])[pred % n_states, 0]
    baf = np.where(pred < n_states, p, 1.0 - p)
    minor: np.ndarray = np.minimum(baf, 1.0 - baf)
    return minor


def normal_baf_outside(u: Units) -> np.ndarray:
    """Per first-pass bin: its pooled normal-spot B count lies outside the beta-binomial interval at
    quality.normal_allele_specific_confidence, mean 1/2 and tau floored at 30 (normal_spot.py:935, 954-1000). The pooled
    counts are the inputs' A and B over the normal candidates, phased per SNP; they must equal what the stage pooled."""
    set_global_config(u.replayed.staged.config)
    pooled_b, pooled = u.per_unit("bins", "normal_b"), u.per_unit("bins", "normal_total")
    given = u.replayed.value("06_normal/normal_baf_bin_filter/in")["args"]
    normal = np.asarray(given[6])
    assert np.array_equal(pooled_b, np.asarray(given[1])[:, 1, normal].sum(axis=1))
    assert np.array_equal(pooled, np.asarray(given[3])[:, normal].sum(axis=1))
    ones = np.ones(pooled.size)
    fit = Weighted_BetaBinom(pooled_b, ones, weights=ones, exposure=pooled).fit(**get_em_solver_params())
    tau = max(fit.params[-1], 30)
    low, high = literal_eval(u.setting("quality.normal_allele_specific_confidence"))
    outside: np.ndarray = (pooled_b < betabinom.ppf(low, pooled, 0.5 * tau, 0.5 * tau)) | (
        pooled_b > betabinom.ppf(high, pooled, 0.5 * tau, 0.5 * tau))
    return outside
