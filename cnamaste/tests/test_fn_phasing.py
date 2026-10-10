"""Per-function rows for phasing: the genetic map, the phase-switch kernel, the phased HMM, the phase vote.

Inputs: the staged transition vectors and gene tables, the phasing stage's replayed
result, and CalicoST's genetic map (read once); else a few-element synthetic
input with a closed form.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import scipy.special
import scipy.stats
from cnamaste.hmm_nophasing import get_log_transmat
from cnamaste.hmm_phased import _switch_betabinom_1d, hmm_phased, update_combined_transmat
from cnamaste.recomb import assign_centiMorgans, compute_numbat_phase_switch_prob, get_sitewise_transmat
from cnamaste.reference import get_reference_recomb_rates

from audit.capture import grch38
from audit.fn import Replay, Row, run, table, unchanged


def close(a: Any, b: Any, atol: float = 1e-10) -> None:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape and np.allclose(a, b, atol=atol, equal_nan=True), f"max |diff| {np.nanmax(np.abs(a - b)):.3e}"


def genetic_map(ctx: Any) -> pd.DataFrame:
    def make() -> pd.DataFrame:
        resources = grch38()
        if resources is None:
            pytest.skip("CalicoST's GRCh38_resources not found")
        return get_reference_recomb_rates(str(resources / "genetic_map_GRCh38_merged.tab.gz"))
    return ctx.once("phasing/genetic_map", make)


# --- oracle --------------------------------------------------------------------


def _switch_law(_: Any) -> None:
    cm = np.array([0.0, 0.5, 0.5001, np.nan, 2.0, 0.0, 3.0])
    contigs = [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0), (2, 1)]
    p = compute_numbat_phase_switch_prob(cm, contigs, nu=1.0, min_prob=0.01)
    want = np.full(7, 0.01)
    want[0] = (1 - np.exp(-2 * 0.5)) / 2
    want[5] = (1 - np.exp(-2 * 3.0)) / 2
    close(p, want)  # NB d = 1e-4 cM floors at min_prob; NaN, the contig change and the last entry take min_prob


def _centimorgans(_: Any) -> None:
    table_ = pd.DataFrame({"chrom": [1, 1, 1, 2, 2], "pos": [100, 200, 400, 50, 150], "pos_cm": [1.0, 2.0, 4.0, 0.5, 1.5]})
    positions = [(1, 50), (1, 150), (1, 300), (2, 100), (2, 150)]
    found = assign_centiMorgans(list(positions), table_)
    want = [0.5, 1.5, 3.0, 1.0, 1.5]  # NB before the first marker: linear from (0, 0)
    close(found, want)


def _switched(_: Any) -> None:
    k, n = np.array([0.0, 3.0, 7.0, 10.0]), np.array([10.0, 10.0, 12.0, 10.0])
    p, tau = np.array([0.2, 0.45]), np.array([30.0, 300.0])
    direct = np.stack([scipy.stats.betabinom.logpmf(k, n, p[i] * tau[i], (1 - p[i]) * tau[i]) for i in range(2)])
    flipped = np.stack([scipy.stats.betabinom.logpmf(k, n, (1 - p[i]) * tau[i], p[i] * tau[i]) for i in range(2)])
    close(_switch_betabinom_1d(direct, k, n, p, tau), flipped, 1e-9)


def _phased_emission(_: Any) -> None:
    rng = np.random.default_rng(9)
    total = rng.integers(5, 40, size=(30, 1)).astype(float)
    x = np.stack([np.zeros((30, 1)), rng.binomial(total.astype(int), 0.3).astype(float)], axis=1)
    p, tau = np.array([[0.2], [0.5]]), np.array([[40.0], [40.0]])
    rdr, baf = hmm_phased.compute_emission_probability_nb_betabinom(x, np.zeros((30, 1)), np.zeros((2, 1)), np.ones((2, 1)), total, p, tau)
    want = np.concatenate([np.stack([scipy.stats.betabinom.logpmf(x[:, 1, 0], total[:, 0], p[i, 0] * tau[i, 0], (1 - p[i, 0]) * tau[i, 0]) for i in range(2)]),
                           np.stack([scipy.stats.betabinom.logpmf(x[:, 1, 0], total[:, 0], (1 - p[i, 0]) * tau[i, 0], p[i, 0] * tau[i, 0]) for i in range(2)])])
    assert rdr.shape == (4, 30, 1) and np.allclose(rdr, 0)
    close(baf[:, :, 0], want, 1e-9)


def phased_case() -> dict[str, Any]:
    rng = np.random.default_rng(12)
    return {"emission": rng.normal(size=(4, 7, 1)), "transmat": get_log_transmat(2, 0.9), "start": np.log(np.array([0.6, 0.4])),
            "switch": np.log(rng.uniform(0.01, 0.5, size=7)), "lengths": np.array([3, 4])}


def combined(transmat: np.ndarray, switch: float) -> np.ndarray:
    a = np.exp(transmat)
    s = np.exp(switch)
    return np.log(np.block([[(1 - s) * a, s * a], [s * a, (1 - s) * a]]))


def _phased_forward(h: dict[str, Any]) -> None:
    found = hmm_phased.forward_lattice(h["lengths"], h["transmat"], h["start"], h["emission"], h["switch"])
    e = h["emission"][:, :, 0]
    want = np.zeros_like(e)
    start = np.log(0.5) + np.concatenate([h["start"], h["start"]])
    for begin, length in ((0, 3), (3, 4)):
        want[:, begin] = start + e[:, begin]
        for t in range(begin + 1, begin + length):
            want[:, t] = scipy.special.logsumexp(want[:, t - 1][:, None] + combined(h["transmat"], h["switch"][t - 1]), axis=0) + e[:, t]
    close(found, want)


def _phased_backward(h: dict[str, Any]) -> None:
    found = hmm_phased.backward_lattice(h["lengths"], h["transmat"], h["start"], h["emission"], h["switch"])
    e = h["emission"][:, :, 0]
    want = np.zeros_like(e)
    for begin, length in ((0, 3), (3, 4)):
        for t in range(begin + length - 2, begin - 1, -1):
            want[:, t] = scipy.special.logsumexp(combined(h["transmat"], h["switch"][t]) + (e[:, t + 1] + want[:, t + 1])[None, :], axis=1)
    close(found, want)


def _combined(_: Any) -> None:
    a = get_log_transmat(3, 0.8)
    out = np.empty((6, 6))
    update_combined_transmat(out, 3, a, np.log(0.9), np.log(0.1), False, np.log(0.5))
    close(out, combined(a, np.log(0.1)))
    assert np.allclose(np.exp(out).sum(axis=1), 1), "rows of the paired chain are distributions"


ORACLE: list[Row] = table(
    "oracle",
    ("recomb:compute_numbat_phase_switch_prob", "synthetic: 7 positions, 2 contigs", lambda c: None, _switch_law, "(1 - e^{-2 nu d}) / 2, floored at min_prob; boundaries, NaN and the end take min_prob"),
    ("recomb:assign_centiMorgans", "synthetic: a 5-marker map, 2 contigs", lambda c: None, _centimorgans, "linear interpolation between markers, from (0, 0) before the first"),
    ("hmm_phased:_switch_betabinom_1d", "synthetic: 4 counts, 2 states", lambda c: None, _switched, "the switched BB is scipy's with alpha and beta exchanged"),
    ("hmm_phased:hmm_phased.compute_emission_probability_nb_betabinom", "synthetic: 30 bins, 2 states", lambda c: None, _phased_emission, "states K..2K-1 are the BB at 1 - p; RDR zero without a baseline"),
    ("hmm_phased:hmm_phased.compute_emission_probability_nb_betabinom_coded", "synthetic: 30 bins, 2 states", lambda c: None, _phased_emission, "the coded phased emission is scipy's, both phases"),
    ("hmm_phased:hmm_phased.forward_lattice", "synthetic: 2 states x 2 phases, 7 bins, 2 segments", lambda c: phased_case(), _phased_forward, "the textbook recursion over the paired chain, switch probability per bin"),
    ("hmm_phased:hmm_phased.backward_lattice", "synthetic: 2 states x 2 phases, 7 bins, 2 segments", lambda c: phased_case(), _phased_backward, "the textbook backward recursion over the paired chain"),
    ("hmm_phased:update_combined_transmat", "synthetic: 3 states", lambda c: None, _combined, "[[(1-s)A, sA], [sA, (1-s)A]]; rows sum to one"),
)


# --- invariants ---------------------------------------------------------------------


def _map_rows(m: pd.DataFrame) -> None:
    assert set(m["chrom"].unique()) == {str(i) for i in range(1, 23)}
    for _, g in m.groupby("chrom"):
        assert np.all(np.diff(g["pos"].to_numpy()) >= 0) and np.all(np.diff(g["pos_cm"].to_numpy()) >= 0), "positions and cM ascend within a contig"


def _map_order(m: pd.DataFrame) -> None:
    order = pd.unique(m["chrom"]).tolist()
    assert order == [str(i) for i in range(1, 23)], f"contig order {order[:12]}"


def _transmat_rows(ctx: Any) -> dict[str, Any]:
    stage = "04_bins"
    frame = ctx.sim.stored(f"{stage}/create_bin_ranges/out")
    return {"log": np.asarray(ctx.sim.stored(f"{stage}/get_sitewise_transmat/out")), "chr": frame.groupby("bin_id")["CHR"].first().to_numpy(),
            "min_prob": float(ctx.config.phasing.min_prob), "shift": float(ctx.config.phasing.logphase_shift)}


def _transmat_bounds(d: dict[str, Any]) -> None:
    assert d["log"].shape == d["chr"].shape
    assert np.all(d["log"] <= np.log(0.5) + 1e-12) and np.all(d["log"] >= np.log(d["min_prob"]) - d["shift"] - 1e-12)
    last = np.r_[d["chr"][1:] != d["chr"][:-1], True]
    assert np.allclose(d["log"][last], np.log(d["min_prob"]) - d["shift"]), "the last bin of a contig takes min_prob"


def _transmat_contigs(d: dict[str, Any]) -> None:
    floored = np.isclose(d["log"], np.log(d["min_prob"]) - d["shift"])
    per = {c: floored[d["chr"] == c].mean() for c in range(1, 23)}
    assert max(per.values()) < 0.9, f"contigs with every bin at the floor: {[c for c, f in per.items() if f == 1.0]}"


def _vote(ctx: Any) -> dict[str, Any]:
    given = ctx("03_phasing/initial_phase_given_partition/in")
    return {"out": ctx.replayed.run("03_phasing/initial_phase_given_partition"), "lengths": np.asarray(given["args"][1]),
            "n_clones": len(given["args"][5]), "n_states": int(given["args"][6]), "config": ctx.config}


def _phase_vote(d: dict[str, Any]) -> None:
    res, phase, refined = d["out"]
    n_states, n_clones = d["n_states"], d["n_clones"]
    pred = np.argmax(np.asarray(res["log_gamma"]), axis=0).reshape(n_clones, -1)
    p = np.asarray(res["new_p_binom"])[pred % n_states, 0]
    model = np.where(pred < n_states, p, 1 - p)
    votes = np.where(np.abs(model - 0.5) < 0.1, np.nan, (pred < n_states).astype(float))
    with np.errstate(invalid="ignore"):
        want = np.where(np.isnan(votes).all(axis=0), 0, np.nanmean(votes, axis=0) >= 0.5)
    assert np.array_equal(np.asarray(phase), want.astype(int)), "the phase is the majority of the clones' non-balanced votes"
    minor = np.minimum(model, 1 - model)
    threshold, size = d["config"].phasing.baf_change_threshold, d["config"].phasing.min_new_segment_size
    want_lengths, offset = [], 0
    for le in d["lengths"]:
        s = 0
        for i in range(le):
            if i > s + size and np.any(np.abs(minor[:, offset + i] - minor[:, offset + i - 1]) >= threshold):
                want_lengths.append(i - s)
                s = i
        want_lengths.append(le - s)
        offset += le
    assert np.array_equal(np.asarray(refined), want_lengths), "segments split where the minor BAF jumps, no shorter than the minimum"


def _phased_init(_: Any) -> None:
    m = hmm_phased(params="sp", t=0.3)
    assert (m.params, m.t) == ("sp", 0.3)


INVARIANT: list[Row] = table(
    "invariant",
    ("reference:get_reference_recomb_rates", "CalicoST genetic map", genetic_map, _map_rows, "chr1-22 only, chr prefix dropped; positions and cM ascend within a contig"),
    ("reference:get_reference_recomb_rates", "CalicoST genetic map", genetic_map, _map_order, "contigs in numeric order, as assign_centiMorgans' sweep needs",
     "Ticket#438: get_reference_recomb_rates sorts contigs as text (1, 10..19, 2, 20..22, 3..9), so assign_centiMorgans' numeric sweep gives chr2-9 one constant cM"),
    ("recomb:get_sitewise_transmat", "04_bins/get_sitewise_transmat out, 04_bins/create_bin_ranges out", _transmat_rows, _transmat_bounds, "one entry per bin in [log min_prob - shift, log 1/2]; contig ends at the floor"),
    ("recomb:get_sitewise_transmat", "04_bins/get_sitewise_transmat out, 04_bins/create_bin_ranges out", _transmat_rows, _transmat_contigs, "no contig has every bin at the min_prob floor",
     "Ticket#438: get_reference_recomb_rates sorts contigs as text, so every bin on chr2-9 (861 of 1,847 on easy) takes the min_prob floor: no phase switch there"),
    ("phasing:initial_phase_given_partition", Replay("03_phasing/initial_phase_given_partition, replayed"), _vote, _phase_vote, "phase = majority vote of non-balanced clones; refined lengths split at minor-BAF jumps (recomputed)"),
    ("hmm_phased:hmm_phased.__init__", "synthetic", lambda c: None, _phased_init, "stores params and t"),
)


# --- captured -------------------------------------------------------------------------


def _replay_transmat(ctx: Any) -> tuple[np.ndarray, np.ndarray]:
    given = ctx.sim.stored("02_blocks/get_sitewise_transmat/in")
    frame = given["args"][1]
    head = frame[frame["block_id"] < 200]

    resources = grch38()
    if resources is None:
        pytest.skip("CalicoST's GRCh38_resources not found")
    out = unchanged(get_sitewise_transmat, "block_id", head, str(resources / "genetic_map_GRCh38_merged.tab.gz"), 1.0, -2.0)
    return out, np.asarray(ctx.sim.stored("02_blocks/get_sitewise_transmat/out"))[:200]


def _prefix(pair: tuple[np.ndarray, np.ndarray]) -> None:
    found, want = pair
    close(found[:-1], want[:-1], 0)  # NB the slice's last block is a contig end only in the slice


CAPTURED: list[Row] = table(
    "captured",
    ("recomb:get_sitewise_transmat", "02_blocks/get_sitewise_transmat in, the first 200 blocks", _replay_transmat, _prefix, "on the first 200 blocks it returns the run's first 199 entries; no input mutation"),
)


@pytest.mark.parametrize("row", ORACLE, ids=[r.id for r in ORACLE])
def test_oracle(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """Closed forms and scipy's switched beta-binomial; the paired lattices by the textbook recursion."""
    run(row, ctx, request)


@pytest.mark.parametrize("row", INVARIANT, ids=[r.id for r in INVARIANT])
def test_invariant(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """Map order, kernel bounds, the phase vote recomputed."""
    run(row, ctx, request)


@pytest.mark.parametrize("row", CAPTURED, ids=[r.id for r in CAPTURED])
def test_captured(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """A slice of the run's kernel, recomputed."""
    run(row, ctx, request)
