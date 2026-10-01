"""The stage files of `port.extensions.outputs`: schema, joins, completeness (#613).

A synthetic run drawn from the shifted NB/BB model the HMM fits: three
clones, three states, two chromosomes of 15 bins, and a capture and a
lattice decode as `run_cnaster_port` keeps them. Its `log_gamma`,
`pred_cnv` and `llf` are the referee's own forward-backward on the pooled
counts, so the run is self-consistent, as `cnaster`'s final fit is not
(`port.extensions.outputs`, `hmrf.py:784`).

The referee recomputes from the written files alone, with `scipy.stats`'
negative binomial, a beta-binomial summed term by term and a NumPy
forward-backward: a second implementation of the emission and of the chain,
not `port`'s.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import gammaln, logsumexp

N_STATES, LENGTHS, N_CLONES, SPOTS_PER_CLONE = 3, np.array([15, 15]), 3, 20
N_OBS = int(LENGTHS.sum())
LOG_MU = np.log([1.0, 1.5, 0.5])
P_BINOM = np.array([0.5, 1.0 / 3.0, 0.1])
ALPHA, TAU, STAY = 0.05, 50.0, 0.9
MAX_TOTAL_COPY, PARSIMONY, DECODE_STAY = 4, 0.5, 1.0 - 1e-4
GENES_PER_BIN = 2
TOLERANCE = 1e-10
"""Relative, on a log-likelihood of about 1e3 nats: summation order."""


# --- the referee -----------------------------------------------------------------


def _nb(x: np.ndarray, mean: np.ndarray, alpha: float) -> np.ndarray:
    alpha = max(alpha, 1e-10)
    out = stats.nbinom.logpmf(x, 1.0 / alpha, 1.0 / (1.0 + alpha * mean))
    return np.where(mean > 0, out, 0.0)


def _rise(x: float, m: int) -> float:
    """`log x (x + 1) ... (x + m - 1)`, summed term by term: exact at any `x`."""
    return float(np.sum(np.log(x + np.arange(m))))


def _bb(k: np.ndarray, n: np.ndarray, p: Any, tau: float) -> np.ndarray:
    """The beta-binomial log pmf by rising factorials, broadcast over `k`, `n`, `p`.

    Not `scipy.stats.betabinom`, whose `lgamma` differences lose about 1e-6
    nats at the decode's fitted `tau` of 1e8 (#561).
    """
    k, n, p = np.broadcast_arrays(np.asarray(k), np.asarray(n), np.asarray(p))
    out = np.empty(k.shape)

    for index in np.ndindex(k.shape):
        a = max(float(p[index]) * tau, 1e-10)
        b = max((1.0 - float(p[index])) * tau, 1e-10)
        kk, nn = int(k[index]), int(n[index])
        out[index] = (
            gammaln(nn + 1)
            - gammaln(kk + 1)
            - gammaln(nn - kk + 1)
            + _rise(a, kk)
            + _rise(b, nn - kk)
            - _rise(a + b, nn)
        )

    return out


def _forward_backward(
    log_emission: np.ndarray,
    log_transmat: np.ndarray,
    log_startprob: np.ndarray,
    lengths: np.ndarray,
) -> tuple[float, np.ndarray]:
    """`(log-likelihood, log posterior)` of an `(n_states, n_obs)` emission."""
    log_gamma = np.empty_like(log_emission)
    total, start = 0.0, 0

    for length in lengths:
        e = log_emission[:, start : start + length]
        alpha, beta = np.empty_like(e), np.zeros_like(e)
        alpha[:, 0] = log_startprob + e[:, 0]
        for t in range(1, length):
            alpha[:, t] = (
                logsumexp(alpha[:, t - 1, None] + log_transmat, axis=0) + e[:, t]
            )
        for t in range(length - 2, -1, -1):
            beta[:, t] = logsumexp(
                log_transmat + (e[:, t + 1] + beta[:, t + 1]), axis=1
            )
        total += float(logsumexp(alpha[:, -1]))
        joint = alpha + beta
        log_gamma[:, start : start + length] = joint - logsumexp(joint, axis=0)
        start += length

    return total, log_gamma


def _hmm_emission(
    counts: pd.DataFrame, states: pd.DataFrame, shift: float
) -> np.ndarray:
    """The HMM's `(n_states, n_obs)` emission of one clone, from file columns."""
    return np.stack(
        [
            _nb(
                counts["X_depth"].to_numpy(),
                counts["base_nb_mean"].to_numpy() * np.exp(row.log_mu - shift),
                row.alphas,
            )
            + _bb(
                counts["X_allele"].to_numpy(),
                counts["total_bb_RD"].to_numpy(),
                row.p_binom,
                row.taus,
            )
            for row in states.sort_values("state").itertuples()
        ]
    )


def _lattice() -> np.ndarray:
    return np.array(
        [
            (a, b)
            for a in range(MAX_TOTAL_COPY + 1)
            for b in range(MAX_TOTAL_COPY + 1)
            if 0 < a + b <= MAX_TOTAL_COPY
        ]
    )


def _decode_emission(counts: pd.DataFrame, clone: pd.Series) -> np.ndarray:
    """The lattice decode's `(n_pairs, n_obs)` emission and prior, from file columns."""
    return _decode_emission_on(_lattice(), counts, clone)


def _chain(n: int, stay: float) -> tuple[np.ndarray, np.ndarray]:
    off = (1.0 - stay) / (n - 1)
    return np.log(np.full((n, n), off) + np.eye(n) * (stay - off)), np.full(
        n, -np.log(n)
    )


# --- the run -----------------------------------------------------------------------


def _draw(seed: int = 7) -> dict[str, Any]:
    """Counts per spot from the shifted model, and the referee's fit of them."""
    rng = np.random.default_rng(seed)
    n_spots = N_CLONES * SPOTS_PER_CLONE
    assignment = rng.permutation(np.repeat(np.arange(N_CLONES), SPOTS_PER_CLONE))
    planted = np.zeros((N_OBS, N_CLONES), dtype=np.int64)
    planted[5:12, 1], planted[18:25, 1] = 1, 2
    planted[3:9, 2], planted[20:28, 2] = 2, 1

    weight = rng.dirichlet(np.full(N_OBS, 20.0))
    depth = rng.uniform(300.0, 600.0, n_spots)
    base = weight[:, None] * depth[None, :]
    log_lambda = np.log(weight)
    shift = np.array(
        [0.0, *(logsumexp(LOG_MU[planted[:, c]] + log_lambda) for c in (1, 2))]
    )

    rate = np.exp(LOG_MU[planted[:, assignment]] - shift[assignment])
    x = rng.negative_binomial(1.0 / ALPHA, 1.0 / (1.0 + ALPHA * base * rate))
    trials = rng.poisson(0.4 * base)
    p = P_BINOM[planted[:, assignment]]
    allele = stats.betabinom.rvs(trials, p * TAU, (1 - p) * TAU, random_state=rng)

    single_x = np.stack([x, allele], axis=1).astype(np.float64)
    log_transmat = np.log(
        np.full((N_STATES, N_STATES), (1 - STAY) / (N_STATES - 1))
        + np.eye(N_STATES) * (STAY - (1 - STAY) / (N_STATES - 1))
    )
    log_startprob = np.full(N_STATES, -np.log(N_STATES))
    log_gamma = np.empty((N_STATES, N_OBS, N_CLONES))
    llf = []

    for c in range(N_CLONES):
        spots = assignment == c
        counts = pd.DataFrame(
            {
                "X_depth": x[:, spots].sum(axis=1),
                "X_allele": allele[:, spots].sum(axis=1),
                "base_nb_mean": base[:, spots].sum(axis=1),
                "total_bb_RD": trials[:, spots].sum(axis=1),
            }
        )
        states = pd.DataFrame(
            {
                "state": np.arange(N_STATES),
                "log_mu": LOG_MU,
                "alphas": ALPHA,
                "p_binom": P_BINOM,
                "taus": TAU,
            }
        )
        ll, log_gamma[:, :, c] = _forward_backward(
            _hmm_emission(counts, states, shift[c]),
            log_transmat,
            log_startprob,
            LENGTHS,
        )
        llf.append(ll)

    res = {
        "n_states": N_STATES,
        "new_log_mu": LOG_MU[:, None],
        "new_alphas": np.full((N_STATES, 1), ALPHA),
        "new_p_binom": P_BINOM[:, None],
        "new_taus": np.full((N_STATES, 1), TAU),
        "new_log_startprob": log_startprob,
        "new_log_transmat": log_transmat,
        "new_log_mu_shift": shift,
        "log_gamma": log_gamma,
        "pred_cnv": np.argmax(log_gamma, axis=0),
        "llf": float(np.sum(llf)),
        "total_llf": np.nan,
        "new_assignment": assignment,
        "prev_assignment": assignment,
    }
    return {
        "single_X": single_x,
        "base": base,
        "total": trials.astype(np.float64),
        "res": res,
        "llf": np.array(llf),
    }


def _lineage(n_obs: int) -> Any:
    from port.extensions.segments import Genes, Lineage, Segmentation

    n_genes = n_obs * GENES_PER_BIN
    chromosome = np.repeat(np.arange(1, LENGTHS.size + 1), LENGTHS * GENES_PER_BIN)
    start = np.concatenate([np.arange(n) for n in LENGTHS * GENES_PER_BIN]) * 500
    genes = Genes(
        chromosome,
        start,
        start + 400,
        np.arange(n_genes),
        np.arange(n_genes),
        np.array([f"G{k}" for k in range(n_genes)], dtype=object),
    )
    bins = np.arange(n_genes) // GENES_PER_BIN
    filtered = bins.copy()
    filtered[7] = -1  # NB one gene the filter drops, inside its bin
    levels = {
        "blocks": Segmentation.of(genes, np.arange(n_genes), name="blocks"),
        "bins": Segmentation.of(genes, bins, name="bins"),
        "bins-filtered": Segmentation.of(genes, filtered, name="bins-filtered"),
        "bins-floored": Segmentation.of(genes, filtered, name="bins-floored"),
        "bins.2": Segmentation.of(genes, filtered, name="bins.2"),
    }
    return Lineage(genes=genes, levels=levels, floor=(0.0, np.full(n_genes, 10.0), 0.0))


def _write(tmp_path: Path) -> tuple[Path, dict[str, Any], Any]:
    """The run directory, written as `cnaster` and `run_cnaster_port` write it."""
    from port.extensions.copy_errors import Captured
    from port.extensions.copy_likelihood import captured_clones, lattice_decode
    from port.extensions.outputs import RunRecord, write_outputs
    from port.extensions.samples import Recorded, Samples
    from port.patch.integer_copy import _modal

    drawn = _draw()
    res = drawn["res"]
    captured = Captured(drawn["single_X"], LENGTHS, drawn["base"], drawn["total"], res)
    decode = lattice_decode(
        captured_clones(captured) or [],
        normal_clone=0,
        max_total_copy=MAX_TOTAL_COPY,
        lengths=LENGTHS,
        stay=DECODE_STAY,
        parsimony=PARSIMONY,
    )

    chromosome = np.repeat(np.arange(1, LENGTHS.size + 1), LENGTHS)
    start = np.concatenate([np.arange(n) for n in LENGTHS]) * 1000
    seglevel = pd.DataFrame({"CHR": chromosome, "START": start, "END": start + 900})
    perstate = pd.DataFrame(index=range(N_STATES))

    for c in range(N_CLONES):
        path = res["pred_cnv"][:, c]
        pairs = decode.pairs[c]
        seglevel[f"clone{c} Z"] = path
        seglevel[f"clone{c} A"], seglevel[f"clone{c} B"] = pairs.T
        modal = _modal(pairs, path, N_STATES)
        perstate[f"clone{c} A"], perstate[f"clone{c} B"] = modal.T

    run = tmp_path / "clone3_rectangle0_w1.0"
    run.mkdir()
    seglevel.to_csv(run / "cnv_seglevel.tsv", sep="\t", index=False)
    perstate.to_csv(run / "cnv_perstate.tsv", sep="\t", index=False)
    np.savez(run / f"rdrbaf_final_nstates{N_STATES}_smp.npz", **res)

    barcodes = np.array([f"spot_{k}" for k in range(res["new_assignment"].size)])
    ids = (np.arange(barcodes.size) % 2).astype(np.int64)
    labels = pd.DataFrame(
        {
            "barcode": barcodes,
            "sample_id": barcodes,
            "x": np.arange(barcodes.size) % 10,
            "y": np.arange(barcodes.size) // 10,
            "clone_label": res["new_assignment"],
        }
    )
    # NB sorted by barcode, as `cnaster` writes it: not the run's spot order.
    labels = labels.sort_values("barcode", key=lambda b: b.astype(str))
    labels.to_csv(run / "clone_labels.tsv", sep="\t", index=False)
    labels.assign(clone_label=np.minimum(labels["clone_label"], 1)).to_csv(
        run / "baf_clone_labels.tsv", sep="\t", index=False
    )

    record = RunRecord(
        lineage=_lineage(N_OBS),
        samples=Recorded(Samples(("S1", "S2"), ids), barcodes),
        captured=captured,
        decode=decode,
    )
    write_outputs(run, flags={"copy_decode": "lattice_decode (lattice)"}, record=record)
    return run, drawn, decode


@pytest.fixture(scope="module")
def written(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict[str, Any], Any]:
    return _write(tmp_path_factory.mktemp("stages"))


def _read(run: Path, name: str) -> pd.DataFrame:
    # NB `round_trip`, so a float reads back as the one the writer held.
    return pd.read_csv(run / name, sep="\t", float_precision="round_trip")


# --- the guards --------------------------------------------------------------------


@pytest.mark.infra
def test_every_file_carries_the_schema_s_columns_and_types(
    written: tuple[Path, dict[str, Any], Any],
) -> None:
    """Each file's header is `SCHEMA`'s, in order; an `int` column reads as
    integers and a `bool` one as booleans; `run.json` is strict JSON with
    `RUN_KEYS` and string keys, and `cnaster`'s files are left as written."""
    from port.extensions.outputs import RUN_KEYS, SCHEMA

    run, _, _ = written

    for name, columns in SCHEMA.items():
        table = _read(run, name)
        names = [c.name for c in columns]
        if name == "cnv_lineage.tsv":
            # NB a step the run did not take has no column, the floor
            #    without `--sal` say; what is written is in the schema's order.
            assert list(table.columns) == [n for n in names if n in table], name
            assert {"gene_index", "segment_final"} <= set(table.columns)
        else:
            assert list(table.columns) == names, name
        for column in columns:
            values = table[column.name]
            if column.kind == "int":
                assert pd.api.types.is_integer_dtype(values), (name, column.name)
            elif column.kind == "bool":
                assert pd.api.types.is_bool_dtype(values), (name, column.name)
            elif column.kind == "float":
                # NB `%.17g` writes an integral float as `20`, read back as int.
                assert pd.api.types.is_numeric_dtype(values), (name, column.name)
                assert not pd.api.types.is_bool_dtype(values), (name, column.name)

    manifest = json.loads((run / "run.json").read_text(), parse_constant=pytest.fail)
    assert tuple(manifest) == RUN_KEYS
    for key in ("clones", "clone_label_decode", "samples"):
        assert all(isinstance(k, str) for k in manifest[key]), key
    assert manifest["lengths"] == LENGTHS.tolist()
    assert manifest["max_total_copy"] == MAX_TOTAL_COPY
    assert pd.read_csv(run / "clone_labels.tsv", sep="\t").columns.tolist() == [
        "barcode",
        "sample_id",
        "x",
        "y",
        "clone_label",
    ]


@pytest.mark.infra
def test_the_stage_files_join_on_their_keys(
    written: tuple[Path, dict[str, Any], Any],
) -> None:
    """Every `(clone, bin)` once in both stages' bin files, every clone in
    both clone files and `run.json`; the label maps are bijections where
    they say so; segments tile each clone's bins and expand to its pairs;
    genes join the lineage's final bin; `mu` is `exp(log_mu - log_mu_shift)`
    (#623) and `rdr_observed` the counts' ratio, both to 1e-12."""
    run, drawn, _ = written
    manifest = json.loads((run / "run.json").read_text())
    clones = sorted(int(k) for k in manifest["clones"])
    keys = {(c, b) for c in clones for b in range(N_OBS)}

    hmm_bins, copy_bins = (
        _read(run, "cnv_hmm_bins.tsv"),
        _read(run, "cnv_copy_bins.tsv"),
    )
    assert (
        set(zip(hmm_bins.clone, hmm_bins.bin, strict=True))
        == keys
        == set(zip(copy_bins.clone, copy_bins.bin, strict=True))
    )
    assert len(hmm_bins) == len(copy_bins) == len(keys)
    assert _read(run, "cnv_bins.tsv")["bin"].tolist() == list(range(N_OBS))
    for name in ("cnv_hmm_clones.tsv", "cnv_copy_clones.tsv"):
        assert sorted(_read(run, name)["clone"]) == clones, name
    assert sorted(manifest["clones"].values()) == list(range(len(clones)))

    states = _read(run, "cnv_hmm_states.tsv")
    copy_states = _read(run, "cnv_copy_states.tsv")
    assert set(zip(copy_states.clone, copy_states.state, strict=True)) == {
        (c, s) for c in clones for s in states.state
    }
    transmat = _read(run, "cnv_hmm_transmat.tsv").pivot_table(
        index="state", columns="state_next", values="log_transmat"
    )
    np.testing.assert_allclose(logsumexp(transmat.to_numpy(), axis=1), 0.0, atol=1e-12)

    labels = _read(run, "spot_labels.tsv")
    decode_map = {int(k): v for k, v in manifest["clone_label_decode"].items()}
    assert labels["clone_label_decode"].tolist() == [
        decode_map[c] for c in labels["clone_label"]
    ]
    assert set(labels["clone_label"]) <= set(clones)
    assert {int(k): v for k, v in manifest["samples"].items()} == {0: "S1", 1: "S2"}
    assert labels["sample"].tolist() == [
        manifest["samples"][str(i)] for i in labels["sample_id"]
    ]
    spot = labels["barcode"].str.removeprefix("spot_").astype(int)
    np.testing.assert_array_equal(
        labels["n_umi"], drawn["single_X"][:, 0, spot].sum(axis=0)
    )
    assert labels["sample_id"].tolist() == (spot % 2).tolist()

    segments = _read(run, "cnv_copy_segments.tsv")
    lineage = _read(run, "cnv_lineage.tsv")
    for c in clones:
        own = segments[segments.clone == c]
        cover = np.concatenate(
            [
                np.arange(a, b + 1)
                for a, b in zip(own.bin_first, own.bin_last, strict=True)
            ]
        )
        np.testing.assert_array_equal(cover, np.arange(N_OBS))
        pairs = copy_bins[copy_bins.clone == c].sort_values("bin")[["A", "B"]]
        np.testing.assert_array_equal(
            np.repeat(own[["A", "B"]].to_numpy(), own.n_bins, axis=0), pairs.to_numpy()
        )
        first = lineage.set_index("gene_index").loc[own.gene_first, "segment_final"]
        np.testing.assert_array_equal(first.to_numpy(), own.bin_first.to_numpy())

    genes = _read(run, "cnv_copy_genes.tsv").merge(
        copy_bins, on=["clone", "bin"], suffixes=("", "_bin")
    )
    np.testing.assert_array_equal(genes[["A", "B"]], genes[["A_bin", "B_bin"]])
    named = lineage.set_index("gene_index").loc[genes.gene_index]
    np.testing.assert_array_equal(genes["gene"], named["gene"])
    np.testing.assert_array_equal(genes["bin"], named["segment_final"])
    kept = lineage[lineage.segment_final >= 0]
    assert len(genes) == len(clones) * len(kept)
    np.testing.assert_array_equal(
        _read(run, "cnv_bins.tsv")["n_genes"], np.bincount(kept.segment_final)
    )

    joined = hmm_bins.merge(states, left_on="hmm_state", right_on="state").merge(
        _read(run, "cnv_hmm_clones.tsv"), on="clone"
    )
    np.testing.assert_allclose(
        joined["mu"], np.exp(joined["log_mu"] - joined["log_mu_shift"]), rtol=1e-12
    )
    np.testing.assert_allclose(
        hmm_bins["rdr_observed"],
        hmm_bins["X_depth"] / hmm_bins["base_nb_mean"],
        rtol=1e-12,
    )


@pytest.mark.oracle
def test_the_hmm_files_recompute_the_run_s_likelihood_and_posterior(
    written: tuple[Path, dict[str, Any], Any],
) -> None:
    """From `cnv_hmm_*` alone, the referee's forward-backward returns each
    clone's `llf` and the run's total to 1e-10 relative, the posterior of
    `hmm_state` to 1e-10 and `hmm_state` itself as its argmax; and
    `outputs.hmm_log_likelihoods` agrees with the referee to the same."""
    from port.extensions.outputs import hmm_log_likelihoods

    run, drawn, _ = written
    manifest = json.loads((run / "run.json").read_text())
    states = _read(run, "cnv_hmm_states.tsv")
    transmat = _read(run, "cnv_hmm_transmat.tsv").pivot_table(
        index="state", columns="state_next", values="log_transmat"
    )
    clones = _read(run, "cnv_hmm_clones.tsv").set_index("clone")
    bins = _read(run, "cnv_hmm_bins.tsv")
    found = {}

    for clone, row in clones.iterrows():
        own = bins[bins.clone == clone].sort_values("bin")
        found[clone], log_gamma = _forward_backward(
            _hmm_emission(own, states, row["log_mu_shift"]),
            transmat.to_numpy(),
            states["log_startprob"].to_numpy(),
            np.asarray(manifest["lengths"]),
        )
        held = np.exp(log_gamma[own.hmm_state.to_numpy(), np.arange(N_OBS)])
        np.testing.assert_allclose(held, own.hmm_state_probability, rtol=TOLERANCE)
        np.testing.assert_array_equal(np.argmax(log_gamma, axis=0), own.hmm_state)

    np.testing.assert_allclose(list(found.values()), drawn["llf"], rtol=TOLERANCE)
    assert sum(found.values()) == pytest.approx(
        manifest["log_likelihood_cnaster"], rel=TOLERANCE
    )
    np.testing.assert_allclose(clones["log_likelihood"], drawn["llf"], rtol=TOLERANCE)
    module = hmm_log_likelihoods(
        states,
        _read(run, "cnv_hmm_transmat.tsv"),
        clones.reset_index(),
        bins,
        manifest["lengths"],
    )
    np.testing.assert_allclose(list(module.values()), drawn["llf"], rtol=TOLERANCE)


@pytest.mark.oracle
def test_the_decode_files_recompute_its_likelihood_and_path(
    written: tuple[Path, dict[str, Any], Any],
) -> None:
    """From `cnv_copy_*` and the pooled counts alone, the referee's lattice
    returns each clone's decoded `(A, B)` as its Viterbi path and the
    decode's own per-clone `log_likelihood` as that path's score, to 1e-10
    relative; `outputs.decode_log_likelihoods` to the same; the predicted `mu` and `baf` are the lattice's at the clone's
    fraction and shift."""
    from port.extensions.copy_likelihood import viterbi_oracle
    from port.extensions.outputs import decode_log_likelihoods

    run, _, decode = written
    manifest = json.loads((run / "run.json").read_text())
    clones = _read(run, "cnv_copy_clones.tsv").set_index("clone")
    copies = _read(run, "cnv_copy_bins.tsv")
    counts = _read(run, "cnv_hmm_bins.tsv")
    lattice = _lattice()

    for clone, row in clones.iterrows():
        own = copies[copies.clone == clone].sort_values("bin")
        held = counts[counts.clone == clone].sort_values("bin")
        emission = _decode_emission(held, row)
        transmat, start = _chain(len(lattice), float(row["stay"]))
        path, score = viterbi_oracle(
            emission, transmat, start, np.asarray(manifest["lengths"])
        )

        np.testing.assert_array_equal(lattice[path], own[["A", "B"]].to_numpy())
        assert score == pytest.approx(decode.log_likelihoods[clone], rel=TOLERANCE)
        assert row["log_likelihood"] == decode.log_likelihoods[clone]

        rho, total = row["tumour_fraction"], own["A"] + own["B"]
        np.testing.assert_allclose(
            own["mu"],
            (rho * total / 2 + 1 - rho) * np.exp(-row["log_mu_shift"]),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            own["baf"],
            (rho * own["A"] + 1 - rho) / (rho * total + 2 * (1 - rho)),
            rtol=1e-12,
        )

    module = decode_log_likelihoods(
        clones.reset_index(),
        copies,
        counts,
        manifest["lengths"],
        manifest["max_total_copy"],
    )
    np.testing.assert_allclose(
        list(module.values()), decode.log_likelihoods, rtol=TOLERANCE
    )
    assert clones["is_normal"].tolist() == [True, False, False]


@pytest.mark.oracle
# NB no tier: `tests.ci` runs every `oracle` test in its release step, and a
#    `merge` mark would run it twice (`tests/test_ci_entry.py`). 36 s.
@pytest.mark.xdist_group("pipeline")
def test_a_run_s_decode_files_recompute_the_decode(tmp_path: Path) -> None:
    """`run_cnaster_port` on the two-state copy lattice writes every stage
    file, and from them alone the referee returns each clone's decoded
    `(A, B)` as its Viterbi path and the decode's per-clone `log_likelihood`
    to 1e-10 relative; the HMM files recompute `cnv_hmm_clones`' own
    `log_likelihood` to the same."""
    import warnings

    import matplotlib as mpl
    from port.extensions.copy_likelihood import viterbi_oracle
    from port.extensions.outputs import SCHEMA, hmm_log_likelihoods, run_directories
    from port.patch import integer_copy
    from port.scripts.run_cnaster import main

    from tests.fixtures import critical_instance
    from tests.run_config import isolated_run, write_run_cnaster_config
    from tests.tmp_inputs import write_tmp_inputs
    from tests.unsegment import unsegment

    mpl.use("Agg")
    truth = critical_instance(copy_lattice=True)
    inputs = write_tmp_inputs(
        truth, unsegment(truth, flip_every=0, unassigned_genes=0), tmp_path
    )
    config = write_run_cnaster_config(
        inputs, truth, max_iter_outer=1, max_iter=3, n_states=2
    )
    with isolated_run(), warnings.catch_warnings(), integer_copy.recorded() as decodes:
        warnings.simplefilter("ignore")
        assert main([str(config), "--no-plots"]) == 0

    (run,) = run_directories(inputs.root / "output")
    assert {p.name for p in run.iterdir()} >= {*SCHEMA, "run.json"}
    manifest = json.loads((run / "run.json").read_text())
    clones = _read(run, "cnv_copy_clones.tsv").set_index("clone")
    copies, counts = _read(run, "cnv_copy_bins.tsv"), _read(run, "cnv_hmm_bins.tsv")
    lattice = np.array(
        [
            (a, b)
            for a in range(manifest["max_total_copy"] + 1)
            for b in range(manifest["max_total_copy"] + 1)
            if 0 < a + b <= manifest["max_total_copy"]
        ]
    )

    for clone, row in clones.iterrows():
        own = copies[copies.clone == clone].sort_values("bin")
        emission = _decode_emission_on(
            lattice, counts[counts.clone == clone].sort_values("bin"), row
        )
        transmat, start = _chain(len(lattice), float(row["stay"]))
        path, score = viterbi_oracle(
            emission, transmat, start, np.asarray(manifest["lengths"])
        )

        np.testing.assert_array_equal(lattice[path], own[["A", "B"]].to_numpy())
        assert row["log_likelihood"] == pytest.approx(score, rel=TOLERANCE)

    assert sum(clones["log_likelihood"]) == pytest.approx(
        decodes[-1].log_likelihood, rel=TOLERANCE
    )
    hmm = _read(run, "cnv_hmm_clones.tsv")
    module = hmm_log_likelihoods(
        _read(run, "cnv_hmm_states.tsv"),
        _read(run, "cnv_hmm_transmat.tsv"),
        hmm,
        counts,
        manifest["lengths"],
    )
    np.testing.assert_allclose(
        list(module.values()), hmm["log_likelihood"], rtol=TOLERANCE
    )


def _decode_emission_on(
    lattice: np.ndarray, counts: pd.DataFrame, clone: pd.Series
) -> np.ndarray:
    """`_decode_emission` over an explicit lattice."""
    rho = float(clone["tumour_fraction"])
    total = lattice.sum(axis=1)
    depth = rho * total / 2.0 + 1.0 - rho
    share = (rho * lattice[:, 0] + 1.0 - rho) / (rho * total + 2.0 * (1.0 - rho))
    emission = _nb(
        counts["X_depth"].to_numpy()[None, :],
        counts["base_nb_mean"].to_numpy()[None, :]
        * (depth * np.exp(-float(clone["log_mu_shift"])))[:, None],
        float(clone["alphas"]),
    ) + _bb(
        counts["X_allele"].to_numpy()[None, :],
        counts["total_bb_RD"].to_numpy()[None, :],
        share[:, None],
        float(clone["taus"]),
    )
    emission = np.where(np.isfinite(emission), emission, -1e10)
    return emission - float(clone["parsimony_weight"]) * np.abs(total - 2)[:, None]


@pytest.mark.infra
def test_docs_outputs_md_states_the_schema() -> None:
    """`docs/outputs.md` has one table per file of `SCHEMA`, rows equal to its
    columns -- name, type, unit, meaning -- and one row per `run.json` key."""
    from port.extensions.outputs import RUN_KEYS, SCHEMA

    text = (Path(__file__).resolve().parent.parent / "docs" / "outputs.md").read_text()
    sections = {part.split("`", 1)[0]: part for part in text.split("\n### `")[1:]}

    for name, columns in SCHEMA.items():
        rows = [line for line in sections[name].splitlines() if line.startswith("| `")]
        assert rows == [
            f"| `{c.name}` | {c.kind} | {c.unit} | {c.meaning} |" for c in columns
        ], name

    keys = [
        key.strip("` ")
        for line in sections["run.json"].splitlines()
        if line.startswith("| `")
        for key in line.split("|")[1].split(",")
    ]
    assert tuple(keys) == RUN_KEYS
