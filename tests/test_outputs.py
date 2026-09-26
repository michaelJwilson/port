"""`port.extensions.outputs`: the fitted and the decoded views of a run (#331).

A synthetic run directory in `cnaster`'s formats -- `cnv_seglevel.tsv`,
`cnv_perstate.tsv` and the `.npz` -- whose clone ids are not their positions,
so each claim below is read against a layout the writer has to resolve rather
than one it can assume.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

N_STATES, N_BINS = 4, 30
# NB `cnaster`'s clone ids, in column order, and their positions in
#    `pred_cnv`: not the identity, so a positional reading fails.
IDS, POSITIONS = ("0", "2", "5"), (1, 2, 0)


def _run(tmp_path: Path, seed: int = 3) -> Path:
    rng = np.random.default_rng(seed)
    n_clones = len(IDS)
    pred_cnv = np.zeros((N_BINS, n_clones), dtype=int)

    for s in range(n_clones):
        # NB runs of a state, as a decoded path is.
        pred_cnv[:, s] = np.repeat(rng.integers(0, N_STATES, size=6), N_BINS // 6)

    log_mu = rng.normal(0.0, 0.3, size=(N_STATES, 1))
    p_binom = rng.uniform(0.2, 0.5, size=(N_STATES, 1))
    # NB each clone's decoder maps states to pairs, two states sharing one.
    pairs = {clone: rng.integers(0, 3, size=(N_STATES, 2)) for clone in IDS}
    for clone in IDS:
        pairs[clone][1] = pairs[clone][0]

    # NB a soft posterior peaked on the decoded state.
    gamma = np.full((N_STATES, N_BINS, n_clones), 0.05)
    for s in range(n_clones):
        gamma[pred_cnv[:, s], np.arange(N_BINS), s] = 1.0
    gamma /= gamma.sum(axis=0, keepdims=True)

    chromosome = np.repeat([1, 2], N_BINS // 2)
    start = np.tile(np.arange(N_BINS // 2) * 1000, 2)
    seglevel = pd.DataFrame({"CHR": chromosome, "START": start, "END": start + 999})
    perstate = pd.DataFrame(index=range(N_STATES))

    for clone, position in zip(IDS, POSITIONS, strict=True):
        path = pred_cnv[:, position]
        seglevel[f"clone{clone} Z"] = path
        seglevel[f"clone{clone} logmu"] = log_mu[path, 0]
        seglevel[f"clone{clone} p"] = p_binom[path, 0]
        seglevel[f"clone{clone} A"] = pairs[clone][path, 0]
        seglevel[f"clone{clone} B"] = pairs[clone][path, 1]

        for name, values in (
            ("logmu", log_mu[:, 0]),
            ("p", p_binom[:, 0]),
            ("A", pairs[clone][:, 0]),
            ("B", pairs[clone][:, 1]),
        ):
            perstate[f"clone{clone} {name}"] = values

    run = tmp_path / "clone3_rectangle0_w1.0"
    run.mkdir()
    seglevel.to_csv(run / "cnv_seglevel.tsv", sep="\t", index=False)
    perstate.to_csv(run / "cnv_perstate.tsv", sep="\t", index=False)
    np.savez(
        run / f"rdrbaf_final_nstates{N_STATES}_smp.npz",
        n_states=N_STATES,
        new_log_mu=log_mu,
        new_p_binom=p_binom,
        log_gamma=np.log(gamma),
        pred_cnv=pred_cnv,
        llf=-10.0,
        total_llf=np.nan,
        new_log_mu_shift=np.array(None, dtype=object),
    )

    return run


def _load(run: Path) -> Any:
    from port.extensions.outputs import _load as load

    return load(run)


@pytest.mark.infra
def test_each_clone_is_matched_to_its_column_by_its_path(tmp_path: Path) -> None:
    """`clone0`, `clone2`, `clone5` sit at positions 1, 2, 0 of `pred_cnv`."""
    from port.extensions.outputs import clone_columns

    seglevel, _, fit = _load(_run(tmp_path))

    assert clone_columns(seglevel, fit["pred_cnv"]) == dict(
        zip(IDS, POSITIONS, strict=True)
    )


@pytest.mark.analytic
def test_the_segments_expand_back_to_every_bin_s_pair(tmp_path: Path) -> None:
    """Each clone's runs tile its bins without gap or overlap, never cross a
    chromosome, and carry the `(A, B)` and states of every bin they span."""
    from port.extensions.outputs import segments

    seglevel, _, fit = _load(_run(tmp_path))
    table = segments(seglevel, fit)

    for clone in IDS:
        runs = table[table.clone == clone]
        expanded = np.repeat(runs[["A", "B"]].to_numpy(), runs.n_bins, axis=0)
        cover = np.concatenate(
            [
                np.arange(a, b + 1)
                for a, b in zip(runs.first_bin, runs.last_bin, strict=True)
            ]
        )

        np.testing.assert_array_equal(cover, np.arange(N_BINS))
        np.testing.assert_array_equal(
            expanded, seglevel[[f"clone{clone} A", f"clone{clone} B"]].to_numpy()
        )
        for _, row in runs.iterrows():
            span = seglevel.iloc[row.first_bin : row.last_bin + 1]
            assert np.all(span.CHR.to_numpy() == span.CHR.to_numpy()[0])
            assert row.states == ",".join(
                str(s) for s in np.unique(span[f"clone{clone} Z"])
            )

        # NB adjacent runs on one chromosome differ in their pair: runs are
        #    maximal.
        same = runs.CHR.to_numpy()[1:] == runs.CHR.to_numpy()[:-1]
        differ = np.any(
            runs[["A", "B"]].to_numpy()[1:] != runs[["A", "B"]].to_numpy()[:-1], axis=1
        )
        assert np.all(differ[same])


@pytest.mark.analytic
def test_the_states_carry_the_fit_the_decoding_and_the_shares(tmp_path: Path) -> None:
    """Per clone, one row per state: `logmu` and `p` from the `.npz`, `(A, B)`
    from `cnv_perstate.tsv`, agreeing with every bin the state holds, and
    shares summing to one."""
    from port.extensions.outputs import states

    seglevel, perstate, fit = _load(_run(tmp_path))
    table = states(seglevel, perstate, fit)

    assert len(table) == N_STATES * len(IDS)

    for clone in IDS:
        rows = table[table.clone == clone].set_index("state")
        np.testing.assert_array_equal(rows.logmu, fit["new_log_mu"][:, 0])
        np.testing.assert_array_equal(rows.p, fit["new_p_binom"][:, 0])
        assert rows.share.sum() == pytest.approx(1.0, abs=1e-12)

        path = seglevel[f"clone{clone} Z"].to_numpy()
        for state in np.unique(path):
            held = seglevel[path == state]
            assert set(held[f"clone{clone} A"]) == {rows.A[state]}
            assert set(held[f"clone{clone} B"]) == {rows.B[state]}


@pytest.mark.analytic
def test_the_posterior_means_lie_within_the_states(tmp_path: Path) -> None:
    """A posterior mean is the posterior's weights on the states' `mu` and
    `p`, to 1e-12 against the weights taken from the `.npz` directly, and so
    lies within the states' range."""
    from port.extensions.outputs import binlevel

    seglevel, _, fit = _load(_run(tmp_path))
    table = binlevel(seglevel, fit)
    mu, p = np.exp(fit["new_log_mu"][:, 0]), fit["new_p_binom"][:, 0]

    for clone in IDS:
        means = table[f"clone{clone} mu"].to_numpy()
        assert np.all((means >= mu.min() - 1e-12) & (means <= mu.max() + 1e-12))
        assert np.all(
            (table[f"clone{clone} p"] >= p.min() - 1e-12)
            & (table[f"clone{clone} p"] <= p.max() + 1e-12)
        )
        position = POSITIONS[IDS.index(clone)]
        weights = np.exp(fit["log_gamma"][:, :, position])
        np.testing.assert_allclose(means, mu @ weights, rtol=1e-12)
        np.testing.assert_allclose(table[f"clone{clone} p"], p @ weights, rtol=1e-12)


@pytest.mark.infra
def test_the_writer_leaves_cnaster_s_files_and_writes_valid_json(
    tmp_path: Path,
) -> None:
    """Four files beside `cnaster`'s, which are byte for byte untouched; the
    manifest is strict JSON, `total_llf`'s NaN written as null."""
    from port.extensions.outputs import run_directories, write_outputs

    run = _run(tmp_path)
    before = {path.name: path.read_bytes() for path in run.iterdir()}

    assert list(run_directories(tmp_path)) == [run]
    written = write_outputs(run, flags={"shift": True})

    assert {path.name for path in written} == {
        "cnv_states.tsv",
        "cnv_segments.tsv",
        "cnv_binlevel.tsv",
        "manifest.json",
    }
    assert {name: (run / name).read_bytes() for name in before} == before

    manifest = json.loads(
        (run / "manifest.json").read_text(), parse_constant=pytest.fail
    )
    assert manifest["total_llf"] is None
    assert manifest["clones"] == dict(zip(IDS, POSITIONS, strict=True))
    assert manifest["run_cnaster_port"] == {"shift": True}


@pytest.mark.analytic
def test_clones_that_decode_alike_are_one_integer_clone() -> None:
    """Equal `(A, B)` at every bin is one clone, named by its smallest id;
    one differing bin keeps two clones apart."""
    from port.extensions.outputs import integer_clones

    base = np.array([[1, 1], [2, 1], [1, 0], [1, 1]])
    frame = pd.DataFrame({"CHR": [1, 1, 2, 2]})
    near = base.copy()
    near[2] = [1, 1]

    for clone, profile in (("5", base), ("0", near), ("2", base), ("7", near)):
        frame[f"clone{clone} A"], frame[f"clone{clone} B"] = profile.T

    assert integer_clones(frame) == {"5": "2", "0": "0", "2": "2", "7": "0"}


@pytest.mark.infra
def test_the_integer_labels_keep_every_spot_s_fitted_label(tmp_path: Path) -> None:
    """`clone_labels_integer.tsv` is `clone_labels.tsv` column for column,
    plus each spot's integer clone; a spot with no clone keeps none."""
    from port.extensions.outputs import integer_clones, write_outputs

    run = _run(tmp_path)
    seglevel, _, _ = _load(run)
    # NB clone 5 decodes as clone 0 does.
    for column in ("A", "B"):
        seglevel[f"clone5 {column}"] = seglevel[f"clone0 {column}"]
    seglevel.to_csv(run / "cnv_seglevel.tsv", sep="\t", index=False)

    labels = pd.DataFrame(
        {
            "barcode": [f"BC{k}" for k in range(5)],
            "sample_id": "S1",
            "x": range(5),
            "y": 0,
            "clone_label": [0, 2, 5, np.nan, 5],
        }
    )
    labels.to_csv(run / "clone_labels.tsv", sep="\t", index=False)
    write_outputs(run)
    written = pd.read_csv(run / "clone_labels_integer.tsv", sep="\t")

    pd.testing.assert_frame_equal(written[labels.columns], labels)
    assert integer_clones(seglevel)["5"] == "0"
    np.testing.assert_array_equal(
        written.integer_clone_label.to_numpy(), [0, 2, 0, np.nan, 0]
    )


def _truth() -> Any:
    from tests.fixtures import core_inference_truth

    return core_inference_truth(
        n_clones=2, n_states=3, lattice=(25, 40), n_obs=40, n_segments=3, seed=11
    )


@pytest.mark.end2end
@pytest.mark.merge
# NB one whole run at a time: four at once exceed 15 GB (#403).
@pytest.mark.xdist_group("pipeline")
def test_a_run_s_outputs_recover_the_planted_clones_and_the_flat_normal(
    cnaster_config: None, tmp_path: Path
) -> None:
    """The writer on a whole `run_cnaster` run, read against the planted truth.

    The round trip's instance (two clones, three states, 40 bins). Each
    fitted clone's spots are one planted clone to 95 per cent (1.000 on this
    host), and the planted normal clone reads flat in `cnv_binlevel.tsv`:
    `mu` constant to 1 per cent about its own mean -- `run_cnaster` leaves
    `mu`'s scale unpinned -- and `p` 1/2 to 0.02 (0.499 here, 0.487 on CI's
    runner). The segments reproduce `cnaster`'s own table bin for bin.

    The tumour clone's amplification is not judged here: at this run's three
    iterations its recovery differs by machine -- `mu` ratio 4.41 on this
    host, 1.18 on CI's runner, and its BAF likewise -- which is `run_cnaster`'s
    fit (#293), not the writer. `test_the_writer_returns_the_planted_states_
    of_a_perfect_decode` judges the writer on the amplification, exactly.
    """
    from port.extensions.outputs import run_directories, write_outputs

    from tests.test_run_cnaster_round_trip import _run
    from tests.tmp_inputs import GENE_SPACING

    truth = _truth()
    # NB `cnaster`'s ICM draws from numpy's global generator unseeded, so the
    #    run is seeded here and the generator put back after.
    state = np.random.get_state()  # noqa: NPY002
    np.random.seed(11)  # noqa: NPY002
    try:
        (run,) = run_directories(
            _run(truth, tmp_path, max_iter_outer=1, max_iter=3, plots=False)
        )
    finally:
        np.random.set_state(state)  # noqa: NPY002
    write_outputs(run)

    bins = pd.read_csv(run / "cnv_binlevel.tsv", sep="\t")
    seglevel = pd.read_csv(run / "cnv_seglevel.tsv", sep="\t", comment="#")
    segments = pd.read_csv(run / "cnv_segments.tsv", sep="\t")
    labels = pd.read_csv(run / "clone_labels.tsv", sep="\t", comment="#")
    offset = np.concatenate([[0], np.cumsum(truth.lengths)[:-1]])
    planted_bin = (
        offset[bins.CHR.to_numpy() - 1] + bins.START.to_numpy() // GENE_SPACING
    )
    normal = 0

    for clone in np.unique(labels.clone_label):
        spots = labels.clone_label.to_numpy() == clone
        planted = pd.Series(truth.labels[spots]).value_counts(normalize=True)
        assert planted.iloc[0] >= 0.95, f"clone {clone}: {planted.to_dict()}"

        runs = segments[segments.clone.astype(str) == str(clone)]
        np.testing.assert_array_equal(
            np.repeat(runs[["A", "B"]].to_numpy(), runs.n_bins, axis=0),
            seglevel[[f"clone{clone} A", f"clone{clone} B"]].to_numpy(),
        )

        if np.all(truth.states[int(planted.index[0]), planted_bin] == 0):
            normal += 1
            mu = bins[f"clone{clone} mu"].to_numpy()
            np.testing.assert_allclose(mu, mu.mean(), rtol=1e-2)
            np.testing.assert_allclose(bins[f"clone{clone} p"], 0.5, atol=0.02)

    assert normal == 1, "the planted normal clone was not recovered as one clone"


@pytest.mark.analytic
def test_the_writer_returns_the_planted_states_of_a_perfect_decode(
    tmp_path: Path,
) -> None:
    """A run directory written from the planted truth itself -- each clone's
    path its planted states, the posterior one-hot on them, one `(A, B)` per
    state -- is read back as the truth: every bin's `mu` and `p` the planted
    state's, the amplification's `mu` 5.0 and `p` 0.88 exactly, and the
    segments the planted runs of state within each chromosome."""
    from port.extensions.outputs import binlevel, segments, states

    truth = _truth()
    n_states, n_bins = len(np.ravel(truth.log_mu)), truth.states.shape[1]
    chromosome = np.repeat(np.arange(1, len(truth.lengths) + 1), truth.lengths)
    start = np.concatenate([np.arange(n) for n in truth.lengths]) * 1000
    pairs = np.array([[1, 1], [2, 1], [4, 1]])[:n_states]
    seglevel = pd.DataFrame({"CHR": chromosome, "START": start, "END": start + 999})
    perstate = pd.DataFrame(index=range(n_states))

    for clone in range(truth.states.shape[0]):
        path = truth.states[clone]
        seglevel[f"clone{clone} Z"] = path
        seglevel[f"clone{clone} A"] = pairs[path, 0]
        seglevel[f"clone{clone} B"] = pairs[path, 1]
        perstate[f"clone{clone} A"] = pairs[:, 0]
        perstate[f"clone{clone} B"] = pairs[:, 1]

    gamma = np.zeros((n_states, n_bins, truth.states.shape[0]))
    for clone in range(truth.states.shape[0]):
        gamma[truth.states[clone], np.arange(n_bins), clone] = 1.0

    fit = {
        "n_states": n_states,
        "new_log_mu": np.ravel(truth.log_mu)[:, None],
        "new_p_binom": np.ravel(truth.p_binom)[:, None],
        "log_gamma": np.log(np.maximum(gamma, 1e-300)),
        "pred_cnv": truth.states.T,
    }
    mu_planted = np.exp(np.ravel(truth.log_mu))
    p_planted = np.ravel(truth.p_binom)
    table = binlevel(seglevel, fit)

    for clone in range(truth.states.shape[0]):
        path = truth.states[clone]
        np.testing.assert_allclose(
            table[f"clone{clone} mu"], mu_planted[path], rtol=1e-12
        )
        np.testing.assert_allclose(
            table[f"clone{clone} p"], p_planted[path], rtol=1e-12
        )

        runs = segments(seglevel, fit)
        runs = runs[runs.clone == str(clone)]
        breaks = np.flatnonzero(
            (path[1:] != path[:-1]) | (chromosome[1:] != chromosome[:-1])
        )
        np.testing.assert_array_equal(runs.first_bin, np.concatenate([[0], breaks + 1]))
        np.testing.assert_allclose(
            runs.mu, mu_planted[path[runs.first_bin]], rtol=1e-12
        )

    amplified = int(np.argmax(mu_planted))
    rows = states(seglevel, perstate, fit)
    held = rows[(rows.state == amplified) & (rows.share > 0)]
    assert len(held) >= 1
    assert np.exp(held.logmu.iloc[0]) == pytest.approx(5.0, rel=1e-12)
    assert held.p.iloc[0] == pytest.approx(0.88, rel=1e-12)
