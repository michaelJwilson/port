"""`port.extensions.outputs`: the fitted and the decoded views of a run (#331, #613).

A synthetic run directory in `cnaster`'s formats -- `cnv_seglevel.tsv`,
`cnv_perstate.tsv` and the `.npz` -- whose clone ids are not their positions,
so each claim below is read against a layout the writer has to resolve rather
than one it can assume. Written without a capture, as `cnaster`'s own run
would be: the counts and the decode's fit are empty, and
`tests/test_output_stages.py` holds the files with them.
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


def _run(tmp_path: Path, seed: int = 3, shift: np.ndarray | None = None) -> Path:
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
        new_alphas=np.full((N_STATES, 1), 0.1),
        new_taus=np.full((N_STATES, 1), 100.0),
        new_log_startprob=np.full(N_STATES, -np.log(N_STATES)),
        new_log_transmat=np.log(np.full((N_STATES, N_STATES), 1.0 / N_STATES)),
        log_gamma=np.log(gamma),
        pred_cnv=pred_cnv,
        llf=-10.0,
        total_llf=np.nan,
        new_assignment=np.repeat(np.arange(n_clones), 2),
        new_log_mu_shift=np.array(None, dtype=object) if shift is None else shift,
    )

    return run


def _written(run: Path, name: str) -> pd.DataFrame:
    from port.extensions.outputs import write_outputs

    if not (run / name).exists():
        write_outputs(run)
    # NB `round_trip`, so a float reads back as the one the writer held.
    return pd.read_csv(run / name, sep="\t", float_precision="round_trip")


def _load(run: Path) -> Any:
    from port.extensions.outputs import _load as load

    return load(run)


@pytest.mark.infra
def test_each_clone_is_matched_to_its_column_by_its_path(tmp_path: Path) -> None:
    """`clone0`, `clone2`, `clone5` sit at positions 1, 2, 0 of `pred_cnv`."""
    from port.extensions.outputs import clone_columns

    seglevel, _, res = _load(_run(tmp_path))

    assert clone_columns(seglevel, res["pred_cnv"]) == dict(
        zip(IDS, POSITIONS, strict=True)
    )


@pytest.mark.analytic
def test_the_segments_expand_back_to_every_bin_s_pair(tmp_path: Path) -> None:
    """Each clone's runs tile its bins without gap or overlap, never cross a
    chromosome, and carry the `(A, B)` and HMM states of every bin they span."""
    run = _run(tmp_path)
    seglevel, _, _ = _load(run)
    table = _written(run, "cnv_copy_segments.tsv")

    for clone in IDS:
        runs = table[table.clone == int(clone)]
        expanded = np.repeat(runs[["A", "B"]].to_numpy(), runs.n_bins, axis=0)
        cover = np.concatenate(
            [
                np.arange(a, b + 1)
                for a, b in zip(runs.bin_first, runs.bin_last, strict=True)
            ]
        )

        np.testing.assert_array_equal(cover, np.arange(N_BINS))
        np.testing.assert_array_equal(
            expanded, seglevel[[f"clone{clone} A", f"clone{clone} B"]].to_numpy()
        )
        assert runs.segment.tolist() == list(range(len(runs)))
        for _, row in runs.iterrows():
            span = seglevel.iloc[row.bin_first : row.bin_last + 1]
            assert np.all(span.CHR.to_numpy() == span.CHR.to_numpy()[0])
            assert str(row.hmm_states) == ",".join(
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
def test_the_states_carry_the_decoding_and_the_bins_each_holds(tmp_path: Path) -> None:
    """Per clone, one row per state: `(A, B)` from `cnv_perstate.tsv`,
    agreeing with every bin the state holds, and bin counts summing to the
    bins; the HMM's parameters once per state, from the `.npz`."""
    run = _run(tmp_path)
    seglevel, _, res = _load(run)
    table = _written(run, "cnv_copy_states.tsv")
    fitted = _written(run, "cnv_hmm_states.tsv")

    assert len(table) == N_STATES * len(IDS)
    np.testing.assert_array_equal(fitted.log_mu, res["new_log_mu"][:, 0])
    np.testing.assert_array_equal(fitted.p_binom, res["new_p_binom"][:, 0])

    for clone in IDS:
        rows = table[table.clone == int(clone)].set_index("state")
        assert rows.n_bins.sum() == N_BINS

        path = seglevel[f"clone{clone} Z"].to_numpy()
        np.testing.assert_array_equal(
            rows.n_bins, np.bincount(path, minlength=N_STATES)
        )
        for state in np.unique(path):
            held = seglevel[path == state]
            assert set(held[f"clone{clone} A"]) == {rows.A[state]}
            assert set(held[f"clone{clone} B"]) == {rows.B[state]}


@pytest.mark.infra
def test_the_writer_leaves_cnaster_s_files_and_writes_valid_json(
    tmp_path: Path,
) -> None:
    """The stage files beside `cnaster`'s, which are byte for byte untouched;
    without a lineage the gene files are not written; `run.json` is strict
    JSON, `total_llf`'s NaN written as null and every map keyed by strings."""
    from port.extensions.outputs import SCHEMA, run_directories, write_outputs

    run = _run(tmp_path)
    before = {path.name: path.read_bytes() for path in run.iterdir()}

    assert list(run_directories(tmp_path)) == [run]
    written = write_outputs(run, flags={"shift": True})

    assert {path.name for path in written} == (
        set(SCHEMA) - {"cnv_lineage.tsv", "cnv_copy_genes.tsv", "spot_labels.tsv"}
    ) | {"run.json"}
    assert {name: (run / name).read_bytes() for name in before} == before

    manifest = json.loads((run / "run.json").read_text(), parse_constant=pytest.fail)
    assert manifest["total_llf"] is None
    assert manifest["log_likelihood_cnaster"] == -10.0
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


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "barcode": [f"BC{k}" for k in range(5)],
            "sample_id": "S1",
            "x": range(5),
            "y": 0,
            "clone_label": [0, 2, 5, np.nan, 5],
        }
    )


@pytest.mark.infra
def test_the_spot_labels_keep_each_spot_s_clone_and_name_its_merge(
    tmp_path: Path,
) -> None:
    """`spot_labels.tsv` keeps every spot's `clone_label` and adds its integer
    clone; a spot with no clone keeps none; `clone_labels.tsv` is
    `cnaster`'s byte for byte though clones merge (#518, #613), and
    `read_run_labels` returns the merged clones, as that file did before."""
    from port.extensions.outputs import integer_clones, read_run_labels

    run = _run(tmp_path)
    seglevel, _, _ = _load(run)
    # NB clone 5 decodes as clone 0 does.
    for column in ("A", "B"):
        seglevel[f"clone5 {column}"] = seglevel[f"clone0 {column}"]
    seglevel.to_csv(run / "cnv_seglevel.tsv", sep="\t", index=False)

    labels = _labels()
    labels.to_csv(run / "clone_labels.tsv", sep="\t", index=False)
    untouched = (run / "clone_labels.tsv").read_bytes()
    written = _written(run, "spot_labels.tsv")

    assert (run / "clone_labels.tsv").read_bytes() == untouched
    assert integer_clones(seglevel)["5"] == "0"
    assert written.barcode.tolist() == labels.barcode.tolist()
    np.testing.assert_array_equal(written.clone_label, labels.clone_label)
    np.testing.assert_array_equal(written.clone_label_decode, [0, 2, 0, np.nan, 0])
    assert written["sample"].tolist() == ["S1"] * 5
    assert written.sample_id.tolist() == [0] * 5
    np.testing.assert_array_equal(
        read_run_labels(run).clone_label, [0, 2, 0, np.nan, 0]
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
    host), and the planted normal clone reads flat in `cnv_hmm_bins.tsv`:
    `mu`, its state's rate with the clone's shift, constant to 1 per cent
    about its own mean -- `run_cnaster` leaves `mu`'s scale unpinned -- and
    `p_binom` 1/2 to 0.02. The segments reproduce `cnaster`'s own table bin
    for bin.

    The tumour clone's amplification is not judged here: at this run's three
    iterations its recovery differs by machine -- `mu` ratio 4.41 on this
    host, 1.18 on CI's runner, and its BAF likewise -- which is `run_cnaster`'s
    fit (#293), not the writer. `test_the_writer_returns_the_planted_states_
    of_a_perfect_decode` judges the writer on the amplification, exactly.
    """
    from port.extensions.outputs import run_directories, write_outputs

    from tests.run_config import run_written
    from tests.tmp_inputs import GENE_SPACING

    truth = _truth()
    # NB `cnaster`'s ICM draws from numpy's global generator unseeded, so the
    #    run is seeded here and the generator put back after.
    state = np.random.get_state()  # noqa: NPY002
    np.random.seed(11)  # noqa: NPY002
    try:
        (run,) = run_directories(
            run_written(
                truth, tmp_path, port=False, plots=False, max_iter_outer=1, max_iter=3
            )
        )
    finally:
        np.random.set_state(state)  # noqa: NPY002
    write_outputs(run)

    bins = pd.read_csv(run / "cnv_hmm_bins.tsv", sep="\t")
    places = pd.read_csv(run / "cnv_bins.tsv", sep="\t")
    seglevel = pd.read_csv(run / "cnv_seglevel.tsv", sep="\t", comment="#")
    segments = pd.read_csv(run / "cnv_copy_segments.tsv", sep="\t")
    labels = pd.read_csv(run / "clone_labels.tsv", sep="\t", comment="#")
    offset = np.concatenate([[0], np.cumsum(truth.lengths)[:-1]])
    planted_bin = (
        offset[places.CHR.to_numpy() - 1] + places.START.to_numpy() // GENE_SPACING
    )
    normal = 0

    for clone in np.unique(labels.clone_label):
        spots = labels.clone_label.to_numpy() == clone
        planted = pd.Series(truth.labels[spots]).value_counts(normalize=True)
        assert planted.iloc[0] >= 0.95, f"clone {clone}: {planted.to_dict()}"

        runs = segments[segments.clone == clone]
        np.testing.assert_array_equal(
            np.repeat(runs[["A", "B"]].to_numpy(), runs.n_bins, axis=0),
            seglevel[[f"clone{clone} A", f"clone{clone} B"]].to_numpy(),
        )

        if np.all(truth.states[int(planted.index[0]), planted_bin] == 0):
            normal += 1
            own = bins[bins.clone == clone].sort_values("bin")
            mu = own["mu"].to_numpy()
            np.testing.assert_allclose(mu, mu.mean(), rtol=1e-2)
            np.testing.assert_allclose(own["p_binom"], 0.5, atol=0.02)

    assert normal == 1, "the planted normal clone was not recovered as one clone"


@pytest.mark.analytic
def test_the_writer_returns_the_planted_states_of_a_perfect_decode(
    tmp_path: Path,
) -> None:
    """A run directory written from the planted truth itself -- each clone's
    path its planted states, the posterior one-hot on them, one `(A, B)` per
    state, a shift per clone -- is read back as the truth: every bin's
    `p_binom` the planted state's, its `log_mu` through `hmm_state` in
    `cnv_hmm_states.tsv` the planted state's, its `mu` the planted rate
    over `exp(shift)` and its `hmm_state_probability` 1, the amplification's
    `mu` 5.0 and `p_binom` 0.88 exactly, and the segments the planted runs
    of state within each chromosome."""
    truth = _truth()
    n_states, n_bins = len(np.ravel(truth.log_mu)), truth.states.shape[1]
    n_clones = truth.states.shape[0]
    chromosome = np.repeat(np.arange(1, len(truth.lengths) + 1), truth.lengths)
    start = np.concatenate([np.arange(n) for n in truth.lengths]) * 1000
    pairs = np.array([[1, 1], [2, 1], [4, 1]])[:n_states]
    seglevel = pd.DataFrame({"CHR": chromosome, "START": start, "END": start + 999})
    perstate = pd.DataFrame(index=range(n_states))

    for clone in range(n_clones):
        path = truth.states[clone]
        seglevel[f"clone{clone} Z"] = path
        seglevel[f"clone{clone} A"] = pairs[path, 0]
        seglevel[f"clone{clone} B"] = pairs[path, 1]
        perstate[f"clone{clone} A"] = pairs[:, 0]
        perstate[f"clone{clone} B"] = pairs[:, 1]

    gamma = np.zeros((n_states, n_bins, n_clones))
    for clone in range(n_clones):
        gamma[truth.states[clone], np.arange(n_bins), clone] = 1.0

    shift = np.array([0.0, 0.2])[:n_clones]
    run = tmp_path / "planted"
    run.mkdir()
    seglevel.to_csv(run / "cnv_seglevel.tsv", sep="\t", index=False)
    perstate.to_csv(run / "cnv_perstate.tsv", sep="\t", index=False)
    np.savez(
        run / f"rdrbaf_final_nstates{n_states}_smp.npz",
        n_states=n_states,
        new_log_mu=np.ravel(truth.log_mu)[:, None],
        new_p_binom=np.ravel(truth.p_binom)[:, None],
        new_alphas=np.full((n_states, 1), 0.1),
        new_taus=np.full((n_states, 1), 100.0),
        new_log_startprob=np.full(n_states, -np.log(n_states)),
        new_log_transmat=np.log(np.full((n_states, n_states), 1.0 / n_states)),
        log_gamma=np.log(np.maximum(gamma, 1e-300)),
        pred_cnv=truth.states.T,
        llf=-1.0,
        total_llf=np.nan,
        new_assignment=truth.labels,
        # NB per position in `pred_cnv`, which here is the clone.
        new_log_mu_shift=shift,
    )
    mu_planted = np.exp(np.ravel(truth.log_mu))
    p_planted = np.ravel(truth.p_binom)
    table = _written(run, "cnv_hmm_bins.tsv")
    fitted = _written(run, "cnv_hmm_states.tsv").set_index("state")
    runs_all = _written(run, "cnv_copy_segments.tsv")

    for clone in range(n_clones):
        path = truth.states[clone]
        own = table[table.clone == clone].sort_values("bin")
        np.testing.assert_allclose(
            fitted.log_mu.to_numpy()[own.hmm_state.to_numpy()],
            np.ravel(truth.log_mu)[path],
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            own["mu"], mu_planted[path] / np.exp(shift[clone]), rtol=1e-12
        )
        np.testing.assert_allclose(own["p_binom"], p_planted[path], rtol=1e-12)
        np.testing.assert_allclose(own["hmm_state_probability"], 1.0, rtol=1e-12)

        runs = runs_all[runs_all.clone == clone]
        breaks = np.flatnonzero(
            (path[1:] != path[:-1]) | (chromosome[1:] != chromosome[:-1])
        )
        np.testing.assert_array_equal(runs.bin_first, np.concatenate([[0], breaks + 1]))
        np.testing.assert_array_equal(
            runs.hmm_states.astype(str), [str(s) for s in path[runs.bin_first]]
        )

    amplified = int(np.argmax(mu_planted))
    assert np.exp(fitted.log_mu[amplified]) == pytest.approx(5.0, rel=1e-12)
    assert fitted.p_binom[amplified] == pytest.approx(0.88, rel=1e-12)


def _profiles(disagreeing: dict[str, int], n_bins: int = 1000) -> pd.DataFrame:
    """Clones `id -> bins that differ from the neutral profile`, over `n_bins`."""
    frame = pd.DataFrame({"CHR": np.ones(n_bins, dtype=int)})

    for clone, differing in disagreeing.items():
        profile = np.ones((n_bins, 2), dtype=int)
        profile[:differing] = [2, 1]
        frame[f"clone{clone} A"], frame[f"clone{clone} B"] = profile.T

    return frame


@pytest.mark.analytic
def test_the_agreement_rule_joins_what_agrees_and_no_less() -> None:
    """At 0.99, profiles differing at 7 of 1,000 bins are one; at 14, two (#518).

    `dev_tree`'s slice-split clone differs at 2 of 2,895 bins (0.9993) and
    the closest distinct pair on the fixtures at 0.9863, so the thresholds
    below bracket both. 0.99 is the default; 1.0 is the exact rule (#344).
    """
    from port.extensions.outputs import integer_clones

    frame = _profiles({"0": 0, "1": 7, "2": 14})

    assert integer_clones(frame) == {"0": "0", "1": "0", "2": "2"}
    assert integer_clones(frame, 1.0) == {"0": "0", "1": "1", "2": "2"}
    assert integer_clones(frame, 0.985) == {"0": "0", "1": "0", "2": "0"}


@pytest.mark.infra
@pytest.mark.parametrize("agreement", [0.0, -0.1, 1.5])
def test_an_agreement_outside_the_unit_interval_is_refused(agreement: float) -> None:
    """A share of bins must be in (0, 1]; 0 would merge every clone into one."""
    from port.extensions.outputs import integer_clones

    with pytest.raises(ValueError, match="merge agreement"):
        integer_clones(_profiles({"0": 0}), agreement)


@pytest.mark.infra
def test_the_configured_agreement_is_what_write_outputs_uses(tmp_path: Path) -> None:
    """`int_copy_num.merge_agreement` reaches `config_keys` and `run.json` (#518)."""
    from port.extensions.outputs import config_keys, write_outputs

    config = tmp_path / "config.yaml"
    config.write_text("int_copy_num:\n  merge_agreement: 0.95\n")
    run = _run(tmp_path)
    write_outputs(run, config)

    assert config_keys(config)["merge_agreement"] == 0.95
    assert json.loads((run / "run.json").read_text())["merge_agreement"] == 0.95


# NB per position in `pred_cnv`, as `new_log_mu_shift` is: nonzero, distinct.
SHIFT = np.array([0.0, 0.11, -0.07])


@pytest.mark.analytic
def test_a_bin_s_mu_is_its_state_s_rate_with_the_clone_s_shift(
    tmp_path: Path,
) -> None:
    """`cnv_hmm_bins.mu` is `exp(log_mu[hmm_state] - shift_c)` to 1e-12, from
    the `.npz` directly, and differs from the posterior mean of the state
    rates where the posterior splits (#613)."""
    run = _run(tmp_path, shift=SHIFT)
    seglevel, _, res = _load(run)
    table = _written(run, "cnv_hmm_bins.tsv")
    log_mu = res["new_log_mu"][:, 0]

    for clone, position in zip(IDS, POSITIONS, strict=True):
        path = seglevel[f"clone{clone} Z"].to_numpy()
        expected = np.exp(log_mu[path] - SHIFT[position])
        mu = table[table.clone == int(clone)].sort_values("bin")["mu"].to_numpy()
        np.testing.assert_allclose(mu, expected, rtol=1e-12)

        averaged = np.exp(log_mu) @ np.exp(res["log_gamma"][:, :, position])
        assert np.all(np.abs(mu - averaged) > 1e-3)


@pytest.mark.infra
def test_written_mu_joins_the_states_and_the_clones_shift(tmp_path: Path) -> None:
    """On file, `mu` is `exp(log_mu - log_mu_shift)`: `log_mu` from
    `cnv_hmm_states.tsv` joined on `hmm_state`, the shift from
    `cnv_hmm_clones.tsv` joined on `clone`; a run recording no shift
    writes `exp(log_mu)` (#613)."""
    for name, shift in (("shifted", SHIFT), ("unshifted", None)):
        (tmp_path / name).mkdir()
        run = _run(tmp_path / name, shift=shift)
        bins = _written(run, "cnv_hmm_bins.tsv")
        joined = bins.merge(
            _written(run, "cnv_hmm_states.tsv"), left_on="hmm_state", right_on="state"
        ).merge(_written(run, "cnv_hmm_clones.tsv"), on="clone")

        np.testing.assert_allclose(
            joined["mu"], np.exp(joined["log_mu"] - joined["log_mu_shift"]), rtol=1e-12
        )
        if shift is None:
            assert (joined["log_mu_shift"] == 0.0).all()


@pytest.mark.oracle
def test_a_bin_s_mu_is_the_rate_the_shifted_emission_evaluates(
    tmp_path: Path,
) -> None:
    """Per clone, port's shifted emission of each bin's state scores the bin
    as `cnaster`'s unshifted emission does at exposure `base * mu` and
    `log_mu = 0`: the written `mu` is the rate the fit used (#613)."""
    from cnaster.hmm_nophasing import hmm_nophasing as upstream
    from port.patch.hmm_nophasing import hmm_nophasing
    from port.pipeline import with_attributes

    run = _run(tmp_path, shift=SHIFT)
    _, _, res = _load(run)
    table = _written(run, "cnv_hmm_bins.tsv")
    shifted = with_attributes(hmm_nophasing, apply_logmu_shift=True)
    rng = np.random.default_rng(7)
    X = np.stack(
        [rng.poisson(40, (N_BINS, 1)), rng.integers(0, 10, (N_BINS, 1))], axis=1
    ).astype(float)
    base = rng.uniform(20, 60, (N_BINS, 1))
    total = np.full((N_BINS, 1), 10.0)
    alphas, taus = np.full((N_STATES, 1), 0.1), np.full((N_STATES, 1), 30.0)
    previous = hmm_nophasing._row_shift

    try:
        for clone, position in zip(IDS, POSITIONS, strict=True):
            own = table[table.clone == int(clone)].sort_values("bin")
            hmm_nophasing._row_shift = np.full(N_BINS, SHIFT[position])
            ours = shifted.compute_emission_probability_nb_betabinom(
                X, base, res["new_log_mu"], alphas, total, res["new_p_binom"], taus
            )[0]
            theirs = upstream.compute_emission_probability_nb_betabinom(
                X,
                base * own["mu"].to_numpy()[:, None],
                np.zeros((N_STATES, 1)),
                alphas,
                total,
                res["new_p_binom"],
                taus,
            )[0]
            path = own["hmm_state"].to_numpy()
            bins = np.arange(N_BINS)
            np.testing.assert_allclose(ours[path, bins], theirs[path, bins], rtol=1e-12)
    finally:
        hmm_nophasing._row_shift = previous
