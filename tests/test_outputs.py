"""`port.extensions.outputs`: the run's integer-clone rule and configuration keys, and a run read from `cnamaste.h5`.

Port writes no table of its own beside `cnaster`'s (T- #817): what
`cnv_states.tsv`, `cnv_segments.tsv`, `cnv_binlevel.tsv`,
`clone_labels_integer.tsv`, `gene_segments.tsv` and `manifest.json` held is in
`cnamaste.h5`. A synthetic fit whose clone ids are not their positions stands
in for a run where a claim is about the arithmetic, not the run.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from cnaster.hmm_nophasing import hmm_nophasing as upstream
from port.extensions import cnamaste
from port.extensions.outputs import config_keys, integer_clones, run_directories
from port.patch.hmm_nophasing import hmm_nophasing
from port.pipeline import with_attributes
from port.sandbox.extensions import copy_errors
from port.sim.run_config import run_written
from port.sim.truth import core_inference_truth

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
        log_gamma=np.log(gamma),
        pred_cnv=pred_cnv,
        llf=-10.0,
        total_llf=np.nan,
        new_log_mu_shift=np.array(None, dtype=object) if shift is None else shift,
    )

    return run


@pytest.mark.analytic
def test_clones_that_decode_alike_are_one_integer_clone() -> None:
    """Equal `(A, B)` at every bin is one clone, named by its smallest id;
    one differing bin keeps two clones apart."""

    base = np.array([[1, 1], [2, 1], [1, 0], [1, 1]])
    frame = pd.DataFrame({"CHR": [1, 1, 2, 2]})
    near = base.copy()
    near[2] = [1, 1]

    for clone, profile in (("5", base), ("0", near), ("2", base), ("7", near)):
        frame[f"clone{clone} A"], frame[f"clone{clone} B"] = profile.T

    assert integer_clones(frame) == {"5": "2", "0": "0", "2": "2", "7": "0"}


def _truth() -> Any:
    return core_inference_truth(
        n_clones=2, n_states=3, lattice=(25, 40), n_obs=40, n_segments=3, seed=11
    )


@pytest.mark.end2end
@pytest.mark.merge
# NB one whole run at a time: four at once exceed 15 GB (#403).
@pytest.mark.xdist_group("pipeline")
def test_a_run_s_outputs_recover_the_planted_clones_and_the_flat_normal(
    tmp_path: Path,
) -> None:
    """`run_cnaster_port` on the round trip's instance, its `cnamaste.h5` read against the planted truth.

    Two clones, three states, 40 bins. Each fitted clone's spots are one
    planted clone to 95 per cent, and the planted normal clone reads flat:
    its rate `exp(log_mu[Z] - shift)` constant to 1 per cent about its mean --
    the fit leaves `mu`'s scale unpinned -- and its state's `p` 1/2 to 0.02.
    Read from `/rdrbaf` and `/clone_assignment`; the per-bin posterior-mean
    `p` the removed `cnv_binlevel.tsv` held is the decoded state's `p` here.
    """

    truth = _truth()
    # NB `cnaster`'s ICM draws from numpy's global generator unseeded, so the
    #    run is seeded here and the generator put back after.
    state = np.random.get_state()  # noqa: NPY002
    np.random.seed(11)  # noqa: NPY002
    try:
        output = run_written(
            truth, tmp_path, port=True, plots=False, max_iter_outer=1, max_iter=3
        )
    finally:
        np.random.set_state(state)  # noqa: NPY002

    h5 = Path(output) / cnamaste.FILE
    spots, _ = cnamaste.read(h5, "inputs")
    final, _ = cnamaste.read(h5, "clone_assignment")
    fit, _ = cnamaste.read(h5, "rdrbaf")
    order = dict(
        zip(final["assignment"].tolist(), fit["assignment"].tolist(), strict=True)
    )
    # NB the spots in the order the instance writes them, which is `/inputs`'
    assert spots["barcodes"].size == truth.labels.size
    labels = truth.labels
    shift = fit.get("logmu_shift", np.zeros(fit["pred_cnv"].shape[1]))
    normal = 0

    for clone in np.unique(final["assignment"]):
        chosen = final["assignment"] == clone
        planted = pd.Series(labels[chosen]).value_counts(normalize=True)
        assert planted.iloc[0] >= 0.95, f"clone {clone}: {planted.to_dict()}"

        column = order[int(clone)]
        path = fit["pred_cnv"][:, column]
        if np.all(truth.states[int(planted.index[0])] == 0):
            normal += 1
            mu = np.exp(fit["log_mu"][path] - shift[column])
            np.testing.assert_allclose(mu, mu.mean(), rtol=1e-2)
            np.testing.assert_allclose(fit["p_binom"][path], 0.5, atol=0.02)

    assert normal == 1, "the planted normal clone was not recovered as one clone"


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

    frame = _profiles({"0": 0, "1": 7, "2": 14})

    assert integer_clones(frame) == {"0": "0", "1": "0", "2": "2"}
    assert integer_clones(frame, 1.0) == {"0": "0", "1": "1", "2": "2"}
    assert integer_clones(frame, 0.985) == {"0": "0", "1": "0", "2": "0"}


@pytest.mark.infra
@pytest.mark.parametrize("agreement", [0.0, -0.1, 1.5])
def test_an_agreement_outside_the_unit_interval_is_refused(agreement: float) -> None:
    """A share of bins must be in (0, 1]; 0 would merge every clone into one."""

    with pytest.raises(ValueError, match="merge agreement"):
        integer_clones(_profiles({"0": 0}), agreement)


@pytest.mark.infra
def test_the_configured_agreement_is_what_config_keys_reads(tmp_path: Path) -> None:
    """`int_copy_num.merge_agreement` reaches `config_keys` (#518)."""

    config = tmp_path / "config.yaml"
    config.write_text("int_copy_num:\n  merge_agreement: 0.99\n")

    assert config_keys(config)["merge_agreement"] == 0.99


@pytest.mark.infra
def test_a_directory_an_earlier_run_left_is_not_this_run_s(tmp_path: Path) -> None:
    """`since` keeps the directories written at or after it (T- #617)."""

    run = _run(tmp_path)
    table = run / "cnv_seglevel.tsv"
    os.utime(table, (1_000.0, 1_000.0))

    assert list(run_directories(tmp_path)) == [run]
    assert list(run_directories(tmp_path, since=1_000.0)) == [run]
    assert list(run_directories(tmp_path, since=1_001.0)) == []


@pytest.mark.infra
def test_copy_sets_go_beside_the_fit_this_run_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not beside the newest fit under `output_dir`, another K's (T- #617)."""

    ours, other = (
        tmp_path / "clone3_rectangle0_w1.0",
        tmp_path / "clone4_rectangle0_w1.0",
    )
    ours.mkdir()
    other.mkdir()
    (ours / "rdrbaf_final_nstates4_smp.npz").write_bytes(b"")
    (other / "rdrbaf_final_nstates7_smp.npz").write_bytes(b"")
    os.utime(ours / "rdrbaf_final_nstates4_smp.npz", (2_000.0, 2_000.0))
    os.utime(other / "rdrbaf_final_nstates7_smp.npz", (3_000.0, 3_000.0))

    placed: list[Path] = []

    def write(run: Path, captured: object, **_: object) -> Path:
        placed.append(Path(run))
        return Path(run) / "cnv_copy_sets.tsv"

    monkeypatch.setattr(copy_errors, "write_copy_sets", write)
    config = tmp_path / "config.yaml"
    config.write_text(f"paths:\n  output_dir: {tmp_path}\nhmm:\n  n_states: 4\n")

    copy_errors.write_beside_final_fit(str(config), ["fit"], since=1_500.0)
    assert placed == [ours]

    copy_errors.write_beside_final_fit(str(config), ["fit"], since=2_500.0)
    assert placed == [ours]


@pytest.mark.infra
def test_a_refused_fit_leaves_the_run_and_says_so(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """T- #599's refusal writes no sets and does not fail the run it follows (#705)."""

    (tmp_path / "rdrbaf_final_nstates4_smp.npz").write_bytes(b"")

    def refuse(run: Path, captured: object, **_: object) -> Path:
        msg = "--copy-errors: the fit's tau 6e+05 >= 100000 (T- #599); refused"
        raise ValueError(msg)

    monkeypatch.setattr(copy_errors, "write_copy_sets", refuse)
    config = tmp_path / "config.yaml"
    config.write_text(f"paths:\n  output_dir: {tmp_path}\nhmm:\n  n_states: 4\n")

    copy_errors.write_beside_final_fit(str(config), ["fit"])

    assert "T- #599); refused; cnv_copy_sets.tsv not written" in capsys.readouterr().err


# NB per position in `pred_cnv`, as `new_log_mu_shift` is: nonzero, distinct.
SHIFT = np.array([0.0, 0.11, -0.07])


@pytest.mark.oracle
def test_a_bin_s_mu_is_the_rate_the_shifted_emission_evaluates(
    tmp_path: Path,
) -> None:
    """Per clone, port's shifted emission of each bin's state scores the bin
    as `cnaster`'s unshifted emission does at exposure `base * mu` and
    `log_mu = 0`, `mu = exp(log_mu[Z] - shift)`: the rate a reader of
    `cnamaste.h5` derives from `/rdrbaf` is the rate the fit used (#613)."""

    run = _run(tmp_path, shift=SHIFT)
    with np.load(
        run / f"rdrbaf_final_nstates{N_STATES}_smp.npz", allow_pickle=True
    ) as held:
        fit = {k: held[k] for k in held.files}
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
        for position in POSITIONS:
            hmm_nophasing._row_shift = np.full(N_BINS, SHIFT[position])
            ours = shifted.compute_emission_probability_nb_betabinom(
                X, base, fit["new_log_mu"], alphas, total, fit["new_p_binom"], taus
            )[0]
            theirs = upstream.compute_emission_probability_nb_betabinom(
                X,
                base
                * np.exp(
                    fit["new_log_mu"][fit["pred_cnv"][:, position], 0] - SHIFT[position]
                )[:, None],
                np.zeros((N_STATES, 1)),
                alphas,
                total,
                fit["new_p_binom"],
                taus,
            )[0]
            path = fit["pred_cnv"][:, position]
            bins = np.arange(N_BINS)
            np.testing.assert_allclose(ours[path, bins], theirs[path, bins], rtol=1e-12)
    finally:
        hmm_nophasing._row_shift = previous


@pytest.mark.patch
def test_the_run_rule_at_one_is_the_exact_rule_it_replaces() -> None:
    """`outputs.integer_clones(frame, 1.0)` against `qa.scoring.integer_clones` as it stood (T- #817).

    The exact rule, inlined as it was deleted: on 2,000 random decodes of 2 to
    9 clones over 1 to 40 bins, copies drawn from 0..3 so that equal profiles
    are common, the two merge every clone alike.
    """

    def exact(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        merged = np.arange(a.shape[1])
        seen: dict[bytes, int] = {}
        for clone in range(a.shape[1]):
            profile = np.stack([a[:, clone], b[:, clone]]).astype(np.int64)
            merged[clone] = seen.setdefault(profile.tobytes(), clone)
        return merged

    rng = np.random.default_rng(817)
    joined = 0
    for _ in range(2_000):
        n_bins, n_clones = int(rng.integers(1, 41)), int(rng.integers(2, 10))
        a = rng.integers(0, 4, size=(n_bins, n_clones)) * (
            rng.random(n_bins)[:, None] < 0.3
        )
        b = rng.integers(0, 4, size=(n_bins, n_clones)) * (
            rng.random(n_bins)[:, None] < 0.3
        )
        frame = pd.DataFrame(
            {
                f"clone{c} {k}": v[:, c]
                for c in range(n_clones)
                for k, v in (("A", a), ("B", b))
            }
        )
        names = integer_clones(frame, 1.0)
        ours = np.array([int(names[str(c)]) for c in range(n_clones)])
        theirs = exact(a, b)
        np.testing.assert_array_equal(ours, theirs)
        joined += int((theirs != np.arange(n_clones)).any())
    # NB 185 of the 2,000 draws merge at least one pair, measured: the comparison is not vacuous
    assert joined > 100
