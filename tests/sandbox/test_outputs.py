"""`port.sandbox.extensions.copy_errors` writes a run's copy sets beside its own fit (T- #617, #705)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from port.sandbox.extensions import copy_errors


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
