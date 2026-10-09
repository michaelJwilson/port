"""`cnaster`'s four lattices from `oxiport`, bitwise against `cnaster`'s `@njit` kernels
(#318).

Under `NUMBA_DISABLE_JIT=1` the referee runs as NumPy and the claim is 1e-14 relative.
The whole-run form is `test_a_rust_run_reproduces_a_numba_one`.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from cnaster.hmm_nophasing import hmm_nophasing
from cnaster.hmm_phased import hmm_phased
from port.patch.hmm_nophasing import hmm_nophasing as shifted
from port.patch.lattice import forward_lattice_rust, rust_lattices
from port.sim.run_config import write_for_run
from port.sim.truth import critical_instance

from tests.builders import masked_lattice
from tests.correspondence import lattice_agree
from tests.figure_checks import compare_run_artifacts
from tests.fixtures import (
    end_to_end_truth,
    partition_ari,
    run_planted_core_inference,
)


@pytest.mark.patch
def test_a_strided_emission_is_read_as_cnaster_reads_it() -> None:
    """A non-contiguous view is copied, not refused and not misread."""

    lengths, log_transmat, log_startprob, wide, log_sitewise = masked_lattice(
        3, (9, 6), 6, phased=False, seed=8
    )
    strided = wide[:, :, ::2]

    assert not strided.flags.c_contiguous

    lattice_agree(
        forward_lattice_rust(lengths, log_transmat, log_startprob, strided, None),
        hmm_nophasing.forward_lattice(
            lengths, log_transmat, log_startprob, strided, log_sitewise
        ),
    )


@pytest.mark.patch
@pytest.mark.parametrize("phased", [False, True], ids=["unphased", "phased"])
def test_installed_every_call_form_returns_cnasters_lattice(phased: bool) -> None:
    """`hmmclass.forward_lattice` and `self.forward_lattice` both return `cnaster`'s lattice once installed."""

    cls = hmm_phased if phased else hmm_nophasing
    arguments = masked_lattice(4, (13, 8), 3, phased=phased, seed=6)
    expected = [
        getattr(cls, w)(*arguments) for w in ("forward_lattice", "backward_lattice")
    ]

    with rust_lattices():
        instance = cls(params="smp", t=0.99)

        for owner in (cls, instance):
            for which, reference in zip(
                ("forward_lattice", "backward_lattice"), expected, strict=True
            ):
                lattice_agree(getattr(owner, which)(*arguments), reference)


@pytest.mark.warning
def test_lengths_that_do_not_cover_the_emission_are_refused() -> None:
    """`cnaster` would index past the end; the binding says why instead."""

    lengths, log_transmat, log_startprob, log_emission, log_sitewise = masked_lattice(
        3, (5, 5), 2, phased=False, seed=1
    )

    with pytest.raises(ValueError, match="sum to the emission's second axis"):
        forward_lattice_rust(
            lengths[:1], log_transmat, log_startprob, log_emission, log_sitewise
        )


@pytest.mark.infra
def test_the_installer_reaches_both_classes_and_restores_them() -> None:
    """Inside the block every caller's lattice is Rust; outside, `cnaster`'s."""

    before = {
        (cls, name): cls.__dict__[name]
        for cls in (hmm_nophasing, hmm_phased)
        for name in ("forward_lattice", "backward_lattice")
    }

    with rust_lattices():
        assert hmm_nophasing.forward_lattice.__name__ == "forward_lattice_rust"
        assert hmm_phased.backward_lattice.__name__ == "backward_lattice_phased_rust"
        # NB `port`'s shifted class subclasses `cnaster`'s and inherits it
        assert shifted.forward_lattice.__name__ == "forward_lattice_rust"

    for (cls, name), original in before.items():
        assert cls.__dict__[name] is original


@pytest.mark.end2end
def test_the_critical_instance_recovers_its_labelling_through_rust(
    cnaster_config: None,
) -> None:
    """ARI 1.000 against the planted clones with every lattice call from Rust (`M = K = 2`, `S = 500`)."""

    truth = critical_instance()

    with rust_lattices():
        result = run_planted_core_inference(truth, max_iter_outer=1, max_iter=3)

    fitted = np.asarray(result.assignment.new_assignment)

    assert np.unique(fitted).size == truth.n_clones
    assert partition_ari(truth.labels, fitted) == pytest.approx(1.0)


@pytest.mark.patch
@pytest.mark.release
def test_a_rust_run_reproduces_a_numba_one(tmp_path: Path) -> None:
    """Two whole `--no-patch` runs, one with `--rust`, equal artifact by artifact, each in its own process."""

    written, config = write_for_run(
        end_to_end_truth(), tmp_path, max_iter_outer=1, max_iter=3
    )
    output = written.root / "output"

    def run(*flags: str) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "port.scripts.run_cnaster", *flags, str(config)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr[-4000:]

    run("--no-patch")
    baseline = tmp_path / "baseline"
    shutil.move(str(output), str(baseline))

    run("--no-patch", "--rust")

    same, differ = compare_run_artifacts(baseline, output)

    assert not differ, f"a Rust run did not reproduce: {differ}"
    assert len(same) >= 25, f"only {len(same)} artifacts compared"
