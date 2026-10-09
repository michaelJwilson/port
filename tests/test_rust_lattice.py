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
from typing import Any

import numba
import numpy as np
import pytest
from cnaster.hmm_nophasing import hmm_nophasing
from cnaster.hmm_phased import hmm_phased
from port.patch import lattice
from port.patch.hmm_nophasing import hmm_nophasing as shifted
from port.patch.lattice import forward_lattice_rust, rust_lattices
from port.sim.run_config import write_for_run
from port.sim.truth import critical_instance

from tests.builders import random_lattice
from tests.figure_checks import compare_run_artifacts
from tests.fixtures import (
    end_to_end_truth,
    partition_ari,
    run_planted_core_inference,
)

CASES = [
    # (n_states, lengths, spots): gate sizes, then one above PARALLEL_WORK
    (3, (7, 11, 5), 3),
    (5, (1, 40, 2, 17), 4),
    (4, (60,), 1),
    (10, (1000,) * 10, 20),
]


def _inputs(
    n_states: int, lengths: tuple[int, ...], spots: int, *, phased: bool, seed: int
) -> tuple[np.ndarray, ...]:
    """A proper transition, a switch kernel that moves, and some `-inf` sites."""
    inputs = random_lattice(
        n_states, lengths, spots, phased=phased, seed=seed, dirichlet=True
    )
    # NB an impossible state at some sites, as a zero-count BAF bin gives `cnaster`.
    inputs.log_emission[0, :: max(inputs.log_emission.shape[1] // 7, 1), 0] = -np.inf
    return inputs.arguments


def _agree(rust: np.ndarray, cnaster: np.ndarray) -> None:
    """Bitwise against compiled `cnaster`; 1e-14 when `numba` is disabled."""

    if not getattr(numba.config, "DISABLE_JIT"):  # noqa: B009 -- numba sets it at import
        np.testing.assert_array_equal(rust, cnaster)
        return

    finite = np.isfinite(cnaster)
    np.testing.assert_array_equal(np.isfinite(rust), finite)
    np.testing.assert_array_equal(rust[~finite], cnaster[~finite])
    np.testing.assert_allclose(rust[finite], cnaster[finite], rtol=1e-14, atol=0.0)


def _ids(case: tuple[Any, ...]) -> str:
    n_states, lengths, spots = case
    return f"K{n_states}-G{sum(lengths)}-S{spots}"


@pytest.mark.patch
@pytest.mark.parametrize("case", CASES, ids=_ids)
@pytest.mark.parametrize("which", ["forward_lattice", "backward_lattice"])
def test_the_unphased_lattice_is_cnasters_bitwise(
    case: tuple[Any, ...], which: str
) -> None:
    """Every entry equal, `-inf` included, at four shapes."""

    arguments = _inputs(*case, phased=False, seed=3)
    rust = getattr(lattice, f"{which}_rust")

    _agree(rust(*arguments), getattr(hmm_nophasing, which)(*arguments))


@pytest.mark.patch
@pytest.mark.parametrize("penalize", [False, True])
@pytest.mark.parametrize("case", CASES, ids=_ids)
@pytest.mark.parametrize("which", ["forward_lattice", "backward_lattice"])
def test_the_phased_lattice_is_cnasters_bitwise(
    case: tuple[Any, ...], which: str, penalize: bool
) -> None:
    """Both settings of `cnaster`'s phase penalty, which builds two transitions."""

    arguments = _inputs(*case, phased=True, seed=4)
    rust = getattr(lattice, which.replace("_lattice", "_lattice_phased") + "_rust")

    _agree(rust(*arguments, penalize), getattr(hmm_phased, which)(*arguments, penalize))


@pytest.mark.patch
def test_a_strided_emission_is_read_as_cnaster_reads_it() -> None:
    """A non-contiguous view is copied, not refused and not misread."""

    lengths, log_transmat, log_startprob, wide, log_sitewise = _inputs(
        3, (9, 6), 6, phased=False, seed=8
    )
    strided = wide[:, :, ::2]

    assert not strided.flags.c_contiguous

    _agree(
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
    arguments = _inputs(4, (13, 8), 3, phased=phased, seed=6)
    expected = [
        getattr(cls, w)(*arguments) for w in ("forward_lattice", "backward_lattice")
    ]

    with rust_lattices():
        instance = cls(params="smp", t=0.99)

        for owner in (cls, instance):
            for which, reference in zip(
                ("forward_lattice", "backward_lattice"), expected, strict=True
            ):
                _agree(getattr(owner, which)(*arguments), reference)


@pytest.mark.infra
def test_lengths_that_do_not_cover_the_emission_are_refused() -> None:
    """`cnaster` would index past the end; the binding says why instead."""

    lengths, log_transmat, log_startprob, log_emission, log_sitewise = _inputs(
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
