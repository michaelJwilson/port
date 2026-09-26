"""`cnaster`'s four lattices from `oxiport`, bitwise (#318).

**The claim is equality to the bit, not a tolerance.** The Rust recursions
run `cnaster`'s arithmetic in `cnaster`'s order -- the same sequential spot
sums, the same `max + ln(sum(exp(a - max)))` over the same buffer -- so any
difference is a defect rather than round-off. `cnaster`'s `@njit` kernels are
the referee; `numba` compiles without `fastmath`, so it does not reassociate.

What moves is where the compile happens (`maturin build`, once) and how the
contigs are scheduled: above `PARALLEL_WORK` they run on separate threads,
each into its own buffer. The stress case here is above that threshold, so
the parallel path is what it pins.

**Under `NUMBA_DISABLE_JIT=1`**, which the drop-in coverage guard sets, the
referee is no longer `cnaster`'s compiled arithmetic: its kernels run as
NumPy, whose reductions accumulate eight-wide and whose `exp`/`log` may be
SIMD. There the claim is 1e-14 relative, against a measured worst of 5.8e-16
over these cases, with non-finite entries still equal; `_agree` says which
applies.

`patch`: each test puts a replacement to the call it replaces. The whole-run
form of the same claim is `test_a_rust_run_reproduces_a_numba_one`.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

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
    generator = np.random.default_rng(seed)
    lengths_array = np.asarray(lengths, dtype=np.int64)
    n_obs = int(lengths_array.sum())
    rows = 2 * n_states if phased else n_states

    log_transmat = np.log(generator.dirichlet(np.ones(n_states), n_states))
    log_startprob = np.log(generator.dirichlet(np.ones(n_states)))
    log_emission = generator.normal(-5.0, 3.0, (rows, n_obs, spots))
    # NB an impossible state at some sites, which is what a zero-count BAF
    #    bin gives `cnaster`: the recursion must carry `-inf` as it does.
    log_emission[0, :: max(n_obs // 7, 1), 0] = -np.inf
    log_sitewise = np.log(generator.uniform(1e-4, 0.3, n_obs))

    return lengths_array, log_transmat, log_startprob, log_emission, log_sitewise


def _agree(rust: np.ndarray, cnaster: np.ndarray) -> None:
    """Bitwise against compiled `cnaster`; 1e-14 when `numba` is disabled."""
    from numba import config

    if not getattr(config, "DISABLE_JIT"):  # noqa: B009 -- numba sets it at import
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
    from cnaster.hmm_nophasing import hmm_nophasing
    from port.patch import lattice

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
    from cnaster.hmm_phased import hmm_phased
    from port.patch import lattice

    arguments = _inputs(*case, phased=True, seed=4)
    rust = getattr(lattice, which.replace("_lattice", "_lattice_phased") + "_rust")

    _agree(rust(*arguments, penalize), getattr(hmm_phased, which)(*arguments, penalize))


@pytest.mark.patch
def test_a_strided_emission_is_read_as_cnaster_reads_it() -> None:
    """A non-contiguous view is copied, not refused and not misread."""
    from cnaster.hmm_nophasing import hmm_nophasing
    from port.patch.lattice import forward_lattice_rust

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
    """`hmmclass.forward_lattice` and `self.forward_lattice`, before and after.

    `cnaster` reaches the lattices both ways -- `hmm.py` through the class,
    `optimize` through the instance -- so the installer is judged on each.
    """
    from cnaster.hmm_nophasing import hmm_nophasing
    from cnaster.hmm_phased import hmm_phased
    from port.patch.lattice import rust_lattices

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
    from port.patch.lattice import forward_lattice_rust

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
    from cnaster.hmm_nophasing import hmm_nophasing
    from cnaster.hmm_phased import hmm_phased
    from port.patch.hmm_nophasing import hmm_nophasing as shifted
    from port.patch.lattice import rust_lattices

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
    """ARI 1.000 against the planted clones, every lattice call from Rust.

    `tests/test_core_inference_end_to_end.py`'s critical claim -- `M = K = 2`,
    `G = 1,000`, `S = 500` -- with `rust_lattices()` installed, so the truth
    that generated the data is the referee rather than `cnaster`'s kernel.
    """
    from port.patch.lattice import rust_lattices

    from tests.fixtures import critical_instance
    from tests.test_core_inference_end_to_end import _adjusted_rand_index, _run

    truth = critical_instance()

    with rust_lattices():
        result = _run(truth, max_iter_outer=1, max_iter=3)

    fitted = np.asarray(result.assignment.new_assignment)

    assert np.unique(fitted).size == truth.n_clones
    assert _adjusted_rand_index(truth.labels, fitted) == pytest.approx(1.0)


@pytest.mark.patch
@pytest.mark.release
def test_a_rust_run_reproduces_a_numba_one(tmp_path: Path) -> None:
    """Two whole `--no-patch` runs, one with `--rust`, artifact by artifact.

    `--no-patch` so the Rust lattices are the only difference between the
    arms; each arm its own process, as `test_patched_entry_point` does and
    for its reason.
    """
    import subprocess
    import sys

    from tests.fixtures import core_inference_truth
    from tests.run_config import write_run_cnaster_config
    from tests.test_patched_entry_point import _compare
    from tests.tmp_inputs import write_tmp_inputs
    from tests.unsegment import unsegment

    truth = core_inference_truth(
        n_clones=2, n_states=3, lattice=(25, 40), n_obs=40, n_segments=3, seed=11
    )
    written = write_tmp_inputs(
        truth, unsegment(truth, flip_every=0, unassigned_genes=0), tmp_path
    )
    config = write_run_cnaster_config(written, truth, max_iter_outer=1, max_iter=3)
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

    same, differ = _compare(baseline, output)

    assert not differ, f"a Rust run did not reproduce: {differ}"
    assert len(same) >= 25, f"only {len(same)} artifacts compared"
