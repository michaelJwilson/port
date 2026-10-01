"""Integer copies from the error bars: every `(A, B)` they admit (#353).

What is pinned here:

- a state's set is the lattice points its covariance admits, the planted one
  among them, and it widens with the covariance (`analytic`);
- the neutral state, pinned to `mu = 1`, decodes on total 2 alone, by its
  allele fraction (`analytic`);
- `--copy-errors` refuses an unshifted fit, whose scale is not the pin's
  (`infra`);
- one realization of the integer genome: each planted pair is in its fitted
  state's set (`end2end`, `release`: a whole run).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _errors(mu: list[float], minor: list[float], sigma: tuple[float, float]) -> object:
    from port.extensions.copy_errors import PinnedErrors

    n_states = len(mu)
    covariance = np.zeros((n_states, 2, 2))
    covariance[:, 0, 0] = sigma[0] ** 2
    covariance[:, 1, 1] = sigma[1] ** 2
    covariance[0, 0, 0] = 0.0

    return PinnedErrors(
        np.asarray(mu),
        np.asarray(minor),
        np.zeros(n_states, dtype=bool),
        covariance,
        0,
        0.0,
    )


@pytest.mark.analytic
def test_a_tight_error_admits_only_the_planted_pair(cnaster_config: None) -> None:
    """At the lattice point with 1 per cent errors, each set is that point."""
    from port.extensions.copy_errors import copy_sets

    errors = _errors([1.0, 1.5, 2.0], [0.5, 1 / 3, 0.25], (0.01, 0.005))
    sets = copy_sets(errors)  # type: ignore[arg-type]

    assert [s.consistent for s in sets] == [((1, 1),), ((2, 1),), ((3, 1),)]


@pytest.mark.analytic
def test_a_wider_error_admits_more_and_keeps_the_planted_pair(
    cnaster_config: None,
) -> None:
    """Sets are nested in the covariance, and the planted pair never leaves."""
    from port.extensions.copy_errors import copy_sets

    sizes = []

    for scale in (0.02, 0.1, 0.3):
        errors = _errors([1.0, 1.5, 2.0], [0.5, 1 / 3, 0.25], (scale, scale / 2))
        sets = copy_sets(errors)  # type: ignore[arg-type]
        assert (2, 1) in sets[1].consistent
        assert (3, 1) in sets[2].consistent
        sizes.append(sum(len(s.consistent) for s in sets))

    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]


@pytest.mark.analytic
def test_the_neutral_state_decodes_on_total_two_by_its_allele_fraction(
    cnaster_config: None,
) -> None:
    """`mu = 1` exactly: balanced reads `(1, 1)`; a lost allele reads `(2, 0)`."""
    from port.extensions.copy_errors import copy_sets

    balanced = copy_sets(_errors([1.0], [0.49], (0.0, 0.01)))  # type: ignore[arg-type]
    lost = copy_sets(_errors([1.0], [0.005], (0.0, 0.01)))  # type: ignore[arg-type]

    assert balanced[0].consistent == ((1, 1),)
    assert lost[0].consistent == ((2, 0),)
    assert balanced[0].threshold == pytest.approx(3.841, abs=1e-3)


@pytest.mark.infra
def test_copy_errors_without_the_shift_is_refused(tmp_path: Path) -> None:
    from port.scripts.run_cnaster import main

    with pytest.raises(SystemExit):
        main([str(tmp_path / "config.yaml"), "--copy-errors", "--no-shift"])


@pytest.mark.end2end
@pytest.mark.release
def test_each_planted_pair_is_in_its_state_s_set(tmp_path: Path) -> None:
    """One realization of the integer genome, decoded from its error bars.

    Tolerance and realized value are in #353's pull request.
    """
    from tests.copy_audit import decode_one

    score = decode_one(0, tmp_path)

    assert all(score["covered"]), score


@pytest.mark.patch
def test_the_capture_reads_the_values_its_copies_held_and_refuses_a_change() -> None:
    """`captured_fits` keeps references; its readers sum the float64 copies' values (#569).

    `pseudobulk` on the captured references against the same sums over
    `np.array(x, float64)`, bitwise, with the assignment changed after the
    capture as the run's floor merge changes it; then an input changed in
    place is refused by `intact`.
    """
    import port.patch.hmrf as patch
    from port.extensions.copy_errors import Captured, captured_fits, pseudobulk

    rng = np.random.default_rng(11)
    n_obs, n_spots = 57, 1_003
    single_x = rng.poisson(3.0, (n_obs, 2, n_spots)).astype(np.int32)
    base = rng.gamma(2.0, 1e-4, (n_obs, n_spots))
    total = (single_x[:, 1, :] + rng.poisson(3.0, (n_obs, n_spots))).astype(np.int32)
    result = {"new_assignment": rng.choice([0, 1, 2], n_spots)}
    lengths = np.array([n_obs])

    original = patch.run_core_inference
    patch.run_core_inference = lambda *_, **__: result
    try:
        with captured_fits() as kept:
            patch.run_core_inference(single_x, lengths, base, total, params="smp")
    finally:
        patch.run_core_inference = original

    result["new_assignment"] = np.where(result["new_assignment"] == 2, 1, 0)
    copies = Captured(
        np.array(single_x, dtype=np.float64),
        lengths,
        np.array(base, dtype=np.float64),
        np.array(total, dtype=np.float64),
        result,
    )
    for key, value in pseudobulk(copies).items():
        np.testing.assert_array_equal(pseudobulk(kept[0])[key], value, err_msg=key)

    single_x[0, 0, 0] += 1
    with pytest.raises(RuntimeError, match="changed in place"):
        pseudobulk(kept[0])
