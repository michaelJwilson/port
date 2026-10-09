"""Recovery against planted truth, and the normal-candidate defect behind it (#313, #320).

Harness: `tests/recovery_audit.py`; tables in `docs/audit-recovery.md`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np
import pytest
from port.qa.audit import audit_truth
from port.sim.truth import COPY_LATTICE, dev_instance

if TYPE_CHECKING:
    from port.qa.audit import Recovery


@pytest.mark.analytic
def test_the_copy_lattice_plants_integer_allele_copies() -> None:
    """Copy lattice plants `2 mu = A + B` to 1e-15 and `p = B / (A + B)` exactly (paper's map)."""

    truth = dev_instance(n_states=len(COPY_LATTICE), copy_lattice=True)
    copies = np.asarray(COPY_LATTICE, dtype=np.float64)
    total = copies.sum(axis=1)

    np.testing.assert_allclose(2.0 * np.exp(truth.log_mu), total, rtol=1e-15, atol=0)
    np.testing.assert_array_equal(truth.p_binom, copies[:, 1] / total)
    assert total.max() <= 6, "cnaster's max_total_copy"
    assert np.all(truth.p_binom >= 0.5)
    assert np.all(truth.p_binom < 1.0)


@pytest.mark.snapshot
def test_the_default_grid_is_unchanged() -> None:
    """`copy_lattice` is off by default, so every other fixture draws as before."""

    truth = dev_instance()

    np.testing.assert_array_equal(
        truth.log_mu, np.concatenate(([0.0], np.log(np.linspace(1.5, 5.0, 9))))
    )
    np.testing.assert_array_equal(
        truth.p_binom, np.concatenate(([0.5], np.linspace(0.58, 0.88, 9)))
    )


@pytest.mark.warning
def test_a_state_count_beyond_the_lattice_is_refused() -> None:
    with pytest.raises(ValueError, match="copy lattice has"):
        dev_instance(n_states=len(COPY_LATTICE) + 1, copy_lattice=True)


def _lattice_run(*, oracle_normal: bool) -> Recovery:
    mpl.use("Agg")
    truth = dev_instance(n_states=len(COPY_LATTICE), copy_lattice=True)
    recovery, _ = audit_truth(
        truth,
        [],
        n_states=len(COPY_LATTICE),
        max_iter_outer=3,
        max_iter=30,
        oracle_normal=oracle_normal,
    )
    return recovery


@pytest.mark.bug
@pytest.mark.release
def test_the_normal_candidates_admit_a_tumor_clone() -> None:
    """Normal candidates admit over 300 of clone 1's spots (#320); fails when fixed."""
    recovery = _lattice_run(oracle_normal=False)

    assert recovery.candidates_tumor > 300, recovery
    assert recovery.copy_exact_altered < 0.6, recovery


@pytest.mark.end2end
@pytest.mark.release
def test_a_clean_baseline_recovers_the_altered_integer_copies() -> None:
    """Planted normal spots recover over 0.7 of altered clone-bins' planted `A + B`."""
    recovery = _lattice_run(oracle_normal=True)

    assert recovery.candidates_tumor == 0
    assert recovery.ari == pytest.approx(1.0)
    assert recovery.copy_exact_altered >= 0.7, recovery
    assert recovery.mu_error_mean <= 0.05, recovery
