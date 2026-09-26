"""Recovery against the planted truth, and the causes #313 attributed (#313, #320).

The harness is `tests/recovery_audit.py`; `docs/audit-recovery.md` carries
its tables. What is pinned here:

- the copy-lattice fixture plants integer allele copies exactly, and the
  default fixture still plants what it planted (`analytic`, `snapshot`);
- `cnaster`'s normal candidates admit a tumor clone at the converged lattice
  configuration (`bug`, #320), written to fail when that is fixed;
- handed the planted normal spots, the same run recovers the integer copies
  of altered bins at 0.749 against 0.484 (`end2end`): the size of what #320
  costs, against the truth.

The last two are whole runs of 75 to 80 s each, so `release`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from tests.recovery_audit import Recovery


@pytest.mark.analytic
def test_the_copy_lattice_plants_integer_allele_copies() -> None:
    """`2 mu = A + B` to 1e-15 and `p = B / (A + B)` exactly, within `cnaster`'s cap.

    The paper's map from `(A, B)` to `(mu, p)` (`integer_copy_numbers.tex`),
    which the default grid does not satisfy: `mu = 1.5` at `p = 0.58`
    is no integer pair. `mu` is stored as its log, so it round-trips to
    round-off rather than to the bit.
    """
    from tests.fixtures import COPY_LATTICE, dev_instance

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
    from tests.fixtures import dev_instance

    truth = dev_instance()

    np.testing.assert_array_equal(
        truth.log_mu, np.concatenate(([0.0], np.log(np.linspace(1.5, 5.0, 9))))
    )
    np.testing.assert_array_equal(
        truth.p_binom, np.concatenate(([0.5], np.linspace(0.58, 0.88, 9)))
    )


@pytest.mark.infra
def test_a_state_count_beyond_the_lattice_is_refused() -> None:
    from tests.fixtures import COPY_LATTICE, dev_instance

    with pytest.raises(ValueError, match="copy lattice has"):
        dev_instance(n_states=len(COPY_LATTICE) + 1, copy_lattice=True)


def _lattice_run(*, oracle_normal: bool) -> Recovery:
    import matplotlib as mpl

    from tests.fixtures import COPY_LATTICE, dev_instance
    from tests.recovery_audit import run_arm

    mpl.use("Agg")
    truth = dev_instance(n_states=len(COPY_LATTICE), copy_lattice=True)
    recovery, _ = run_arm(
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
    """399 of clone 1's 400 spots enter the baseline beside the 480 normal (#320).

    Clone 1's events are mostly the balanced (3, 3) state, which BAF cannot
    see, so the BAF-only stage merges it with the normal clone and the
    percentile filter relaxes to take the whole merged clone. Stated as
    more than 300 tumor spots so a change of a few spots fails no-one, and
    written to fail when the selection stops admitting the clone.
    """
    recovery = _lattice_run(oracle_normal=False)

    assert recovery.candidates_tumor > 300, recovery
    assert recovery.copy_exact_altered < 0.6, recovery


@pytest.mark.end2end
@pytest.mark.release
def test_a_clean_baseline_recovers_the_altered_integer_copies() -> None:
    """0.749 of altered clone-bins at their planted `A + B`, against 0.484.

    The same run with the planted normal spots as candidates. The tolerance
    is 0.7: realized 0.749, and 0.690 with `--sal`, so a few bins' drift
    fails no-one while the 0.484 the contaminated baseline reaches fails.
    The ARI is 1.000 either way; this is a copy-number claim, not a clone one.
    """
    recovery = _lattice_run(oracle_normal=True)

    assert recovery.candidates_tumor == 0
    assert recovery.ari == pytest.approx(1.0)
    assert recovery.copy_exact_altered >= 0.7, recovery
    assert recovery.mu_error_mean <= 0.05, recovery
