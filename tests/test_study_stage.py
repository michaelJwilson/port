"""Study harness scoring and Baum-Welch starts, each against an independent answer (#730)."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from tests import ROOT, TESTS


@pytest.mark.oracle
def test_missed_is_the_fewest_misses_over_every_matching_of_states() -> None:
    """Against brute force over all permutations of 4 states."""
    from port.studies.stage import missed

    rng = np.random.default_rng(1)
    truth = rng.integers(0, 4, 200)
    label = np.where(rng.random(200) < 0.7, (truth + 1) % 4, rng.integers(0, 4, 200))
    brute = min(
        int((np.asarray(perm)[label] != truth).sum())
        for perm in itertools.permutations(range(4))
    )

    assert missed(label, truth) == brute


@pytest.mark.bug
@pytest.mark.parametrize("name", ["cnaster-gmm", "distinct", "calicost-gmm", "prior"])
def test_every_start_returns_one_state_per_planted_state(name: str) -> None:
    """Two stacked clones: `n_states` states with positive rates and p in (0, 1)."""
    from port.extensions.copy_starts import CopyCall
    from port.studies.copy_state_stream import seed_states

    rng = np.random.default_rng(0)
    n = 300
    clone = np.repeat([0, 1], n)
    state = np.where((np.arange(2 * n) % n < n // 2) & (clone == 1), 1, 0)
    exposure = np.full(2 * n, 300.0)
    trials = rng.integers(20, 40, 2 * n).astype(float)
    total = rng.poisson(exposure * np.where(state == 1, 0.5, 1.0)).astype(float)
    b = rng.binomial(trials.astype(int), np.where(state == 1, 0.1, 0.5)).astype(float)
    config = (TESTS / "data" / "zenodo_sim_config.yaml").read_text()
    raw = {"X": np.stack([total, b], axis=1)[:, :, None], "base_nb_mean": exposure[:, None],
           "total_bb_RD": trials[:, None], "lengths": np.array([n, n]), "log_sitewise_transmat": np.zeros(2 * n),
           "params": "smp", "config": config}  # fmt: skip
    call = CopyCall("rdrbaf", 2, total, b, exposure, trials, clone, np.full(2 * n, "1"),
                    np.tile(np.arange(n) * 1_000_000, 2), np.full(2 * n, 1_000_000),
                    np.column_stack([1 - state, np.ones(2 * n, int)]), raw)  # fmt: skip
    log_mu, p = seed_states(name, call, np.random.default_rng(1))

    assert log_mu.size == p.size == 2
    assert np.isfinite(log_mu).all()
    assert ((p > 0) & (p < 1)).all()


@pytest.mark.release
@pytest.mark.patch
def test_the_stage_is_the_runs_baum_welch_at_the_planted_clones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dev_tree_1s_hard r0: the replayed `--sal` Baum-Welch call reproduces itself bitwise (#730)."""
    from port.studies import stage
    from port.studies.copy_state_stream import oracle_states, scored, truth_label

    # NB manifests extend relative to the repository root.
    monkeypatch.chdir(ROOT)
    manifest = Path("sim/manifests/dev_tree_1s_hard.toml")
    member = next(stage.members(manifest, tmp_path / "sim", n=1))
    assert member.hash == "9ec90dc2"

    def study(found: stage.Stage) -> tuple[float, float, int, int]:
        one, two = found.run(), found.run()
        truth = truth_label(found)
        planted = scored(found, *oracle_states(found), truth, polish=True)
        own = scored(
            found,
            found.arguments["init_log_mu"],
            found.arguments["init_p_binom"],
            truth,
            polish=True,
        )
        assert found.X.shape[0] == found.planted.shape[0] == found.clone.size
        assert found.n_clones == 4
        assert found.arguments["t"] == pytest.approx(1.0 - 1e-7)
        # NB `cnaster`'s initializers read this configuration globally.
        assert yaml.safe_load(found.config)["hmm"]["t"] == pytest.approx(1.0 - 1e-7)
        return float(one.llf), float(two.llf), planted["missed"], own["missed"]

    one, two, planted, own = stage.at_oracle_clones(
        member.sample, study, root=tmp_path / "run"
    )

    assert one == two
    assert planted < own


@pytest.mark.release
@pytest.mark.oracle
def test_the_field_is_cnasters_at_the_planted_clones_less_the_clone_shift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dev_tree_1s_hard r0: `--sal`'s field is `cnaster`'s bitwise, less the shift (#362, #735)."""
    import port.patch.hmrf.clone_assignment as assignment
    import scipy.sparse as sp
    from port.studies import stage

    monkeypatch.chdir(ROOT)
    member = next(
        stage.members(
            Path("sim/manifests/dev_tree_1s_hard.toml"), tmp_path / "sim", n=1
        )
    )

    def fields(args: tuple[Any, ...], arguments: dict[str, Any], installed: Any) -> Any:
        _, shifted, _ = installed(*args, **arguments)
        _, theirs, _ = assignment.UPSTREAM(*args, **arguments)
        with monkeypatch.context() as unshift:
            unshift.setattr(assignment, "_clone_shifts", lambda *_: None)
            _, unshifted, _ = installed(*args, **arguments)
        bound = stage._bind(args, arguments)
        return np.asarray(shifted), np.asarray(theirs), np.asarray(unshifted), bound

    shifted, theirs, unshifted, bound = stage._drive(
        member.sample, "rdrbaf", None, tmp_path / "run", lambda _: {}, fields
    )
    found = stage.at_clone_assignment(
        member.sample, lambda f: f, root=tmp_path / "again"
    )

    np.testing.assert_array_equal(unshifted, theirs)
    assert np.abs(shifted - theirs).max() > 0
    assert (shifted.argmax(1) == theirs.argmax(1)).mean() > 0.9
    np.testing.assert_array_equal(found.field, shifted)
    np.testing.assert_array_equal(found.planted, np.asarray(bound["prev_assignment"]))
    adjacency = sp.csr_matrix((found.weights, found.indices, found.indptr))
    assert (adjacency != sp.csr_matrix(bound["adjacency_mat"])).nnz == 0
    assert found.spatial_weight == bound["spatial_weight"]
