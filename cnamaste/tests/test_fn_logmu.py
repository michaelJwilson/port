"""`hmm_nophasing`'s read-depth emission under each clone's library normalizer, `hmrf.apply_logmu_shift` (L2, T- #276).

Per-function rows: `hmrf.apply_logmu_shift` is False on easy, so `run_cnamaste` reaches the shifted path but easy does
not execute it, and every row is synthetic. The generative model is the referee: clone `c` draws its
pseudobulk counts from `NB(base_g mu_{s_c(g)} / Z_c)`, `base_g = N_c lambda_g`, so its counts sum to `N_c` in
expectation whatever its copy numbers; `log Z_c` is `clone_log_normalizers`'.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from cnamaste.config import YAMLConfig, set_global_config
from cnamaste.count_encoder import CountEncoder
from cnamaste.hmm import pipeline_baum_welch
from cnamaste.hmm_nophasing import clone_log_normalizers, compute_logmu_shifts, hmm_nophasing

from fn import Row, run, table


ROOT = Path(__file__).resolve().parents[1]
MU = np.array([0.5, 1.0, 1.5, 2.0])
ALPHA = 0.005


def _config() -> None:
    """The packaged configuration, the encoder's rounding at 12 decimals so the coded emission is the dense one's; `run`
    reinstalls the staged configuration for the next row."""
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    config["hmm"]["compression_decimals"] = 12
    set_global_config(YAMLConfig(config))


def simulate(seed: int = 7, n_obs: int = 600, totals: tuple[float, ...] = (2e5, 5e5, 1e6)) -> dict[str, Any]:
    """Three clones, neutral (`mu = 1`) but for 3-5 segments each, at three library sizes."""
    rng = np.random.default_rng(seed)
    n_clones = len(totals)
    lam = rng.gamma(4.0, size=n_obs)
    lam /= lam.sum()
    paths = np.ones((n_obs, n_clones), dtype=np.int64)
    for c in range(n_clones):
        for _ in range(3 + c):
            a = rng.integers(0, n_obs - 60)
            paths[a : a + rng.integers(30, 120), c] = rng.choice([0, 2, 3])
    log_z = np.log((lam[:, None] * MU[paths]).sum(axis=0))
    base = lam[:, None] * np.asarray(totals)[None, :]
    mean = base * MU[paths] / np.exp(log_z)
    X = rng.negative_binomial(1 / ALPHA, 1 / (1 + ALPHA * mean))
    return {"lam": lam, "paths": paths, "log_z": log_z, "base": base, "X": X}


def _stacked(d: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Clones concatenated along the genomic axis, as `hmrf` stacks them: counts, baseline, copy states, lengths."""
    n_obs, n_clones = d["X"].shape
    X = np.zeros((n_obs * n_clones, 2, 1))
    X[:, 0, 0] = d["X"].T.reshape(-1)
    return X, d["base"].T.reshape(-1, 1), d["paths"].T.reshape(-1), np.full(n_clones, n_obs)


def _coded(d: dict[str, Any], log_mu: np.ndarray, copy_states: np.ndarray | None) -> np.ndarray:
    X, base, states, lengths = _stacked(d)
    nb, bb = CountEncoder(X[:, 0, :], base), CountEncoder(X[:, 1, :], np.zeros_like(base))
    k = log_mu.size
    rdr, _ = hmm_nophasing().compute_emission_probability_nb_betabinom_coded(
        nb, bb, log_mu.reshape(-1, 1), np.full((k, 1), ALPHA), np.full((k, 1), 0.5), np.ones((k, 1)),
        normal_log_lambda=np.tile(np.log(d["lam"]), lengths.size), clone_lengths=lengths, copy_states=copy_states,
    )  # fmt: skip
    return rdr


def _shift_is_the_normalizer(d: dict[str, Any]) -> None:
    """The coded emission under the true paths is the dense emission at baseline `base / Z_c`, `log Z_c` the
    generative one, to 1e-9; its shifts are `clone_log_normalizers`', bitwise."""
    _config()
    X, base, states, lengths = _stacked(d)
    shifts = compute_logmu_shifts(np.log(MU), states, np.tile(np.log(d["lam"]), lengths.size), lengths)
    assert np.array_equal(shifts[:: d["X"].shape[0]], clone_log_normalizers(np.log(MU), d["paths"], d["base"] / d["base"].sum(axis=0)))
    assert np.allclose(shifts[:: d["X"].shape[0]], d["log_z"], rtol=0.0, atol=1e-12)
    k = MU.size
    dense, _ = hmm_nophasing.compute_emission_probability_nb_betabinom(
        X, base * np.exp(-shifts)[:, None], np.log(MU).reshape(-1, 1), np.full((k, 1), ALPHA), np.zeros_like(base),
        np.full((k, 1), 0.5), np.ones((k, 1)),
    )  # fmt: skip
    assert np.allclose(_coded(d, np.log(MU), states), dense.reshape(k, -1), rtol=0.0, atol=1e-9)


def _common_scale_is_invisible(d: dict[str, Any]) -> None:
    """Shifted, `mu -> c mu` leaves the emission unchanged (1e-9): `mu` has no scale of its own, which L3's pin
    supplies. Unshifted, it moves the emission by more than 1."""
    _config()
    states = _stacked(d)[2]
    for scale in (-0.7, 0.4):
        assert np.allclose(_coded(d, np.log(MU) + scale, states), _coded(d, np.log(MU), states), rtol=0.0, atol=1e-9)
        assert np.max(np.abs(_coded(d, np.log(MU) + scale, None) - _coded(d, np.log(MU), None))) > 1.0


def _fit(d: dict[str, Any], shifted: bool) -> Any:
    X, base, states, lengths = _stacked(d)
    log_gamma = None
    if shifted:
        log_gamma = np.full((MU.size, states.size), -50.0)
        log_gamma[states, np.arange(states.size)] = 0.0
    return pipeline_baum_welch(
        None, X, lengths, MU.size, base, np.zeros_like(base), np.zeros(states.size), None, hmmclass=hmm_nophasing,
        params="sm", init_log_mu=np.log(MU), init_p_binom=np.full(MU.size, 0.5), normal_lambda=d["lam"],
        clone_lengths=lengths, init_log_gamma=log_gamma,
    )  # fmt: skip


def _shifted_fit_recovers_the_truth(d: dict[str, Any]) -> None:
    """Started at the truth, with the true paths as the previous copy states: the shifted fit keeps `mu / mu_1`
    within 2%, decodes every bin, reports each `log Z_c` to 0.02 after the scale `mu_1` takes, and its likelihood
    exceeds the unshifted fit's, which loses the truth: a clone's library size and its copy numbers are not
    separable without `Z_c`."""
    _config()
    states = _stacked(d)[2]
    shifted, unshifted = _fit(d, True), _fit(d, False)
    mu = np.exp(np.asarray(shifted["new_log_mu"]).reshape(-1))
    assert np.allclose(mu / mu[1], MU, rtol=0.02, atol=0.0)
    assert np.array_equal(np.asarray(shifted["pred_cnv"]), states)
    assert np.allclose(shifted.params.new_log_mu_shift - np.log(mu[1]), d["log_z"], rtol=0.0, atol=0.02)
    assert shifted["llf"] > unshifted["llf"] + 100.0
    lost = np.exp(np.asarray(unshifted["new_log_mu"]).reshape(-1))
    assert unshifted.params.new_log_mu_shift is None and not np.allclose(lost / lost[1], MU, rtol=0.1, atol=0.0)
    print(f"shifted: mu/mu_1 {np.round(mu / mu[1], 4).tolist()}, llf {shifted['llf']:.2f}; "
          f"unshifted: {np.round(lost / lost[1], 4).tolist()}, llf {unshifted['llf']:.2f}")  # fmt: skip


SIM = "synthetic: 3 clones x 600 bins, mu (0.5, 1, 1.5, 2), libraries 2e5, 5e5, 1e6"

ORACLE: list[Row] = table(
    "oracle",
    ("hmm_nophasing:compute_logmu_shifts", SIM, lambda c: simulate(), _shift_is_the_normalizer,
     "each clone's shift is the generative log Z_c (1e-12) and clone_log_normalizers' (bitwise)"),
    ("hmm_nophasing:hmm_nophasing.compute_emission_probability_nb_betabinom_coded", SIM, lambda c: simulate(), _shift_is_the_normalizer,
     "with copy states: the dense emission at baseline base / Z_c, 1e-9"),
    ("hmm_nophasing:hmm_nophasing.get_copy_states", SIM, lambda c: simulate(), _shifted_fit_recovers_the_truth,
     "shifted, from the truth: mu / mu_1 within 2%, every bin decoded, log Z_c to 0.02; llf above the unshifted fit's"),
)

INVARIANT: list[Row] = table(
    "invariant",
    ("hmm_nophasing:hmm_nophasing.compute_emission_probability_nb_betabinom_coded", SIM, lambda c: simulate(), _common_scale_is_invisible,
     "with copy states, mu -> c mu leaves the emission unchanged (1e-9); without, it moves it by > 1"),
)


@pytest.mark.parametrize("row", ORACLE, ids=[r.id for r in ORACLE])
def test_oracle_agrees(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """The generative model's normalizers and truth."""
    run(row, ctx, request)


@pytest.mark.parametrize("row", INVARIANT, ids=[r.id for r in INVARIANT])
def test_invariants_hold(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """A common scale on mu."""
    run(row, ctx, request)
