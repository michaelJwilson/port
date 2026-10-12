"""`hmm_nophasing.clone_log_normalizers`: each clone's library normalizer `log Z_c` (L1, Ticket#276).

Plain tests until the read-depth emission applies it (L2) and the run reaches it. The referee is the sum written out:
`log Z_c = log sum_g lambda_g mu_{s_c(g)}`, `lambda` the spot-summed baseline normalized to 1.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from cnamaste.hmm_nophasing import clone_log_normalizers, compute_logmu_shifts

from fn import fitted


def _brute(log_mu: np.ndarray, paths: np.ndarray, base: np.ndarray) -> np.ndarray:
    profile = base.sum(axis=1)
    weights = profile / profile.sum()
    return np.array([np.log(np.sum(weights * np.exp(log_mu[paths[:, c]]))) for c in range(paths.shape[1])])


def _case(seed: int = 41) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    log_mu = rng.normal(scale=0.5, size=6)
    paths = rng.integers(0, 6, size=(50, 3))
    base = rng.gamma(2.0, size=(50, 20))
    base[7] = 0.0  # NB a bin with no baseline contributes nothing
    return log_mu, paths, base


def test_the_normalizer_is_the_sum() -> None:
    """`log Z_c` equals the sum written out, per clone, to rtol 1e-12, a bin with no baseline included.

    Fails where lambda is not normalized, a clone reads another's path, or a bin is dropped or counted twice."""
    log_mu, paths, base = _case()
    assert np.allclose(clone_log_normalizers(log_mu, paths, base), _brute(log_mu, paths, base), rtol=1e-12, atol=0.0)


def test_the_normalizer_is_cnasters_shift() -> None:
    """Each clone's `compute_logmu_shifts` block is that clone's normalizer, repeated along its bins."""
    log_mu, paths, base = _case()
    profile = base.sum(axis=1)
    with np.errstate(divide="ignore"):
        log_lambda = np.log(profile / profile.sum())
    blocks = compute_logmu_shifts(log_mu, np.ascontiguousarray(paths.T).reshape(-1), np.tile(log_lambda, 3), np.full(3, 50))
    assert np.array_equal(blocks, np.repeat(clone_log_normalizers(log_mu, paths, base), 50))


def test_a_common_scale_moves_every_normalizer_by_it() -> None:
    """`mu -> c mu` moves every `log Z_c` by `log c`: the shifted mean `mu / Z_c` cannot see a common scale, which
    is why L3 pins the neutral state."""
    log_mu, paths, base = _case()
    for shift in (-1.3, 0.7, 4.0):
        moved = clone_log_normalizers(log_mu + shift, paths, base) - clone_log_normalizers(log_mu, paths, base)
        assert np.allclose(moved, shift, rtol=0.0, atol=1e-12)


def test_no_baseline_is_none() -> None:
    log_mu, paths, base = _case()
    assert clone_log_normalizers(log_mu, paths, np.zeros_like(base)) is None


def test_on_easys_rdr_fit(ctx: Any) -> None:
    """On easy's RDR fit: its `mu`, each clone's decoded path and the baseline give the sum written out (1e-12); the
    normalizers printed, for L2-L5 to cite."""
    f = fitted(ctx, "08_rdr")
    res = f["res"]
    n_obs = f["single_X"].shape[0]
    log_mu = np.asarray(res["new_log_mu"]).reshape(-1)
    pred = np.asarray(res["pred_cnv"]) % log_mu.size
    paths = pred.reshape(-1, n_obs).T if pred.ndim == 1 else pred
    found = clone_log_normalizers(log_mu, paths, f["single_base_nb_mean"])
    print(f"easy RDR: log Z_c {np.round(found, 6).tolist()}, mu {np.round(np.exp(log_mu), 4).tolist()}")
    assert np.allclose(found, _brute(log_mu, paths, f["single_base_nb_mean"]), rtol=1e-12, atol=0.0)
