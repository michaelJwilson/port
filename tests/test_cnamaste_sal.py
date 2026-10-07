"""`cnamaste`'s `sal` rows (T- #670 PR6b): the dense emission and the copy-state starts.

`cnamaste` declares `snakes_and_ladders` (the owner's decision on T- #670),
so the rows PR5 and PR6 left out move in, each off by default as `port`
leaves it without its shift: `hmm_nophasing.emission_kernels="sal"` (#425),
and `gmm_init`'s `start` / `baf_start`, `sal`'s `kmeans++x5+em` (#489) and
the lattice (#540), which fix #236. Referees, one per test:

- `patch`: `port`'s `hmm_nophasing` under `LOG_SPACE_SWAPS` and `port`'s
  `sal_mixture.gmm_init`, bitwise;
- `oracle`: `scipy.stats` for the dense emission; `sal`'s own EM on the
  raw exposure and trials for the moved start;
- `bug`: #236, the GMM's equal vote per bin, against `cnaster`'s start;
- `end2end`: the planted states of dev_tree_1s_hard r0 (`9ec90dc2`), at the run's stage.

The gate instance's (`350fbd2b`) byte pin against absorbed `cnaster` is
`test_cnamaste_copy.py`'s, unchanged: every row here is off by default.
"""

from __future__ import annotations

import hashlib
import tomllib
import warnings
from typing import Any

import numpy as np
import pytest
from scipy import stats

from tests.test_cnamaste_hmm import N_STATES, _configs, _fit, _stacked
from tests.test_cnamaste_init import _installed

__all__ = ["_configs"]

# --- the dense emission (#425) -------------------------------------------------


@pytest.mark.patch
@pytest.mark.cnamaste
@pytest.mark.usefixtures("_configs")
@pytest.mark.parametrize("shift", [False, True], ids=["unshifted", "shifted"])
def test_the_sal_emission_fit_is_ports(shift: bool) -> None:
    """`emission_kernels="sal"`: `cnamaste.hmm_nophasing`'s fit against
    `port`'s, every returned array bitwise, the shift off and on. The fit
    differs from the `cnaster` kernels' fit, so the equality is not that one
    twice."""
    from cnamaste.hmm_nophasing import hmm_nophasing as own
    from port.patch.hmm_nophasing import hmm_nophasing as ports
    from port.pipeline import LOG_SPACE_SWAPS, patched

    problem = _stacked(per_clone=200)
    options = {"apply_logmu_shift": shift, "emission_kernels": "sal"}
    ours = _fit(own, problem, **options)
    with patched(LOG_SPACE_SWAPS):
        theirs = _fit(ports, problem, **options)

    assert sorted(ours) == sorted(theirs)
    for name, value in ours.items():
        np.testing.assert_array_equal(value, theirs[name], err_msg=name)

    kernels = _fit(own, problem, apply_logmu_shift=shift)
    assert not np.array_equal(kernels["new_log_mu"], ours["new_log_mu"])


EMISSION_TOLERANCE = 1e-12
"""Relative, on log emissions of -47 to -0.77 nats; 6.2e-14 (NB) and 3.2e-14
(BB) realized."""


@pytest.mark.oracle
@pytest.mark.cnamaste
@pytest.mark.usefixtures("_configs")
def test_the_sal_emission_is_scipys() -> None:
    """The coded emission under `emission_kernels="sal"` against
    `scipy.stats.nbinom` and `betabinom`, every state at every bin of the
    3-clone fixture at seed 1, 200 bins per clone."""
    from cnamaste.count_encoder import CountEncoder
    from cnamaste.hmm_nophasing import hmm_nophasing as own

    problem = _stacked(per_clone=200)
    # NB whole exposures: `CountEncoder` rounds a fractional one to its
    #    `decimals`, which moves the score by 1.3e-8 relative under either kernel.
    X, base, total = problem["X"], np.rint(problem["base"]), problem["total"]
    log_mu = np.linspace(-0.6, 0.5, N_STATES)[:, None]
    alphas = np.full((N_STATES, 1), 0.05)
    p_binom = np.linspace(0.15, 0.5, N_STATES)[:, None]
    taus = np.full((N_STATES, 1), 40.0)

    model = type("sal", (own,), {"emission_kernels": "sal"})(params="smp")
    rdr, baf = model.compute_emission_probability_nb_betabinom_coded(
        CountEncoder(X[:, 0, :], base),
        CountEncoder(X[:, 1, :], total),
        log_mu,
        alphas,
        p_binom,
        taus,
    )

    mean = np.exp(log_mu) * base[:, 0][None, :]
    size = 1.0 / alphas
    nb = stats.nbinom.logpmf(X[:, 0, 0][None, :], size, size / (size + mean))
    bb = stats.betabinom.logpmf(
        X[:, 1, 0][None, :], total[:, 0][None, :], p_binom * taus, (1 - p_binom) * taus
    )

    np.testing.assert_allclose(rdr, nb, rtol=EMISSION_TOLERANCE)
    np.testing.assert_allclose(baf, bb, rtol=EMISSION_TOLERANCE)


# --- the copy-state starts (#489, #540) ----------------------------------------

MU = np.array([0.5, 1.0, 1.5])
"""The planted read-depth ratios of `_depths`' three states."""

P = np.array([0.1, 0.5, 1.0 / 3.0])
"""Their planted B-allele frequencies."""

SHALLOW, DEEP = 4.0, 200.0
"""`_depths`' two exposures, a 50x span."""


def _depths(seed: int, deep_share: float) -> tuple[Any, ...]:
    """`gmm_init`'s positional arguments on 2,000 bins of one clone.

    Each bin is one of three states at random, `MU` and `P`; its exposure is
    `DEEP` with probability `deep_share`, `SHALLOW` otherwise. Negative
    binomial depth at dispersion 1e-3, so a shallow bin's log ratio is
    noisier than a deep one's by `sqrt(DEEP / SHALLOW)` (#236); binomial B
    counts over 20 to 59 trials.
    """
    generator = np.random.default_rng(seed)
    state = generator.integers(0, 3, 2_000)
    exposure = np.where(generator.random(state.size) < deep_share, DEEP, SHALLOW)
    trials = generator.integers(20, 60, state.size).astype(np.float64)
    size = 1.0 / 1e-3
    total = generator.negative_binomial(size, size / (size + MU[state] * exposure))
    b = generator.binomial(trials.astype(np.int64), P[state])
    X = np.stack([total, b], axis=1).astype(np.float64)[:, :, None]
    return (
        3, X, exposure[:, None], trials[:, None], "smp",
        np.array([state.size]), None, np.zeros(state.size),
    )  # fmt: skip


def _hash(arguments: tuple[Any, ...]) -> str:
    """The drawn counts, exposures and trials, as 8 hex digits of their SHA-256."""
    digest = hashlib.sha256()
    for array in arguments[1:4]:
        digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
    return digest.hexdigest()[:8]


DEPTHS = {(0, 0.5): "dbe87caa"}
"""`_hash` of each `_depths(seed, deep_share)` a test reads by name."""

START = {"random_state": 0, "in_log_space": False, "only_minor": False}
"""How `run_core_inference` calls its initializer (`hmrf.py:511`)."""


def _baf_only(arguments: tuple[Any, ...]) -> tuple[Any, ...]:
    """The BAF-only stage's call: `params` without `m`, the depth zeroed as `cnaster` does."""
    n_states, X, exposure, *rest = arguments
    X = X.copy()
    X[:, 0, :] = 0.0
    return (n_states, X, np.zeros_like(exposure), rest[0], "sp", *rest[2:])


@pytest.mark.patch
@pytest.mark.cnamaste
@pytest.mark.usefixtures("_configs")
@pytest.mark.parametrize(
    ("stage", "start"),
    [("rdrbaf", "kmeans++x5+em"), ("rdrbaf", "lattice"), ("baf", "lattice")],
)
def test_the_starts_are_ports(stage: str, start: str) -> None:
    """`gmm_init`'s `start` (BAF + RDR) and `baf_start` (BAF only) against
    `port.patch.hmm_initialize.sal_mixture.gmm_init`'s, `(log_mu, p_binom)`
    bitwise on `_depths(0, 0.5)`."""
    from cnamaste.hmm_initialize import gmm_init
    from port.patch.hmm_initialize.sal_mixture import gmm_init as ports

    arguments = _depths(0, 0.5)
    assert _hash(arguments) == DEPTHS[0, 0.5]
    key = "start" if stage == "rdrbaf" else "baf_start"
    if stage == "baf":
        arguments = _baf_only(arguments)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ours = gmm_init(*arguments, **START, **{key: start})
        theirs = ports(*arguments, **START, **{key: start})

    for left, right in zip(ours[:2], theirs[:2], strict=True):
        np.testing.assert_array_equal(left, right)
    assert ours[2:] == theirs[2:] == (None, None)


START_TOLERANCE = 5e-4
"""Absolute, on `log mu` and `p`: the start's polish stops at a relative
log-likelihood change of 1e-6 (13 nats x 1e-6 here) and `sal`'s EM at 1e-10;
2.3e-4 realized."""


@pytest.mark.oracle
@pytest.mark.cnamaste
@pytest.mark.usefixtures("_configs")
def test_the_start_is_sals_mixture_fit() -> None:
    """`kmeans++x5+em` through `gmm_init` lands where `sal`'s EM lands on the
    raw counts, exposures and trials from the planted states.

    The start seeds in `sal`'s rate space, with the exposure divided by
    `EXPOSURE_SCALE` and the B column over the common trial count (#547);
    the referee is `sal.opt.emission_mixture.expectation_maximization` with
    neither, run to its own tolerance. The same model, so the same maximum.
    """
    from cnamaste.hmm_initialize import gmm_init
    from sal.opt.emission_mixture import CountPairSeeding, expectation_maximization

    arguments = _depths(0, 0.5)
    assert _hash(arguments) == DEPTHS[0, 0.5]
    _, X, exposure, trials, *_ = arguments

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_mu, p_binom, _, _ = gmm_init(*arguments, **START, start="kmeans++x5+em")

    at = CountPairSeeding(dispersion=10.0, concentration=1_000.0, joint=False, trials=1)
    fit = expectation_maximization(
        X[:, :, 0],
        np.full(3, 1.0 / 3.0),
        at(np.column_stack([MU, P])),
        covariate=np.column_stack([exposure, trials]),
    )
    assert fit.termination is not None
    assert fit.termination.converged

    components: Any = fit.components
    mean = np.asarray(components.total.mean, dtype=np.float64).reshape(-1)
    rate = np.asarray(components.rate, dtype=np.float64).reshape(-1)
    ours, theirs = np.argsort(log_mu.ravel()), np.argsort(mean)
    np.testing.assert_allclose(
        log_mu.ravel()[ours], np.log(mean[theirs]), rtol=0, atol=START_TOLERANCE
    )
    np.testing.assert_allclose(
        p_binom.ravel()[ours], rate[theirs], rtol=0, atol=START_TOLERANCE
    )


def _error(log_mu: np.ndarray) -> float:
    """The largest distance of the three states' `mu` from `MU`, sorted."""
    return float(np.abs(np.sort(np.exp(np.ravel(log_mu))) - MU).max())


@pytest.mark.bug
@pytest.mark.cnamaste
@pytest.mark.usefixtures("_configs")
def test_the_gmm_gives_a_shallow_bin_a_deep_bins_vote_and_the_start_does_not() -> None:
    """#236: `cnaster`'s start fits a Gaussian to `X / base_nb_mean`, so a bin
    at exposure 4 votes as one at 200 with 7x its noise. With every bin deep
    it places the three states within 0.012 of `MU`; with half of them
    shallow, 0.037 to 0.108 off on seeds 0 to 2. `kmeans++x5+em`, which fits
    the negative binomial with each bin's exposure, stays within 0.014.
    """
    from cnamaste.hmm_initialize import gmm_init
    from cnaster.hmm_initialize import gmm_init as cnasters

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in (0, 1, 2):
            deep = _depths(seed, 1.0)
            mixed = _depths(seed, 0.5)
            gmm = cnasters(*mixed[:4], "sm", *mixed[5:], **START)
            start = gmm_init(*mixed[:4], "sm", *mixed[5:], **START, start=DEFAULT)

            assert _error(cnasters(*deep[:4], "sm", *deep[5:], **START)[0]) < 0.02
            assert _error(gmm[0]) > 0.03, seed
            assert _error(start[0]) < 0.02, seed


DEFAULT = "kmeans++x5+em"
"""`--sal`'s start (#489)."""


HARD_HASH = "9ec90dc2"
"""dev_tree_1s_hard r0's `realization_hash`, as its manifest states it."""


@pytest.mark.end2end
@pytest.mark.cnamaste
@pytest.mark.merge
def test_the_moved_starts_place_the_planted_states_of_dev_tree_1s_hard() -> None:
    """dev_tree_1s_hard r0 (`9ec90dc2`) at its planted clones, as the run
    builds it (`test_cnamaste_init.hard_call`): 7,284 rows, 6,989 neutral and
    4 other phase-free states on 23 to 134 rows.

    A state is placed within 0.1 in log mu and 0.05 in folded p
    (`port.sandbox.extensions.copy_starts.found`). `kmeans++x5+em` and the
    lattice, through `cnamaste.hmm_initialize.gmm_init`, each place 3 of the
    5: the neutral state, the one-copy loss `(0, 1)` and `(1, 2)`; PR6's GMM
    start places 2, the neutral state and `(1, 2)`.
    """
    import port.sandbox.extensions.copy_starts as study
    import yaml
    from cnamaste.hmm_initialize import gmm_init
    from port.extensions.copy_starts import CopyStart

    from tests.test_cnamaste_init import HARD, hard_call

    assert tomllib.loads(HARD.read_text())["sample"]["r0_hash"] == HARD_HASH
    call = hard_call()
    planted = study.planted_states(call)
    assert call.total.size == 7_284
    raw = call.raw
    arguments = (
        call.n_states, raw["X"], raw["base_nb_mean"], raw["total_bb_RD"],
        raw["params"], raw["lengths"], None, raw["log_sitewise_transmat"],
    )  # fmt: skip

    placed = {}
    with _installed(yaml.safe_load(raw["config"])), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for start in PLACED:
            log_mu, p_binom, _, _ = gmm_init(*arguments, **START, start=start)
            found = study.found(
                CopyStart(start, "rdrbaf", np.ravel(log_mu), np.ravel(p_binom),
                          np.nan, 0.0, 0.0),
                planted,
            )  # fmt: skip
            placed[start] = sorted(state for state, hit in found.items() if hit)

    assert placed == PLACED


PLACED: dict[str, list[tuple[int, int]]] = {
    "kmeans++x5+em": [(0, 1), (1, 1), (1, 2)],
    "lattice": [(0, 1), (1, 1), (1, 2)],
}
"""The planted states each start places on dev_tree_1s_hard r0 at the run's
stage (`9ec90dc2`): 3 of 5 each, against the GMM's 2 (`test_cnamaste_init`).
On `port.sandbox.known_copy`'s rebuild (`d2938975`, before #730) they placed
1 and 2, the GMM 1."""
