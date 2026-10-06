"""`cnamaste`'s HMM initialization (T- #670 PR6).

`cnamaste.hmm_initialize.gmm_init` takes `port`'s distinct start (#348) as
the option `distinct`, off as `cnaster` is; `cnamaste.hmm.pipeline_baum_welch`
initializes itself (#143). `--sal`'s starts, `sal`'s `kmeans++x5+em` and the
lattice, need `sal` and are not moved. Referees, one per test:

- `patch`: `port.patch.hmm_initialize.distinct.gmm_init` and `cnaster`'s
  `gmm_init`, bitwise;
- `bug`: #143, against `cnaster` and against the same fit given the start;
- `end2end`: the planted states of dev_tree_1s_hard r0 (`d2938975`).

The 300 normal-UMI segment floor's end-to-end pin is `test_cnamaste_copy.py`'s.
"""

from __future__ import annotations

import tomllib
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

HARD = (
    Path(__file__).resolve().parents[1] / "sim/manifests/baseline/dev_tree_1s_hard.toml"
)
HARD_HASH = "d2938975"
"""dev_tree_1s_hard r0's `realization_hash`, as the manifest states it and
`test_sim_r0_hash.py` pins it."""


@contextmanager
def _installed(document: dict[str, Any]) -> Iterator[None]:
    """`document` as both packages' global configuration, restored after."""
    from cnamaste import config as own
    from port.sim.inputs import written_config

    saved = own._global_config
    with written_config(document):
        own.set_global_config(own.YAMLConfig(document))
        try:
            yield
        finally:
            own.set_global_config(saved)


@pytest.fixture
def _configs(tmp_path: Path) -> Iterator[None]:
    """`cnaster`'s test configuration, installed in both packages and restored."""
    from tests.conftest import SHIPPED_EM_FTOL, cnaster_test_config

    with _installed(cnaster_test_config(tmp_path, SHIPPED_EM_FTOL, 100)):
        yield


def _call(seed: int = 0, size: float = 30.0, n_states: int = 6) -> tuple[Any, ...]:
    """`gmm_init`'s positional arguments on one clone of 2,000 bins.

    1,600 neutral bins, 250 at a one-copy loss (`mu` 0.5, BAF 0.05) and 150
    at a gain (`mu` 1.5, BAF 1/3), each BAF on either haplotype at random;
    negative binomial depth at size `size` over exposures 300 to 600, and
    100 to 199 trials. At the defaults `cnaster`'s start keeps two slices of
    the neutral cluster and `distinct`'s merges them (#348).
    """
    generator = np.random.default_rng(seed)
    mu = np.r_[np.full(1600, 1.0), np.full(250, 0.5), np.full(150, 1.5)]
    p = np.r_[np.full(1600, 0.5), np.full(250, 0.05), np.full(150, 1 / 3)]
    p = np.where(generator.random(mu.size) < 0.5, 1 - p, p)
    exposure = generator.uniform(300, 600, mu.size)
    trials = generator.integers(100, 200, mu.size).astype(np.float64)
    total = generator.negative_binomial(size, size / (size + mu * exposure))
    b = generator.binomial(trials.astype(np.int64), p)
    X = np.stack([total, b], axis=1).astype(np.float64)[:, :, None]
    return (
        n_states,
        X,
        exposure[:, None],
        trials[:, None],
        "smp",
        np.array([mu.size]),
        None,
        np.zeros(mu.size),
    )


START = {"random_state": 0, "in_log_space": False, "only_minor": False}
"""How `run_core_inference` calls its initializer (`hmrf.py:511`)."""


def _same(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    return all(np.array_equal(a, b) for a, b in zip(left[:2], right[:2], strict=True))


@pytest.mark.patch
@pytest.mark.cnamaste
@pytest.mark.usefixtures("_configs")
def test_the_distinct_start_is_ports_and_off_it_is_cnasters() -> None:
    """`distinct=True` returns `port`'s distinct `(log_mu, p_binom)` bit for
    bit; at the default, `cnaster`'s. On `_call`'s instance the two differ, so
    the first equality is not `cnaster`'s start twice."""
    from cnamaste.hmm_initialize import gmm_init
    from cnaster.hmm_initialize import gmm_init as cnasters
    from port.patch.hmm_initialize.distinct import gmm_init as ports

    arguments = _call()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = gmm_init(*arguments, **START)
        distinct = gmm_init(*arguments, **START, distinct=True)
        assert _same(default, cnasters(*arguments, **START))
        assert _same(distinct, ports(*arguments, **START))

    assert not _same(default, distinct)


@pytest.mark.bug
@pytest.mark.cnamaste
@pytest.mark.usefixtures("_configs")
def test_pipeline_baum_welch_initializes_itself() -> None:
    """#143: given no start, `cnaster`'s `pipeline_baum_welch` raises
    `TypeError` calling `gmm_init`; `cnamaste`'s fits from `gmm_init`'s start,
    bit for bit the fit given that start."""
    from cnamaste.hmm import pipeline_baum_welch
    from cnamaste.hmm_initialize import gmm_init
    from cnamaste.hmm_nophasing import hmm_nophasing
    from cnaster.hmm import pipeline_baum_welch as cnasters
    from cnaster.hmm_nophasing import hmm_nophasing as cnasters_class

    n_states, X, exposure, trials, params, lengths, _, sitewise = _call(n_states=3)
    fit = {
        "hmmclass": hmm_nophasing,
        "params": params,
        "max_iter": 2,
        "clone_lengths": lengths,
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(TypeError, match="gmm_init"):
            cnasters(
                None, X, lengths, n_states, exposure, trials, sitewise,
                **{**fit, "hmmclass": cnasters_class},
            )  # fmt: skip

        alone = pipeline_baum_welch(
            None, X, lengths, n_states, exposure, trials, sitewise, **fit
        )
        log_mu, p_binom, _, _ = gmm_init(
            n_states, X, exposure, trials, params, lengths, None, sitewise,
            random_state=0, in_log_space=True, only_minor=False,
        )  # fmt: skip
        given = pipeline_baum_welch(
            None, X, lengths, n_states, exposure, trials, sitewise, **fit,
            init_log_mu=log_mu, init_p_binom=p_binom,
        )  # fmt: skip

    for name in ("new_log_mu", "new_p_binom", "new_alphas", "new_taus", "log_gamma"):
        np.testing.assert_array_equal(alone[name], given[name], err_msg=name)


@pytest.mark.end2end
@pytest.mark.cnamaste
@pytest.mark.merge
def test_the_starts_place_only_the_neutral_state_on_dev_tree_1s_hard() -> None:
    """dev_tree_1s_hard r0 (`d2938975`) at its planted clones: 7,688 rows,
    7,562 of them neutral, and 4 other phase-free states on 20 to 47 rows.

    `cnamaste`'s start at the study's configuration, `distinct` on or off,
    places one of the 5 planted states, the neutral one, within 0.1 in log
    mu and 0.05 in folded p (`port.sandbox.extensions.copy_starts.found`).
    `distinct` merges 1 of the 14 components and returns the same 7 states
    bit for bit. `--sal`'s `kmeans++x5+em`, which is not moved, also places
    1 of 5 (T- #670 PR6).
    """
    import port.sandbox.extensions.copy_starts as study
    from cnamaste.hmm_initialize import gmm_init
    from port.extensions.copy_starts import CopyStart
    from port.sandbox.known_copy import problems
    from port.studies.copy_state_stream import _call as call_of

    assert tomllib.loads(HARD.read_text())["sample"]["r0_hash"] == HARD_HASH
    call = call_of(next(problems(HARD, n=1)))
    planted = study.planted_states(call)
    assert call.total.size == 7_688
    assert planted[(1, 1)][2] == 7_562

    raw = call.raw
    arguments = (
        call.n_states, raw["X"], raw["base_nb_mean"], raw["total_bb_RD"],
        raw["params"], raw["lengths"], None, raw["log_sitewise_transmat"],
    )  # fmt: skip

    starts = {}
    # NB the configuration the study's calls carry, `zenodo_sim_config.yaml`.
    with _installed(yaml.safe_load(raw["config"])), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for distinct in (False, True):
            starts[distinct] = gmm_init(*arguments, **START, distinct=distinct)

    assert _same(starts[False], starts[True])
    found = study.found(
        CopyStart("distinct", "rdrbaf", np.ravel(starts[True][0]),
                  np.ravel(starts[True][1]), np.nan, 0.0, 0.0),
        planted,
    )  # fmt: skip
    assert [state for state, placed in found.items() if placed] == [(1, 1)]
    assert len(found) == 5
