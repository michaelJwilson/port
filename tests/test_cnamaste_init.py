"""`cnamaste`'s HMM initialization (T- #670 PR6).

`cnamaste.hmm_initialize.gmm_init` takes `port`'s distinct start (#348) as
the option `distinct`, off as `cnaster` is; `cnamaste.hmm.pipeline_baum_welch`
initializes itself (#143). `--sal`'s starts, `sal`'s `kmeans++x5+em` and the
lattice, need `sal` and are not moved. Referees, one per test:

- `patch`: `port.patch.hmm_initialize.distinct.gmm_init` and `cnaster`'s
  `gmm_init`, bitwise;
- `bug`: #143, against `cnaster` and against the same fit given the start;
- `end2end`: the planted states of dev_tree_1s_hard r0 (`9ec90dc2`), at the run's stage.

The 300 normal-UMI segment floor's end-to-end pin is `test_cnamaste_copy.py`'s.
"""

from __future__ import annotations

import functools
import tomllib
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

HARD = Path(__file__).resolve().parents[1] / "sim/manifests/dev_tree_1s_hard.toml"
HARD_HASH = "9ec90dc2"
"""dev_tree_1s_hard r0's `realization_hash`, as the manifest states it and
`test_sim_r0_hash.py` pins it."""


@functools.cache
def hard_call() -> Any:
    """dev_tree_1s_hard r0's copy-state problem at its planted clones, as the run builds it.

    `run_cnaster_port --sal` to the RDR + BAF stage's first Baum-Welch
    (`port.studies.stage.at_oracle_clones`, #730), read as a `CopyCall`
    (`copy_state_stream._call`): the problem the copy-state study measures.
    Before #730 it was `port.sandbox.known_copy`'s rebuild of it.
    """
    import tempfile

    from port.studies import stage
    from port.studies.copy_state_stream import _call

    root = Path(tempfile.mkdtemp())
    member = next(stage.members(HARD, root / "sim", n=1))
    assert member.hash == HARD_HASH
    return stage.at_oracle_clones(member.sample, _call, root=root / "run")


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
def test_the_starts_place_the_planted_states_of_dev_tree_1s_hard() -> None:
    """dev_tree_1s_hard r0 (`9ec90dc2`) at its planted clones, as the run
    builds it (`hard_call`): 7,284 rows, 6,989 of them neutral, and 4 other
    phase-free states on 23 to 134 rows.

    `cnamaste`'s start at the run's configuration places the neutral state
    and `(1, 2)` within 0.1 in log mu and 0.05 in folded p
    (`port.sandbox.extensions.copy_starts.found`); `distinct` places `(2, 2)`
    as well, 3 of the 5. On `port.sandbox.known_copy`'s rebuild of r0
    (`d2938975`, before #730) both placed the neutral state alone and
    returned the same states bit for bit; on the run's problem they differ.
    """
    import port.sandbox.extensions.copy_starts as study
    from cnamaste.hmm_initialize import gmm_init
    from port.extensions.copy_starts import CopyStart

    assert tomllib.loads(HARD.read_text())["sample"]["r0_hash"] == HARD_HASH
    call = hard_call()
    planted = study.planted_states(call)
    assert call.total.size == 7_284
    assert planted[(1, 1)][2] == 6_989

    raw = call.raw
    arguments = (
        call.n_states, raw["X"], raw["base_nb_mean"], raw["total_bb_RD"],
        raw["params"], raw["lengths"], None, raw["log_sitewise_transmat"],
    )  # fmt: skip

    starts = {}
    # NB the run's own configuration, as its stage carries it
    with _installed(yaml.safe_load(raw["config"])), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for distinct in (False, True):
            starts[distinct] = gmm_init(*arguments, **START, distinct=distinct)

    placed = {}
    for distinct, (log_mu, p_binom, *_) in starts.items():
        found = study.found(
            CopyStart("distinct", "rdrbaf", np.ravel(log_mu), np.ravel(p_binom),
                      np.nan, 0.0, 0.0),
            planted,
        )  # fmt: skip
        assert len(found) == 5
        placed[distinct] = [state for state, hit in found.items() if hit]
    assert not _same(starts[False], starts[True])
    assert placed == {False: [(1, 1), (1, 2)], True: [(1, 1), (1, 2), (2, 2)]}
