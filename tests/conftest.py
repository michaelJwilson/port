"""Shared fixtures.

`cnaster` reads configuration from a module-level global rather than from
its arguments, so a test touching a path that consults it has to set one.
The fixture below does that and restores what was there, because a global
left behind is a test that passes alone and fails in a suite.
"""

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

COMPRESSION_DECIMALS = 6
"""Places `CountEncoder` rounds to before deduplicating.

The only key the paths under test read. Six is `cnaster`'s own default and
is far enough below float64's precision that the rounding is not what any
comparison here measures.
"""


MIN_PHASE_SWITCH_PROB = 1e-10
"""Floor `compute_numbat_phase_switch_prob` clamps to when none is passed.

Far below any distance the fixtures use, so the clamp is observable as a
floor rather than mistaken for a computed value.
"""


BETABINOM_START_PARAMS = "0.1,0.3,0.5,0.7,0.9"
"""Starting `p` per state, read by `get_betabinom_start_params`.

`cnaster` reads a fixed list from configuration and
`Weighted_BetaBinom_mix.get_default_params` slices `[:num_states]` from it,
so the list has to be at least as long as the widest state space any test
fits. Five is that today.

The values are spread across `(0, 1)` and deliberately do **not** sit at the
planted ones: a start at the answer is a test of the objective's value, not
of the solve that is supposed to find it.
"""

BETABINOM_START_DISPERSION = 10.0
"""Starting `tau`, read by `get_betabinom_start_params`.

Away from the fixtures' planted concentration for the same reason.
"""


SHIPPED_EM_FTOL = 1e-6
"""`cnaster`'s own `em_ftol`, and what `cnaster_config` installs.

Read by `get_em_solver_params` and passed to `L-BFGS-B`, whose `ftol` is
**relative to the objective's magnitude**. The objective is a sum over
observations, so this criterion tightens with nothing and loosens with the
data: `tests/test_m_step.py` measures the fit stopping 27 nats short of its
own maximum at 9,600 observations, and reporting that it converged.

Installed as the default anyway, because a test that silently configures a
better solver than the one `cnaster` ships is testing a program nobody runs.
Where a test needs the maximum rather than where `cnaster` stops, it takes
`cnaster_converged_config` and says so.
"""

CONVERGED_EM_FTOL = 1e-9
"""An `em_ftol` at which the solve reaches its maximum.

Not a recommendation for `cnaster` -- a relative criterion is the wrong
shape whatever its value, and three decades is only what these fixtures
need. It is the setting under which "do the two implementations find the
same point" is a question about the implementations rather than about where
one of them gave up.
"""


def install_cnaster_config(tmp_path: Path, em_ftol: float, em_maxiter: int) -> None:
    """Set the global config `cnaster` reads instead of taking arguments."""
    from cnaster.config import YAMLConfig, set_global_config

    set_global_config(
        YAMLConfig(
            {
                "phasing": {"min_prob": MIN_PHASE_SWITCH_PROB},
                # NB `hmm_utils` reads the solver name and then the
                #    `em_`-prefixed option for each keyword that solver takes.
                "hmm": {
                    "compression_decimals": COMPRESSION_DECIMALS,
                    "solver": "L-BFGS-B",
                    "em_maxiter": em_maxiter,
                    "em_ftol": em_ftol,
                    "em_disp": 0,
                    "em_xrtol": 1e-5,
                    "em_xtol": 1e-5,
                },
                "betabinom": {
                    "start_params": BETABINOM_START_PARAMS,
                    "start_disp": BETABINOM_START_DISPERSION,
                },
                # NB `hmm_emission.flush_perf` reads this to count the rows
                #    already written. It does not write here; see
                #    `cnaster_perf_sink`.
                "paths": {"perf_path": str(tmp_path / "cnaster.perf")},
            }
        )
    )


@pytest.fixture
def cnaster_config(tmp_path: Path) -> Iterator[None]:
    """Install a minimal `cnaster` global config, and put back what was there.

    At `cnaster`'s own solver settings. A global left behind is a test that
    passes alone and fails in a suite, so what was there is restored.
    """
    from cnaster.config import get_global_config, set_global_config

    previous = get_global_config()
    install_cnaster_config(tmp_path, em_ftol=SHIPPED_EM_FTOL, em_maxiter=100)
    try:
        yield
    finally:
        set_global_config(previous)


@pytest.fixture
def cnaster_converged_config(tmp_path: Path) -> Iterator[None]:
    """As `cnaster_config`, but at a criterion the solve actually reaches.

    For a comparison of two implementations. Where `cnaster` stops under its
    shipped criterion is a separate question, and the tests that ask it take
    `cnaster_config` instead.
    """
    from cnaster.config import get_global_config, set_global_config

    previous = get_global_config()
    install_cnaster_config(tmp_path, em_ftol=CONVERGED_EM_FTOL, em_maxiter=5_000)
    try:
        yield
    finally:
        set_global_config(previous)


@pytest.fixture
def cnaster_perf_sink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run in a scratch directory, because `fit` writes a file into the one it is in.

    `Weighted_BetaBinom_mix.fit` calls `hmm_emission.flush_perf`
    unconditionally, and `flush_perf` opens the **literal relative path**
    `"cnaster.perf"` -- it reads `config.paths.perf_path` only to count the
    existing rows and to decide whether to write a header. So the configured
    path and the written path are two different files, and a fit performed
    from a checkout drops a timing log into it.

    Confirmed by observation, not by reading alone: an early run of this
    work left `cnaster.perf` in the repository root.

    Two consequences for a test, and this fixture handles both. The estimator
    has a side effect, so it cannot be called from wherever pytest happens to
    start; and the header is written when the *configured* file is absent
    while the appended-to file may already exist, so a header can land in the
    middle. Returns the scratch directory, so a test may assert on what was
    written.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def cnaster_config_switch(tmp_path: Path) -> Iterator[Callable[[float, int], None]]:
    """Reinstall the config mid-test, for a comparison *between* criteria.

    The two fixtures above each pin one solver setting, which is what a test
    of `cnaster` at that setting wants. A test asking how far the shipped
    criterion falls short of the maximum needs both within one body, because
    the shortfall is a difference and neither run alone is the answer.
    """
    from cnaster.config import get_global_config, set_global_config

    previous = get_global_config()

    def switch(em_ftol: float, em_maxiter: int) -> None:
        install_cnaster_config(tmp_path, em_ftol=em_ftol, em_maxiter=em_maxiter)

    try:
        yield switch
    finally:
        set_global_config(previous)
