"""Shared fixtures.

`cnaster` reads configuration from a module-level global rather than from
its arguments, so a test touching a path that consults it has to set one.
The fixture below does that and restores what was there, because a global
left behind is a test that passes alone and fails in a suite.
"""

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

NUMBA_SEED = 314159
"""What `numba`'s own generator is seeded with for the whole session (#264).

Any value works; what matters is that there is one. The figure the coverage
guards record is a function of it, so changing it moves them.
"""

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
                    # NB `gmm_init` clips the observed allele share before
                    #    fitting, and reads the bounds from here rather than
                    #    taking them as arguments (`hmm_initialize.py:362`).
                    #    Wide enough to clip nothing a fixture plants, so the
                    #    initializer's start is the data's and not the clip's.
                    "gmm_min_binom_prob": 0.01,
                    "gmm_max_binom_prob": 0.99,
                    "gmm_maxiter": 100,
                },
                # NB `run_core_inference` reads the outer loop's own settings
                #    from here: `inertia` decides whether a uniform prior over
                #    clones is added to the field, `fixed_assignment` whether
                #    the label solve runs at all, and `ari_tolerance` when the
                #    loop stops. All three are the shipped defaults.
                "hmrf": {
                    "inertia": False,
                    "fixed_assignment": False,
                    "ari_tolerance": 0.99,
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


_COLLECTED: list[pytest.Item] = []
"""Every test collected this session, before any `-m` deselected one.

`session.items` is the *selected* slice, so a guard reading it under
`-m critical` would see only the tier it is supposed to be auditing and pass
for that reason.
"""


@pytest.hookimpl(tryfirst=True)
@pytest.fixture(scope="session", autouse=True)
def numba_seeded() -> None:
    """Seed `numba`'s generator, so the coverage guards are reproducible.

    **#264.** `cnaster/icm.py:888` takes an epsilon-greedy branch under
    `np.random.rand()` inside an `@njit`. Numba keeps its own generator,
    seeded from entropy on first use and **not** by `np.random.seed` from
    Python -- so whether that branch fires is a coin flip per process.

    Measured before this fixture: five runs of the `end2end` selection at a
    fixed thread count read 40.27 four times and 40.21 once, a 5-statement
    swing in `icm.py` alone. `tests/check_badges.py` compares a recorded
    figure against the measured one, so it refused two consecutive CI runs of
    the same code, in opposite directions.

    Seeded here rather than in the tests that reach the solver, because any
    test that ever reaches it moves the figure for the whole session.
    `cnaster`'s own entry point does the same thing at
    `scripts/run_cnaster.py:66`; this is that call, made where the suite can
    rely on it.
    """
    from cnaster.scripts.run_cnaster import set_numba_seed

    set_numba_seed(NUMBA_SEED)


@pytest.fixture(scope="session", autouse=True)
def cnaster_runs() -> None:
    """Install the compatibility rows for the whole session.

    **#259.** `cnaster`'s `port` branch passes keywords its own signatures no
    longer take, so a fit raises `TypeError` before it computes anything and
    the emission's shift raises a numba `TypingError` before that. Without
    `port.pipeline.COMPAT_SWAPS` there is nothing for a referee to referee:
    the tests that call `cnaster` directly, deliberately unpatched, are
    testing a dependency that does not run.

    Session-scoped and never restored, because the rows change no number --
    each is a keyword translation or the suppression of a result upstream
    discards -- so leaving them installed cannot move a comparison. `install`
    rather than `patched` for the same reason: there is no state here worth
    unwinding, and a fixture that restored them would leave the last test in
    a session unable to run.

    `install(())` asks for no replacements: `_with_compat` adds the
    compatibility rows to whatever is requested, so an empty request is
    exactly these and nothing else.
    """
    from port.pipeline import install

    install(())


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Record the collection before pytest's own mark deselection runs."""
    _COLLECTED[:] = items


@pytest.fixture
def collected_items() -> list[pytest.Item]:
    """The whole suite's items, or a skip when the run is a narrowed one.

    Selecting a file or a `-k` expression collects less than the tree, and a
    guard over that subset would say something weaker than it claims while
    reporting green. The claim is about the suite, so it is made only when
    the suite is what was collected.
    """
    items = list(_COLLECTED)
    collected_modules = {item.nodeid.split("::")[0] for item in items}
    on_disk = {
        f"tests/{path.name}" for path in (Path(__file__).parent).glob("test_*.py")
    }

    missing = on_disk - collected_modules
    if missing:
        pytest.skip(f"narrowed collection; {len(missing)} test module(s) absent")

    return items


@pytest.fixture(autouse=True, scope="session")
def _keep_the_perf_log_out_of_the_checkout(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[None]:
    """Run the whole suite from a scratch directory.

    `hmm_emission.flush_perf` opens the **literal relative path**
    `"cnaster.perf"`, reading `config.paths.perf_path` only to count rows and
    decide on a header, so every fit drops a timing log into whatever
    directory pytest was started from. `cnaster_perf_sink` has handled that
    for the tests that ask for it since the M-step work; the pipeline tests do
    not ask, because they reach `fit` several stages down and have no reason
    to know.

    The result was a tracked `cnaster.perf` modified by every run. Making the
    protection automatic is the fix: nothing in this suite resolves a path
    relative to the working directory -- `tmp_path` is absolute and
    `tests/test_coverage_scope.py` anchors on `__file__` -- so the directory
    is free to move.
    """
    import os

    previous = Path.cwd()
    os.chdir(tmp_path_factory.mktemp("cwd"))
    try:
        yield
    finally:
        os.chdir(previous)
