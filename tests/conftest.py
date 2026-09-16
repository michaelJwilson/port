"""Shared fixtures.

`cnaster` reads configuration from a module-level global rather than from
its arguments, so a test touching a path that consults it has to set one.
The fixture below does that and restores what was there, because a global
left behind is a test that passes alone and fails in a suite.
"""

from collections.abc import Iterator

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


@pytest.fixture
def cnaster_config() -> Iterator[None]:
    """Install a minimal `cnaster` global config, and put back what was there."""
    from cnaster.config import YAMLConfig, get_global_config, set_global_config

    previous = get_global_config()
    set_global_config(
        YAMLConfig(
            {
                "phasing": {"min_prob": MIN_PHASE_SWITCH_PROB},
                # NB `hmm_utils` reads the solver name and then the
                #    `em_`-prefixed option for each keyword that solver takes.
                "hmm": {
                    "compression_decimals": COMPRESSION_DECIMALS,
                    "solver": "L-BFGS-B",
                    "em_maxiter": 100,
                    "em_ftol": 1e-6,
                    "em_disp": 0,
                    "em_xrtol": 1e-5,
                    "em_xtol": 1e-5,
                },
            }
        )
    )
    try:
        yield
    finally:
        set_global_config(previous)
