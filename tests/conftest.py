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


@pytest.fixture
def cnaster_config() -> Iterator[None]:
    """Install a minimal `cnaster` global config, and put back what was there."""
    from cnaster.config import YAMLConfig, get_global_config, set_global_config

    previous = get_global_config()
    set_global_config(
        YAMLConfig({"hmm": {"compression_decimals": COMPRESSION_DECIMALS}})
    )
    try:
        yield
    finally:
        set_global_config(previous)
