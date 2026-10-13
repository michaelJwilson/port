"""The compiled `port.oxiport` extension against the arithmetic it claims; needs `uv sync`."""

import pytest
from port import double
from port.oxiport import double as double_from_extension


@pytest.mark.oracle
@pytest.mark.critical
def test_double() -> None:
    """The extension doubles, against the arithmetic it claims to do."""
    assert double_from_extension(21) == 42
    assert double_from_extension(0) == 0
    assert double_from_extension(-3) == -6


@pytest.mark.smoke
def test_top_level_reexport_is_the_extension_function() -> None:
    """`port.__init__` re-exports the binding rather than shadowing it."""
    assert double is double_from_extension
