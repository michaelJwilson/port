"""Top-level package. Re-exports only port's own utilities, not submodule
contents -- import submodules explicitly (e.g. `from port.x import ...`)."""

from .oxiport import double

__all__ = ["double"]
