"""A wrapper that takes exactly what the `cnaster` function it wraps takes (#517).

The body receives the arguments by name as given (defaults not filled in), and
`port`'s keyword-only options, every one present, in a second mapping.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any

__all__ = ["as_upstream"]

Body = Callable[..., Any]


def as_upstream(
    upstream: Callable[..., Any], **options: Any
) -> Callable[[Body], Callable[..., Any]]:
    """Decorate `body(arguments)`, or `body(arguments, options)`, into `upstream`'s shape.

    `options` names each `port` option and its default.
    """
    base = inspect.signature(upstream)
    signature = base.replace(
        parameters=[
            *base.parameters.values(),
            *(
                inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=default)
                for name, default in options.items()
            ),
        ]
    )

    def decorate(body: Body) -> Callable[..., Any]:
        @functools.wraps(body)
        def call(*args: Any, **kwargs: Any) -> Any:
            given = dict(signature.bind(*args, **kwargs).arguments)

            if not options:
                return body(given)

            chosen = {
                name: given.pop(name, default) for name, default in options.items()
            }
            return body(given, chosen)

        call.__signature__ = signature  # type: ignore[attr-defined]
        return call

    return decorate
