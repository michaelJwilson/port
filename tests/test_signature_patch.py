"""`as_upstream` binds and refuses exactly as `cnaster.hmrf.reindex_clones`'s signature does (#517)."""

from __future__ import annotations

import inspect
from typing import Any

import pytest


@pytest.mark.patch
def test_the_wrapper_binds_and_refuses_as_cnaster_does() -> None:
    from cnaster.hmrf import reindex_clones
    from port.patch._signature import as_upstream

    seen: list[dict[str, Any]] = []

    @as_upstream(reindex_clones)
    def wrapped(arguments: dict[str, Any]) -> None:
        seen.append(arguments)

    assert inspect.signature(wrapped) == inspect.signature(reindex_clones)

    wrapped("res", None, single_tumor_prop="tp")
    assert seen == [
        {"res_combine": "res", "posterior": None, "single_tumor_prop": "tp"}
    ]

    for args, kwargs in (((), {}), (("a", "b", "c", "d"), {}), (("a",), {"x": 1})):
        with pytest.raises(TypeError):
            inspect.signature(reindex_clones).bind(*args, **kwargs)
        with pytest.raises(TypeError):
            wrapped(*args, **kwargs)


@pytest.mark.patch
def test_an_option_is_keyword_only_and_arrives_with_its_default() -> None:
    from cnaster.hmrf import reindex_clones
    from port.patch._signature import as_upstream

    seen: list[tuple[dict[str, Any], dict[str, Any]]] = []

    @as_upstream(reindex_clones, mode="cnaster")
    def wrapped(arguments: dict[str, Any], options: dict[str, Any]) -> None:
        seen.append((arguments, options))

    parameter = inspect.signature(wrapped).parameters["mode"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY

    wrapped("res")
    wrapped("res", mode="port")
    assert seen == [
        ({"res_combine": "res"}, {"mode": "cnaster"}),
        ({"res_combine": "res"}, {"mode": "port"}),
    ]

    with pytest.raises(TypeError):
        wrapped("res", None, None, "port")
