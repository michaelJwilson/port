"""#619: `[cna.length] law = "lognormal"` against `scipy.stats.lognorm` and the derived sigma."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from port.sim.draw import _event_length, extended, from_document, lognormal_median
from port.sim.fixtures import SIM_ROOT
from port.sim.laws import lognormal_sigma
from scipy import stats

MANIFESTS = SIM_ROOT / "manifests"
DRAWS = 20_000
SHARE, BELOW = 0.10, 0.5
"""The criterion every lognormal manifest states: 10% below half the median."""


def _lognormal_manifests() -> list[Path]:
    """Return every manifest whose own `[cna.length]` states `law = "lognormal"`."""
    return [
        p
        for p in sorted(MANIFESTS.rglob("*.toml"))
        if tomllib.loads(p.read_text()).get("cna", {}).get("length", {}).get("law")
        == "lognormal"
    ]


@pytest.mark.analytic
@pytest.mark.parametrize(("share", "below"), [(0.10, 0.5), (0.05, 0.25), (0.3, 0.9)])
def test_lognormal_sigma_puts_share_below_the_fraction_of_the_median(
    share: float, below: float
) -> None:
    """`scipy.stats.lognorm.cdf(below m)` at the returned sigma is `share`, to 1e-12."""
    sigma = lognormal_sigma(share, below)
    median = 1e7

    found = stats.lognorm.cdf(below * median, s=sigma, scale=median)

    assert sigma > 0
    assert abs(found - share) < 1e-12


@pytest.mark.analytic
@pytest.mark.parametrize(
    ("manifest", "key"),
    [("dev_tree.toml", "mean"), ("dev_tree_1s_hard.toml", "median")],
)
def test_lognormal_lengths_keep_the_stated_quantity_and_the_short_tail(
    manifest: str, key: str
) -> None:
    """20,000 draws keep the stated mean or median within 1%, the tail at 0.10 +- 0.01, KS at 1%."""
    drawn = from_document(extended(MANIFESTS / manifest), MANIFESTS)
    law: dict[str, Any] = drawn.cna["length"]
    rng = np.random.default_rng(619)
    lengths = np.array([_event_length(drawn, rng) for _ in range(DRAWS)], float)
    median = lognormal_median(law)

    stated = float(law[key])
    measured = lengths.mean() if key == "mean" else float(np.median(lengths))
    log_scores = (np.log(lengths) - np.log(median)) / float(law["sigma"])

    assert key in law
    assert abs(measured / stated - 1.0) < 0.01
    assert abs((lengths < median / 2).mean() - SHARE) < 0.01
    assert stats.kstest(log_scores, "norm").pvalue > 0.01


@pytest.mark.snapshot
def test_every_lognormal_manifest_states_the_derived_sigma() -> None:
    """Every lognormal manifest states `round(lognormal_sigma(0.10, 0.5), 3)`."""
    paths = _lognormal_manifests()
    expected = round(lognormal_sigma(SHARE, BELOW), 3)

    assert [p.name for p in paths] == [
        "dev_tree.toml",
        "dev_tree_1s.toml",
        "dev_tree_1s_dense.toml",
        "dev_tree_1s_easy.toml",
        "dev_tree_1s_hard.toml",
        "study10.toml",
        "study15.toml",
    ]
    for path in paths:
        length = tomllib.loads(path.read_text())["cna"]["length"]
        assert length["sigma"] == expected, path


@pytest.mark.infra
@pytest.mark.parametrize("keys", [("mean", "median"), ()])
def test_a_lognormal_law_states_exactly_one_of_mean_or_median(
    keys: tuple[str, ...],
) -> None:
    """Both keys, or neither, is refused, naming the two."""
    document = extended(MANIFESTS / "dev_tree.toml")
    document["cna"]["length"] = {"law": "lognormal", "sigma": 0.541, "minimum": 1e6}
    document["cna"]["length"] |= {k: 1e7 for k in keys}

    with pytest.raises(ValueError, match=r"exactly one of \('mean', 'median'\)"):
        from_document(document, MANIFESTS)


@pytest.mark.infra
def test_a_stated_law_replaces_the_base_length_table_whole() -> None:
    """A stated law replaces the base length table whole."""
    length = extended(MANIFESTS / "dev_tree_1s_hard.toml")["cna"]["length"]

    assert set(length) == {"law", "median", "sigma", "minimum"}
