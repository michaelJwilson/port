"""The badges `README.md` carries, and the one place their values live.

**A badge is a claim in the most-read file in the repository, and a stale one
is worse than none.** `CLAUDE.md` puts a measurement's conditions beside it
and refuses a claim with no number behind it; a shields badge has room for
neither, so this module is where the number and its conditions are kept and
the badge is what is derived.

Run as `python -m tests.badges` to rewrite `.badges/*.json` from
`.badges/measurements.json`. `tests/test_badges_agree.py` is what refuses a
drift between the two, the same way
`tests/test_planning_documents_agree.py` refuses a milestone edited in one
document alone.

## Why the values are committed rather than computed in the badge

shields.io renders whatever JSON it is pointed at. Pointing it at a live
service would make the README's numbers depend on a run nobody can see; a
committed file makes them depend on a commit, and a commit is reviewable.
The cost is that they can go stale, and the test is what pays it.

## What updates each one

| badge | measured by | tier |
| --- | --- | --- |
| judged, oracle, reach | the three coverage guards | per pull request |
| runtime, memory | a whole `run_cnaster`, both arms | `release` |

The two ratio badges carry the instance they were read at, because
`CLAUDE.md` is explicit that a ratio read at a gate size decides nothing.
A badge saying "1.15x" with no size on it is exactly the claim that rule
forbids.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
BADGES = ROOT / ".badges"
MEASUREMENTS = BADGES / "measurements.json"

__all__ = [
    "BADGES",
    "MEASUREMENTS",
    "Badge",
    "badges",
    "load",
    "write",
]


@dataclass(frozen=True)
class Badge:
    """One shields.io endpoint badge.

    `label` is the left half and `message` the right. `color` is chosen from
    the value rather than fixed, so a floor that stops being cleared shows up
    in the README before anyone reads the CI log.
    """

    name: str
    label: str
    message: str
    color: str

    def payload(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "label": self.label,
            "message": self.message,
            "color": self.color,
        }


def _coverage_colour(value: float, floor: float) -> str:
    """Green clear of the floor, amber within a point of it, red below.

    The amber band is deliberate: a guard sitting a tenth of a point above
    its floor is one test away from red and the badge should say so.
    """
    if value < floor:
        return "red"

    return "green" if value >= floor + 1.0 else "orange"


def _ratio_colour(value: float) -> str:
    """`CLAUDE.md`'s bar, rendered.

    A speedup claim is 2x at a stress size; below that a patch is a
    simplification and the badge must not read as a win it did not earn.
    Under 1.0 it is a regression and reads red.
    """
    if value < 1.0:
        return "red"

    return "brightgreen" if value >= 2.0 else "orange"


def _coverage_badge(key: str, guard: dict[str, Any]) -> Badge:
    """One guard, or the fact that it has not been measured.

    A guard with no `percent` renders "not measured" rather than a zero or a
    last-known figure. Guard 3 is the live case: #159 proposes it and CI does
    not run it, so the README should say that instead of carrying a number
    from a commit nobody can name.
    """
    if guard.get("percent") is None:
        return Badge(f"coverage-{key}", guard["label"], "not measured", "lightgrey")

    return Badge(
        name=f"coverage-{key}",
        label=guard["label"],
        message=f"{guard['percent']:.2f}%",
        color=_coverage_colour(guard["percent"], guard["floor"])
        if guard["floor"] is not None
        else "blue",
    )


def load() -> dict[str, Any]:
    """The recorded measurements, with their conditions."""
    return json.loads(MEASUREMENTS.read_text())  # type: ignore[no-any-return]


def badges(measurements: dict[str, Any] | None = None) -> tuple[Badge, ...]:
    """Every badge `README.md` carries, derived from the measurements."""
    recorded = load() if measurements is None else measurements

    guards = recorded["coverage"]
    run = recorded["whole_run"]

    rendered = [_coverage_badge(key, guard) for key, guard in guards.items()]

    if run.get("ratio") is None:
        # NB a comparison that did not produce a ratio is not a zero, and a
        #    badge that rendered it as one would be the worst kind of stale.
        #    `note` says what happened instead.
        rendered.extend(
            [
                Badge(f"run-{axis}", axis, run["note"], "lightgrey")
                for axis in ("runtime", "memory")
            ]
        )
    else:
        rendered.extend(
            [
                Badge(
                    name=f"run-{axis}",
                    label=f"{axis} @ {run['instance']}",
                    message=f"{run['ratio'][axis]:.2f}x",
                    color=_ratio_colour(run["ratio"][axis]),
                )
                for axis in ("runtime", "memory")
            ]
        )

    return tuple(rendered)


def write() -> tuple[Path, ...]:
    """Rewrite every badge file, and return what was written."""
    BADGES.mkdir(parents=True, exist_ok=True)

    written = []

    for badge in badges():
        path = BADGES / f"{badge.name}.json"
        path.write_text(json.dumps(badge.payload(), indent=1) + "\n")
        written.append(path)

    return tuple(written)


def main() -> None:
    for path in write():
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
