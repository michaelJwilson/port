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
| runtime, memory, instance | a whole `run_cnaster`, both arms | `release` |
| patched | `python -m tests.patched_share`, one unpatched run (#302) | by hand |
| port, sal | `python -m tests.recovery_audit`, default and `--sal` (#313) | by hand |

The `instance` badge carries the size the two ratios were read at, because
`CLAUDE.md` is explicit that a ratio read at a gate size decides nothing.
A badge saying "1.15x" with no size beside it is exactly the claim that
rule forbids, so the two are rendered together or not at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

UNMEASURED = "/"
"""What a badge reads before its measurement exists.

Not "0", which is a claim, and not a last-known figure from a commit nobody
can name, which is the staleness this module exists to prevent. `/` is the
one rendering that asserts nothing.
"""

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


RATIOS = (("speed", "runtime"), ("mem", "memory"))
"""The badge name and the axis of `measurements.json` it reads."""


def _ratio_badge(name: str, axis: str, run: dict[str, Any]) -> Badge:
    """One whole-run ratio, or the fact that it has not been measured.

    The label is the bare axis. `CLAUDE.md` is explicit that a ratio read at
    a gate size decides nothing, so one must never be shown without the size
    beside it -- but the `instance` badge carries that now, and in more
    detail than a tier name could. Repeating `@ stress` here said less than
    `stress: 4000x1980x5` does one badge along, and cost the width twice.

    `test_a_ratio_is_never_rendered_without_its_instance` is what keeps the
    pair honest, so dropping the tier from the label did not drop the rule.
    """
    if run.get("ratio") is None:
        return Badge(f"run-{name}", name, UNMEASURED, "lightgrey")

    value = run["ratio"][axis]

    return Badge(f"run-{name}", name, f"{value:.2f}X", _ratio_colour(value))


def _instance_badge(run: dict[str, Any]) -> Badge:
    """The tier the two ratio badges were read at.

    `speed` and `mem` carry a ratio and no context, and a ratio read at a
    gate size decides nothing -- `CLAUDE.md`'s Measurement rule establishes
    a speedup at a stress size alone. So this says which tier they came
    from, and it is what makes the other two readable rather than
    decorative.

    **The tier alone, not the shape.** A tier is not a shape and two
    instances both called stress can differ by more than the patch being
    measured does, so the size stays in `.badges/measurements.json` and in
    the README's table, where there is room to say what it means. A badge
    reading `stress: 4000x1980x5` spent its width on digits nobody can
    interpret in place.

    It asserts nothing, so it is blue rather than coloured by a threshold.
    """
    if not run.get("instance"):
        return Badge("instance", "instance", UNMEASURED, "lightgrey")

    return Badge("instance", "instance", str(run["tier"]), "blue")


def _coverage_badge(key: str, guard: dict[str, Any]) -> Badge:
    """One guard, or the fact that it has not been measured.

    A guard with no `percent` renders "not measured" rather than a zero or a
    last-known figure. Guard 3 is the live case: #159 proposes it and CI does
    not run it, so the README should say that instead of carrying a number
    from a commit nobody can name.
    """
    if guard.get("percent") is None:
        return Badge(f"coverage-{key}", guard["label"], UNMEASURED, "lightgrey")

    return Badge(
        name=f"coverage-{key}",
        label=guard["label"],
        message=f"{guard['percent']:.2f}%",
        color=_coverage_colour(guard["percent"], guard["floor"])
        if guard["floor"] is not None
        else "blue",
    )


def _patched_badge(record: dict[str, Any] | None) -> Badge:
    """The share of executed `cnaster` lines a default run has replaced.

    Blue: it measures how much of the subject `port` has taken over, which
    is neither good nor bad on its own, so no threshold colours it.
    """
    if not record or record.get("percent") is None:
        return Badge("patched", "patched", UNMEASURED, "lightgrey")

    return Badge("patched", record["label"], f"{record['percent']:.1f}%", "blue")


def _recovery_badge(arm: str, record: dict[str, Any] | None) -> Badge:
    """Clone and copy-state recovery against the planted truth, one arm.

    Two adjusted Rand indices against the fixture that generated the data,
    both on the integer decode: the fitted clone labels over spots after
    merging clones of one decoded `(A, B)` profile (#344), and each matched
    clone-bin's decoded `(A, B)` against the state the fixture painted there.
    The continuous indices are recorded beside them, not shown. Blue: the instance and configuration
    they were read at are in `measurements.json`, and a badge has no room
    for them.
    """
    name = f"recovery-{arm}"
    values = (record or {}).get("arms", {}).get(arm)

    if not values:
        return Badge(name, arm, UNMEASURED, "lightgrey")

    return Badge(
        name,
        arm,
        f"ARI clones {values['ari_integer']:.3f} / copies {values['copy_ari']:.3f}",
        "blue",
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

    rendered.extend(_ratio_badge(name, axis, run) for name, axis in RATIOS)
    rendered.append(_instance_badge(run))
    rendered.append(_patched_badge(recorded.get("patched")))
    rendered.extend(
        _recovery_badge(arm, recorded.get("recovery")) for arm in ("port", "sal")
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
