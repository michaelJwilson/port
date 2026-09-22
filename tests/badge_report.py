"""What this branch moved, rendered where a reviewer will see it (#271).

Every badge URL in `README.md` is pinned to `/main/.badges/`, so a pull
request's README renders `main`'s figures rather than its own. The figures
the branch actually changed live in `.badges/measurements.json`, inside
notes that run to several hundred words, and in three derived files that
repeat them. A reviewer who wants to know what moved diffs two floats by eye.

This renders that comparison instead: base against head, one row per guard,
the delta as the claim. It is what `check_badges` cannot be -- that module
**refuses** a figure that moved and was never written down, and must keep
refusing. This one reports what *was* written down, and gates nothing.

## Why base-against-head rather than the head figures

A report that restates the head figures is the JSON diff with borders. The
statement a reviewer needs is what this branch changed, which needs the base
to compare against -- so the base file is an input, and the report says so
when it is missing rather than inventing a zero.

Run as `python -m tests.badge_report --base <path>`; CI writes the result to
the job summary and to one upserted pull request comment.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tests.badges import MEASUREMENTS, UNMEASURED

__all__ = ["Row", "render", "rows"]

NO_BASE = (
    "No base measurements to compare against, so this branch's figures are "
    "reported alone."
)
"""What the report says when the base file could not be read.

A first commit, a renamed file or a shallow checkout all reach this. Stated
rather than rendered as a move from zero, which would be a claim about a
measurement nobody took -- the same reason `UNMEASURED` is `/`.
"""


@dataclass(frozen=True)
class Row:
    """One guard, base against head."""

    guard: str
    label: str
    base: float | None
    head: float | None
    floor: float | None
    base_denominator: str | None
    head_denominator: str | None

    @property
    def moved(self) -> bool:
        """Did anything a reader would act on change?"""
        return self.base != self.head or self.base_denominator != self.head_denominator

    @property
    def delta(self) -> float | None:
        """Head minus base, or `None` where either is unmeasured."""
        if self.base is None or self.head is None:
            return None

        return self.head - self.base

    @property
    def clears_floor(self) -> bool | None:
        """Is the head figure above its own floor? `None` where either is absent."""
        if self.head is None or self.floor is None:
            return None

        return self.head >= self.floor


def _percent(value: float | None) -> str:
    return UNMEASURED if value is None else f"{value:.2f}%"


def _delta(row: Row) -> str:
    """The delta, signed, or why there is not one."""
    if row.delta is None:
        return UNMEASURED
    if row.delta == 0.0:
        return "--"

    return f"{row.delta:+.2f}"


def _floor(row: Row) -> str:
    if row.floor is None:
        return "none"

    return f"{row.floor:.2f}%" + ("" if row.clears_floor else " **not cleared**")


def rows(base: dict[str, Any] | None, head: dict[str, Any]) -> tuple[Row, ...]:
    """One `Row` per guard in `head`, matched by key against `base`.

    Driven by `head` rather than by the union, so a guard deleted on this
    branch does not render as a move to nothing -- the deletion is the diff's
    to show, and a table row claiming a figure fell to `/` would be wrong.
    """
    baseline = (base or {}).get("coverage", {})
    rendered = []

    for guard, recorded in head.get("coverage", {}).items():
        was = baseline.get(guard, {})

        rendered.append(
            Row(
                guard=guard,
                label=recorded.get("label", guard),
                base=was.get("percent"),
                head=recorded.get("percent"),
                floor=recorded.get("floor"),
                base_denominator=was.get("denominator"),
                head_denominator=recorded.get("denominator"),
            )
        )

    return tuple(rendered)


def render(base: dict[str, Any] | None, head: dict[str, Any]) -> str:
    """The markdown a reviewer reads, as a whole document."""
    measured = rows(base, head)
    moved = [row for row in measured if row.moved]

    lines = ["### Coverage guards, this branch against its base", ""]

    if base is None:
        lines += [NO_BASE, ""]

    if not moved and base is not None:
        lines += ["**No guard moved.** The recorded figures are the base's.", ""]

    lines += [
        "| guard | base | head | delta | floor |",
        "| --- | ---: | ---: | ---: | --- |",
    ]

    for row in measured:
        mark = "**" if row.moved and base is not None else ""
        lines.append(
            f"| {mark}{row.label}{mark} | {_percent(row.base)} | "
            f"{_percent(row.head)} | {_delta(row)} | {_floor(row)} |"
        )

    denominators = [
        row for row in measured if row.base_denominator != row.head_denominator
    ]

    if denominators and base is not None:
        lines += [
            "",
            "**The denominator moved**, so the percentages are not "
            "measured against the same set:",
        ]
        lines += [
            f"- `{row.label}`: {row.base_denominator or UNMEASURED} "
            f"-> {row.head_denominator or UNMEASURED}"
            for row in denominators
        ]

    lines += [
        "",
        "Rendered from `.badges/measurements.json`, which carries the "
        "selection and denominator behind each figure. The README's badges "
        "are pinned to `main`, so they do not show this branch (#271). This "
        "report gates nothing -- `tests/check_badges.py` is what refuses a "
        "figure that moved and was not written down.",
    ]

    return "\n".join(lines)


def _read(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())  # type: ignore[no-any-return]
    except (OSError, json.JSONDecodeError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        help=(
            "the base revision's measurements.json. Missing or unreadable is "
            "not an error: the report says so and renders the head alone."
        ),
    )
    parser.add_argument(
        "--head",
        type=Path,
        default=MEASUREMENTS,
        help="this branch's measurements.json (default: the repository's)",
    )

    arguments = parser.parse_args(argv)
    head = _read(arguments.head)

    if head is None:
        print(f"cannot read {arguments.head}", file=sys.stderr)
        return 1

    print(render(_read(arguments.base) if arguments.base else None, head))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
