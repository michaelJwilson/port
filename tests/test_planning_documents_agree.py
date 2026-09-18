"""The three planning documents name the same work (issue #74).

`ROADMAP.md` says what the stages are, `TICKETS.md` says what is filed under
each, and `STATUS.md` says what each has established. Three documents carrying
one list of milestones drift the moment one is edited alone, and the drift is
invisible: each file reads correctly on its own.

**The referee is the documents against each other**, which is an invariant
rather than a second implementation, so these carry `analytic`. What would have
to be wrong for them to fail: a milestone renamed in one file, added to one
file, reordered in one file, or a ticket cited in a form the parser cannot
follow.

`CLAUDE.md`'s Documentation Sync is the rule this enforces. Without it the rule
is satisfied by default, which is #48's finding and #72's.
"""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.analytic
"""The referee is the documents against each other, which is an invariant."""

ROOT = Path(__file__).resolve().parents[1]

DOCUMENTS = ("ROADMAP.md", "TICKETS.md", "STATUS.md")
"""The three that carry milestone headings. `README.md` indexes, and indexes
are allowed to be shorter than what they point at."""

MILESTONE = re.compile(r"^## (Milestone \d+\.\d+ — .+)$", re.MULTILINE)
"""`## Milestone 1.1 — Name`, with an em dash.

The heading is the join key, so its shape is pinned rather than guessed at: a
hyphen where an em dash belongs would silently give three empty lists and a
passing test.
"""

CITATION = re.compile(r"\(#(\d+)\)")
"""`(#123)`, which `TICKETS.md` declares is how a filed issue is cited."""


def _milestones(name: str) -> list[str]:
    return MILESTONE.findall((ROOT / name).read_text())


@pytest.mark.infra
def test_every_document_carries_milestones() -> None:
    """Guards the parser: an empty match is not agreement.

    Pinned because the failure mode of a join on a regular expression is that
    it matches nothing in every file and reports that they agree.
    """
    for name in DOCUMENTS:
        assert len(_milestones(name)) >= 10, f"{name} parsed no milestones"


@pytest.mark.infra
def test_the_three_documents_name_the_same_milestones() -> None:
    """Same headings, same order, in all three."""
    roadmap = _milestones("ROADMAP.md")

    for name in DOCUMENTS[1:]:
        other = _milestones(name)

        missing = [m for m in roadmap if m not in other]
        added = [m for m in other if m not in roadmap]

        assert not missing, f"{name} is missing {missing}"
        assert not added, f"{name} adds {added}, which ROADMAP.md does not name"
        assert other == roadmap, f"{name} orders the milestones differently"


@pytest.mark.infra
def test_the_milestones_are_numbered_in_order() -> None:
    """`1.1` before `1.2` before `2.1`, so a reader can find one by its number."""
    numbers = [
        tuple(int(part) for part in m.split()[1].split("."))
        for m in _milestones("ROADMAP.md")
    ]

    assert numbers == sorted(numbers), f"out of order: {numbers}"
    assert len(set(numbers)) == len(numbers), "a milestone number is used twice"


def _bullets(text: str) -> list[str]:
    """Each `- ` item with its continuation lines folded in."""
    items: list[str] = []
    for line in text.splitlines():
        if line.startswith("- "):
            items.append(line[2:].strip())
        elif line.startswith("  ") and items:
            items[-1] += " " + line.strip()
    return items


@pytest.mark.infra
def test_every_filed_ticket_is_cited_at_the_end_in_parentheses() -> None:
    """A bullet naming a filed issue ends with `(#n)`, and carries one.

    `TICKETS.md` says a parenthesized number is a filed issue and a bullet
    without one is work nobody has filed. That distinction is the file's only
    piece of state, so it has to be machine-readable.

    A bare `#n` **inside** a bullet is not a citation and is not refused: the
    bullets are issue titles verbatim, and several titles reference another
    issue in prose -- "the live caller of #24's M step". Rewriting those to
    satisfy a parser would make this file disagree with the tracker, which is
    the thing it exists to mirror.
    """
    text = (ROOT / "TICKETS.md").read_text()
    body = text[text.index("## Milestone") :]
    bullets = _bullets(body)

    assert len(bullets) >= 40, f"parsed {len(bullets)} bullets, expected the filed set"

    for bullet in bullets:
        found = CITATION.findall(bullet)

        assert len(found) <= 1, f"more than one citation: {bullet!r}"
        if found:
            assert bullet.endswith(f"(#{found[0]})"), (
                f"citation is not last: {bullet!r}"
            )


@pytest.mark.infra
def test_the_documents_agree_with_the_readme_index() -> None:
    """`README.md` links all three, so a reader arrives at them.

    The index is what makes the other three findable; a planning document
    nothing points at is a document nobody opens.
    """
    readme = (ROOT / "README.md").read_text()

    for name in DOCUMENTS:
        assert f"({name})" in readme, f"README.md does not link {name}"
