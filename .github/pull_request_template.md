<!--
Adapted from `snakes_and_ladders/.github/pull_request_template.md`. The rows
upstream's map carries and `port` does not -- `DEV.md`, `ROADMAP.md`,
`STATUS.md`, `TICKETS.md`, `docs/tex/` -- are dropped rather than left to be
ticked "None applicable" every time; `CLAUDE.md`'s Repository Map says to add
a document when the content exists, and this list grows with it.

This template mirrors CLAUDE.md's "Definition of Done" and "Documentation
Sync". Fill in every section; delete none of them.

The body carries at most 40 content lines, 60 with extended discussion
(CLAUDE.md, Writing Style 1). Blank lines, these comments and the headings
are not charged. What does not fit belongs where it is refereed -- a
measurement in a `docs/` audit, a sweep in a benchmark file.

Open with the result (Writing Style 8): the number, the decision, or what
broke, before any context.
-->

### Description

<!-- What changed, and why. Link the issue this closes, if any. -->

### Benchmark

<!--
Required whenever a hot function is new or changed (CLAUDE.md, Definition of
Done 2). Report baseline against new, at the sizes the Measurement rule
requires -- a ratio read at a gate size decides nothing, and a speedup is
established at a stress size alone. Say which size each row is.

If this PR touches no hot path, write "N/A -- no hot-path change" and delete
the table rather than leaving it rendered with blank cells.
-->

| Function | Size | Baseline | This PR | Delta |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

<!--
Required whenever this PR pins output against a reference (CLAUDE.md, "Pin to
Independent Sources", "Cross-precision agreement is a tolerance"). Name the
referee -- `cnaster`, `snakes_and_ladders`, the paper, an analytic property,
brute force -- and report the **realized** value, not only the tolerance it
was checked against. Bitwise is what a comparison strives for; a stated
tolerance is the floor it may back off to.
-->

| Test | Referee | Tolerance | Realized |
| --- | --- | --- | --- |
|  |  |  |  |

### Stated differences

<!--
CLAUDE.md, "The paper, `cnaster` and `snakes_and_ladders` agree, to a stated
tolerance": a difference in the problem itself counts once it is stated where
the work is. Name which reference differs, in what regime, and why it is a
choice rather than a defect. An unstated difference is a defect by default.

Also state a `sandbox/` or `deprecated/` excursion here: which tree, and why
the question could not be answered from the installed path.
-->

- [ ] None -- every comparison is in a regime where the references correspond.

### Documentation Sync

- [ ] `README.md`
- [ ] `CLAUDE.md`
- [ ] `docs/` (an audit this change makes inaccurate)
- [ ] `changelog.d/` (see `changelog.d/README.md` for what counts as user-visible)
- [ ] None applicable

### Follow-up / Deferred Work

- [ ] None -- nothing deferred
- **What's deferred:**
- **Why it's deferred rather than done here:**
- **Tracking issue:**

### Definition of Done

- [ ] **Regression test:** asserts scientific validity, not shape or absence of an exception, and pins the expected output.
- [ ] **Benchmark:** new or changed hot functions carry a `pytest-benchmark`, with the numbers above at gate and stress sizes.
- [ ] **Coverage:** the `--cov-fail-under` gate is maintained or raised.
- [ ] **Docs & tooling:** `ruff`, `mypy --strict` and `cargo` pass locally; Documentation Sync is satisfied.
- [ ] **Dependency hygiene:** new dependencies are OSI-licensed, justified, and flagged under 1,000 stars.
