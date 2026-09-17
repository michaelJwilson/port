# ROADMAP: validating and optimizing a dependency this repository does not own

`port` exists to hold one implementation — `cnaster` — against two references:
the paper that states the method and `snakes_and_ladders`, which implements the
parts of it that are not application specific. This file says what the stages
are and what a change passes through on its way in. `TICKETS.md` says what is
filed. `STATUS.md` says what has been established, with the number that
established it.

# 0. The development loop

Development is agent-assisted. The claim is not that an agent wrote the code
but that the loop validates it, so each stage below is a gate and they are
ordered so a claim is refused before it is written rather than after it is
published.

## 0.1 The ticket

A unit of work, naming the outcome so it can be checked, what it unblocks, and
how it will be validated. A ticket opens with its result — the number, the
decision, or what broke — in `O(1)` lines (`CLAUDE.md`, Writing Style 8), and
is limited to 40 lines.

A finding with no number is not yet a ticket. What the finding costs, how far
off it is, or how often it fires is what turns an observation into work someone
can rank.

## 0.2 The plan

Posted to the thread before code exists, stating how each step will be
validated and what is still open. A plan whose questions are all answered says
so under that heading rather than dropping it.

## 0.3 The pull request

One per ticket, opened as a draft when work starts so the ticket reports as it
goes. `.github/pull_request_template.md` is the layout: the result first, a
benchmark table at the sizes the Measurement rule requires, a tolerance table
naming the referee and the **realized** value, the differences the work states,
Documentation Sync, and the Definition of Done.

## 0.4 Validation

Two CI jobs on `ubuntu-latest`: Python syncs with `--locked` — so a stale
`uv.lock` fails before a merge rather than after — then runs `ruff`,
`ruff format`, `mypy --strict` and `pytest -m "not release"` under the coverage
gate; Rust runs `cargo fmt --check` and `cargo clippy --all-targets -D warnings`.

Every test names its referee in a registered marker: `upstream`, `cnaster`,
`planted`, `analytic`. `benchmark` records a baseline and asserts no ratio.
`release` is over the per-pull-request budget and CI deselects it.

The coverage floor is measured against the whole of `cnaster` rather than the
modules a test imports, so it never falls for bringing a new module under test
and there is no case where lowering it is legitimate.

# 1. Objectives

Establish what `cnaster` computes, against a referee that is not `cnaster`, and
make the parts that are worth making faster faster — at a ratio measured at a
stress size, or not at all.

The constraint that shapes every stage: **`cnaster` and `snakes_and_ladders`
are read only.** `port` can pin them, measure them and report on them. It
cannot land a change in either, so an upstream requirement is a report with
evidence attached, never a branch.

# Stage 1 — The ladder

`run_core_inference` composes an emission, a lattice, an initializer, a spatial
graph, a label solver and an outer loop. A discrepancy at the top is
attributable to none of them, so each layer is refereed alone before the layer
containing it. The first rung that breaks names the layer. #14 is the umbrella.

## Milestone 1.1 — Fixtures and planted truth

Truth is planted with a `snakes_and_ladders` simulator and a seed, never
recovered from the data, and converted to `cnaster`'s arguments by an adapter
that every later rung extends. Two modelling decisions belong here rather than
to the code that consumes them: which overdispersion latent the draw plants,
and which segmentation.

## Milestone 1.2 — The emission and the phased HMM

Three emission implementations and a phased lattice, scored at fixed
parameters so a disagreement is a defect in one of them and can be nothing
else.

## Milestone 1.3 — The M step and the EM loop

Re-estimation both implementations perform by optimization, the monotonicity
the loop is supposed to have, and the solver settings that decide whether
either reaches its maximum.

## Milestone 1.4 — The spatial graph and the label solve

The graph the solvers run on, and the solvers, against exhaustive Potts energy
where enumeration reaches.

## Milestone 1.5 — Scale, uncertainty and integer copy

The per-clone normalization that fixes the absolute scale, intervals on the
fitted parameters, and the integer copy fit that consumes them.

## Milestone 1.6 — End to end

`run_core_inference` against a planted fixture, recovering labels, states and
parameters within stated tolerances, with its cost recorded. The fixture claim
and the oracle claim are separate claims and are not asserted by one test.

# Stage 2 — The denominator

Coverage is measured against the whole of `cnaster`, so the figure says how
much of the subject is validated. It is low by construction and rises only by
validating more.

## Milestone 2.1 — Coverage over the whole of cnaster

The tiers that raise it without inventing fixtures, and the modules whose
referee is a round trip or a refusal rather than a second implementation.

## Milestone 2.2 — The input path and the entry points

The console scripts the wheel installs, the loader beneath them, and the
synthetic dataset that makes either testable. The weakest assurance per
statement in the repository, and the ticket that says so.

# Stage 3 — The audits

An audit names its upstream correspondence: what `snakes_and_ladders` would
have to change for the problem to be expressible there, or a record that it
would not and why.

## Milestone 3.1 — The three references

Where the paper, `cnaster` and upstream describe one quantity, all three
compute it and the agreement is a number. Where they differ, the difference is
stated where the work is; an unstated difference is a defect by default.

## Milestone 3.2 — Runtime and memory audits

`cProfile` first, the algorithmic cut ahead of the mechanical one, and a
dismissal carries the number that dismissed it.

# Stage 4 — Optimization

A simplification lands on its evidence of equivalence. A speedup claim is held
to 2x at a stress size, and a ratio read at a gate size decides nothing.

## Milestone 4.1 — The HMM/spatial boundary

The seam between the HMM and the label solver: its layout, what it
materializes, and its interface.

## Milestone 4.2 — The emission and the M step

The two stages a profile puts the time in.

## Milestone 4.3 — Defects found in the subject

Reproducibility, in-place mutation of a caller's arrays, criteria that report
convergence they did not reach. Reported with the evidence; landed by the
repository that owns them.

# Stage 5 — Working against a repository you do not own

## Milestone 5.1 — Upstream reports

What `snakes_and_ladders` and `cnaster` would each need, stated with the
measurement, and the pin this repository controls as the only lever it has.

## Milestone 5.2 — The documents and the gate

The rules this repository declares and the machinery that makes them true. A
declared requirement with nothing behind it reads as satisfied, which is worse
than an absent one.
