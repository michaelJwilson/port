# TICKETS

The work that stands between `STATUS.md` and `ROADMAP.md`, as titles. Each line
is one filed issue — the outcome, the non-goals and how it will be validated
are written there, not here.

Ordering within a milestone is by dependency, not priority. A parenthesized
number **at the end of a bullet** is an issue already filed; a bullet without
one is work this file names and nobody has filed yet. A bare `#n` inside a
bullet belongs to the title and is not a citation. `tests/test_planning_documents_agree.py` keeps the
milestone headings here, in `ROADMAP.md` and in `STATUS.md` naming the same
work, and keeps the parenthesis the only way a ticket is cited.

51 issues are open. #14 is the umbrella for Stage 1 and holds no work of its
own.

## Milestone 1.1 — Fixtures and planted truth

- Decided: the overdispersion latent is shared across a clone's spots, and the
  fixture plants it here (#66)
- Segment lengths are ragged, and the fixture has to declare which
  segmentation it plants (#67)

## Milestone 1.2 — The emission and the phased HMM

- Audit the emission calculation and the phased HMM: three implementations,
  none pinned against another (#9)
- The phased transfer matrix: hoisting and factorization both measure below the
  bar (#16)
- `assign_centiMorgans` sorts its caller's list in place, and
  `get_sitewise_transmat` depends on it (#20)
- Stage 1: pin that `CountEncoder`'s compression does not move the score (#39)

## Milestone 1.3 — The M step and the EM loop

- Stage 1: pin Baum-Welch monotonicity, where #30 surfaces (#36)
- Stage 1: referee `_run_optimization_pipeline` by the separability of its
  objective (#37)
- Stage 1: validate `normal_baf_bin_filter`, the live caller of #24's M step
  (#38)

## Milestone 1.4 — The spatial graph and the label solve

- Audit the clone label optimization against `snakes_and_ladders`': solvers,
  bounds, and where the time goes (#8)
- Audit the spatial graph and smoothing: the input #8 compares solvers on (#12)
- Stage 2: the label solvers, against exhaustive Potts energy (#40)

## Milestone 1.5 — Scale, uncertainty and integer copy

- Per-clone library normalization of `log_mu`: pin the absolute scale, and the
  `logmu_shifts` that would set it (#5)
- Error bars on HMM parameters, and an integer copy fit that respects them (#6)
- Referee the integer copy fit against exhaustive enumeration (#25)
- Model selection by Pareto front is implemented on neither side, and its cost
  argument rests on a wrong number (#54)

## Milestone 1.6 — End to end

- Validate `cnaster`'s `run_core_inference` against `snakes_and_ladders`'
  spatio-sequential simulator (#4)
- Separate the two uses of `snakes_and_ladders`: a fixture that plants truth,
  and an oracle that referees an implementation (#69)

## Milestone 2.1 — Coverage over the whole of cnaster

- A coverage floor over `cnaster`, and the per-function tests the HMM fixtures
  reach (#17)
- Tier one: raise coverage without importing anything new (#18)
- Tier two: bring further `cnaster` modules under test (#19)
- Cover `cnaster`'s result containers and file loaders (#26)
- Cover the plotting modules by running them on validated fixtures (#34)

## Milestone 2.2 — The input path and the entry points

- The coverage gate cannot see `cnaster`'s console entry points (#35)
- Stage 3: the input path, where the only referee is an invariant (#41)
- Run `run_cnaster` end to end from temporary fixture files (#68)

## Milestone 3.1 — The three references

- Audit the paper against itself: notation, unreferenced sections, and a
  runtime it admits is wrong (#27)
- Audit the paper against `cnaster`: does the code implement what the methods
  state? (#28)
- Audit the paper against `snakes_and_ladders`: which of its methods already
  have a backend (#29)
- Audit: what `snakes_and_ladders` needs before it can referee the whole of
  `cnaster` (#32)
- The external field the HMM hands ICM is not the one the paper defines (#58)

## Milestone 3.2 — Runtime and memory audits

- Audit the runtime and memory of `load_input_data`: densification, filter
  order, and sparse layout (#7)
- Audit initialization: two initializers, no comparison, and the seed decides
  the optimum (#10)
- Audit the block aggregation: the consumer of the dense matrices in #7 (#11)
- Triage: outer-loop warm starts, per-iteration recomputation, and the dead
  code around them (#13)

## Milestone 4.1 — The HMM/spatial boundary

- Audit the HMM/spatial boundary: layout, materialization and a 16-argument
  seam (#59)

## Milestone 4.2 — The emission and the M step

- Close the 12.8x M-step gap: the design matrix is `K` times the problem (#47)

## Milestone 4.3 — Defects found in the subject

- The emission M step stops short of its maximum and reports that it converged
  (#30)
- `icm_sweep_deque` is not reproducible, and mutates its caller's array (#45)
- `Weighted_BetaBinom_mix.fit` writes to the working directory, and sends
  `scipy` an option it rejects (#46)
- `cnaster.wolff` raises `ImportError` on import, is shipped in the wheel, and
  is in the coverage denominator

## Milestone 5.1 — Upstream reports

- Upstream's covariate stops at the emission family; four modules above it do
  not pass one (#57)
- A covariate per channel, and a varying kernel that reaches a fit (#65)
- What the oracle needs once the latent is clone-shared, and
  `baum_welch_family` cannot take a channel axis (#70)
- `cnaster` is a namespace package: recommend `__init__.py`, worth +919
  statements to the gate (#73)
- `cnaster` packaging: no licence, four dependencies with no importer, two
  imports with no declaration (#75)

## Milestone 5.2 — The documents and the gate

- `changelog.d` does not exist, so Documentation Sync has been skipped silently
  (#48)
- CI runs on `ubuntu-latest` only; decide whether macOS is in scope (#49)
- Tickets and PRs here exceed Writing Style's line limits; comply or record why
  not (#50)
- Build `cnaster`'s API docs here, in a subtree that ports back unchanged (#71)
- `CLAUDE.md` names `sphinx-build -W` and there is nothing to build (#72)
- Add `ROADMAP.md`, `STATUS.md` and `TICKETS.md`, and make `README.md` their
  index (#74)
