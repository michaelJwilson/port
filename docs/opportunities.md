# Where the next gains are, ranked by measurement

**3,932 statements run in the suite and are judged by nothing. The whole run
is 192 s and 11.35 GB, of which the figure swaps alone are 70 s and 7.7 GB.
Neither number needed a new experiment to find; both came out of measurements
this repository already had and had not compared.**

Measured at `48c7101`, stress instance `4,000 x 1,980 x 5` on an idle host,
guards over `cnaster` + `python/port`.

The tracker holds 107 open tickets and `docs/audit-ticket-register.md` says
its problem is that it is over-filed rather than behind. So this document
**ranks what already exists** and names the few gaps nothing covers. A row
here is not a new ticket; the ticket column says who owns it.

---

## A. Coverage — the cheapest 3,932 statements

`e2e` counts a statement only when a test whose referee is outside `cnaster`
executes it. `all` counts it when anything executes it. The difference is
code that **already runs in the suite and is refereed by nothing**: it needs
an oracle, not new machinery, which is what makes it the cheap end.

| rank | what | unjudged | reached | owner |
| ---: | --- | ---: | ---: | --- |
| 1 | `cnaster/scripts/run_cnaster.py` | 899 | 899 | #153, #223 |
| 2 | plotting: `plotting`, `plot_genomic`, `plot_copy_number_profile`, `plot_validation_stats`, `plot_loh_density` | 1,243 | 1,243 | #34 **vs** #103 |
| 3 | `python/port/patch/*` — port's own replacements | 819 | 819 | **unticketed** |
| 4 | `hmm_initialize` | 142 | 267 | #160 |
| 5 | `hmrf` | 127 | 477 | #140 |
| 6 | `recomb`, `cna_hmrf_result`, `utils` | 173 | 344 | #26, #111 |

### A1. The entry point is the single biggest item, and #223 taxes it

899 statements, every one executed, none judged. It is also the module whose
import decides the denominator (#223), which makes the arithmetic perverse:
an `end2end` test reaching it adds 899 statements to `e2e`'s denominator, so
unless that test judges more than 44.81 per cent of them **the guard falls
for validating the most live code the subject has**.

So #223 is not a tidy-up that can follow this work. It gates it, and doing
A1 first would punish the person who did it.

### A2. Plotting: two open tickets want opposite things

1,243 statements, all reached, none judged. #34 says cover them by running
them on validated fixtures; #103 says take them out of the denominator. Both
are open, and they cannot both be right.

The measurement that decides it is now in hand: plotting is **68 per cent of
this run's peak memory** (11.35 GB to 3.69 GB by changing only figure
rendering) and about a third of its wall time. Code that dominates the
resource envelope is not code to excuse from the denominator, so the
evidence points to #34 and against #103 -- but that is a decision to take
explicitly, not to let the two tickets sit contradicting each other.

### A3. `port`'s own code is judged by nothing — unticketed

819 statements across `patch/omics` (293), `patch/spatial` (151),
`patch/summaries` (150), `patch/figures` (57), `patch/icm_interface` (43),
`patch/clone_assignment` (32), `patch/lattice` (28), `patch/reference` (24),
`patch/hmrf_invariants` (22) and `patch/emission` (19).

They are covered by `patch`-marked tests, and `CLAUDE.md` is explicit that
those do not count: "a patch agrees with the call it replaces without either
being right". That is the correct rule. The consequence is that **the code
`port` actually ships is the least-judged code in the repository**, which no
ticket says.

It is sharper now that `FIGURE_SWAPS` is in the default: `patch/figures` is
57 unjudged statements that change what every user's output looks like.

---

## B. Runtime and memory

| what | measured | owner |
| --- | --- | --- |
| figure swaps, whole run | 192.06 s to 120.48 s, 11.35 GB to 3.69 GB | landed |
| `SWAPS`, whole run | 192.06 s to 156.63 s, 11.35 GB to 11.33 GB | landed |
| the remaining ~120 s | **unattributed** | #218 |
| emission, at the boundary | 93 % of 13.1 s | #90, #47 |

### B1. #90's premise does not hold at this instance, and says so about regime

#90 is "the emission array is what caps the instance: 8 GB at the declared
scale". At `4,000 x 1,980 x 5` the emission array is about 0.3 GB against a
peak of 11.35 GB, and changing **only figure rendering** takes the peak to
3.69 GB. Here plotting caps the run and the emission array does not come
close.

That is a difference in regime rather than a contradiction: #90's 8 GB is at
`10,000 x 5,000`, where the array is 25x larger and plotting is not. Both
can be true, and `CLAUDE.md` asks for exactly this to be stated where the
work is rather than left for the next reader to rediscover. What it does
mean is that **"the emission array caps the instance" is not a claim that
transfers**, and #90's table needs the regime on it.

### B2. Most of the run is unprofiled

The whole run is 192 s. `SWAPS` accounts for 33 s of it and the figure swaps
for about 70 s. That leaves **roughly 90 s nobody has attributed**, at an
instance where `STATUS.md`'s only profile covers the HMM/spatial boundary --
which, if it were the whole of the rest, would make the boundary far larger
a share than the 33 s the patches recover.

#218 drills `phasing`, `integer_copy` and `hmrf`'s wrappers. That is the
right next step and it is filed; this is the measurement that says how much
is at stake.

### B3. The M step is where the profile puts the time and is unmeasured

`STATUS.md` 4.2 says so plainly, and #47 carries the 12.8x design-matrix
gap. Nothing here changes that; it is listed so the ranking is honest about
what has not been touched.

---

## C. `snakes_and_ladders`

`CLAUDE.md`: an optimization is considered first as functionality that
exists upstream, then as one that could, and only then as one written here.
Every row below is the second case.

| what | evidence | owner |
| --- | --- | --- |
| `likelihood.spatio_sequential` and `search.spatio_sequential` at 0 % | the fit-level half of #97's rung | #77 item 1 |
| a covariate per channel | four modules above the emission family pass none | #57, #65 |
| what #674 makes newly possible | pin moved; not re-read since | #109 |
| the referee surface is 43.76 % and 0.06 above its floor | guard 2, measured | #128, #139 |

**Guard 2 sits 0.06 points above its floor.** That is one test away from red
through no fault of anyone here -- the floor follows a dependency this
repository does not own, and `.coveragerc-oracle` already records one recut
made for exactly that reason. It is the most fragile number on the README
and nothing tracks the fragility.

---

## What this says to do next, in order

1. **#223**, because it taxes A1 and makes the largest coverage win read as
   a regression while it stands.
2. **A3**, a ticket nobody has filed: `port`'s own shipped code is the
   least-judged code here, and `patch/figures` now changes every user's
   output.
3. **Decide #34 against #103** on the memory evidence, rather than leaving
   two open tickets asking for opposite things.
4. **#218**, to attribute the ~90 s the whole-run measurement cannot see.

Nothing in 1-4 needs a new dependency, a new fixture or a new framework.
