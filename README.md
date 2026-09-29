# port

[![e2e](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/michaelJwilson/port/main/.badges/coverage-judged.json)](#what-the-badges-mean)
[![oracle](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/michaelJwilson/port/main/.badges/coverage-oracle.json)](#what-the-badges-mean)
[![drop-in](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/michaelJwilson/port/main/.badges/coverage-dropin.json)](#what-the-badges-mean)
[![all](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/michaelJwilson/port/main/.badges/coverage-reach.json)](#what-the-badges-mean)
[![speedup](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/michaelJwilson/port/main/.badges/run-speed.json)](#what-the-badges-mean)
[![mem](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/michaelJwilson/port/main/.badges/run-mem.json)](#what-the-badges-mean)
[![instance](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/michaelJwilson/port/main/.badges/instance.json)](#what-the-badges-mean)
[![port ARI](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/michaelJwilson/port/main/.badges/recovery-port.json)](#what-the-badges-mean)
[![sal ARI](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/michaelJwilson/port/main/.badges/recovery-sal.json)](#what-the-badges-mean)
[![patched](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/michaelJwilson/port/main/.badges/patched.json)](#what-the-badges-mean)

A scientific repository built on
[`snakes_and_ladders`](https://github.com/michaelJwilson/snakes_and_ladders),
holding the same separation of infrastructure from application and the same
standard: correctness and reproducibility of numerical results are required.

`port` adds a job upstream does not have: it **validates and optimizes a
dependency it does not own**. `cnaster` is the subject, the
[`cna-maste-paper`](https://github.com/michaelJwilson/cna-maste-paper) states
the method, and `snakes_and_ladders` implements the parts of that method which
are not application specific. Where all three describe one quantity, all three
compute it, and the agreement is reported as a number rather than as the word
"matches".

Neither dependency can host this comparison; each would have to depend on the
other. Both are **read only** here: `port` pins them, measures them and reports
on them, and lands nothing in either.

Python lives under `python/port/`; the CPU-bound work belongs in the Rust crate
under `src/`, exposed to Python as `port.oxiport`.

## What the badges mean

Ten numbers, and each is a claim rather than a decoration.
`.badges/measurements.json` holds every value with the selection, denominator
and commit that produced it, `python -m tests.badges` derives the badges from
it, and `tests/test_badges_agree.py` fails when the two disagree -- the same
guard `tests/test_planning_documents_agree.py` puts on the planning
documents.

**Four coverage guards, because one figure would answer four questions
badly** (#159, #281). Three measure a dependency this repository does not
own and are low by construction; the fourth measures `port`'s own
replacements and is high for the same reason. One of the four is off: see
the table.

| badge | selection | denominator | what it says |
| --- | --- | --- | --- |
| **e2e** | `end2end` | `cnaster` + `python/port` | how much of the subject is **validated end to end**, against the truth that generated the data. `oracle` is excluded because the badge beside it claims that word |
| **oracle** | the referee's own reach | `snakes_and_ladders` | **disabled** (#282), so it renders `/` rather than a figure nothing measures. It said how much of upstream is used as a referee, separately so it could not rise by importing more of upstream |
| **all** | the other eight markers | `cnaster` + `python/port` | how much is merely **run**, rather than judged against anything outside `cnaster` |
| **drop-in** | `patch or cnaster` | `python/port/patch` | how much of what `port` wrote to replace something is reached by the test comparing it with the something. The one guard whose denominator is ours, so the one with a high floor |

**`speedup`** is CalicoST's wall over `run_cnaster_port --sal`'s on
`dev_tree` r0 (60 x 50 per slice, 6,000 spots): CalicoST's 20,243 s
(#532, 3 cores) taken as its one-core time, which favours CalicoST, against
`port` pinned to one core. The sample, both walls and the commit are in
`measurements.json`.

**`mem`** is patched `run_cnaster` against `--no-patch`: peak resident
memory, each arm in its own process, in ratio units. It reads `/` until both
arms are measured on `dev_tree`; the last ratio, 3.21X at 3800 x 1980 x 5,
is kept in `measurements.json` as superseded, since a ratio read at another
instance is not comparable.

**`instance`** names the sample the badges are read at, `dev_tree` r0, and
`CLAUDE.md` is explicit that a ratio read at a gate size decides nothing. It
asserts nothing and is blue for that reason.

**`patched`** is how much of what a run executes `port` has replaced: of
the `cnaster` lines an unpatched `run_cnaster` executes on the dev instance
(`numba` disabled, so a kernel's body counts), the share inside a function a
default row of `run_cnaster_port` replaces -- for a class, its overridden
methods (#302). Measured by `python -m tests.patched_share`, not per pull
request, since it is a whole run; blue, because it asserts nothing.

**`port ARI` and `sal ARI`** are recovery against the planted truth on
`dev_tree` r0, for `run_cnaster_port`'s default and for `--sal`, as
`(clones, copies)`: the adjusted Rand index of the run's `clone_labels.tsv`
against the planted clones over spots -- after #518's merge of clones whose
decoded `(A, B)` agree at 0.99 of bins -- and of each clone-bin's phased
`(A, B)` against the state the fixture painted there. Measured by
`python -m tests.sim_audit --sample generated/dev_tree/r0`; the sample,
configuration and commit are in `measurements.json`. Not per pull request,
since each is a whole run; orange, a fixed colour that no threshold
decides.

`tests/test_badges_agree.py` is what keeps them together. It refuses a
recorded ratio that does not name its instance, carry exactly two arms, and
show both arms exiting 0 -- a ratio from an arm that did not complete is not
a ratio -- and it refuses a ratio rendered while `instance` still reads `/`.

**The badges are pinned to `main`, so a pull request does not show its own
figures** -- the ten URLs above all read `/main/.badges/`, and a README
cannot render a branch-relative badge without making `main`'s README wrong.
CI closes that with a report instead (#271): `tests/badge_report.py` renders
this branch's guards against its base, delta first, into the job summary and
into one pull request comment rewritten in place on each push. It reports and
never gates -- `tests/check_badges.py` is what fails the job on a figure that
moved and was never written down, and it runs first so the report cannot
stand in for it. A denominator that moved is called out beside the delta,
because two percentages over different denominators are not comparable and a
bare `-3.43` invites exactly that mistake.

Neither ratio is re-measured by CI, and that is deliberate rather than
pending. A whole `run_cnaster` on a shared two-core runner is moved more by
the runner than by the patch, so a figure from there would be a number
`CLAUDE.md` would not let this repository report. They are measured by hand
on a quiet host, and the recorded commit is what says which tree they
describe.

**`e2e` and `all` are not subtractable** (#223). They report against
different denominators -- 7,030 and 7,329 statements on the same tree with
the same config -- because a `cnaster` subdirectory module enters the figure
only when some test imports it. `cnaster` carries no `__init__.py`, so
coverage's directory scan never reaches `scripts/`, but the tracer measures
whatever runs and the source prefix then admits it. The 299-statement
difference is `scripts/run_cnaster.py`, which guard 3's selection imports
and guard 1's does not.

The direction is what makes it worth fixing rather than noting: bringing that
entry point under an `end2end` test would add its statements to `e2e`'s
denominator, and unless the test covered more than 45.69 per cent of them the
guard would **fall** for validating the most live code the subject has.

**A badge reading `/` has no measurement yet**, and that is the point: not a
zero, which is a claim, and not a last-known figure from a commit nobody can
name. `all` is unwired (#159's "Done when" asks for it and is unmet) and the
ratio pair is #91.

## Prerequisites

| Tool | Version | Notes |
| --- | --- | --- |
| Python | >= 3.12.2 | `requires-python` in `pyproject.toml` |
| Rust | 1.94.1 | pinned by `rust-toolchain.toml`; `rustup` installs it automatically |
| [uv](https://docs.astral.sh/uv/) | >= 0.8.17 | |

`maturin` is the PEP 517 build backend, so a Rust toolchain is required even
for a Python-only workflow: any install compiles the crate.

## Install

```
uv sync --locked --all-extras
source .venv/bin/activate
```

That builds the Rust extension, installs `port` in editable form, and
resolves `snakes_and_ladders` from git against `uv.lock`. The first sync
compiles two Rust crates and downloads PyTorch, so allow several minutes;
later syncs are cached.

To install a single extra rather than all six (`dev`, `test`, `docs`,
`notebooks`, `calicost`, `track`):

```
uv sync --locked --extra test
```

`calicost` is the odd one. It pins
[CalicoST](https://github.com/raphael-group/CalicoST), the program `cnaster`
was rewritten from, at a commit rather than a branch, because it is a
reference this repository **reads and does not run** -- see
`docs/audit-logmu-shift-calicost.md`,
`docs/audit-integer-copy-calicost.md`,
`docs/audit-cnaster-calicost-divergence.md` and, stage by stage against
`--sal`, `docs/audit-calicost-methods.md` (#509). It is an extra rather than a
dependency because nothing on the default path imports it.

`track` is the other odd one, and it is the one to read before running
`--all-extras`. It pins [Aim](https://github.com/aimhubio/aim), the run
store behind `snakes_and_ladders.track`, which #251 records
`run_cnaster`'s optimizations through. `track.Run` is a Protocol written
with `aim.Run`'s own signatures, so the recording path imports, types and
tests with `aim` absent; only a reader who wants the UI installs it.

Two things come with it. The resolution goes **194 packages to 219** --
`aim` carries the web server `aim up` runs, which the recording path never
touches. And that server holds two advisories with no fixed version,
**PYSEC-2026-1087** (XSS in the report endpoint) and **PYSEC-2026-1088** (a
sandbox escape in the query handler). CI syncs `--extra dev --extra test`
and so never installs it; `uv sync --locked --all-extras` does, and a
`pip-audit` after that command reports both. Sync without this extra before
auditing.

Without `uv`, any PEP 517 front end works, but the git dependency is then
unpinned and the extension is rebuilt rather than reused:

```
pip install -e '.[dev,test]'
```

After changing a dependency, run `uv lock` and commit the updated `uv.lock`
in the same change.

## Checks

CI runs locally, through one entry point (#403). Each step prints its seconds.

```
uv run python -m tests.ci                  # gate: ruff, mypy, critical + untiered tests; <= 60 s
uv run python -m tests.ci --badges         # judged and drop-in coverage; --record writes them
uv run python -m tests.ci --full           # gate, badges, then `merge` tests and benchmarks
uv run python -m tests.ci --release        # `release` and `oracle`
uv run python -m tests.ci --figures        # redraw docs/plots
uv run python -m tests.ci --install        # once per clone: the `badges` merge driver
cargo clippy --all-targets -- -D warnings  # Rust lint
cargo fmt --check                          # Rust format
```

Every test sits in at most one tier -- `critical`, none, `merge`, `release`,
`deprecate` -- and each step selects one, so no step repeats another's tests.
A `deprecate` test runs only in the change that touches its module (against
`--base`, `origin/main`) and at a release. The gate
is `pytest -n 4`; a whole-pipeline test (`xdist_group("pipeline")`) runs one
at a time, since four exceed 15 GB. Badges are measured and recorded locally
by the change that moves them. `.gitattributes` sends `.badges/*.json` and
`docs/plots/*` to the `badges` driver, which keeps the branch's copy on a
merge; `--badges --record` and `--figures` then regenerate them.

`mypy` reads its paths from `pyproject.toml` (`python/`, `tests/`). The
compiled extension is typed by the hand-written stub
`python/port/oxiport.pyi`, which must be kept in step with the
`#[pyfunction]` definitions in `src/lib.rs`.

## Running the pipeline patched

```
run_cnaster_port config.yaml                 # cnaster's pipeline, port's replacements
run_cnaster_port --no-patch config.yaml      # the same run, nothing rebound
run_cnaster_port --no-figure-swaps config.yaml  # the replacements that reproduce bitwise
run_cnaster_port --no-plots config.yaml      # build every figure, write none; for a run whose claim is not a figure
run_cnaster_port --no-rust config.yaml       # cnaster's numba lattices instead of oxiport's
run_cnaster_port --sal config.yaml           # snakes_and_ladders routines where port measured a gain
run_cnaster_port --no-copy-cap config.yaml   # cnaster's integer copy caps, A + B <= 6, whatever the config states
run_cnaster_port --sample-layout 3,1 config.yaml  # clone spatial plots, one panel per sample
run_cnaster_port --genomic-colours states config.yaml  # clones_genomic coloured per fitted state, not per integer pair
run_cnaster_port --copy-decode shared config.yaml  # one integer pair per fitted state; default: lattice Viterbi per clone
run_cnaster_port --time-stages config.yaml   # what the replacements cost in the run
run_cnaster_port --floor-merge --refinement-mask config.yaml  # #348's clone patches, opt-in; --no-distinct-init drops the third
python -m port.sandbox.np_merge config.yaml # sandbox: CalicoST's Neyman-Pearson merge of clones that decode alike (#497), not installed by default
run_cnaster_port --hmm-start kmeans++x5+em config.yaml  # the read-depth HMM's start from sal's covariate mixture (#489); on with --sal
run_calicost config.yaml                     # CalicoST on the same fixture files, at port's configuration
run_calicost --shipped configuration_cna config.yaml  # CalicoST's own configuration file, the run's paths (#494)
run_cnaster_port --no-outputs config.yaml    # skip the fitted/decoded tables below
run_cnaster_port --list                      # what would be rebound, and why
run_cnaster_port --audit-config config.yaml # what the config states that cnaster does not use (#324)
```

**The options, their defaults and what measured them.** `run_cnaster_port
--help` gives each in one line; the numbers are here. A tri-state option
follows the arm unless asked: on in a patched run, off with `--no-patch`.

| Option | Default | What | Measured |
| --- | --- | --- | --- |
| `--figure-swaps` | on; off with `--no-patch` | `FIGURE_SWAPS`: dpi and raster groups (#195), the genomic RDR line (#299), tiles and the copy profile (#309) | 47 per cent of a run; figure rendering 8,287 MB to 1,036 MB (#195) |
| `--shift` | on; off with `--no-patch` | `SHIFT_SWAPS`: the per-clone `log Z_c` in the fit, the normal clone pinned to `mu = 1` (#276, #299) | without it a clone's rates return divided by its own normalizer |
| `--sal-emission` | on where the shift is | sal's dense log-emission for the coded NB/BB (#425) | 3.2e-12 of `cnaster`'s kernels, 3.5e-9 at the dispersion floor; no faster end to end |
| `--distinct-init` | on where the shift is | the HMM starts from distinct GMM components (#348) | copy-state ARI 0.896 to 0.997 on `calicost_instance` |
| `--copy-cap` | on; off with `--no-patch` | the likelihood decode under the configured cap (#313, #362) | `cnaster`'s decoders read no cap: A + B <= 6 |
| `--copy-decode` | `lattice` | per-clone lattice Viterbi with tumour fraction (#370), or one pair per state, `shared` (#327) | |
| `--rust` | on; off with `--no-patch` | `cnaster`'s four lattices from `oxiport` (#318) | bitwise; compiled at build, not per process |
| `--sal` | off | alpha expansion with the Rust cut for the labelling (#312), and the next two | a lower Potts energy on every problem measured |
| `--refinement-mask` | off; on with `--sal` | each read-depth sub-clone kept in its BAF clone, a 100-nat penalty (#348, #467) | with the floor merge, CalicoST hard clone ARI 0.303 to 0.982 |
| `--floor-merge` | off; on with `--sal` | the clone-size floor met smallest first (#348) | alone, #338's three-sample instance: 2 planted clones fitted as 6 |
| `--hmm-start` | `none`; `kmeans++x5+em` with `--sal` | the read-depth HMM's start from sal's covariate mixture (#489) | CalicoST hard clone ARI 0.8652 to 0.9829 |
| `--copy-errors` | off | `cnv_copy_sets.tsv`: every `(A, B)` in each state's 95 per cent credible region (#353) | differentiates the whole objective once |
| `--png-copies` | off | a PNG without metadata beside each PDF, for `docs/plots` (#452) | two runs of the same code write the same bytes |
| `--sample-layout`, `--genomic-colours` | unset | one panel per sample (#328); bins coloured per fitted state | |
| `--warm-up` | off | compile every kernel before the clock starts (#211) | |
| `--no-plots`, `--no-outputs`, `--time-stages`, `--audit-config`, `--list` | off | build figures and write none (#403); skip port's tables (#331); cost per swapped name; unused config (#324); the table | |

**A patched run also writes the seam between the fit and the integers**
(#331): beside `cnaster`'s files, `port.extensions.outputs` writes
`cnv_states.tsv` (each fitted state, the `(A, B)` each clone decodes it to,
and its share of the clone's bins), `cnv_segments.tsv` (runs of equal
`(A, B)`), `cnv_binlevel.tsv` (the posterior-mean `mu` and `p` per bin),
`clone_labels_integer.tsv` (each spot's clone named by its integer copy
profile: clones whose `(A, B)` agree at no less than
`int_copy_num.merge_agreement` of bins, 0.99 unless stated, are one clone,
#344, #518) and `manifest.json` (states, clones, likelihoods, the
configuration's caps and the flags). Where that merge joins clones it also
rewrites `clone_labels.tsv`: `clone_label` is the merged clone and
`cnaster_clone_label` keeps `cnaster`'s, since the merge stands in for the
Neyman-Pearson merge `--sal` no longer installs (#497). Off with
`--no-patch`, so the baseline arm writes what `cnaster` writes.

`port.pipeline.SWAPS` is the table -- one row per `cnaster` name `port`
replaces, each naming the ticket that measured it -- and `patched()` is the
context manager that installs and restores it. The rebinding follows a name
wherever it has been imported, because `run_cnaster` holds its own
`from cnaster.omics import ...`.

**Every row of `SWAPS` reproduces `cnaster` artifact by artifact**
(`tests/test_patched_entry_point.py`), which is the claim that makes the
speed claims worth reading. `FIGURE_SWAPS` is a second table that does not:
lowering the dpi and merging the rasterizing groups writes a different file
by design (#195). It is **in the default** because it is the largest win
here, and `--no-figure-swaps` is the arm that reproduces bitwise.

**The figure swaps also set one face for every figure**, `cnaster`'s and
`port`'s: `[tool.port.figures]` in `pyproject.toml` names it (STIX, with its
math fonts, by default; the alternatives are commented there), and
`port.extensions.figure_style` refuses a face matplotlib cannot find rather
than falling back to another.

Measured at 4,000 x 1,980 x 5, against `--no-patch`:

| installed | wall | peak RSS |
| --- | ---: | ---: |
| nothing | 192.06 s | 11.35 GB |
| `SWAPS` | 156.63 s | 11.33 GB |
| `SWAPS` + `FIGURE_SWAPS` | 120.48 s | 3.69 GB |

So the figure swaps are most of the runtime win and all of the memory one.

**`--copy-cap` is on by default** (#313). `cnaster` decodes integer copies
under `A + B <= 6` and `A, B <= 5` and reads no key that changes them, so a
planted total of 10 cannot be decoded. `COPY_SWAPS` reads
`int_copy_num.max_total_copy` and applies it to both caps; a configuration
without the key, or with `none`, decodes exactly as `cnaster` does, and a
value that is not an integer of at least 2 is refused at start. The MILP decoder, called
as `run_cnaster` calls it, returns planted totals of 10 to 12 exactly at a
stated 12, and none of them at `cnaster`'s 6 (`tests/test_integer_copy_patch.py`).

**Several samples run as is, with shared clones** (#328).
`tests/multisample.py` places three realizations of one genome side by side,
with one empty column between them and an integer `sample_label` per spot.
`run_cnaster_port` recovers the planted clones in every sample at ARI 1.000.
Spatial edges stay within a sample.
`port.extensions.multisample.cross_sample_adjacency` is the placeholder for
edges between samples, and nothing installs it. `--sample-layout 3,1` draws
the clone spatial plots one panel per sample, each in its own coordinates.

**`--genomic-colours` chooses how `clones_genomic` colours bins** (#333).
`integer` colours by decoded `(A, B)`, so fitted states that oversample one
pair share a colour. `states` colours each fitted state separately, with its
continuous `2mu` and `p` in the legend. Unset, the choice is `cnaster`'s:
integer copies where the figure has them, states elsewhere.

**Integer copies are decoded by likelihood, per clone and bin** (#367, #371).
The copy rows (on with `--copy-cap`) run `lattice_decode`: one state per
`(A, B)` with `A + B` at most the configured cap, Viterbi along each clone's
bins on its own pseudobulk NB/BB counts, and an EM fitting each clone's
shift and tumour fraction `rho` against a diploid normal. So a loss and an
LOH the HMM fitted as one state still decode apart by depth, and admixed
normal cells no longer read as balanced states. The normal clone, held at
`(1, 1)`, shift 0 and `rho = 1`, is the one with the largest share of
balanced bins, as `clone_shifts` names it (#389). `--copy-decode shared`
keeps one pair per fitted state (#327). `dev_tree` r0 at 60 x 50 under
`--sal`: copy ARI 0.981, 0.936 of altered bins exact up to phase, against
cnaster's decoder's 0.956 and 0.707 at 42 x 42.

**`--rust` is on by default** (#318). It runs `cnaster`'s four
forward/backward lattices from `port.oxiport`, bitwise `cnaster`'s
(`tests/test_rust_lattice.py`, and a whole `--no-patch` run reproduced
artifact by artifact). `cnaster`'s unphased pair is `@njit` without a cache,
so every process compiled it: 4.1 s and 0.5 s of first call, against 0.5 ms
from Rust. On the dev instance a default run takes 29.8 s against 36.5 s
with `--no-rust`. The four kernels are 4.3x to 6.1x faster warm at
`K = 10`, 10,000 bins and 20 spots, on four cores.

**`--sal-emission` is on by default** (#425). It scores the coded NB/BB
emission with `snakes_and_ladders`' dense log-emission, to 3.2e-12 of
`cnaster`'s kernels. On the dev instance the default run's clone ARI rises
0.7927 to 0.8653 and its integer ARI 0.9242 to 0.9905, fitting 5 clones for 4
rather than 6; the lattice fixture is unchanged at 0.9985. It is not faster end
to end. `--no-sal-emission` restores `cnaster`'s kernels. It rides on the
shift rows, so `--no-shift` turns it off too, and `--distinct-init` with it.

**The M step's gradient is closed form** (#433). `cnaster` fits the
emission by BFGS with a finite-difference gradient, one objective call per
packed coordinate; `port` supplies the derivative instead
(`port.patch.hmm_nophasing.gradient`), pinned against `jax`'s. On the dev
instance the M step falls from 11.1 s to 2.0 s. The row's
`analytic_gradient=False` option restores `cnaster`'s gradient.

**The spatial graph is validated before the HMRF sees it** (#417). The run
builds it from `port.extensions.adjacency`: by default each spot's `k`
nearest (`PORT_ADJACENCY=knn`), with `k` the lattice's coordination -- 8 on a
square grid (Moore; `PORT_SQUARE_NEIGHBOURHOOD=square` for 4), 6 on a Visium
hexagon. On a square grid that is `cnaster`'s own graph entry for entry.
`PORT_ADJACENCY=lattice` builds the neighbourhood's offsets instead,
symmetric with boundary edges reinforced. The guard refuses a self loop, and
for `knn` a row without `k` unit edges or a graph under 0.6 reciprocated.

**`--sal` is off by default** (#312). It admits `snakes_and_ladders`
routines only on `port`'s measurement, and admits one today: the clone
labelling. That row fuses alpha expansion (Rust minimum cut) with sal's
descent from the field's argmax, then applies sal's `merge_small_labels` at
`cnaster`'s 200-spot floor (#410), so no `cnaster` ICM runs. On the dev instance it recovers the planted clones at
ARI 1.000 against the default's 0.7927, in 24.8 s against 44.9 s.
`port.extensions.sal` lists what was measured and not admitted.
`docs/audit-recovery.md` carries its recovery against the planted truth at
four configurations (#313).
At this instance the emission array is about 0.3 GB against an 11.35 GB
peak, which says plotting caps this run rather than the emission array --
a different regime from #90's declared scale, not a contradiction of it.

**Three clone-assignment patches, one on by default** (#348). On
`tests.fixtures.calicost_instance`, `cnaster` ends with one clone (ARI 0.000):
`run_cnaster.py:1105` drops the read-depth refinement's allowed-clone mask, and
`icm_sweep_deque` then moves every clone under 200 spots at random into the one
that reached 200. `--floor-merge` keeps the floor but merges the smallest clone
first, into each spot's best remaining clone. `--refinement-mask` passes the
mask. `--distinct-init` stops `gmm_init` keeping near-duplicate normal
components as separate states. With all three on, the run recovers clone ARI
0.774 (1.000 integer) and copy-state ARI 0.997. The floor merge alone removes
the collapse. `--floor-merge` and `--refinement-mask` are opt-in on the
default arm, where each alone splits #338's three-sample instance, 2 planted
clones into 6 fitted, and on with `--sal` (#467). There the mask is a 100-nat
penalty rather than `-inf`, so the read-depth stage can still move a spot the
BAF stage misplaced: with `--sal`, CalicoST hard goes from clone ARI 0.303 (2
clones for 4) to 0.982, easy from 0.944 to 0.986, and `dev`, r0 and the
three-sample instance stay at 1.000, 0.998 and 1.000.
`--distinct-init` is on by default.
The mask and the floor are read by port's clone assignment alone, so
`--no-patch` refuses them, and where a tumour proportion hands the assignment
to `cnaster` the run warns that they, and the per-clone shift, are not applied
(#466).

**`run_calicost`** (#347) translates the same YAML and runs CalicoST in-process
on the same files, into `<output_dir>_calicost`. `--align` (the default)
replaces the CalicoST constants that have a `cnaster` counterpart;
`--no-align` keeps CalicoST's own. It refuses the initial-clone layout on which
CalicoST's `rectangle_initialize_initial_clone` never returns (`cnaster` #248).
`python -m tests.recovery_audit --calicost` scores it with port's scorer.
`--shipped FILE` runs CalicoST's own configuration file instead, taking only
the paths from the YAML; a sheet of several slices takes
`configuration_cna_multi`. `docs/final-benchmark.md` compares it with `--sal`.

**`port.sim.draw`** (#445) draws new samples from a version-3 manifest:
clones from CalicoST's `shared.unique` counts or a mutation tree
(`snakes_and_ladders`' `random_topology`, rooted at `normal`), fixed or
exponential event lengths, one or more slices with clones on N-gon regions,
Visium barcodes with hexadecimal `sample_id`s, and phase switches at the
genetic map's Haldane rate. Every assumption is a TOML key, and a manifest
that omits one is refused; `sim/manifests/calicost_grch38.toml` states
CalicoST's and a manifest `extends` it. Counts follow `sim/normal_baseline.txt.gz`
(λ per gene) and `sim/normal_coverage.toml`, both fitted on CalicoST's normal
spots by `port.sim.normal_fit` (#455). A spot's genes are
`Multinomial(N_s, p_s)`, `p_s ~ Dirichlet(κ λ d_c)` with `d` the clone's
`(A + B) / 2` and `N_s` the `spot_umi` law times `Σ λ d_c`, so a gain grows
the library; κ = 100 matches the normal spots' per-(gene, spot) nonzero share
(0.90%) and `log10` moments. Each (SNP, spot) is drawn independently from the
`snp_spot_umi` law. `[sample] realizations` redraws the counts and phase over
the same clones and layout, each a complete sample in
`sim/generated/<name>/r<k>/`, untracked; `dev_tree` draws in 23.0 s. Both dev
manifests use CalicoST's array, 60 rows of 50 per slice, and plant every
clone above cnaster's fixed 200-spot ICM floor (#468): `dev_tree` 6,000
spots over two slices, `dev_shared_unique` 3,000 on one.
`tests.sim_audit` runs and scores one realization:

    python -m port.sim.draw sim/manifests/dev_tree.toml
    python -m tests.sim_audit --sample generated/dev_tree/r0 -- --sal

`port.sandbox.sim_from_run` (#460, set aside) writes a version-3 manifest
from a finished run's `clone_labels.tsv` and `cnv_segments.tsv`: its clones,
shared and unique events, states, array and slices. What a run does not
measure -- slice offsets, the tree, normal fractions -- is left commented, so
the manifest draws only once the offsets are stated:

    python -m port.sandbox.sim_from_run <run dir> > sim/manifests/<name>.toml

`tests/test_file_sizes.py` refuses a tracked or addable file above
`[tool.port] max_file_bytes` (5 MB): GitHub rejects 100 MiB, and a clone keeps
every version. CalicoST's samples and `sim/normal_baseline.txt.gz` are
stored compressed (`port.sim.files`, deterministic gzip where the format is
not compressed already); `tests.sim_fixtures.stage` writes a run's inputs
out plain under the names `cnaster` opens.

`cnaster` appends a fit record to `cnaster.perf` in the repository root on
every run. It is **not tracked** (#222): nothing reads it, no test
references it, and its rows carry no commit or instance, so it is a log
rather than a measurement record. A tracked file that changes on every run
trains a reader to ignore `git status`.

## Layout

| Path | Contents |
| --- | --- |
| `python/port/` | The Python package; `python-source` in `pyproject.toml` |
| `src/` | The Rust crate `oxiport`, bound as `port.oxiport` |
| `tests/` | The suite; `testpaths` in `pyproject.toml` |
| `sim/` | CalicoST's simulated samples, their normal fits, and `manifests/` that draw them |
| `Cargo.toml` | The single source of the version, which maturin reads across |

# Infrastructure

## The contract an agent reads first

| Contract | Governs |
| --- | --- |
| [`CLAUDE.md`](CLAUDE.md) | Every rule the repository is developed under. It mirrors upstream's section for section, so the two diff against each other; where they disagree, this one wins and the disagreement is the reason it is written down. |

One package and one crate, so one file is the whole of the guidance. A
submodule `CLAUDE.md` is added when a directory has details this file should
not carry, not before.

## What enforces the claims

| Mechanism | What it refuses |
| --- | --- |
| Two CI jobs | A stale `uv.lock`, a lint or format failure, an untyped definition, a failing test, a `clippy` warning |
| Registered markers | A test not checked against exactly one of `end2end`, `oracle`, `analytic`, `patch`, `backend`, `bug`, `warning`, `snapshot`, `smoke`, `infra` (#157), plus the second axes `critical` and `cnaster`, and the tiers `release`, `preprocessing` and `benchmark`. Only `end2end` and `oracle` count toward coverage, and CI runs neither the `oracle` tests nor their guard while #282 holds |
| `--cov-fail-under` over the whole of `cnaster` | A figure that rises for importing less. The denominator is the dependency, so the number says how much of the subject is validated |
| [`tests/test_coverage_scope.py`](tests/test_coverage_scope.py) | A gate silently measuring a fraction of the subject after a Python version bump |
| [`tests/test_planning_documents_agree.py`](tests/test_planning_documents_agree.py) | The three planning documents naming different work |
| [`.github/pull_request_template.md`](.github/pull_request_template.md) | A ratio with no pinned output, a patch with no ratio, an unstated difference between the references |

## The documents

| Document | Contents |
| --- | --- |
| [ROADMAP.md](ROADMAP.md) | The stages, and the loop a change passes through |
| [TICKETS.md](TICKETS.md) | What is filed and not done, grouped by the milestone it serves |
| [STATUS.md](STATUS.md) | What has landed, with the measurement that established it |
| [CLAUDE.md](CLAUDE.md) | The rules |
| [docs/measurements.md](docs/measurements.md) | The timings, ratios and histories the package docstrings cited, by module and object (#517) |
| [docs/metrics.md](docs/metrics.md) | One row per recovery run: commit, timestamp, fixture hash, test, arguments, clone/copy/state ARI, wall, peak, note (#409) |
| [docs/templates/](docs/templates/README.md) | Templates for documents made outside the code: the work-in-flight page (#335) |

`DEV.md`, `INSTALL.md` and `CHANGELOG.md` are added when the content for them
exists, not ahead of it: `README.md` still carries installation and
development, and `towncrier` has no release to build.

# Application

## What exists, measured

Coverage is the four guards above, each recorded with its selection,
denominator and commit in `.badges/measurements.json`; the `e2e` badge is
the validated share of `cnaster` and `python/port`. Low by construction, and
it rises only by validating more of the subject.

| Claim | Realized |
| --- | --- |
| The emission, scored at fixed parameters against upstream | max abs diff `1.5e-13`; total log-likelihood `8.0e-13`; the phased total exact at every parameter tried |
| The combined transition against `cnaster`'s own builder | `2e-17` |
| Two M steps sharing no solver, parameterization or start | agree to `5e-5` relative on `(alpha, beta)` at their maxima |
| `cnaster`'s shipped EM criterion | stops `7.48` nats short at 19 per cent error in `alpha`, reporting `converged: True` |
| `cnaster` maximises what upstream minimises | cost plus energy is `0` to `7.1e-15` across every labelling tried |

`STATUS.md` carries the rest, and says which rows are unmeasured.
