# Measurements

The timings, speedups, memory figures, recovery scores and histories that
the package's docstrings and comments cited, moved here verbatim so a
docstring carries only its function's contract (#517). Each `##` section
names the module and each `###` heading the function, class or constant the
passage came from; that docstring points back here. A figure is as current
as the commit that recorded it, and re-measuring one updates it here rather
than in the code.

## `port.extensions.parameter_errors`

### module docstring

The previous work is `cnaster/sandbox/hmm_nophasing_jax.py`, named because
`CLAUDE.md` puts a dependency's `sandbox/` out of scope by default and an
excursion has to say which tree it read and why. It could not be answered
from the installed path: that tree is not in the wheel, so nothing installed
differentiates this objective.

`jax` is a dependency of this repository as of #287, added with permission
because the alternative was writing the objective a third time.

## `port.patch.reference`

### module docstring

**13.3x and 55.2 MB to 23.3 MB at 250,000 transcripts, which is a human
reference's size.** The function reads one tab-separated file and hands back
six columns, and `pandas` is 367 ms of that on its own.

Measured at 250,000 transcripts, warm, best of five:

    cnaster                430.65 ms   49.54 MB
    column by column        42.84 ms   23.28 MB   10.1x, 2.1x less
    through Arrow           60.53 ms   15.49 MB    7.1x, 3.2x less

and at the dev instance's 1,213 transcripts, 3.94 / 1.90 / 4.03 ms.

So it is **1.41x slower than the route it replaces at a stress size and a
wash at a gate size**, for 33 per cent less peak and a function that is one
expression rather than seven. `CLAUDE.md` is what decides which of those
wins: a speedup claim needs 2x at a stress size and this is not offered as
one; a simplification needs evidence of equivalence, which is the bitwise
test. The cost is stated rather than buried -- `pyarrow` is 152 MB installed
and the largest wheel in the environment, against 7.8 MB of peak saved on a
stage that is 0.17 s of a whole run.

## `port.patch.hmrf.invariants`

### module docstring

**Measured** by `pytest-benchmark`, both count passes together, minimum
over the rounds it took:

| `n_obs` | `n_spots` | per iteration |
| ---: | ---: | ---: |
| 240 | 160 | 0.041 ms |
| 3,000 | 5,000 | 25.2 ms |
| 10,000 | 2,500 | 48.5 ms |

Against a boundary costing about 16 s that is under two tenths of a per
cent, so this is a **simplification** rather than a speedup and is offered
as one: the value is that a quantity which cannot change stops being
recomputed, and the ratio is incidental. It multiplies by `max_iter_outer`,
which is the only reason the absolute number is worth writing down at all.

## `port.extensions.integer_copy`

### module docstring

`tests/test_integer_copy.py` measures what the scale costs, and the answer is
not the obvious one. **A scale error corrupts the confidence rather than the
answer.** At `(4, 2)` with `sigma_mubar = 0.08` the argmin stays `(4, 2)` at
every error up to twenty per cent -- the lattice spacing in `mubar` is `0.5`
and no competitor gets closer -- while the squared residual runs `0.00`,
`0.56`, `3.52`, `9.00`, `56.25` and the credible set empties at eight per
cent.

That is the argument for the one-to-many map in one line: an argmin-only
decoder returns the correct pair at a twenty per cent scale error with
nothing to say the fit is fifty-six chi-square units from explaining it.

## `port.patch.hmm_nophasing.bb_logpmf`

### `rises_on_distinct`

The integer-copy likelihood (`port.extensions.copy_likelihood.pseudobulk_log_pmf`)
takes three rising factorials per `(state, bin)`. Over the 7,287 calls of a
dev_tree r0 `--sal` run (#702), the median call is 4 states over 655
pseudobulk bins with 220 distinct counts, and the largest 25 states over
2,829 bins with 594. `tests/test_rises_on_distinct_bench.py`, minimum of
rounds: 91.7 to 56.1 us at the former (1.64x) and 1,589 to 430 us at the
latter (3.69x). Bitwise (`tests/test_rises_on_distinct.py`).

Since T- #781 the gather evaluates `sal`'s `log_rising`, not port's
retired `rises`. At sal `72428f3b` (its #1330: the plain `lgamma`
difference wherever its bound meets 1e-14, one compiled pass per element),
`tests/test_rises_on_distinct_bench.py`, minimum of rounds, load1 1.05: the
gathered call is 382 us at the latter and 51.7 us at the former, against
1,614 and 227 us at sal `21f7013e` and 1,007 and 135 us for port's form
(`83ed58b`). The compile is its own cost, paid once per process and not
cached: 0.78-0.88 s for `log_rising` and 0.19-0.23 s for `digamma_rising`
on the first call (3 processes).

### `_bb_logpmf_1d`, `_dense_bb_logpmf`, the field's rise tables

Every beta-binomial rise is `sal`'s `log_rising` since T- #781; port's
numba `rise` is gone. One process per arm on the same host and `sal`
(`72428f3b`), main's port kernels against these, `log_space=True`, warm,
minimum of 3, gate `5 x 400 x 300 x 2` and stress `7 x 3,000 x 5,000 x 4`
(states x bins x spots x clones):

| Call | Gate (s) | Stress (s) |
| --- | --- | --- |
| `tabulated_field`, the run's | 0.0052 -> 0.0034 | 0.884 -> 0.821 |
| `fused_field`, non-integer counts only | 0.032 -> 0.070 | 8.09 -> 10.28 (1.27x slower) |
| `_dense_bb_logpmf` | 0.066 -> 0.081 | 10.86 -> 31.07 (2.86x slower) |

`_dense_bb_logpmf` was a 4-thread `prange` over states; it is now one
thread over `sal`'s kernel. The values agree with port's former kernels to
2.7e-14 over `max(|f|, 1)` (field 2.7e-15), inside `log_rising`'s 1e-14
promise summed over three rises (PR- #786).

## `port.patch.hmm_nophasing.logmu_shift`

### module docstring

It stays a `numba` kernel: the vectorized form the comment sketches is
measurably slower than the compiled loop, so what this removes is the
broadcast write, not the loop.

On unequal clones: the earlier version kept a `CloneStack` view for equal
lengths and it is gone with the vectorized reduction that needed it.

### _per_clone

Kept as `numba` and kept as upstream's shape of loop, because that is
what the measurement says: a `scipy.special.logsumexp` over per-clone
views is **2.1x slower** at the stress size and 3.9x at the gate one.

### shifts

At the segment count `expected_runtime.tex` derives, 2.9e5, the difference
is also 2.3 MB against a handful of numbers.

A comment above the `_per_clone` call, describing the rectangular path the
module docstring records as removed:

    # NB the rectangular fast path, detected rather than assumed. Where the
    #    clones are equal the whole reduction is one call on a view that
    #    copies nothing; where they are not, a view cannot exist and the
    #    per-clone slices are still each contiguous.

## `port.patch.hmrf.field`

### module docstring

**Measured**, minimum of three runs, `n_states = 7`:

| `n_obs` | `n_spots` | clones | `cnaster` | this | ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3,000 | 5,000 | 4 | 69.6 ms | 17.8 ms | 3.90 |
| 3,000 | 5,000 | 7 | 198.7 ms | 36.1 ms | 5.50 |
| 10,000 | 2,500 | 7 | 483.6 ms | 67.4 ms | 7.17 |

Above `CLAUDE.md`'s 2x bar at every size measured, and it rises with both
extents: the stride is `n_spots` and the number of strided probes is
`n_obs`, so `cnaster`'s form worsens as either grows.
`expected_runtime.tex`'s own derivation puts the genome at 2.9e5 segments
rather than the 3.2e3 it states (#27), so the
range measured understates the size the method is aimed at.

Three other forms were tried and are recorded so they are not retried:

| form | ratio |
| --- | ---: |
| transposing the emission to `(n_states, n_spots, n_obs)` | 1.18-7.46, and needs the producer changed |
| transposing at run time, then that kernel | 0.11 |
| a `numpy` gather, `rdr[pred[:, c], arange(n_obs), :].sum(0)` | 0.32 |
| the same gather on the transposed array | 0.09 |

The transpose is the instructive one: it fixes the contiguity and leaves the
scalar reduction, and measures roughly half what the reorder does at the same
sizes -- 61.8 ms against 36.1 ms at 3,000 x 5,000 x 7. Both gathers
materialize a fancy-indexed `(n_obs, n_spots)` copy per clone that a compiled
loop does not have.

**What the profile says next.** The producer that fills these arrays costs
4,129 ms at 3,000 x 5,000 where the field costs 70 ms, so the field is under
two per cent of the boundary and this patch moves about one per cent of it.
It lands because it is free and bitwise, not because it moves the boundary.
The algorithmic cut is issue #59 item 2 -- not materializing the
`(n_states, n_obs, n_spots)` array at all.

## `port.patch.icm.interface`

### module docstring

The fold does turn one indexed add per clone per spot *visit* into one
vectorized add per spot *sweep*, and that measures 1.19x at 400 spots and
1.38x at 20,000 -- below the 2x bar, so it is reported rather than claimed.

## `port.patch.utils`

### module docstring

**20.34 s of plotting to 3.84 s, and 8,287 MB of `RendererAgg` to 1,036 MB.**
Measured over a whole `run_cnaster` on the dev instance, 19 figures, with
every figure's groups and buffers counted as it was written:

| arm | plotting | groups | allocated | PDF |
| --- | ---: | ---: | ---: | ---: |
| `cnaster` | 20.34 s | 120 | 8,287 MB | 1,954 KB |
| `dpi=150` (#209) | 5.87 s | 120 | 2,072 MB | 1,048 KB |
| `dpi=150`, `sink` | **3.84 s** | **60** | **1,036 MB** | **839 KB** |
| `dpi=150`, `sweep` | 5.16 s | 60 | 1,036 MB | 1,021 KB |
| `dpi=150`, nothing rasterized | 5.08 s | 0 | 0 MB | 1,054 KB |

**Two of the ticket's three claims did not survive being measured**, and
both are recorded here rather than carried forward.

*The group count is not the artist count.* The ticket read mixed mode as
allocating a full-figure buffer per rasterized artist. `matplotlib`'s
`allow_rasterization` starts rasterizing at the first rasterized artist and
stops at the first one that is **not**, so a run of them shares one buffer.
What splits `cnaster`'s runs is a gridline: `_format_track_axis` adds
`ax.axhline(..., c="lightgray", linewidth=0.5, zorder=0)` per y tick
(`plot_genomic.py:70`), between the rasterized errorbar at zorder 0 and the
rasterized scatter at zorder 1. So the floor is one group per axes, not one
per figure, and a collapse that refuses to touch the drawing refuses
everywhere -- measured at 120 groups before and 120 after, byte for byte the
same files.

*Rasterizing is worth it, above about 500 bins.* The ticket asked whether
these panels need rasterizing at all. At the dev instance's 1,000 bins the
vector arm allocates nothing and is still slower, and it gets worse with the
bin count, which is `CLAUDE.md`'s rule that cost depends on the data rather
than only on its size. One panel, two clones, `dpi=150`:

| bins | `cnaster` | `sink` | vector |
| ---: | ---: | ---: | ---: |
| 250 | 0.406 s / 56 KB | 0.292 s / 41 KB | **0.275 s / 31 KB** |
| 1,000 | 0.436 s / 151 KB | **0.344 s / 115 KB** | 0.546 s / 107 KB |
| 4,000 | 0.729 s / 407 KB | **0.603 s / 312 KB** | 1.645 s / 387 KB |
| 16,000 | 1.509 s / 871 KB | **1.361 s / 635 KB** | 6.050 s / 1,506 KB |

The crossover is between 250 and 1,000 bins, and `expected_runtime.tex`'s own
derivation puts a genome at 2.9e5 segments (#27), so every instance the method is aimed at is far above it. The decision
is **keep rasterizing**, and it is a decision rather than a default because
nothing had measured it.

A third hypothesis died earlier and is kept for the same reason:
`bbox_inches="tight"` renders the figure twice and looked like the cost. It
is **1.35x faster**, because the tight bbox shrinks the area that gets
rasterized. Dropping it is worth having only alongside a dpi change; on its
own it is a regression.

### collapse_rasterizing_groups

**The ticket's mechanism was wrong, and the measurement is why this
function exists at all.** It read mixed mode as allocating a buffer per
rasterized artist -- "four rasterized collections in one axes cost four
full-figure buffers". `matplotlib` does not.

Two groups per axes, measured: a whole run allocates **120 groups over 60
rasterized artists on 39 axes** -- exactly two per artist, one per artist
per `bbox_inches` pass -- and 8,287 MB of `RendererAgg` at `cnaster`'s dpi.

`strict` on `cnaster`'s own figures: measured at 120 groups before and 120
after, byte for byte the same files.

### write_fig

**Releasing the renderer each `Text` cached (T- #692 part 2).** Peak RSS of
one whole `run_cnaster`, sampled every 50 ms, one run per arm under
`host.lock`, `cnaster` pin `4adad4d`, `matplotlib` 3.11.2. `release` is
`cnaster`'s figures written by this function at `cnaster`'s defaults, its
"before" `cnaster`'s own `write_fig`; `port`
is `FIGURE_SWAPS`.

| instance | arm | peak before | peak after | write_fig before / after |
| --- | --- | ---: | ---: | ---: |
| gate (`350fbd2b`) | `release` | 5.06 GiB | 1.89 GiB | 11.3 / 11.2 s |
| gate (`350fbd2b`) | `port` | 1.91 GiB | 1.51 GiB | 3.3 / 3.2 s |
| CalicoST easy (`2d4ce9a9`) | `release` | 11.82 GiB | 4.22 GiB | 60.8 / 39.9 s |
| CalicoST easy (`2d4ce9a9`) | `port` | 3.96 GiB | 3.25 GiB | 11.8 / 11.8 s |

Before the change RSS rose by about 1.4 GiB per genomic figure at 300 dpi on
easy and fell to 1.21 GiB only after `plt.close("all")` and `gc.collect()`.
Each `Text` holds the `MixedModeRenderer`, which holds the `PdfFile`, whose
`_images` are views of each rasterizing group's full-page `RendererAgg`
buffer; a `gc.collect()` in this function freed nothing, because the
caller's `fig` still reaches them. All 25 output files of every pair above
are byte-identical. The easy `write_fig` times are one run each and are not
a speedup claim.

### discard_fig

Only the rendering is skipped, which is where a small run spends 31 to 45
per cent of its time, in PDF text layout.

## `port.patch.hmrf.adjacency`

### module docstring

**This is a simplification, and the speedup is beside the point.**
`CLAUDE.md` separates the two: a patch that makes the code plainer lands on
its evidence of equivalence alone. The ratio is large and the saving is not:

| spots | non-zeros | `cast_csr` + `unpack_adjacency` | this | ratio |
| ---: | ---: | ---: | ---: | ---: |
| 1,200 | 7,192 | 2.8 ms | 0.014 ms | 202 |
| 5,000 | 29,987 | 11.4 ms | 0.040 ms | 282 |
| 20,000 | 119,994 | 52.7 ms | 0.951 ms | 55 |

11 ms per outer iteration against a boundary that costs about 16 s
(`docs/`, issue #59 item 1's profile) is under a tenth of a per cent. Landing
it for the ratio would be reporting a number that does not matter; landing it
because two Python loops become three array expressions is the argument.

## `port.pipeline`

### FIGURE_DPI

Measured on one figure with four rasterized collections, written to PDF:

    dpi=300, tight bbox -- cnaster   2,033 ms   35.1 MB
    dpi=150, tight bbox                692 ms    9.9 MB   2.9x
    dpi=150, no tight bbox             489 ms    9.8 MB   4.2x
    dpi=300, tight, not rasterized   1,379 ms    2.1 MB

### T- #692 part 2: one genomic figure's peak

`tests/test_figure_memory.py` carries the table. On CalicoST easy
(`2d4ce9a9`), a captured `plot_clones_genomic` call replayed alone costs
+116 MB to draw under either implementation. Writing it costs +1,577 MB at
`cnaster`'s `write_fig` and +311 MB at this module's row (5.1x). Four calls
replayed twice in one process retain at most +94 MB, so no figure is left
open. `cnaster`'s 11.8-13.9 GB arms are the run's arrays plus the 1.6 GB
write.

### FIGURE_SWAPS

`write_fig` is 47 per cent of a run (#195).

Together they take a run's plotting from 20.34 s to 3.84 s and its renderer
buffers from 8,287 MB to 1,036 MB.

Merging the two would have bought the same 47 per cent and cost the claim.

### SHIFT_SWAPS

On #292's genome the fit returned `mu / Z_c`, state by state, rather than
the planted `mu` (#293).

### REFINEMENT_SWAPS

On `calicost_instance` it merged all 16 sub-clones into one (ARI 0.000).

#466: this said "on", the CLI never did.

## `port.patch.lattice`

### module docstring

It is not offered as a speedup and does not measure as one. At `K = 7`,
`G = 3,000`, `S = 50`, minimum over the rounds `pytest-benchmark` took:

| chain | pass | `cnaster` | this | ratio |
| --- | --- | ---: | ---: | ---: |
| unphased | forward | 5.098 ms | 4.339 ms | 1.18 |
| unphased | backward | 9.454 ms | 8.350 ms | 1.13 |
| phased | forward | 14.174 ms | 14.345 ms | 0.99 |
| phased | backward | 31.535 ms | 32.743 ms | 0.96 |

**0.96x to 1.18x, and none of it is the claim.** `CLAUDE.md` puts a speedup
at 2x measured at a stress size and this is nowhere near it in either
direction; what the table says is that one implementation for four costs
nothing. The unphased rows gain what the phased rows lose: the transition is
copied into a buffer once instead of being indexed out of `log_transmat` per
step, which helps where the transition is constant and is dead weight where
it is rebuilt per site anyway.

## `port.patch.hmrf.fused_field`

### module docstring

n_clones` fewer evaluations. Measured against item 1's reordered field:

| `n_obs` | `n_spots` | states | clones | two-step | fused | ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3,000 | 5,000 | 7 | 4 | 16,046 ms | 7,698 ms | 2.08 |
| 3,000 | 5,000 | 7 | 7 | 16,102 ms | 14,043 ms | 1.15 |
| 10,000 | 2,500 | 7 | 4 | 26,735 ms | 14,449 ms | 1.85 |

`CLAUDE.md` puts a speedup claim at 2x, and this clears it **only where
`n_clones < n_states`**. At `n_clones == n_states` there is no flop to save
and the 1.15x is the materialization alone. Stated rather than averaged:
the ratio is `n_states / n_clones` and a reader can compute their own.

Measured at `n_states = 7`, `n_obs = 30,000`, `n_spots = 5,000` on a machine
with 13 GB free: the fused form completed in **67.9 s**; the two-step
allocated its first 8.4 GB channel and the process was **killed by the
kernel** on the second. That is a size the two-step cannot run and this can,
which is a capability rather than a ratio -- and the kill is worth naming,
because an OOM death reads as infrastructure breaking rather than as a stated
limit.

## `port.patch.count_encoder`

`CountEncoder` against `cnaster`'s, on the fit's arrays at the planted
clones of the stream's r0 (`port.studies.stage.members`, `at_oracle_clones`),
7 states, one process, warm, minimum of 9; cnaster / port (T- #799):

| | dev_tree_1s_easy (`7ba9b01f`), n 7,244 | dev_tree (`3339b9a0`), n 10,484 |
| --- | --- | --- |
| NB codes | 6,799 | 9,844 |
| BB codes | 3,842 | 6,017 |
| Construct, ms (NB, BB) | 4.38 / 4.11, 4.82 / 4.48 | 6.89 / 6.47, 7.09 / 6.64 |
| Decode (7 x codes), ms | 0.120 / 0.039, 0.097 / 0.039 | 0.173 / 0.055, 0.145 / 0.055 |
| Encode (7 x n), ms | 0.112 / 0.109, 0.108 / 0.099 | 0.169 / 0.162, 0.154 / 0.143 |
| Map, bytes | 115,908 / 28,976 | 167,748 / 41,936 |

Decode bitwise, encode equal on both fixtures (0 difference). The decode is
3.1x at dev_tree, the map a quarter the bytes (an `int32` index against
float64 data, `int32` indices and an `indptr`); construction and encode are
within 10%: a simplification, not offered as a speedup.
