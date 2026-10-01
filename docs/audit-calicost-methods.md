# Audit: `cnaster` + port `--sal` against CalicoST, by method (#509)

**TL;DR:** outside the NB/BB emission formula, the 7-state fixed-diagonal
transition matrix, the nophasing lattice and the `merge_by_minspots` body,
the two pipelines differ at every stage. Four of the differences decide
what a benchmark ARI can attribute:

1. CalicoST's BAF stage starts from `n_clones` random rectangles.
   `cnaster`'s starts from `npart_phasing²` grid cells per slice, and it
   never reads `hmrf.n_clones`.
2. CalicoST fits a separate read-depth HMRF and HMM per BAF clone and
   finishes with one reassignment sweep over all clones. `cnaster` fits one
   global HMM, and its 7 states are shared by every read-depth clone.
3. The integer copy objectives differ. CalicoST's is L2 on `(mu, p)`,
   solved by hill climbing. `cnaster`'s live path is L1 solved as a MILP.
   port's is the NB/BB likelihood per bin, with purity and shift refitted.
4. `tests.sim_audit` could not read a finished CalicoST run (fixed in #507).
   No CalicoST copy ARI or integer clone ARI has been measured.

This audit comes from reading the code, not running it. Every "effect" below
is a hypothesis for the plan to measure.

CalicoST `c1abcae`; `cnaster` `4adad4d`; port at #507 (`4f7c8b9`); `sal`
0.3.0. Paths: `C/` = `calicost/src/calicost`, `N/` = `cnaster`, `P/` =
`python/port`. It extends `audit-cnaster-calicost-divergence.md`,
`audit-integer-copy-calicost.md` and `audit-logmu-shift-calicost.md`
(#131), and corrects one statement in the second of them (§8).

## 0. Which configuration is compared

| key | CalicoST shipped `configuration_cna` | #494 `--sal` run config | `run_calicost` aligned arm |
| --- | --- | --- | --- |
| initial clones | `n_clones` 3 | 2 × 2 grid per slice (`npart_phasing` 2) | 2 × 2 (`npart` aligned) |
| `n_clones_rdr` | 2 | 2 | 2 |
| `t` | 1 − 1e-5 | 1 − 1e-7 | config's |
| `max_iter`, `tol` | 30, 1e-4 | 100, 1e-3 | config's |
| `max_iter_outer` | 20 (stage 2: 10, hard-coded) | 25 | config's |
| `min_spots_per_clone` | 100 | 50 | config's |
| node potential | `weighted_sum` | `max` | `max` |
| spot pooling | `maxspots_pooling` 7 | 1 (hard-coded, `N/scripts/run_cnaster.py:531`) | 1 |
| NP merge threshold | 1.0 | 2.0 (hard-coded, `P/extensions/np_merge.py`) | −inf (off) |
| HMRF stop | ARI > 0.99 (hard-coded) | ARI ≥ 1.0 | CalicoST's 0.99 |

#494's `--shipped` arm uses the left-hand column, with `--no-align`. The
aligned arm (`P/scripts/run_calicost.py:131-199`) is the only
configuration-for-configuration comparison. Even there, `UNALIGNED`
(`:77-87`) lists 9 constants it cannot reach.

## 1. Loading and filtering

| item | CalicoST | `cnaster` (+ port) | verdict |
| --- | --- | --- | --- |
| spot filter | single slice: total UMI `>` threshold, SNP UMI `>=` (`C/utils_IO.py:66,72`); joint: both `>=` (`:231,239`) | both `>=` 50 (`N/io.py:659-670`); port bitwise (`P/patch/io.py`) | same, except spots exactly at the threshold on a single slice |
| expressed genes, Ig list, HLA regions, LOF (200 neighbours) | `C/utils_IO.py:78-115` | `N/io.py:698-836`, `N/filter.py:12-22` | same |
| SNP → gene | looks back 50 rows (`C/utils_IO.py:529`) | 100 rows (`N/omics.py:126,210`) | different, minor |

## 2. Segmentation and bins

| item | CalicoST | `cnaster` (+ port) | verdict |
| --- | --- | --- | --- |
| initial blocks | extended until SNP UMI ≥ 15, hard-coded (`C/utils_IO.py:540,596`) | ≥ `phasing_min_snp_umis` 50 (`N/omics.py:634-716`); port prefix sums, bitwise | different: blocks about 3× larger, so fewer phasing units |
| bin grouping | one criterion (SNP UMI ≥ `secondary_min_umi`); breaks at `max_binlength` 5 Mb regardless (`C/utils_IO.py:857-868`) | three criteria (gene, SNP 200, normal UMI 100); breaks at length only once all three are met (`N/omics.py:51-106`) | different: `cnaster` bins can exceed 5 Mb in sparse regions |
| re-binning after normal selection | none; bins with normal count < 20 have RDR zeroed and are kept (`C/calicost_main.py:146-152`) | `create_bin_ranges(key="bin_id")` re-merges to SNP 200 and normal 100 (`N/scripts/run_cnaster.py:983-1029`), then zeroes bins with normal count < 100 (`N/normal_spot.py:143-162`) | different: fewer, longer read-depth bins. Whether phasing breakpoints survive is to be measured (plan 2) |

## 3. Phasing

| item | CalicoST | `cnaster` (+ port) | verdict |
| --- | --- | --- | --- |
| genetic map → cM | chromosome cast to int, then sorted (`C/utils_phase_switch.py:20-22`) | chromosome sorted as a string, walked as int (`N/reference.py:100-102`, `N/recomb.py:74-107`); **port fixes this per contig** (`P/extensions/segments.py:316-341`) | `cnaster` defect; port matches CalicoST |
| switch probability floor | 1e-20 (`C/utils_phase_switch.py:46-64`) | `phasing.min_prob` 0.01 (`N/recomb.py:30-66`) | different |
| phasing HMM | one fit per partition clone: 5 states, 30 iterations, tol 1e-3; clones below 2e3 SNP UMIs skipped (`C/parse_input.py:100-101`, `C/phasing.py:60-76`) | all partition clones stacked in one fit sharing states: 7 states, 100 iterations (`N/phasing.py:94-136`); port Rust lattices, bitwise | different: shared `p_binom` across partitions; up to 3.3× the EM iterations |
| phase vote | weighted by spot count, balanced band 0.05 (`C/phasing.py:73-85`) | unweighted over clones, band 0.1 (`N/phasing.py:141-176`) | different |
| segment breaks | minor-BAF step > 0.1 (`C/phasing.py:88-95`) | ≥ 0.05 (`N/phasing.py:293-327`) | different |

## 4. Normal spots and baseline

| item | CalicoST | `cnaster` (+ port) | verdict |
| --- | --- | --- | --- |
| candidates | spots of the near-normal BAF clone below the 40th percentile of `std(log1p(X·smooth))`, raised in steps of 10 until they hold 200·n_bins UMIs (`C/calicost_main.py:116-125`) | same statistic. The UMI stop is `if False and` and the alternative compares against `prior_stdthreshold = inf`, never updated (`N/normal_spot.py:62-97`), so the loop ends at 100 %. `dev_tree` r0: "All 2317 spots for clone 1 considered to be normal". Not swapped | different: the whole near-normal clone. Harmless where that clone is pure (all three #494 fixtures); contaminating where it holds balanced-BAF tumour |
| normal-BAF bin filter | 5 %/95 % beta-binomial ppf (`C/utils_IO.py:970-998`) | configured CI (0.01, 0.99); port's cdf identity is bitwise (`P/patch/normal_spot.py`) | same test at a different CI |
| DE-gene filter | `filter_de_genes_tri`, applied (`C/utils_IO.py:1078-1155`) | computed, then made inert by a separator mismatch and an overwrite (`N/omics.py:262`, `N/normal_spot.py:904`, `N/scripts/run_cnaster.py:1031`); **port reconnects it** (#440) | `cnaster` differs; port matches CalicoST |

## 5. Spatial model

| item | CalicoST | `cnaster` | port `--sal` | verdict |
| --- | --- | --- | --- | --- |
| adjacency | kernel `exp(−(d²/bw)^5)`, `bw` grown until the median row sum is ≥ 6; weighted, symmetric, dense (`C/utils_hmrf.py:46-91`) | 8-NN, unit weights, directed (`N/spatial.py:278-327`) | 6-NN on the Visium hex lattice, 8-NN on square grids, sparse (`P/patch/spatial.py:426-487`) | different in all three: Potts coupling differs at equal `spatial_weight` |
| spot pooling | `smooth_mat`: rings grown to about 7 spots (`C/utils_hmrf.py:30-90`) | identity | identity | different from shipped: each node potential comes from 1 spot, not about 7 |
| across-slice edges | PASTE alignment, if given (`C/utils_IO.py:202-228`) | `NotImplementedError` (`N/io.py:461-467`) | — | not exercised |

## 6. Clone and copy-state inference

| item | CalicoST | `cnaster` | port `--sal` | verdict |
| --- | --- | --- | --- | --- |
| BAF-stage start | `n_clones` Dirichlet(10) rectangles, seed 0 (`C/utils_hmrf.py:177-231`) | `npart²` linspace grid per slice (`N/scripts/run_cnaster.py:250-276`) | same as `cnaster`; initializer bitwise | different |
| HMM fit | Baum–Welch; M step statsmodels Nelder–Mead from 2 starts, maxiter 1,500 (`C/utils_hmm.py:869-970`); stops at `tol` | one BFGS over the expected complete log-likelihood, finite-difference gradient, E step every 2nd iteration (`N/hmm_nophasing.py:788-1035`) | analytic gradient (`P/patch/hmm_nophasing/gradient.py`) | different optimizer. Same local objective |
| HMM start | GMM, 7 components, `max_iter` 1, BAF clipped to [0.1, 0.9] (`C/utils_hmm.py:163-225`) | GMM, 28 components → 14 → 7 (`N/hmm_initialize.py:534-735`) | BAF stage `distinct` (#348); read-depth stage `sal` count-pair mixture, `kmeans++x5+em`, 160 s budget (#489) | different |
| dispersion start | α 0.1, τ 30; after stage 2 all clones are set to max α and min τ (`C/calicost_main.py:266-267`) | α 0.5, τ 1,000 | same as `cnaster` | different start |
| RDR cap | none | bins with RDR > 5 are excluded from the fit (`N/hmm_nophasing.py:808`) | inherited | different: high amplifications drop out of the fit |
| per-clone library shift | only on the `_mix` path | not supported | per-clone `logmu_shift`, neutral state pinned to μ = 1 (#370/#375) | different; see `audit-logmu-shift-calicost.md` |
| node potential | `weighted_sum`: logsumexp over states under γ (`C/hmrf.py:118-150`) | `max`: along the argmax path (`N/hmrf.py:103-171`) | `max`, fused, plus a 100-nat refinement-mask penalty | different from shipped |
| prior | Potts + log clone frequency, floor −50 (`C/hmrf.py:514-523`) | Potts only | Potts only | different: CalicoST penalizes small clones |
| label solver | one Gauss–Seidel sweep per outer iteration, a Python loop (`C/hmrf.py:128-147`) | ICM to convergence; fixed floor of 200 with random reassignment; greedy pair merges (`N/icm.py`) | α-expansion (Rust) fused with ICM, then floor merge at `min_spots_per_clone` (#492) | different: CalicoST takes 1 sweep, the others minimize the energy |
| read-depth stage | per BAF clone with ≥ 20·n_obs B reads: its own HMRF + HMM, its own 7 states; NP and minspots merges; HMM refit; `combine_similar_states_across_clones` (0.1); final reassignment sweep over all clones (`C/calicost_main.py:163-297`) | one global `run_core_inference` over n_baf × `n_clones_rdr` clones, one 7-state table; allowed-clone mask computed and dropped (`:1105`); minspots merge only plotted; no refit, no final sweep | mask kept as a penalty; NP merge (#497) across all sub-clones, with no refit after it | different: a shared state table caps the number of distinct copy states |
| NP merge | threshold 1.0, minlength 10, max-clique grouping (`C/hmm_NB_BB_phaseswitch.py:643-722`) | commented out (`N/scripts/run_cnaster.py:743-758`) | ported; threshold 2.0, plus a short-event evidence rule; can merge sub-clones of different BAF clones | port partially matches CalicoST |

## 7. Normal clone

- **CalicoST:** clone 0 is the BAF clone closest to balanced (`C/utils_hmrf.py:345-381`). The integer decode forces to (1, 1) the balanced state with the smallest `log_mu` that holds ≥ 10 % of bins (`C/find_integer_copynumber.py:73-81`).
- **`cnaster`:** clone ordering is the same (`N/hmrf.py:802-891`). The integer decode forces the balanced state with `mu` nearest 1 and has no share floor. #362 shows this decodes tumour (2, 2) as (1, 1).
- **port:** pins the neutral state at the fit (μ = 1) and holds the most-balanced clone at purity 1 and shift 0 in the lattice decode (`P/extensions/copy_likelihood.py:625-646`).

## 8. Integer copy number

| item | CalicoST | `cnaster` (live path) | port (default, `--sal` or not) |
| --- | --- | --- | --- |
| objective | **L2**: `(0.3(mu − frac_rdr))² + (p − frac_baf)²`, weighted by bins per state, plus order and ploidy penalties (`C/find_integer_copynumber.py:100-115`) | **L1**: `0.3\|mu − frac_rdr\| + \|p − frac_baf\|` (`N/integer_copy.py:571-599,661-663`, `cost_type="L1"`) | NB + BB log-likelihood of the clone's pseudobulk counts, plus `−0.5\|A+B−2\|` per bin (`P/extensions/copy_likelihood.py:101-148,229-231`) |
| solver | 20 restarts × ≤ 10 hill-climb sweeps; local optimum | scipy `milp`, one solve per ploidy 1–4; exact | Viterbi over the (A, B) lattice with EM on dispersions; Brent on shift and purity |
| unit | one pair per state per clone | one pair per state per clone | **one pair per bin** per clone (`PairsByBin`) |
| purity | none without a `tumorprop_file` | none | one fraction per clone, fitted |
| caps | A + B ≤ 6, allele ≤ 5 | same | total from `int_copy_num.max_total_copy` (6 here); allele cap not applied, so (6, 0) is allowed |

**Correction to `audit-integer-copy-calicost.md`.** That audit says
`cnaster` "changed the solver and kept the objective". The squared form it
quotes is `cnaster`'s hill climb. The function `run_cnaster` calls,
`hill_climbing_integer_copynumber_fixdiploid_milp`
(`N/scripts/run_cnaster.py:1406`), defaults to `cost_type="L1"`. The live
objective is therefore L1, where CalicoST's is L2. This is only material
under `--no-patch`: port's decode replaces both.

Shipped `nonbalance_bafdist 1.0` and `nondiploid_rdrdist 10.0` never apply
in either program, since |p − 0.5| ≤ 0.5 and μ > 11 is needed.

## 9. Runtime by construction

| | CalicoST | `cnaster` | port `--sal` |
| --- | --- | --- | --- |
| HMM fits | stage 1: ≤ 20; stage 2: per BAF clone ≤ 10, + 1 refit | ≤ (outer + 2) per stage, 2 stages | same as `cnaster` |
| per fit | ≤ 30 EM iterations; 4 Nelder–Mead M-step fits per iteration | ≤ 100 BFGS iterations × about P + 1 objective evaluations (finite differences) | 1 evaluation per iteration (analytic gradient); Rust lattices |
| label step | Python loop over N spots, each making 2 × n_states scipy log-pmf calls over n_obs bins; 1 sweep per outer iteration | numba field; numba ICM; Python `merge_assignment`, O(N · nnz) per call (`N/icm.py:207`) | fused numba field; Rust max-flow |
| adjacency | dense O(N²), rebuilt up to 24 times | dense | sparse |
| one-off | — | — | `sal` start ≤ 160 s per read-depth call; lattice decode |

Measured in #494 (4 cores, `--time-stages`, one core in parentheses):

- **port `--sal` on `dev_tree` 60 × 50 (6,000 spots):** 239 s (235 s).
  - `sal` start: 53 s.
  - lattice decode: 48 s.
  - clone assignment: 32 s.
  - `filter_normal_diffexp`: 22 s. RSS peaks here, at 6.27 GB.
- **CalicoST shipped on the same sample:** loading took 1,006 s. Its BAF-stage HMRF rounds took 250–1,300 s each. It finished 2 rounds in 30 min and 3 in 48 min.
  - Most of that cost is the Python per-spot label loop and the dense adjacency. That attribution is to be profiled (plan 4).

## 10. Scoring

`tests.sim_audit.score` scores both tools on each tool's own `cnv_seglevel.tsv`:

- clone ARI, and integer clone ARI (clones with identical (A, B) at every bin merged);
- copy ARI over Hungarian-matched clone-bins (phased `A·1000 + B`, not length-weighted);
- exact altered, and its phase-free variant.

The asymmetries:

| # | issue | favours |
| --- | --- | --- |
| a | CalicoST's `clone_labels.tsv` is indexed `BARCODES`; the reader expected `barcode`. **Fixed in #507** (`tests/test_sim_audit_reader.py`) | blocked CalicoST |
| b | a clone whose integer fit CalicoST skipped has no columns. **Fixed in #507**: reads as −1 | blocked CalicoST |
| c | each tool is scored on its own bins, so denominators differ in count and bin length | either |
| d | only matched planted clones enter copy ARI; at `n_clones 3` against 4 planted, one planted clone leaves the copy score | CalicoST |
| e | copy ARI uses phased pairs; phase-free exact is reported beside it, phase-free copy ARI is not | the tool whose phase matches the planted A |
| f | copy ARI is dominated by the ~96 % neutral bins: an altered bin decoded (1, 1) creates ~6,500 wrong agreeing pairs (#494: easy 0.8984 → 0.9861 if its 39 such bins were right) | the tool with fewer altered-as-neutral bins |
| g | unfinished CalicoST runs are scored at the BAF stage | port |

## Upstream correspondence

None of these differences needs a change in `snakes_and_ladders`. The
label solver and the HMM start already use it (#492, #489). The remaining
upstream candidates are:

- a pooled (`weighted_sum`) node potential expressed as a `sal` field;
- a count-pair mixture weight for #502.

The `cnaster` defects listed in §2–§4 and §6 belong to `cnaster`, reported
here and not landed (CLAUDE.md, dependency repositories are read only).
