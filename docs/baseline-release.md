# Baseline release: `run_cnaster_port --sal` at `9a47d97`

**TL;DR:** 18 fixtures, one commit, idle host. `--sal` recovers the planted
4 clones at integer clone ARI 0.937–1.0 on 15; `dev_tree_1s_hard` fails at
0.2187, 0.2217 and 0.0 (2, 2 and 1 integer clones; #575). Losses decode as
copy-neutral LOH and balanced gains as neutral (#471, #573). Wall 99–274 s,
peak ≤ 6.15 GB; against `efdebfd` on 14 shared fixtures the wall ratio is
0.79–1.02 at identical scores. CalicoST: `docs/calicost-benchmark.md`.

## Conditions

- port at `9a47d97` (`origin/main` `dbfff01` merged), `uv.lock` as
  committed; Intel Xeon @ 2.80 GHz, 4 cores, `NUMBA_NUM_THREADS` default.
- Serial, one job at a time under a host lock. Each run waited (≤ 5 min)
  for 1-minute load < 0.5: load1 at start 0.42–0.48, at end 1.01–1.56 (the
  run's own load). Runs 2026-10-01 21:02–22:15 UTC.
- Numba warm: an untimed easy run first took 99.6 s, against 99.2 s for the
  timed one, so no stage paid a compile.
- `python -m tests.sim_audit --sample SAMPLE --root ROOT -- --sal --no-plots`,
  scored by `tests.sim_audit.score`. Peak is the child's maximum RSS.
- CalicoST easy and hard are the shipped samples. `dev_tree*` are drawn by
  `python -m port.sim.draw sim/manifests/baseline/st_*.toml` (r0–r2) and,
  for the lognormal segment lengths of #621, by `sim/manifests/<name>.toml`
  (r0). The hash is `port.sim.fixtures.realization_hash`; the 12 `st_*`
  draws reproduce the hashes recorded at `ac00498`, and
  `dev_tree_1s_easy_ln_r0` the `22a5eb85` of #627.
- Every run is in `docs/metrics/` at `9a47d97`, under the fixture names
  below. The three r0 hashes the ledger already named keep those names
  (`dev_tree_1s_r0`, `dev_tree_1s_easy_r0`, `dev_tree_1s_hard_r0`): one hash,
  one name (#588).

## Results

Metric names are the ledger's (`port.qa.ledger.METRICS`); `_pf` is phase-free,
`—` a class not planted. `copy_ari_bgain` is undefined on every fixture
(one balanced pair planted). Clones: fitted / integer.

### CalicoST samples

| fixture | hash | clone_ari | clone_ari_int | copy_ari | copy_ari_pf | copy_ari_loh | copy_ari_loh_pf | copy_ari_ugain | copy_ari_ugain_pf | state_ari | exact | exact_altered | exact_altered_pf | exact_loh | exact_loh_pf | exact_bgain | exact_bgain_pf | exact_ugain | exact_ugain_pf | exact_neutral | clones (fit/int) | bins | wall_s | peak_gb | load1 start–end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `easy` | `2d4ce9a9` | 0.9861 | 0.9861 | 0.9035 | 0.9031 | 0.4515 | 0.2090 | 0.9782 | 0.9632 | 0.0273 | 0.9706 | 0.2667 | 0.7231 | 0.5474 | 0.7263 | 0.0000 | 0.0000 | 0.0000 | 0.9863 | 0.9992 | 4/4 | 1248 | 99.2 | 3.22 | 0.45–1.21 |
| `hard` | `8797710b` | 0.9829 | 0.9829 | 0.9181 | 0.9181 | 0.2413 | 0.2158 | 0.7341 | 0.7803 | 0.0278 | 0.9780 | 0.1654 | 0.7244 | 0.2195 | 0.6707 | 0.0000 | 0.0000 | 0.0750 | 0.9250 | 0.9990 | 4/4 | 1263 | 101.8 | 3.22 | 0.44–1.17 |

### `dev_tree` 60 × 50, two slices

| fixture | hash | clone_ari | clone_ari_int | copy_ari | copy_ari_pf | copy_ari_loh | copy_ari_loh_pf | copy_ari_ugain | copy_ari_ugain_pf | state_ari | exact | exact_altered | exact_altered_pf | exact_loh | exact_loh_pf | exact_bgain | exact_bgain_pf | exact_ugain | exact_ugain_pf | exact_neutral | clones (fit/int) | bins | wall_s | peak_gb | load1 start–end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dev_tree_r0` | `3381575a` | 1.0000 | 1.0000 | 0.9825 | 0.9825 | 0.7654 | 0.7755 | — | — | 0.0787 | 0.9948 | 0.9040 | 0.9213 | 0.9128 | 0.9302 | 0.0000 | 0.0000 | — | — | 0.9995 | 4/4 | 2624 | 253.4 | 6.15 | 0.43–1.08 |
| `dev_tree_r1` | `563661f1` | 0.9993 | 0.9993 | 0.9765 | 0.9765 | 0.8970 | 0.9048 | — | — | 0.0637 | 0.9966 | 0.9442 | 0.9572 | 0.9531 | 0.9662 | 0.0000 | 0.0000 | — | — | 0.9995 | 4/4 | 2608 | 259.4 | 6.05 | 0.43–1.12 |
| `dev_tree_r2` | `c7f1ec6b` | 0.9986 | 0.9986 | 0.9784 | 0.9784 | 0.7371 | 0.7403 | — | — | 0.0872 | 0.9941 | 0.8935 | 0.9065 | 0.9053 | 0.9186 | 0.0000 | 0.0000 | — | — | 0.9996 | 4/4 | 2600 | 261.5 | 6.12 | 0.48–1.06 |

### `dev_tree_1s`

| fixture | hash | clone_ari | clone_ari_int | copy_ari | copy_ari_pf | copy_ari_loh | copy_ari_loh_pf | copy_ari_ugain | copy_ari_ugain_pf | state_ari | exact | exact_altered | exact_altered_pf | exact_loh | exact_loh_pf | exact_bgain | exact_bgain_pf | exact_ugain | exact_ugain_pf | exact_neutral | clones (fit/int) | bins | wall_s | peak_gb | load1 start–end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dev_tree_1s_r0` | `4687b541` | 0.9986 | 0.9986 | 0.9743 | 0.9744 | 0.6899 | 0.6972 | — | — | 0.0635 | 0.9925 | 0.8661 | 0.8845 | 0.8753 | 0.8939 | 0.0000 | 0.0000 | — | — | 0.9994 | 4/4 | 1835 | 140.1 | 3.65 | 0.46–1.32 |
| `dev_tree_1s_r1` | `f5585f31` | 0.9994 | 0.9994 | 0.9731 | 0.9731 | 0.7667 | 0.7744 | — | — | 0.0902 | 0.9941 | 0.8987 | 0.9200 | 0.9084 | 0.9299 | 0.0000 | 0.0000 | — | — | 0.9993 | 4/4 | 1809 | 138.5 | 3.65 | 0.48–1.38 |
| `dev_tree_1s_r2` | `479beca0` | 0.9980 | 0.9980 | 0.9804 | 0.9804 | 0.7543 | 0.7555 | — | — | 0.0678 | 0.9944 | 0.9027 | 0.9216 | 0.9126 | 0.9317 | 0.0000 | 0.0000 | — | — | 0.9993 | 4/4 | 1829 | 164.3 | 3.65 | 0.47–1.26 |

### `dev_tree_1s_easy` (`st_easy_bb01`)

| fixture | hash | clone_ari | clone_ari_int | copy_ari | copy_ari_pf | copy_ari_loh | copy_ari_loh_pf | copy_ari_ugain | copy_ari_ugain_pf | state_ari | exact | exact_altered | exact_altered_pf | exact_loh | exact_loh_pf | exact_bgain | exact_bgain_pf | exact_ugain | exact_ugain_pf | exact_neutral | clones (fit/int) | bins | wall_s | peak_gb | load1 start–end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dev_tree_1s_easy_r0` | `d08e3a1b` | 0.9721 | 0.9721 | 0.9730 | 0.9752 | 0.0114 | — | — | — | 0.0240 | 0.9842 | 0.5246 | 0.9549 | 0.4278 | 0.9778 | 0.8793 | 0.8793 | 0.0000 | 1.0000 | 1.0000 | 4/4 | 1838 | 188.2 | 3.65 | 0.42–1.23 |
| `dev_tree_1s_easy_bb01_r1` | `8d0bbfda` | 0.9710 | 0.9710 | 0.9622 | 0.9643 | 0.0252 | — | — | — | 0.0315 | 0.9821 | 0.4919 | 0.9512 | 0.3867 | 0.9724 | 0.8793 | 0.8793 | 0.0000 | 1.0000 | 0.9994 | 4/4 | 1801 | 125.2 | 3.64 | 0.42–1.17 |
| `dev_tree_1s_easy_bb01_r2` | `e2ce4e06` | 0.9710 | 0.9710 | 0.9868 | 0.9888 | 0.0244 | — | — | — | 0.0272 | 0.9839 | 0.5184 | 0.9796 | 0.3771 | 0.9771 | 0.9839 | 0.9839 | 0.0000 | 1.0000 | 1.0000 | 4/4 | 1830 | 171.4 | 3.65 | 0.48–1.12 |

### `dev_tree_1s_hard` (`st_hard_bb01`)

| fixture | hash | clone_ari | clone_ari_int | copy_ari | copy_ari_pf | copy_ari_loh | copy_ari_loh_pf | copy_ari_ugain | copy_ari_ugain_pf | state_ari | exact | exact_altered | exact_altered_pf | exact_loh | exact_loh_pf | exact_bgain | exact_bgain_pf | exact_ugain | exact_ugain_pf | exact_neutral | clones (fit/int) | bins | wall_s | peak_gb | load1 start–end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dev_tree_1s_hard_r0` | `d2938975` | 0.2187 | 0.2187 | 0.4019 | 0.4018 | 0.1705 | — | -0.0212 | 0.1457 | 0.0021 | 0.9847 | 0.0000 | 0.1358 | 0.0000 | 0.2609 | 0.0000 | 0.0000 | 0.0000 | 0.1852 | 0.9998 | 2/2 | 1786 | 177.9 | 3.65 | 0.48–1.56 |
| `dev_tree_1s_hard_bb01_r1` | `764709dc` | 0.2217 | 0.2217 | 0.3820 | 0.3818 | 0.3694 | — | 0.0190 | 0.2221 | 0.0067 | 0.9755 | 0.0000 | 0.2466 | 0.0000 | 0.6364 | 0.0000 | 0.0000 | 0.0000 | 0.1667 | 0.9997 | 2/2 | 1510 | 139.6 | 3.65 | 0.43–1.47 |
| `dev_tree_1s_hard_bb01_r2` | `3adf249a` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | — | — | 0.0000 | 0.0000 | -0.0035 | 0.9814 | 0.0000 | 0.0000 | — | — | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1/1 | 2044 | 116.5 | 3.60 | 0.45–1.01 |

### Lognormal segment lengths (#621), r0

| fixture | hash | clone_ari | clone_ari_int | copy_ari | copy_ari_pf | copy_ari_loh | copy_ari_loh_pf | copy_ari_ugain | copy_ari_ugain_pf | state_ari | exact | exact_altered | exact_altered_pf | exact_loh | exact_loh_pf | exact_bgain | exact_bgain_pf | exact_ugain | exact_ugain_pf | exact_neutral | clones (fit/int) | bins | wall_s | peak_gb | load1 start–end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dev_tree_ln_r0` | `3339b9a0` | 0.9976 | 0.9976 | 0.9870 | 0.9876 | 0.4648 | 0.7175 | 1.0000 | 1.0000 | 0.0744 | 0.9863 | 0.7778 | 0.9492 | 0.6697 | 0.9039 | — | — | 0.8990 | 1.0000 | 0.9996 | 4/4 | 2619 | 274.2 | 6.13 | 0.44–1.31 |
| `dev_tree_1s_ln_r0` | `df3cc0ab` | 0.9973 | 0.9973 | 0.9780 | 0.9787 | 0.5056 | 0.8087 | 0.9360 | 0.9360 | 0.0607 | 0.9865 | 0.7679 | 0.9506 | 0.6667 | 0.9189 | — | — | 0.8907 | 0.9891 | 0.9994 | 4/4 | 1819 | 168.5 | 3.64 | 0.48–1.21 |
| `dev_tree_1s_easy_ln_r0` | `22a5eb85` | 0.9780 | 0.9780 | 0.9768 | 0.9784 | -0.0069 | — | — | — | 0.0285 | 0.9854 | 0.5310 | 0.9646 | 0.5867 | 0.9733 | 0.8889 | 0.8889 | 0.0000 | 1.0000 | 0.9999 | 4/4 | 1831 | 172.0 | 3.64 | 0.44–1.35 |
| `dev_tree_1s_hard_ln_r0` | `9ec90dc2` | 0.9372 | 0.9372 | 0.7618 | 0.7614 | 0.0769 | — | 0.6844 | 0.6438 | 0.0156 | 0.9753 | 0.4034 | 0.6345 | 0.4783 | 0.7826 | 0.2300 | 0.2300 | 0.4970 | 0.8563 | 0.9991 | 4/4 | 1811 | 124.0 | 3.65 | 0.45–1.31 |

## Runtime against `efdebfd`

`efdebfd` ran the same 14 fixtures on 2026-10-01 16:53–17:35 UTC with other
agents on the host, so its walls are confounded; the ledger notes this.
Every score but `state_ari` (2 fixtures, a state relabelling) is identical
between the two commits, so the ratio compares the same output.

| fixture | wall_s `efdebfd` | wall_s `9a47d97` | ratio |
| --- | --- | --- | --- |
| `easy` | 107.0 | 99.2 | 0.93 |
| `hard` | 108.3 | 101.8 | 0.94 |
| `dev_tree_r0` | 282.8 | 253.4 | 0.90 |
| `dev_tree_r1` | 319.6 | 259.4 | 0.81 |
| `dev_tree_r2` | 272.6 | 261.5 | 0.96 |
| `dev_tree_1s_r0` | 137.9 | 140.1 | 1.02 |
| `dev_tree_1s_r1` | 139.9 | 138.5 | 0.99 |
| `dev_tree_1s_r2` | 187.9 | 164.3 | 0.87 |
| `dev_tree_1s_easy_r0` | 192.6 | 188.2 | 0.98 |
| `dev_tree_1s_easy_bb01_r1` | 127.5 | 125.2 | 0.98 |
| `dev_tree_1s_easy_bb01_r2` | 215.8 | 171.4 | 0.79 |
| `dev_tree_1s_hard_r0` | 181.3 | 177.9 | 0.98 |
| `dev_tree_1s_hard_bb01_r1` | 146.2 | 139.6 | 0.96 |
| `dev_tree_1s_hard_bb01_r2` | 123.0 | 116.5 | 0.95 |
| total | 2542.6 | 2337.1 | 0.92 |

## Failures and causes

A failure is integer clone ARI < 0.9, or exact altered < 0.8 phased or
< 0.9 phase-free. Evidence is the planted → decoded `(A, B)` confusion in
each run's `SIM` record.

- **`dev_tree_1s_hard` r0–r2: clones.** 2, 2 and 1 integer clones of 4
  (ARI 0.2187, 0.2217, 0.0). The integer-clone merge collapses them (#575);
  r2 merges all four, and every bin decodes `(1, 1)`. In r0 and r1 the
  planted gains `(1, 2)`, `(2, 1)`, `(2, 2)` decode `(1, 1)` at 1.0, so
  exact altered is 0.000; that follows from the merge.
- **Losses decoded as copy-neutral LOH (#471).** Planted `(0, 1)` / `(1, 0)`
  / `(2, 0)` → `(0, 2)`: easy 1.0, hard 1.0 and 0.82; `dev_tree_1s` r0–r2
  0.15–0.24 of `(0, 1)`, which puts r0 at 0.8845 phase-free.
- **Balanced gains decoded as neutral (#573).** Planted `(2, 2)` → `(1, 1)`
  at 1.0 on easy, hard, all of `dev_tree` and `dev_tree_1s`;
  0.016–0.12 on `dev_tree_1s_easy`; 0.77 on `dev_tree_1s_hard_ln_r0`.
- **Phase orientation.** Exact altered 0.49–0.53 phased against 0.95–0.98
  phase-free on `dev_tree_1s_easy` r0–r2 and `dev_tree_1s_easy_ln_r0`,
  0.77–0.78 against 0.95 on `dev_tree_ln_r0` and `dev_tree_1s_ln_r0`:
  planted `(1, 0)` → `(0, 1)` and `(3, 1)` → `(1, 3)` at 0.85–1.0. Easy
  and hard add `(1, 2)` → `(2, 1)` at 1.0 and 0.90 of `(2, 1)` → `(1, 2)`.
  The copies are right and the haplotype label is swapped. Cause not
  established.
- **`dev_tree_1s_hard_ln_r0`: unbalanced gains.** Exact altered 0.634
  phase-free; planted `(2, 1)` → `(1, 1)` at 0.77. Cause not established.

## CalicoST

`docs/calicost-benchmark.md`, from committed outputs, no rerun: on its
shipped configuration CalicoST finishes none of easy, hard and `dev_tree_r0`
in 1,800 s. Uncapped, with `n_clones 5`, it takes 20,243 s on `dev_tree_r0`
for clone ARI 0.8538 (6 clones), copy ARI 0.9075 and exact altered 0.7095,
against 253.4 s, 1.0 (4), 0.9825 and 0.9040 for `--sal` above.

## Known failures at this release

- `dev_tree_1s_hard`: 2, 2 and 1 integer clones of 4 (#575).
- `--sal --no-shift` stops in the integer decode with no captured fit (#576).
- `cnaster --no-patch` stops on a `bin_id` index (#105).

## `snakes_and_ladders` `3ad4b04` → `b61dfba` (T- #632 PR A)

**TL;DR:** every scored metric equals PR #631's (`9a47d97`) on 3 fixtures;
clone labels and integer A/B copy numbers are bitwise unchanged. The HMM's
fitted parameters move, by at most 1.6e-2 relative (τ, easy), through sal
#1136's beta-binomial density above shape 100. Wall is 1.03–1.06× the old
pin on one host, one run each, which this measurement does not establish.

- Host: Xeon @ 2.10 GHz, 4 cores (PR #631 ran on a 2.80 GHz host). Each pin
  ran here, serially under the host lock, numba warmed by an untimed easy
  run; load1 0.44–0.49 at each start. The old pin (`dbfff01`) reproduces
  PR #631's output files bitwise, so the host moves wall and nothing else.
- Same command and scoring as above. Ledger: `6a9f3a8-*` (b61dfba) and
  `dbfff01-*` (3ad4b04 control), 2026-10-01 23:47–00:00 UTC.

| fixture | hash | clone_ari | clone_ari_int | copy_ari | copy_ari_pf | exact_altered | exact_altered_pf | bins | wall_s #631 / 3ad4b04 / b61dfba | peak_gb #631 / 3ad4b04 / b61dfba |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `easy` | `2d4ce9a9` | 0.9861 | 0.9861 | 0.9035 | 0.9031 | 0.2667 | 0.7231 | 1248 | 99.2 / 57.2 / 60.4 | 3.22 / 3.19 / 3.20 |
| `hard` | `8797710b` | 0.9829 | 0.9829 | 0.9181 | 0.9181 | 0.1654 | 0.7244 | 1263 | 101.8 / 62.0 / 64.1 | 3.22 / 3.19 / 3.19 |
| `dev_tree_1s_easy_ln_r0` | `22a5eb85` | 0.9780 | 0.9780 | 0.9768 | 0.9784 | 0.5310 | 0.9646 | 1831 | 172.0 / 102.2 / 105.3 | 3.64 / 3.62 / 3.62 |

Scores are one column because old and new are equal to 4 decimals on all
24 scored values, `clone_of` and the confusion tables. What moved, b61dfba
against the same-host 3ad4b04 run (`rdrbaf_final_nstates7_smp.npz`):

| fixture | log-likelihood | τ (BB), max rel | log μ, max abs | HMM state of a bin × clone |
| --- | --- | --- | --- | --- |
| `easy` | −50198.673 → −50198.947 | 1.6e-2 (6006 → 6099) | 2.1e-3 | 187 of 4992 differ; A, B equal |
| `hard` | Δ 1.5e-6 | 7.8e-8 | 1.3e-7 | equal |
| `dev_tree_1s_easy_ln_r0` | Δ 7.2e-8 | 5.4e-9 | 8.2e-10 | equal |

τ is 6,006, 1,524 and 1,211, above sal's Stirling-difference threshold of
100, so the beta-binomial density is the stage that moved; the
negative-binomial shape 1/α is 2.2, 2.1 and 11.3, below it. The log-likelihoods are
of two density implementations and are not ranked against each other.

## `snakes_and_ladders` `b61dfba` → `253c84f` (T- #671)

**TL;DR:** on 3 fixtures every `--sal` score equals the old pin's, and on the
two simulated ones every output `.tsv` and `.npz` is bitwise equal (12 of 12
files each). Wall is 1.04–1.06× the old pin, one run each, which does not
establish a ratio.

- Host: 4 cores; each pin ran serially under the host lock, numba warmed by
  an untimed `dev` run, load1 1.3–2.1 at each start, 2026-10-05
  22:35–23:07 UTC. The old pin ran `cb0baca` (`main`), the new one the
  T- #671 branch at `10888f7`, whose `python/port/` differs from `main`
  only in a sandbox docstring.
- `dev` is `tests.recovery_audit --instance dev --states 8 --outer 1
  --iterations 3 -- --sal`; the others are `tests.sim_audit -- --sal
  --no-plots`. The scores also equal PR- #656's tier R at `e112e0a`.

| fixture | hash | clone_ari | clone_ari_int | copy_ari | copy_ari_pf | exact_altered | exact_altered_pf | bins | wall_s b61dfba / 253c84f |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dev` | `07b82e92` | 1.0 | 1.0 | 0.9992 | — | — | — | — | 111.9 / 117.0 |
| `dev_tree_1s_easy_r0` | `d08e3a1b` | 0.9721 | 0.9721 | 0.9730 | 0.9752 | 0.5246 | 0.9549 | 1838 | 140.2 / 146.3 |
| `dev_tree_1s_hard_r0` | `d2938975` | 0.2187 | 0.2187 | 0.4019 | 0.4018 | 0.0 | 0.1358 | 1786 | 133.0 / 141.1 |

## `snakes_and_ladders` `253c84f` → `006e49d` (T- #707)

**TL;DR:** on `dev_tree_1s_easy_r0` and `dev_tree_1s_hard_r0` every `--sal`
score and all 12 output `.tsv`/`.npz` files are bitwise equal. On `dev` the
scores are equal; fitted `mu` and `p` differ at the 3rd decimal, inside the
spread of 3 runs of the old pin (T- #696).

- Host: 4 cores; one fixture at a time under the host lock, 2026-10-06
  15:38–16:29 UTC, load1 1.1–5.0 at each start. The old pin ran `66111c9`
  (`main`), the new one the bump alone at `69be257`. Both read the same
  drawn samples. Commands as for T- #671 above.
- `dev`, fitted `mu` of state 1: old pin 2.0158, 1.9935 and 2.0262 (3
  runs); new pin 2.0105. `mu_error_mean` 0.0827–0.0836 old, 0.0831 new.
  Every other score field is equal across the 4 runs.

| fixture | hash | clone_ari | clone_ari_int | copy_ari | copy_ari_pf | exact | exact_altered | bins | wall_s 253c84f / 006e49d |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dev` | `07b82e92` | 1.0 | 1.0 | 0.9992 | — | — | — | — | 113.2 / 114.9 |
| `dev_tree_1s_easy_r0` | `7ba9b01f` | 0.9798 | 0.9798 | 0.9326 | 0.9335 | 0.9690 | 0.1473 | 1816 | 121.6 / 114.0 |
| `dev_tree_1s_hard_r0` | `9ec90dc2` | 0.9372 | 0.9372 | 0.7618 | 0.7614 | 0.9753 | 0.4034 | 1811 | 87.4 / 88.0 |
