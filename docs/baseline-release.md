# Baseline release: `run_cnaster_port --sal` on `main` `ac00498`

**TL;DR:** (#577) `--sal` recovers the planted clones at ARI ≥ 0.969 on 5 of 6
fixtures, in 67–264 s and ≤ 6.1 GB; CalicoST finishes none of the three it was
run on within 30 minutes (`docs/final-benchmark.md`). `dev_tree_1s_hard`
fails at 0.21 (2 clones of 4), by the integer-clone merge (#575). Balanced
gains are recovered at 0.000 on five fixtures and 0.878 on
`dev_tree_1s_easy` r0 (#573).

## Conditions

- port at `ac00498` (#545), `uv.lock` as committed; 4 cores, `NUMBA_NUM_THREADS`
  default; one job at a time under a host lock, each started at 1-minute load
  below 3.
- `python -m tests.sim_audit --sample SAMPLE --root ROOT -- --sal --no-plots`,
  scored by `tests.sim_audit.score`. The numba cache is warm: a first,
  untimed easy run compiled it (287 s). Peak is the child's maximum RSS.
- CalicoST easy and hard are the shipped samples (`tests.sim_audit.SAMPLES`),
  run three times. The others are drawn by `python -m port.sim.draw
  sim/manifests/baseline/<manifest> --into DIR`, three realizations each, run
  once.
- `sim/manifests/baseline/` holds the manifests as run. `dev_tree_1s*` are
  #556's (#558), at the gamma counts sampler `main` carries rather than #549's
  urn; `st_*` wrap each at 3 realizations. The fixture hash is
  `tests.sim_stages.realization_hash` of the drawn realization; redrawing
  `st_dt` and `st_hard_bb01` from this directory reproduces all six hashes.

## Results

Clone ARI (clones), integer copy ARI, exact altered (phase-free), wall, peak.

| fixture | manifest | r | hash | clone ARI | copy ARI | exact altered | wall [s] | peak [GB] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CalicoST easy | shipped | ×3 | `2d4ce9a9` | 0.9861 (4) | 0.8984 | 0.252 (0.630) | 74.2 / 69.9 / 73.2 | 3.2 |
| CalicoST hard | shipped | ×3 | `8797710b` | 0.9829 (4) | 0.9055 | 0.036 (0.627) | 69.7 / 67.4 / 68.7 | 3.2 |
| `dev_tree` 60 × 50 | `st_dt` | 0 | `3381575a` | 1.0 (4) | 0.9828 | 0.920 (0.935) | 232.1 | 6.1 |
| | | 1 | `563661f1` | 0.9993 (4) | 0.9781 | 0.918 (0.933) | 241.9 | 6.1 |
| | | 2 | `c7f1ec6b` | 0.9986 (4) | 0.9751 | 0.886 (0.899) | 263.8 | 6.1 |
| `dev_tree_1s` | `st_dt1s` | 0 | `4687b541` | 0.9986 (4) | 0.9757 | 0.871 (0.890) | 147.4 | 3.6 |
| | | 1 | `f5585f31` | 0.9994 (4) | 0.9760 | 0.902 (0.921) | 132.6 | 3.6 |
| | | 2 | `479beca0` | 0.9980 (4) | 0.9818 | 0.885 (0.905) | 135.2 | 3.6 |
| `dev_tree_1s_easy` | `st_easy_bb01` | 0 | `d08e3a1b` | 0.9721 (4) | 0.9714 | 0.541 (0.952) | 111.3 | 3.6 |
| | | 1 | `8d0bbfda` | 0.9689 (4) | 0.9733 | 0.530 (0.965) | 124.7 | 3.6 |
| | | 2 | `e2ce4e06` | 0.9710 (4) | 0.9692 | 0.458 (0.750) | 156.5 | 3.6 |
| `dev_tree_1s_hard` | `st_hard_bb01` | 0 | `d2938975` | **0.2172 (2)** | 0.4000 | 0.000 (0.140) | 139.3 | 3.6 |
| | | 1 | `764709dc` | **0.2199 (2)** | 0.4099 | 0.000 (0.265) | 140.7 | 3.6 |
| | | 2 | `3adf249a` | **0.2067 (2)** | 0.4426 | 0.000 (0.250) | 114.7 | 3.7 |

The three repeats of easy and hard score identically. Every fixture plants 4
clones.

## Reading

- **Clones.** Five fixtures recover 4 clones at ARI 0.969–1.0. On
  `dev_tree_1s_hard` r0 the integer clones fall from 4 (0.907) at #520 to 2
  (0.217) at #522, which re-landed the integer-clone merge, while the fitted
  clones hold at 0.885 (5); #542 made the integer clones the output (#575,
  #565).
- **Copy states.** Copy ARI is 0.97–0.98 on the `dev_tree` family and 0.90 on
  CalicoST. Phased exact altered on hard, 0.036 against 0.627 phase-free, has
  been at that level since #487's lattice decode (#565).
- **Classes.** Balanced gains (A = B > 1) score 0.000 on CalicoST easy and
  hard and on r0 of `dev_tree`, `dev_tree_1s` and `dev_tree_1s_hard`, and
  0.878 on `dev_tree_1s_easy` r0. Each run puts the planted (2, 2) bins in the same
  HMM state as that clone's neutral bins, so only the per-bin decode can
  separate them (#573).
- **Runtime.** 67–74 s on CalicoST's 1-slice samples, 111–157 s on the
  `dev_tree_1s` family, 232–264 s on 2-slice `dev_tree`. CalicoST, uncapped,
  takes 20,243 s on `dev_tree` r0 for 0.8538 (6 clones).

## Known failures at this release

- `dev_tree_1s_hard`: 2 clones of 4 (#575).
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
