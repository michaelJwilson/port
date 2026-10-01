# Baseline release: `run_cnaster_port --sal` on `main` `ac00498`

**TL;DR:** (#577) `--sal` recovers the planted clones at ARI ≥ 0.969 on 5 of 6
fixtures, in 67–264 s and ≤ 6.1 GB; CalicoST finishes none of the three it was
run on within 30 minutes (`docs/calicost-benchmark.md`). `dev_tree_1s_hard`
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
