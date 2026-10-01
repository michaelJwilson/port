# Final benchmark: `--sal` against CalicoST (#494)

**TL;DR:** `run_cnaster_port --sal` recovers the planted clones at ARI
0.9861 / 0.9829 / 1.0 on CalicoST easy, hard and `dev_tree` 60 × 50 r0, in
95 / 83 / 239 s and 3.3 / 3.3 / 6.4 GB. CalicoST on its shipped
configuration finishes none of them in 30 minutes. The last clone assignment
it wrote, its BAF stage, scores 0.6686 (3 clones), 0.4675 (2) and 0.8601 (3).

## Conditions

- Host: 4 cores, `NUMBA_NUM_THREADS` default, one job at a time.
- port: commit 21845d2 (#489 + #501 on #500), `python -m tests.final_benchmark
  SAMPLE port`: `tests.sim_audit --sal`, three repeats on a warm numba cache,
  median wall. The first call's compilation is not in the figure.
- CalicoST: c1abcae, this branch's driver, `python -m tests.final_benchmark SAMPLE calicost`:
  `run_calicost --shipped configuration_cna --no-align --no-figures`, once,
  under `timeout 1800`. Every value but the paths is CalicoST's shipped file,
  among them `n_clones 3`, `n_clones_rdr 2`, `npart_phasing 3`,
  `max_iter_outer 20`, `np_threshold 1.0`. The samples plant 4 clones.
  `dev_tree` has two slices and runs on CalicoST's joint file,
  `configuration_cna_multi`, with no alignment files. Its drawn slices name
  spots `<barcode>_<sample_id>`, which CalicoST's joint loader suffixes again,
  so the benchmark stages a copy with the suffix removed
  (`final_benchmark.joint_inputs`); the counts are unchanged.
- Peak is the child's maximum RSS.
- A run stopped by the cap is scored on the last clone assignment it wrote
  (`final_benchmark.baf_stage`): the BAF stage after CalicoST's
  Neyman–Pearson merge (`mergedallspots_nstates7_sp.npz`), or before it
  (`allspots_nstates7_sp.npz`, which CalicoST rewrites after every HMRF
  round, scored at its last completed round) where the merge had not run. It is not comparable with a final ARI: the read-depth stage had not
  finished.

## Results

Clone ARI (clones), copy ARI, exact altered (phase-free), wall, peak.

| sample | tool | clone ARI | copy ARI | exact altered | wall | peak |
| --- | --- | --- | --- | --- | --- | --- |
| easy | port `--sal` | 0.9861 (4) | 0.8984 | 0.252 (0.6299) | 94.9 s | 3.34 GB |
| easy | CalicoST | not finished; BAF stage 0.6686 (3) | – | – | > 1800 s | 2.80 GB |
| hard | port `--sal` | 0.9829 (4) | 0.9055 | 0.0355 (0.6272) | 82.7 s | 3.30 GB |
| hard | CalicoST | not finished; BAF stage 0.4675 (2) | – | – | > 1800 s | 2.80 GB |
| 60 × 50 | port `--sal` | 1.0 (4) | 0.9828 | 0.9197 (0.9348) | 238.7 s | 6.37 GB |
| 60 × 50 | CalicoST | not finished; BAF stage before the merge 0.8601 (3) | – | – | > 1800 s | 5.41 GB |

port's walls per repeat: easy 107.8 / 92.4 / 94.9 s, hard 82.7 / 89.0 /
81.4 s, 60 × 50 239.1 / 230.1 / 238.7 s.

## Reading

- **CalicoST stops in the read-depth stage on easy and hard**, having
  written `clone0_nstates7_smp.npz`, the first of its clones' refinements.
  On 60 × 50 it stops inside the BAF stage's HMRF, after 2 of at most 20
  rounds (easy had run 8 and hard 11 before their merges).
- **Its BAF stage asks for 3 clones**, as shipped, against 4 planted, so
  0.6686 and 0.8601 measure that setting as much as the fit. On hard the
  merge joined two of the three (0.4685 before it, 0.4675 after).
- **port's exact altered is low on easy and hard** (0.252, 0.0355) against
  0.63 phase-free: exact altered scores the phased (A, B) pair, and the
  phase-free form ignores which haplotype carries the change. The gap is
  phase orientation, recorded in #467 and not changed here.
- **60 × 50 costs 65 s more than #500's 174 s**: #489's start, 160 s budget
  per call, is most of it. The start is what lifts hard from 0.8652 to
  0.9829.

## Uncapped CalicoST on `dev_tree` (#532)

CalicoST ran to completion on its shipped `configuration_cna_multi` with
`n_clones 5` and no cap. It took 20,243 s (5.62 h) on 3 cores, which is
132× the 153.6 s port `--sal` took on 1 core (395× in core-seconds). port was
run at `main` after #515, #516 and #522, with `merge_agreement 0.99` in the
config; under the exact rule it scores the same.

| tool | clone ARI | integer clone ARI | copy state ARI | copy ARI | exact altered | bins | wall | peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CalicoST | 0.8538 (6) | 0.8538 (6) | 0.0889 | 0.9075 | 0.7095 (0.7095) | 2505 | 20,243 s | 3.43 GB |
| port `--sal` | 0.8612 (5) | 1.0 (4) | 0.0682 | 0.9828 | 0.9197 (0.9348) | 2895 | 153.6 s | 6.11 GB |

- CalicoST was resumed twice from its npz checkpoints after the process was
  killed. Its wall is summed to the last checkpoint of each segment
  (5894 + 4697 + 9652 s). Read-depth fitting is 13,269 s of it (66%); #532
  has the timing for each stage.
- Each tool is scored on its own bins. Copy state ARI compares the HMM state
  index with the planted (A, B), so it is low for both tools by construction.

## Reproduce

```
python -m tests.final_benchmark numcnas1.2_cnasize5e7_ploidy2_random0 port
python -m tests.final_benchmark numcnas1.2_cnasize5e7_ploidy2_random0 calicost
```

and the same for `numcnas6.3_cnasize1e7_ploidy2_random0` and `dev_tree` r0
at 60 × 50, drawn from `claude/467-dev-tree-size`'s manifest (a22d5ab) by
`python -m port.sim.draw sim/manifests/dev_tree.toml --into DIR`; the sample
argument is then `DIR/dev_tree/r0`.
