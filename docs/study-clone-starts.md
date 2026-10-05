# Study: the BAF stage's clone start (#490)

**TL;DR:** on the current `--sal` pipeline (#500 + #503), `cnaster`'s own 2 × 2
phasing grid matches or beats every other start on all four samples:
1.0 / 0.9861 / 0.9861 / 0.9829 clone ARI on 60 × 50, `dev_shared_unique`, easy
and hard. No start is adopted. Without #476's clone flags,
`dev_shared_unique` collapses to one clone from every start. The UMI-grown
start is seed-dependent: hard ends at 0.7267 (3 clones) from seed 0.
#541's notebook (`docs/nb/clone_label_study.ipynb`) separates the start from
the solver and the refit this study measured together.

**#541 rerun at 7d1ba8b (sal b61dfba):** the notebook's end-to-end table, at default threads, reproduces
`--sal`'s own start on all four samples: dev_tree 1.0 (4), CalicoST easy 0.9861 (4) and hard 0.9829 (4);
`dev_shared_unique` reads 0.9983 (4), 0.9971 in #554. No start is adopted. At 1 thread, easy reads 0.9851 (#638).
Key figure: `docs/plots/paper/key_studies/554_clone-starts.png`.

## Method

`python -m tests.studies.clone_starts SAMPLE START SEED [FLAGS]`, one arm per
row, `--sal --no-plots` on #503's pipeline. The starts:
- `grid2` / `grid3`: `cnaster`'s rectangles per slice, `npart_phasing` 2 or 3.
- `grow`: `port.sandbox.wolff_init.umi_grow`, clones grown from the deepest
  spots to equal SNP UMIs, at seeds 0–2.
- `normal-first`: a first `--sal` pass names the normal clone, the largest
  share of (1, 1) bins. On 60 × 50 that is exactly the 2,317 planted normal
  spots. Its spots are one start clone and the grid partitions the rest; the
  wall includes the first pass.

Two arms: `--sal`, and `--sal --no-refinement-mask --no-floor-merge` (#476's
flags off). BAF-stage and final clone ARI are against the planted labels.

## Results

BAF-stage ARI → final ARI (clones), copy ARI, phase-free exact altered, wall.

### `--sal`

| start | 60 × 50 | `dev_shared_unique` | easy | hard |
| --- | --- | --- | --- | --- |
| `grid2` (`cnaster`) | 1.0 → **1.0** (4), 0.9828, 0.9348, 276 s | 0.9861 → **0.9861** (4), 0.8471, 0.8165, 50 s | 0.9861 → **0.9861** (4), 0.8984, 0.6299, 112 s | 0.9829 → **0.9829** (4), 0.9055, 0.6272, 72 s |
| `grid3` | 1.0 → 1.0 (4), 0.9823, 0.9164, 203 s | 0.8719 → 0.8719 (4), 0.8743, 0.8091, 44 s | 0.9849 → 0.9849 (4), 0.8882, 0.6166, 130 s | 0.9794 → 0.9794 (4), 0.8955, 0.5805, 84 s |
| `normal-first` | 1.0 → 1.0 (4), 346 s | 0.9861 → 0.9861 (4), 87 s | 0.9809 → 0.9809 (4), 189 s | 0.9829 → 0.9829 (4), 123 s |
| `grow` seeds 0 / 1 / 2 | 1.0 / 0.9996 / 0.9996 | 0.9816 / 0.9861 / 0.9861 | 0.9861 / 0.9814 / 0.9853 | **0.7267 (3)** / 0.9674 / 0.9829 |

### #476's flags off

| start | 60 × 50 | `dev_shared_unique` | easy | hard |
| --- | --- | --- | --- | --- |
| `grid2` | 1.0 → 0.9981 (4) | 0.0 → 0.0 (1) | 0.9861 → 0.9736 (4) | 0.9829 → 0.9274 (4) |
| `grid3` | 1.0 → 0.9990 (4) | 0.0 → 0.0 (1) | 0.9849 → 0.9797 (4) | 0.9794 → 0.9351 (4) |
| `normal-first` | 1.0 → 0.9977 (4) | 0.0 → 0.0 (1) | 0.698 → 0.6972 (3) | 0.9838 → 0.8806 (4) |
| `grow` seed 0 | 1.0 → 0.9979 (4) | refused by `sal`'s ICM | 0.9861 → 0.9745 (4) | 0.7267 → 0.6963 (3) |

The wall of every `--sal` 60 × 50 arm ran beside other jobs on a 4-core host.
The seconds order the starts within a sample and are not a benchmark (#494 is).

## Reading

- **The flags carry `dev_shared_unique`.** Without them, `cnaster`'s fixed
  200-spot floor dissolves every clone of the 3,000-spot sample.
- **One arm crashes.** With the UMI-grown start, `sal`'s floored ICM refuses
  ("sweep 1 left no state holding min_sites=200 sites") instead of
  collapsing. A refusal is the better failure, but the arm has no result.
- **The UMI-grown start is seed-dependent** on hard: 0.7267 / 0.9674 /
  0.9829. That spread is what #491's choice by likelihood would have to
  resolve. The grid reaches the best of the three with no seed to choose.
- **`normal-first` costs a second run** and matches the grid everywhere
  except easy, where it is 0.005 lower. With the flags off it fails on easy
  (0.698, 3 clones): the first pass's normal clone absorbed a tumour clone.
- **`grid3` loses** on `dev_shared_unique` (0.8719 against 0.9861) and by
  0.001–0.004 elsewhere.

No start matches or beats the grid on every sample and arm, so none is
adopted. `sandbox/wolff_init.umi_grow` and `sandbox/normal_candidates` stay
in the sandbox, their numbers here.

## Upstream correspondence

None of these starts uses `snakes_and_ladders`: the grid is `cnaster`'s and the
other two are port's sandbox. The ICM refusal is `sal`'s
(`search.icm.numba.icm_sweeps_checked`) and is correct as `sal` states it: a
floor above `ceil(n_nodes / n_states)` can leave no survivor.
