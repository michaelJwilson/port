# Study: Potts solvers on the clone assignment (#492)

**TL;DR:** the clone-assignment problems `--sal` solves are easy. TRW-S
certifies the ground state of all 12 captured calls, with a gap of 0 in
0.01–0.36 s. `--sal`'s row, `alpha-rust-fuse-merge`, reaches that optimum on
every call in 0.02–0.06 s. So does every graph-cut row, `sal`'s and port's.
`cnaster`'s ICM ends 305–1,095 nats above it. `--sal` keeps its row: the
solver is not what limits clone recovery (#497 was).

## Method

`python -m tests.studies.potts_solvers capture | per-call`.

1. **Capture.** One `--sal --no-plots` run per sample (#487 + #496's
   loader), pickling every problem `pipeline_clone_assignment` hands its
   solver: the folded field with the refinement mask's penalty, the kNN
   graph, the start and β = 1.
   - 60 × 50 `dev_tree` r0: 10 calls on 6,000 spots.
   - CalicoST easy: 11 calls on 3,000 spots.
   - CalicoST hard: 21 calls on 3,000 spots.
2. **Per call:** the first and last call of each stage, 4 per sample.
   - Each call becomes `sal`'s `Problem` through port's own
     `potts_graph_from`, so the graph is `(A + Aᵀ)β/2` as the rows see it
     (#483).
   - Every `sal.search.ground_state.METHODS` entry and every port row is a
     start of `sal.opt.starts.StartsBenchmark`, polished by ICM, as `sal`'s
     `docs/nb/potts_starts.ipynb` runs them. The budget is 1,000 heat-bath
     sweeps of site visits per `sal` method, with 3 seeds and 4 workers.
   - `sal` methods start from their own labelling. Port rows start from the
     captured assignment and run floorless: the floor is a constraint the
     energy does not carry.
3. **Reference:** TRW-S's lower bound on each call, so every gap is
   certified. TRW-S closed its own gap on all 12 calls (|gap| < 1e-8).

## Results per call

The largest final gap in nats over a sample's 4 calls, with mean seconds per
start. The host had 4 cores and the runs were serialized; the seconds are the
host's.

| arm | 60 × 50 | easy | hard |
| --- | --- | --- | --- |
| `sal` alpha-expansion | 0.0 / 0.02 s | 0.0 / 0.01 s | 0.0 / 0.01 s |
| `sal` alpha-beta-swap | 0.0 / 0.03 s | 0.0 / 0.02 s | 0.0 / 0.01 s |
| port `alpha` (Python cut) | 0.0 / 2.75 s | 0.0 / 1.17 s | 0.0 / 1.14 s |
| port `alpha-rust` | 0.0 / 0.04 s | 0.0 / 0.02 s | 0.0 / 0.02 s |
| port `alpha-rust-icm` | 0.0 / 0.09 s | 0.0 / 0.05 s | 0.0 / 0.04 s |
| port `alpha-rust-merge` | 0.0 / 0.05 s | 0.0 / 0.02 s | 0.0 / 0.02 s |
| port `alpha-rust-fuse-merge` (`--sal`) | 0.0 / 0.06 s | 0.0 / 0.02 s | 0.0 / 0.02 s |
| `sal` anneal | 122.3 / 1.03 s | 40.9 / 0.34 s | 25.0 / 0.34 s |
| `sal` tempering | 207.9 / 0.97 s | 94.1 / 0.35 s | 84.4 / 0.32 s |
| `sal` max-product | 239.3 / 6.92 s | 64.6 / 1.40 s | 66.9 / 1.31 s |
| `sal` swendsen-wang | 269.9 / 0.74 s | 95.8 / 0.27 s | 93.8 / 0.28 s |
| `sal` wolff | 543.8 / 0.28 s | 444.8 / 0.16 s | 336.9 / 0.15 s |
| `sal` icm | 528.2 / 0.13 s | 355.8 / 0.15 s | 686.5 / 0.15 s |
| `sal` field_argmax | 856.5 / 0.48 s | 494.8 / 0.46 s | 373.2 / 0.50 s |
| port `icm-numba` | 997.6 / 0.04 s | 373.8 / 0.01 s | 305.3 / 0.01 s |
| port `icm` (`cnaster`'s) | 1094.7 / 0.11 s | 371.0 / 0.05 s | 320.2 / 0.04 s |
| `sal` icm-random | 1123.1 / 0.81 s | 677.3 / 0.32 s | 1090.7 / 0.25 s |
| `sal` bifurcation | 2502.2 / 0.50 s | 1613.9 / 0.17 s | 1217.9 / 0.17 s |

The runtime-against-gap figures this study drew are retired: #541's
notebook (`docs/nb/clone_label_study.ipynb`) runs every solver from every
clone-label start, on fields built from #540's copy states, and draws the
figure that replaces them (`docs/plots/studies/clone_label_study.png`).

## End to end

`--sal` on #500 (+ #496's loader, `distinct` start), `PORT_LABEL_SOLVER` as
named. Clone ARI (clones) / copy ARI / phase-free exact altered / wall.

| row | 60 × 50 | easy | hard |
| --- | --- | --- | --- |
| `alpha-rust-fuse-merge` (`--sal`) | 1.0 (4) / 0.9828 / 0.9348 / 174 s | 0.9861 (4) / 0.8984 / 0.6299 / 88 s | 0.8652 (5) / 0.8652 / 0.4970 / 88 s |
| `alpha-rust-icm` | 1.0 (4) / 0.9828 / 0.9348 / 153 s | **0.9868 (4)** / 0.8984 / 0.6299 / 86 s | **0.9822 (4) / 0.9055 / 0.6272** / 67 s |
| `alpha-rust-merge` | 1.0 (4) / 0.9828 / 0.9348 / 153 s | 0.9861 (4) / 0.8984 / 0.6299 / 80 s | 0.8922 (5) / 0.9002 / 0.5444 / 86 s |
| `icm-numba-floor` | 1.0 (4) / 0.9828 / 0.9348 / 198 s | 0.9189 (4) / 0.8979 / 0.6260 / 91 s | 0.7143 (5) / 0.9072 / 0.5503 / 143 s |
| `icm` (`cnaster`'s) | 1.0 (4) / 0.9828 / 0.9348 / 239 s | 0.9281 (4) / 0.8979 / 0.6260 / 224 s | 0.7097 (5) / 0.8673 / 0.5266 / 241 s |

The α rows reach the same certified ground state per call. They differ in how
they meet the clone-size floor after it:
- `alpha-rust-icm` hands the labelling to `cnaster`'s ICM at its floor;
- `-merge` and `-fuse-merge` merge the smallest clones into their best
  neighbour.

On hard, the ICM hand-over is the one that keeps clone_0 whole. ICM from the
previous labelling, without a cut, loses 0.07–0.28 clone ARI off 60 × 50, at
1.3–3.6x the wall.

**With #489's start** (`--hmm-start kmeans++x5+em`, #503) the two leading
rows tie:

| row | 60 × 50 | easy | hard |
| --- | --- | --- | --- |
| `alpha-rust-fuse-merge` (`--sal`) | 1.0 (4) / 0.9828 | 0.9861 (4) / 0.8984 | 0.9829 (4) / 0.9055 |
| `alpha-rust-icm` | 1.0 (4) / 0.9828 | 0.9868 (4) / 0.8984 | 0.9822 (4) / 0.9055 |

**Decision:** `--sal` keeps `alpha-rust-fuse-merge`. With the start the rows
agree within 0.0007 clone ARI on every sample. Without it, `alpha-rust-icm` is
the more robust (hard 0.9822 against 0.8652), which is recorded here rather
than adopted, because the start is what `--sal` now carries. No row loses on
every sample, since all tie on 60 × 50, so none moves to `sandbox/`.

## Upstream correspondence

Nothing in `snakes_and_ladders` needs to change to serve this problem:
`Problem`, `METHODS`, `StartsBenchmark` and `trws` take the captured graph and
field as they are. The floor and the refinement mask stay port's
(`enforce_floor`, `MASK_PENALTY`), because they constrain the labelling and
not the energy. A floor-aware expansion would be the upstream change that
serves them; this study gives no reason to want it.

## Found on the way

`tests.sim_fixtures.stage` staged a CalicoST sample loaded by absolute path
into itself: `into / "/abs"` is `/abs`. That unlinked every committed input
and linked it to itself. The target is now `into / Path(name).name`, and
staging into the sample is refused (`tests/test_stage_inputs.py`).
