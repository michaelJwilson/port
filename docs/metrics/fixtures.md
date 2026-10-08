# Fixtures

**TL;DR:** 9 supported fixtures, 13 retired generations. A fixture is a ledger
name and a content hash (`port.sim.fixtures.realization_hash`). It is
**supported** while a manifest under `sim/manifests/` (or a committed
CalicoST sample) draws that hash, and **retired** once the manifest moves to
`sim/sandbox/` or draws another hash.

- **Defined as:** one line in commit-subject form, `<parent>: <what differs>`,
  naming the manifest the fixture extends and what it changes. The manifest is
  the full statement.
- **Created:** the first run that `docs/metrics/runs.tsv` records on the
  hash, as its UTC date and commit.
- **Retired:** the date of the commit that moved its manifest to
  `sim/sandbox/`, or the first run on the hash that superseded it under the
  same name.
- **Runs:** the number of distinct `run_id`s in `ledger.tsv`.

## Supported

| fixture | hash | defined as | source | created | runs |
| --- | --- | --- | --- | --- | --- |
| `easy` | `2d4ce9a9` | `calicost: easy, 1 shared + 2 clone-specific CNAs per clone, 50 Mb each` | CalicoST sample `numcnas1.2_cnasize5e7_ploidy2_random0` | 2026-09-28 `7073269` | 35 |
| `hard` | `8797710b` | `calicost: hard, 6 shared + 3 clone-specific CNAs per clone, 10 Mb each` | CalicoST sample `numcnas6.3_cnasize1e7_ploidy2_random0` | 2026-09-28 `7073269` | 36 |
| `dev_tree_r0` | `3381575a` | `calicost_grch38: 3 tumour clones on a mutation tree, two 60x50 slices, exponential lengths; frozen as CalicoST's benchmark` | `sim/manifests/baseline/dev_tree.toml` r0 (`fixtures.R0_HASH`) | 2026-10-01 `efdebfd` | 7 |
| `dev_tree_r0` | `3339b9a0` | `dev_tree: lognormal CNA lengths, mean 50 Mb (#621)` | `sim/manifests/dev_tree.toml` r0 | 2026-10-01 `9a47d97` | 5 |
| `dev_tree_1s_r0` | `df3cc0ab` | `dev_tree: one 60x50 slice, urn counts, BB overdispersion 0.01; lognormal mean 50 Mb, sigma 0.541` | `sim/manifests/dev_tree_1s.toml` r0 | 2026-10-01 `9a47d97` | 5 |
| `dev_tree_1s_easy_r0` | `7ba9b01f` | `dev_tree_1s: CalicoST easy's law, Felsenstein 7 CNAs, lognormal mean 50 Mb; cell admixture 14% normal` | `sim/manifests/dev_tree_1s_easy.toml` r0 | 2026-10-05 `dbf5436` | 25 |
| `dev_tree_1s_hard_r0` | `9ec90dc2` | `dev_tree_1s: CalicoST hard's law, 6 shared + 3 unique CNAs, lognormal median 10 Mb; cell admixture 14%` | `sim/manifests/dev_tree_1s_hard.toml` r0 | 2026-10-01 `9a47d97` | 5 |
| `dev_tree_1s_dense_r0` | `33e3471e` | `dev_tree_1s_easy: CNAs over >80% of the genome, 60 expected, mean 150 Mb, irreversible LOH (T- #698)` | `sim/manifests/dev_tree_1s_dense.toml` r0 | 2026-10-07 `83ed58b` | 4 |
| `study15_r0` | `784e4ebf` | `calicost_grch38: study draw, 2-4 clones, polygons r ~ N(0.3, 0.1), tree CNAs lognormal median 15 Mb, 26 (A, B) states, cell admixture 14%, phase switches` | `sim/manifests/study15.toml` r0, the first of the study stream (T- #807) | 2026-10-08 `4c05c5f` | 1 |

Drawn by live manifests but never run into the ledger, so not supported:
`dev_tree_easy` r0 (`e4bbe03f`), `dev_shared_unique`, `study10` r0
(`38b57c75`, study15 at a 10 Mb median), `study15_2s` r0 (`ed6f692f`,
study15 on two slices).

`dev` (`07b82e92`) is `run_audit --recovery --instance dev`, an instance
built by `port.sim.truth` rather than a drawn sample. It is outside this
table's manifest rule; its last run was 2026-09-26.

## Retired

| fixture | hash | defined as | source | created | retired | why | runs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `r0` | `93398396` | `dev_tree: an earlier r0 generation, recorded as r0` | `sim/generated/dev_tree/r0` | 2026-09-28 `7073269` | 2026-10-01 | superseded by `dev_tree_r0` `3381575a` | 1 |
| `dev_tree_r1` | `563661f1` | `baseline/dev_tree: realization 1` | `st_dt.toml` r1 | 2026-10-01 `efdebfd` | 2026-10-08 | moved to `sim/sandbox/` (#808) | 5 |
| `dev_tree_r2` | `c7f1ec6b` | `baseline/dev_tree: realization 2` | `st_dt.toml` r2 | 2026-10-01 `efdebfd` | 2026-10-08 | moved to `sim/sandbox/` (#808) | 5 |
| `dev_tree_1s_r0` | `4687b541` | `dev_tree_1s: exponential lengths, mean 50 Mb, phase switches` | `baseline/dev_tree_1s.toml`, `st_dt1s.toml` r0 | 2026-10-01 `ac00498` | 2026-10-08 | moved to `sim/sandbox/` (#808) | 12 |
| `dev_tree_1s_r1` | `f5585f31` | `baseline/dev_tree_1s: realization 1` | `st_dt1s.toml` r1 | 2026-10-01 `efdebfd` | 2026-10-08 | moved to `sim/sandbox/` (#808) | 5 |
| `dev_tree_1s_r2` | `479beca0` | `baseline/dev_tree_1s: realization 2` | `st_dt1s.toml` r2 | 2026-10-01 `efdebfd` | 2026-10-08 | moved to `sim/sandbox/` (#808) | 5 |
| `dev_tree_1s_easy_r0` | `d08e3a1b` | `dev_tree_1s_easy: exponential lengths about 50 Mb` | `baseline/dev_tree_1s_easy.toml`, `st_easy_bb01.toml` r0 | 2026-10-01 `ac00498` | 2026-10-08 | moved to `sim/sandbox/` (#808) | 13 |
| `dev_tree_1s_easy_r0` | `22a5eb85` | `dev_tree_1s_easy: first lognormal generation (#621), recorded as _ln_r0` | `dev_tree_1s_easy.toml` r0 | 2026-10-01 `0eaac23` | 2026-10-05 | superseded by `7ba9b01f` | 4 |
| `dev_tree_1s_easy_bb01_r1` | `8d0bbfda` | `baseline/dev_tree_1s_easy: realization 1` | `st_easy_bb01.toml` r1 | 2026-10-01 `efdebfd` | 2026-10-08 | moved to `sim/sandbox/` (#808) | 5 |
| `dev_tree_1s_easy_bb01_r2` | `e2ce4e06` | `baseline/dev_tree_1s_easy: realization 2` | `st_easy_bb01.toml` r2 | 2026-10-01 `efdebfd` | 2026-10-08 | moved to `sim/sandbox/` (#808) | 5 |
| `dev_tree_1s_hard_r0` | `d2938975` | `dev_tree_1s_hard: exponential lengths about 10 Mb` | `baseline/dev_tree_1s_hard.toml`, `st_hard_bb01.toml` r0 | 2026-10-01 `ac00498` | 2026-10-08 | moved to `sim/sandbox/` (#808) | 13 |
| `dev_tree_1s_hard_bb01_r1` | `764709dc` | `baseline/dev_tree_1s_hard: realization 1` | `st_hard_bb01.toml` r1 | 2026-10-01 `efdebfd` | 2026-10-08 | moved to `sim/sandbox/` (#808) | 5 |
| `dev_tree_1s_hard_bb01_r2` | `3adf249a` | `baseline/dev_tree_1s_hard: realization 2` | `st_hard_bb01.toml` r2 | 2026-10-01 `efdebfd` | 2026-10-08 | moved to `sim/sandbox/` (#808) | 5 |
