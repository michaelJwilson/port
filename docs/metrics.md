# Metrics

One row per measured run (#409): the commit, the UTC timestamp, the fixture
and a digest of the data it built, the test that computed the metrics
(`path::function`), the arguments that reproduce it, the tracked metrics, and
a note of at most 72 characters, written as a commit subject, stating what
change the row measures. `python -m tests.metrics --record --note "..." --
[flags]` appends a row; `--best clone_ari --fixture dev` reads one back. `—`
is unmeasured.

The first two rows predate the `timestamp` column: theirs is the time the
rows were committed (b85346b), within 3 minutes after the runs.

Columns: `clone_ari` fitted clone labels against planted, per spot;
`clone_ari_int` the same after merging clones of one decoded `(A, B)` profile
(#344); `copy_ari` decoded phased `(A, B)` against planted state, per
clone-bin; `state_ari` the continuous state before integer decoding; `exact_altered` the share of planted-altered
clone-bins decoded as exactly the planted `(A, B)`, and `exact_altered_pf` the same up to phase
(`tests.sim_audit`'s `exact_altered_minor`), simulated samples only;
`wall_s` the run; `peak_gb` its resident peak; `note` what the row measures.

| commit | timestamp | fixture | fixture_hash | test | args | clone_ari | clone_ari_int | copy_ari | state_ari | exact_altered | exact_altered_pf | wall_s | peak_gb | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c84aa06 | 2026-09-26T17:00Z | dev | 07b82e92 | tests/recovery_audit.py::main | --states 8 --outer 1 --iterations 3 -- | 0.7927 | 0.9242 | 0.9982 | 0.4298 | — | — | 44.9 | 5.25 | Baseline: run_cnaster_port defaults on dev, before any opt-in flag |
| c84aa06 | 2026-09-26T17:00Z | dev | 07b82e92 | tests/recovery_audit.py::main | --states 8 --outer 1 --iterations 3 -- --sal | 0.9795 | 0.9795 | 0.9971 | 0.2183 | — | — | 25.9 | 3.30 | --sal: snakes_and_ladders routines lift clone ARI 0.79->0.98, wall -42% |
| 7073269 | 2026-09-28T22:51Z | r0 | 93398396 | tests/sim_audit.py::main | -- --sal | 0.9997 | 0.9997 | 0.9560 | 0.0743 | — | — | 240.0 | 5.73 | Baseline --sal on main e5ee447, r0 at 42x42 (#470) |
| 7073269 | 2026-09-28T22:55Z | easy | 1065eb5b | tests/sim_audit.py::main | -- --sal | 0.9440 | 0.9440 | 0.8191 | 0.0767 | — | — | 232.9 | 4.02 | Baseline --sal on main e5ee447, CalicoST easy |
| 7073269 | 2026-09-28T23:00Z | hard | 79cf7e62 | tests/sim_audit.py::main | -- --sal | 0.3031 | 0.3031 | 0.5733 | 0.0466 | — | — | 298.7 | 3.68 | Baseline --sal on main e5ee447, CalicoST hard: 2 clones for 4 |
| 7073269 | 2026-09-28T23:04Z | hard | 79cf7e62 | tests/sim_audit.py::main | -- --sal --refinement-mask --floor-merge | 0.9820 | 0.9820 | 0.4146 | 0.0513 | — | — | 242.0 | 4.02 | --refinement-mask --floor-merge restore hard's 4 clones (#468) |
| ac00498 | 2026-10-01T00:46Z | easy | 23989aa4 | tests/sim_audit.py::main | -- --sal --no-plots | 0.9861 | 0.9861 | 0.8984 | 0.0318 | 0.2520 | 0.6299 | 102.8 | 3.23 | HISTORY #545 truth_combined: a realization's tree, profiles, tracks and… |
| 4899b5f | 2026-10-01T00:47Z | easy | 23989aa4 | tests/sim_audit.py::main | -- --sal --no-plots | 0.9861 | 0.9861 | 0.8984 | 0.0318 | 0.2520 | 0.6299 | 99.1 | 3.23 | HISTORY #542 Integer-clone merge at 0.99 by default, clone_labels.tsv c… |
| aec3403 | 2026-10-01T00:49Z | easy | 23989aa4 | tests/sim_audit.py::main | -- --sal --no-plots | 0.9861 | 0.9861 | 0.8984 | 0.0318 | 0.2520 | 0.6299 | 98.8 | 3.23 | HISTORY #537 Re-land #530, #531, #533, #535 on main: #517 steps 5–8 |
| 558a172 | 2026-10-01T00:51Z | easy | 23989aa4 | tests/sim_audit.py::main | -- --sal --no-plots | 0.9861 | 0.9861 | 0.8984 | 0.0318 | 0.2520 | 0.6299 | 101.6 | 3.23 | HISTORY #529 #517 step 4: records are NamedTuples; copy_likelihood take… |
| c954cf2 | 2026-10-01T00:53Z | easy | 23989aa4 | tests/sim_audit.py::main | -- --sal --no-plots | 0.9861 | 0.9861 | 0.8984 | 0.0318 | 0.2520 | 0.6299 | 101.5 | 3.23 | HISTORY #528 #517 step 3: one implementation per concept |
| dea5754 | 2026-10-01T00:54Z | easy | 23989aa4 | tests/sim_audit.py::main | -- --sal --no-plots | 0.9861 | 0.9861 | 0.8984 | 0.0318 | 0.2520 | 0.6299 | 98.7 | 3.23 | HISTORY #527 #517 steps 1–2: exact signatures and bound options; hmm_no… |
| f613bdc | 2026-10-01T00:56Z | easy | 23989aa4 | tests/sim_audit.py::main | -- --sal --no-plots | 0.9861 | 0.9861 | 0.8984 | 0.0318 | 0.2520 | 0.6299 | 98.6 | 3.23 | HISTORY #536 Re-land #523 on main: #517 step 1, cnaster signatures and … |
| 4cbd805 | 2026-10-01T00:58Z | easy | 23989aa4 | tests/sim_audit.py::main | -- --sal --no-plots | 0.9861 | 0.9861 | 0.8984 | 0.0318 | 0.2520 | 0.6299 | 100.5 | 3.23 | HISTORY #519 #517 step 0: guards E1–E5 before the CLEAN refactor |
| 5244002 | 2026-10-01T01:00Z | easy | 23989aa4 | tests/sim_audit.py::main | -- --sal --no-plots | 0.9861 | 0.9861 | 0.8984 | 0.0318 | 0.2520 | 0.6299 | 99.7 | 3.23 | HISTORY #524 Re-land #469 on main: #466 steps 1–3 |
| a49efb2 | 2026-10-01T01:01Z | easy | 23989aa4 | tests/sim_audit.py::main | -- --sal --no-plots | 0.9861 | 0.9861 | 0.8984 | 0.0318 | 0.2520 | 0.6299 | 99.4 | 3.23 | HISTORY #480 normalidx_file: the named normal spots reach the loader an… |
| 6f06591 | 2026-10-01T01:03Z | easy | 23989aa4 | tests/sim_audit.py::main | -- --sal --no-plots | 0.9861 | 0.9861 | 0.8984 | 0.0318 | 0.2520 | 0.6299 | 100.2 | 3.23 | HISTORY #478 load_input_data: read a range file with bare-integer chrom… |
