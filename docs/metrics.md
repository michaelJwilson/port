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
