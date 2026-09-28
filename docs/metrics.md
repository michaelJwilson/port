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
clone-bin; `state_ari` the continuous state before integer decoding;
`wall_s` the run; `peak_gb` its resident peak; `note` what the row measures.

| commit | timestamp | fixture | fixture_hash | test | args | clone_ari | clone_ari_int | copy_ari | state_ari | wall_s | peak_gb | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c84aa06 | 2026-09-26T17:00Z | dev | 07b82e92 | tests/recovery_audit.py::main | --states 8 --outer 1 --iterations 3 -- | 0.7927 | 0.9242 | 0.9982 | 0.4298 | 44.9 | 5.25 | Baseline: run_cnaster_port defaults on dev, before any opt-in flag |
| c84aa06 | 2026-09-26T17:00Z | dev | 07b82e92 | tests/recovery_audit.py::main | --states 8 --outer 1 --iterations 3 -- --sal | 0.9795 | 0.9795 | 0.9971 | 0.2183 | 25.9 | 3.30 | --sal: snakes_and_ladders routines lift clone ARI 0.79->0.98, wall -42% |
