# Metrics

One row per measured run (#409): the commit, the fixture and a digest of the
data it built, the `tests.recovery_audit` arguments that reproduce it, then
the tracked metrics. `python -m tests.metrics --record -- [flags]` appends a
row; `--best clone_ari --fixture dev` reads one back. `—` is unmeasured.

Columns: `clone_ari` fitted clone labels against planted, per spot;
`clone_ari_int` the same after merging clones of one decoded `(A, B)` profile
(#344); `copy_ari` decoded phased `(A, B)` against planted state, per
clone-bin; `state_ari` the continuous state before integer decoding;
`wall_s` the run; `peak_gb` its resident peak.

| commit | date | fixture | fixture_hash | args | clone_ari | clone_ari_int | copy_ari | state_ari | wall_s | peak_gb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
