# solver_combined.png

Drawn at code `982c7c2` from two records on `sim/manifests/dev_tree_1s_hard.toml`, each problem built by
`run_cnaster_port --sal` at the planted clones (`port.studies.stage`, #730, #742):

| Panel | Record | Data hash | Problems | Starts | Samplers |
| --- | --- | --- | --- | --- | --- |
| (a) spatial solvers | `port.studies.potts_stream` | `bc6c88a8` | 5 | 5 | settings file |
| (b) copy-state starts | `port.studies.copy_state_stream` | `341a2115` | 5 | 5 | settings file |

Each sampler reads its settings file (`potts_sampler_settings.json`, `copy_sampler_settings.json`),
tuned by #723 on the run's own problems, realizations 0-2 held out from these panels, on a quiet
host; Wolff keeps the settings tuned before the harness, since its tune ran 68 min unfinished.
