# solver_combined_forward.png

Drawn at code `ad1ce10` from two `port.studies.copy_state_stream` records on `sim/manifests/dev_tree_1s_hard.toml`,
one start and seed per row in both, each problem built by `run_cnaster_port --sal` at the planted
clones (`port.studies.stage`, #730). Each gap is to the best log-likelihood either record reached on
the realization (`copy_state_plot.shared_best`). The polish is `port.studies.forward_polish` (#748).

| Panel | Polish | Data hash | Problems | Seeds | Median polish [s] | Median passes | Median forward-backward |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (a) | EM | `341a2115` | 5 | 5 | 8.70 | not recorded | not recorded |
| (b) | Forward L-BFGS | `c222981c` | 3 | 5 | 92.88 | 429 | 430 |
