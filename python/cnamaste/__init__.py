"""`cnamaste`: `cnaster`'s forward path, copied to be developed and shipped on its own (T- #670).

The modules here are the 31 `cnaster` modules `cnaster.scripts.run_cnaster`
reaches, and that script as `cnamaste.run`, copied from the locked pin
`4adad4d` with one change: each live `cnaster` import reads `cnamaste`.
`run_cnamaste` (`cnamaste.run:main`) is `run_cnaster` under the copy.

T- #670's later PRs move `port`'s replacements in one stage at a time, each
departure stated where it is made. PR2: the figures (`write_fig`,
`plot_clones_genomic`, `plot_clones_spatial`, `plot_copy_number_profile`),
`run_cnamaste --no-plots`, and the fixes for #105 and #113. PR3: the
preprocessing (`docs/port-forward.md` rows 6 and 8-23), with T- #692's
terminating rectangular initializer.
`tests/test_module_roles.py` says which modules are still copies.

Nothing here imports `cnaster` or `port`. The installed `cnaster`, with the
`port` rows moved in installed over it, is what the copy is pinned against
(`tests/test_cnamaste_copy.py`).

Out of the copy, because `run_cnaster` reaches none of them: `adjacency`,
`hmrf_result`, `plot_loh_density`, `plot_validation_stats`, `sim`, `wolff`,
the other three scripts, `deprecated/` and `sandbox/`.
"""
