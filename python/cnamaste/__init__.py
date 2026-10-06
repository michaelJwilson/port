"""`cnamaste`: `cnaster`'s forward path, copied to be developed and shipped on its own (T- #670).

The modules here are the 31 `cnaster` modules `cnaster.scripts.run_cnaster`
reaches, and that script as `cnamaste.run`, copied from the locked pin
`4adad4d` with one change: each live `cnaster` import reads `cnamaste`.
`run_cnamaste` (`cnamaste.run:main`) is `run_cnaster` under the copy.

Nothing here imports `cnaster` or `port`. The installed `cnaster` stays the
unpatched oracle the copy is pinned against (`tests/test_cnamaste_copy.py`),
and T- #670's later PRs replace one stage at a time, each departure stated
in its module's docstring.

Out of the copy, because `run_cnaster` reaches none of them: `adjacency`,
`hmrf_result`, `plot_loh_density`, `plot_validation_stats`, `sim`, `wolff`,
the other three scripts, `deprecated/` and `sandbox/`.
"""
