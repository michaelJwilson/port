# ruff: noqa
# mypy: ignore-errors
# Design-study probe (#380), kept as run; not package code.
import sys, runpy, numpy as np
from pathlib import Path

sys.path[:0] = [str(Path.cwd() / "python"), str(Path.cwd())]
import port.extensions.clone_mixture as cm
import os

cm.ROW_MODE = os.environ.get("MIX_ROW_MODE", cm.ROW_MODE)
cm.ENTROPY_WEIGHT = float(os.environ.get("MIX_ENTROPY", cm.ENTROPY_WEIGHT))
if os.environ.get("MIX_STARTS"):
    import numpy as _np

    cm.ADMIXTURE_STARTS = tuple(float(x) for x in os.environ["MIX_STARTS"].split(","))
print(
    f"MIXCONF row_mode={cm.ROW_MODE} entropy={cm.ENTROPY_WEIGHT} starts={cm.ADMIXTURE_STARTS}"
)
sys.argv = ["sim_audit"] + sys.argv[1:]
try:
    runpy.run_module("tests.sim_audit", run_name="__main__")
finally:
    for n, f in enumerate(cm.FITS):
        w = f.weights
        print(
            f"MIXFIT {n} K={w.shape[0]} cap={f.cap} gain={f.end - f.start:.2f} sweeps={f.sweeps} W={np.round(w, 3).tolist()}"
        )
