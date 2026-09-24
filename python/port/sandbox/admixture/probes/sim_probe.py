# ruff: noqa
# mypy: ignore-errors
"""`tests.sim_audit` with the clone mixture's fits printed (#380 design study).

Run from the repository root, as `run_sim.sh` does. `MIX_VARIANTS=1` routes
`--clone-mixture` through `port.sandbox.admixture.variants`, whose knobs
`MIX_ROW_MODE`, `MIX_ENTROPY` and `MIX_STARTS` set; otherwise the package's
`port.extensions.clone_mixture` runs as shipped.
"""

import os
import runpy
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path.cwd() / "python"), str(Path.cwd())]

import port.extensions.clone_mixture as package
import port.sandbox.admixture.variants as variants

cm = package
if os.environ.get("MIX_VARIANTS") == "1":
    cm = variants
    package.clone_mixture = variants.clone_mixture
    cm.ROW_MODE = os.environ.get("MIX_ROW_MODE", cm.ROW_MODE)

cm.ENTROPY_WEIGHT = float(os.environ.get("MIX_ENTROPY", cm.ENTROPY_WEIGHT))
if os.environ.get("MIX_STARTS"):
    cm.ADMIXTURE_STARTS = tuple(float(x) for x in os.environ["MIX_STARTS"].split(","))

print(
    f"MIXCONF module={cm.__name__} row_mode={getattr(cm, 'ROW_MODE', 'entropy')} "
    f"entropy={cm.ENTROPY_WEIGHT} starts={cm.ADMIXTURE_STARTS}",
    flush=True,
)
sys.argv = ["sim_audit", *sys.argv[1:]]

try:
    runpy.run_module("tests.sim_audit", run_name="__main__")
finally:
    from port.patch.integer_copy import DECODED

    for n, fit in enumerate(cm.FITS):
        print(
            f"MIXFIT {n} K={fit.weights.shape[0]} gain={fit.end - fit.start:.2f} "
            f"sweeps={fit.sweeps} W={np.round(fit.weights, 3).tolist()}"
        )
    for n, decode in enumerate(DECODED):
        print(
            f"DECODE {n} purity={np.round(np.asarray(decode.purity), 3).tolist()} "
            f"shifts={np.round(np.asarray(decode.shifts), 3).tolist()}"
        )
