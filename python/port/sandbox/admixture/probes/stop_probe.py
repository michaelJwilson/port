# ruff: noqa
# mypy: ignore-errors
# Design-study probe (#380), kept as run; not package code.
"""sim_probe, stopped after cnaster's normal-candidate step; scores BAF-only stage."""

import json, os, runpy, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path[:0] = [str(Path.cwd() / "python"), str(Path.cwd())]
import port.extensions.clone_mixture as cm

cm.ROW_MODE = os.environ.get("MIX_ROW_MODE", cm.ROW_MODE)
if os.environ.get("MIX_STARTS"):
    cm.ADMIXTURE_STARTS = tuple(float(x) for x in os.environ["MIX_STARTS"].split(","))
print(f"MIXCONF row_mode={cm.ROW_MODE} starts={cm.ADMIXTURE_STARTS}", flush=True)
import cnaster.scripts.run_cnaster as rc


class Stop(Exception):
    pass


seen = {}
original = rc.determine_normal_candidates


def stopped(config, merged_res, *args, **kwargs):
    out = original(config, merged_res, *args, **kwargs)
    seen["config"], seen["normal"] = config, np.asarray(out, dtype=bool)
    seen["assignment"] = np.asarray(merged_res["new_assignment"])
    raise Stop


rc.determine_normal_candidates = stopped
sys.argv = ["sim_audit"] + sys.argv[1:]
try:
    runpy.run_module("tests.sim_audit", run_name="__main__")
except Stop:
    pass
finally:
    from sklearn.metrics import adjusted_rand_score

    fits = list(cm.FITS)
    for n, f in enumerate(fits):
        print(
            f"MIXFIT {n} W={np.round(f.weights, 3).tolist()} gain={f.end - f.start:.1f}"
        )
    if "config" in seen:
        config = seen["config"]
        output = Path(config.paths.output_dir)
        labels = pd.read_csv(output / "baf_clone_labels.tsv", sep="\t", index_col=0)
        sheet = pd.read_csv(config.paths.sample_sheet, sep="\t")
        truth = pd.read_csv(
            Path(sheet["spaceranger_dir"][0]) / "truth_clone_labels.tsv",
            sep="\t",
            index_col=0,
        )
        truth = truth.iloc[:, 0]
        names = sorted(truth.unique(), key=lambda s: (s != "normal", s))
        code = {c: i for i, c in enumerate(names)}
        barcodes = (
            labels.index.astype(str).str.split("_").str[0]
            if not labels.index.isin(truth.index).all()
            else labels.index
        )
        t = truth.reindex(labels.index).map(code).to_numpy()
        p = seen["assignment"]
        ari = adjusted_rand_score(t, p)
        # inferred clone -> majority truth clone
        mapping = {
            int(c): names[int(np.bincount(t[p == c]).argmax())] for c in np.unique(p)
        }
        normal = seen["normal"]
        is_normal = t == 0
        tp = int((normal & is_normal).sum())
        rec = {
            "baf_ari": round(ari, 4),
            "n_clones": int(np.unique(p).size),
            "clone_of": mapping,
            "normal_candidates": int(normal.sum()),
            "true_normal": int(is_normal.sum()),
            "precision": round(tp / max(normal.sum(), 1), 4),
            "recall": round(tp / max(is_normal.sum(), 1), 4),
        }
        print("STOP " + json.dumps(rec), flush=True)
