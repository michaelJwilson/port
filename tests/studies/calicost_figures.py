"""#556: CalicoST's easy and hard samples staged as `port.sim` realizations, and their truth figures drawn, each on its own.

`python -m tests.studies.calicost_figures OUT_DIR [easy|hard ...]` writes
`OUT_DIR/<key>/qa/*.png` and `truth_combined.pdf`.

CalicoST writes the labels and the (A, B) profile, not a tree, a phase or a
manifest. Staged:

- the counts, reindexed to the normal baseline's genes by name
  (`baseline_counts`): `genomic_truth` places genes by the baseline's row
  order. The first of a duplicated name is kept; a gene the baseline lacks is
  dropped, one the counts lack is zero;
- the tree, from the profile (`tree_from`): the founder carries the segments
  every tumour clone shares away from (1, 1), each clone the segments where it
  departs from the founder -- the star under a founder that CalicoST's shared
  + local events draw;
- no phase: CalicoST writes A/B phased by haplotype, so `truth_phase.npy` is
  all False and `phase.png` is skipped;
- the manifest: `sim/manifests/dev_tree_1s.toml`'s genome, references and
  model, with one slice holding the sample's clones.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SAMPLES = {
    "easy": "numcnas1.2_cnasize5e7_ploidy2_random0",
    "hard": "numcnas6.3_cnasize1e7_ploidy2_random0",
}
"""CalicoST's committed samples, by the name the studies give them."""

SID = "calicost"
"""The staged slice's sample id."""


def baseline() -> pd.DataFrame:
    """`sim/normal_baseline.txt.gz`: the genes, their loci and lambda."""
    from port.sim.fixtures import SIM_ROOT

    return pd.read_csv(SIM_ROOT / "normal_baseline.txt.gz", sep="\t", comment="#")


def baseline_counts(adata: Any, genes: pd.Series) -> Any:
    """Spots x `genes` CSR UMI: `adata`'s columns by name, the first of a duplicate kept, a missing gene zero."""
    import scipy.sparse as sp

    first = pd.Series(np.arange(adata.n_vars), index=adata.var_names)
    first = first[~first.index.duplicated()]
    column = first.reindex(genes.astype(str)).to_numpy()
    kept = ~np.isnan(column)
    source = sp.csc_matrix(adata.X)
    picked = source[:, column[kept].astype(np.int64)].tocoo()
    target = np.flatnonzero(kept)
    rows, cols, vals = picked.row, target[picked.col], picked.data
    return sp.csr_matrix((vals, (rows, cols)), shape=(adata.n_obs, len(genes)))


def tree_from(profile: pd.DataFrame, clones: list[str]) -> pd.DataFrame:
    """`truth_tree.tsv` for a shared + local profile: a founder under `normal`, every clone under the founder."""
    rows: list[dict[str, Any]] = [
        {"node": "normal", "parent": None},
        {"node": "founder", "parent": "normal"},
    ]
    rows += [{"node": c, "parent": "founder"} for c in clones]
    states = {
        c: list(zip(profile[f"{c}_A_copy"], profile[f"{c}_B_copy"], strict=True))
        for c in clones
    }
    founder = []
    for k, (_, seg) in enumerate(profile.iterrows()):
        shared = {states[c][k] for c in clones}
        state = shared.pop() if len(shared) == 1 else (1, 1)
        founder.append(state)
        if state != (1, 1):
            rows.append({"node": "founder", "parent": "normal", "chr": seg["chr"], "start": seg["start"],
                         "end": seg["end"], "A": state[0], "B": state[1]})  # fmt: skip
    for c in clones:
        for k, (_, seg) in enumerate(profile.iterrows()):
            if states[c][k] != founder[k]:
                rows.append({"node": c, "parent": "founder", "chr": seg["chr"], "start": seg["start"],
                             "end": seg["end"], "A": states[c][k][0], "B": states[c][k][1]})  # fmt: skip
    return pd.DataFrame(
        rows, columns=["node", "parent", "chr", "start", "end", "A", "B"]
    )


def stage(key: str, out: Path) -> Path:
    """CalicoST `key` written as a realization directory under `out/key`."""
    import anndata
    from port.sim.fixtures import SIM_ROOT

    src = SIM_ROOT / SAMPLES[key]
    root = out / key
    (root / SID).mkdir(parents=True, exist_ok=True)
    (root / "snp").mkdir(exist_ok=True)
    adata = anndata.read_h5ad(src / "filtered_feature_bc_matrix.h5ad")
    truth = pd.read_csv(src / "truth_clone_labels.tsv", sep="\t", index_col=0)
    # NB the counts, the SNP matrices and the truth share one spot order: checked, since every figure reads it
    if list(adata.obs_names) != list(truth.index) or (
        src / "barcodes.txt"
    ).read_text().split() != list(truth.index):
        msg = f"{key}: the counts', SNPs' and truth's spot orders differ"
        raise ValueError(msg)
    genes = baseline()["gene"]
    staged = anndata.AnnData(baseline_counts(adata, genes), obs=adata.obs,
                             var=pd.DataFrame(index=genes.astype(str).to_numpy()))  # fmt: skip
    staged.write_h5ad(root / SID / "filtered_feature_bc_matrix.h5ad")
    labels = pd.DataFrame({"barcode": truth.index, "labels": truth["labels"], "x": truth["x"],
                           "y": truth["y"], "sample_id": SID})  # fmt: skip
    labels.to_csv(root / "truth_clone_labels.tsv", sep="\t", index=False)
    for name in ("cell_snp_Aallele.npz", "cell_snp_Ballele.npz", "unique_snp_ids.npy"):
        (root / "snp" / name).unlink(missing_ok=True)
        (root / "snp" / name).symlink_to(src / name)
    profile = pd.read_csv(src / "truth_acn_profile.tsv", sep="\t")
    profile.to_csv(root / "truth_acn_profile.tsv", sep="\t", index=False)
    clones = sorted(set(labels["labels"]) - {"normal"})
    tree_from(profile, clones).to_csv(root / "truth_tree.tsv", sep="\t", index=False)
    n_snps = np.load(src / "unique_snp_ids.npy", allow_pickle=True).size
    np.save(root / "truth_phase.npy", np.zeros(n_snps, dtype=bool))
    manifest = _manifest(clones, key)
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def _manifest(clones: list[str], key: str) -> dict[str, Any]:
    from port.sim import draw
    from port.sim.fixtures import REPOSITORY

    tables = draw.read_manifest(REPOSITORY / "sim/manifests/dev_tree_1s.toml").tables
    tables = json.loads(json.dumps(tables, default=str))
    tables["slice"] = [{"clones": clones, "offset": [0.0, 0.0]}]
    tables["sample"] = {"name": f"calicost {key}", "seed": None, "realization": 0}
    return dict(tables)


def main(argv: list[str] | None = None) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    from port.sim import analysis
    from port.sim.truth_figure import write_truth_combined

    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("keys", nargs="*", default=list(SAMPLES))
    arguments = parser.parse_args(argv)
    for key in arguments.keys:
        root = stage(key, arguments.out)
        realization = analysis.read(root)
        for plot in analysis.PLOTS:
            if plot is not analysis.plot_phase:
                print(key, plot(realization, root / "qa"))
        print(
            key, write_truth_combined(realization, root / "qa" / "truth_combined.pdf")
        )


if __name__ == "__main__":
    main()
