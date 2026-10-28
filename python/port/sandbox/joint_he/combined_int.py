"""combined.png for an integer-copy joint result (study_int.py), drawn by cnamaste's own page (layout.plot_combined_page).

The copy pairs are the model's own: each clone's Viterbi path over (A, B), no rounding. A state's line on the tracks
is its emission mean at the fitted global (omega, rho): mu(a, b) e^omega and p(a, b, 0).

Ticket: PR- #943.
Measurement: PR- #943.
Exit: with the package (`port.sandbox.joint_he`).

Usage: python -m port.sandbox.joint_he.combined_int <draw> <baseline run dir> <arm json> <out.png>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

from port.sandbox.joint_he import joint_int as JI
from port.sandbox.joint_he import study as S
from port.sandbox.joint_he.study import from_npz


def main() -> None:
    """Draw the page for one arm's result."""
    ap = argparse.ArgumentParser()
    for name in ("draw", "run_dir", "arm", "out"):
        ap.add_argument(name, type=Path)
    args = ap.parse_args()
    draw, run_dir, arm, out = args.draw, args.run_dir, args.arm, args.out

    # NB `cnamaste` is vendored and built separately, not installed in port's
    #    environment, so it is imported only when the page is drawn.
    from cnamaste.he import get_he_image
    from cnamaste.layout import plot_combined_page, write_page

    data = np.load(run_dir / "binned_data.npz", allow_pickle=True)
    X = from_npz(data, "single_X")
    rdr, baf = S.dense(X.rdr), S.dense(X.baf)  # (G, N), the run's barcode order
    base = S.dense(from_npz(data, "single_base_nb_mean"))
    total = S.dense(from_npz(data, "single_total_bb_RD"))
    lengths = np.asarray(data["lengths"])
    barcodes = np.asarray(data["barcodes"]).astype(str)
    single_X = np.stack([rdr, baf], axis=1)
    G, N = rdr.shape

    # NB merged labels name their representative clone, whose path is the clone's; renumbered 0.. by size.
    lab = pd.Series(
        np.load(arm.with_suffix(".labels.npy")),
        index=np.load(arm.with_suffix(".barcodes.npy")).astype(str),
    )
    lab = lab.reindex(barcodes).to_numpy()
    paths = np.load(arm.with_suffix(".paths.npy"))  # (M, G) copy-state index per clone
    codes, counts = np.unique(lab, return_counts=True)
    order = codes[np.argsort(-counts)]
    assignment_int = np.array([int(np.flatnonzero(order == c)[0]) for c in lab])
    pred = np.stack([paths[c] for c in order], axis=1)  # (G, clones)

    last = json.loads(arm.read_text())["rounds"][-1]
    theta = {
        "omega": jnp.asarray(last["omega"]),
        "logit_rho": jnp.asarray(
            np.log((last["rho"] - JI.RHO_MIN) / (1.0 - last["rho"]))
        ),
        "log_alpha": jnp.log(last["alpha"]),
        "log_tau": jnp.log(last["tau"]),
    }
    rho = float(JI.purity(theta))
    mu = (rho * JI.T_OF / 2.0 + 1.0 - rho) * np.exp(last["omega"])
    p = np.asarray(JI.state_p(theta))[:, 0]

    seg = pd.read_csv(run_dir / "cnv_diploid_seglevel.tsv", sep="\t")[
        ["CHR", "START", "END"]
    ]
    if len(seg) != G:
        msg = f"{len(seg)} segments in cnv_diploid_seglevel.tsv for {G} bins"
        raise ValueError(msg)
    df_cnv = seg.copy()
    for c in range(len(order)):
        k = pred[:, c]
        df_cnv[f"clone{c} Z"] = k
        df_cnv[f"clone{c} logmu"] = np.log(mu[k])
        df_cnv[f"clone{c} p"] = p[k]
        df_cnv[f"clone{c} A"] = JI.COPY[k, 0]
        df_cnv[f"clone{c} B"] = JI.COPY[k, 1]
    res_combine = {
        "new_assignment": assignment_int,
        "pred_cnv": pred,
        "new_log_mu": np.log(mu),
        "new_p_binom": p,
    }

    cl = (
        pd.read_csv(run_dir / "clone_labels.tsv.gz", sep="\t")
        .set_index("barcode")
        .reindex(barcodes)
    )
    coords = cl[["x", "y"]].to_numpy().astype(float)
    sample_list = list(dict.fromkeys(cl["sample_id"].astype(str)))
    sample_ids = np.array([sample_list.index(s) for s in cl["sample_id"].astype(str)])
    frames, x_offset = [], 0.0
    for s, sid in enumerate(sample_list):
        fr = get_he_image(
            spaceranger_dir=draw / sid, res="hires", pos=None, num_labels=4
        )
        cols = fr["array_col"].to_numpy()
        scale = float(fr["x"].to_numpy()[cols > 0][0] / cols[cols > 0][0])
        fr = fr.copy()
        fr["array_col"] = fr["array_col"] + round(x_offset / scale)
        fr["x"] = fr["array_col"] * scale
        frames.append(fr)
        x_offset += np.max(coords[sample_ids == s, 0]) + 10
    he_frame = pd.concat(frames, ignore_index=True)

    assignment = pd.Series([f"clone {x}" for x in assignment_int])
    fig = plot_combined_page(
        lengths,
        single_X,
        base,
        total,
        df_cnv,
        res_combine,
        coords,
        assignment,
        sample_list=sample_list,
        sample_ids=sample_ids,
        he_frame=he_frame,
    )
    write_page(str(out), fig)
    print(
        json.dumps(
            {
                "clones": len(order),
                "rho": rho,
                "omega": last["omega"],
                "non_diploid_share": {
                    c: float(
                        np.mean(
                            ~(
                                (JI.COPY[pred[:, c], 0] == 1)
                                & (JI.COPY[pred[:, c], 1] == 1)
                            )
                        )
                    )
                    for c in range(len(order))
                },
            }
        )
    )


if __name__ == "__main__":
    main()
