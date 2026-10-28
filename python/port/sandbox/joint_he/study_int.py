"""Study: the integer-copy mixture objective (joint_int.py) on a draw with simulated H&E, against the baseline run.

EM on L(r, theta, psi): per round, theta by Adam (gradient by the Fisher identity, spot chunks), then the
responsibilities r by damped mean-field sweeps (prior log pi + expected log-likelihood + Potts), then the CNN psi by
cross-entropy towards r. Scored by clone ARI and by integer copies against the planted profile.

Ticket: PR- #943.
Measurement: PR- #943.
Exit: with the package (`port.sandbox.joint_he`).

Usage: python -m port.sandbox.joint_he.study_int <draw> <baseline run dir> <out json>
    [--beta B] [--clones M] [--rounds R] [--seed S]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from port.sandbox.joint_he import joint as J
from port.sandbox.joint_he import joint_int as JI
from port.sandbox.joint_he import study as S
from port.sandbox.joint_he.joint import Psi, Theta

T_PHASE = 0.99999
"""The phase's stay probability, the baseline config's hmm.t_phaseing."""


def truth_per_bin(draw: Path, seg: pd.DataFrame) -> dict[str, NDArray[np.int64]]:
    """The planted (major, minor) per bin per truth clone: the narrowest truth row holding each bin's midpoint."""
    t = pd.read_csv(draw / "truth_acn_profile.tsv", sep="\t")
    clones = [c[: -len("_A_copy")] for c in t.columns if c.endswith("_A_copy")]
    mid = (seg["START"].to_numpy() + seg["END"].to_numpy()) / 2
    out = {c: np.ones((len(seg), 2), int) for c in clones}
    span = np.full(len(seg), np.inf)
    for _, row in t.iterrows():
        hit = (
            (seg["CHR"].astype(str).to_numpy() == str(row["chr"]))
            & (mid >= row["start"])
            & (mid < row["end"])
        )
        hit &= (row["end"] - row["start"]) <= span
        span[hit] = row["end"] - row["start"]
        for c in clones:
            a, b = row[f"{c}_A_copy"], row[f"{c}_B_copy"]
            out[c][hit] = (max(a, b), min(a, b))
    return out


def score_copies(
    truth_labels: NDArray[Any],
    labels: NDArray[Any],
    calls: dict[Any, NDArray[Any]],
    truth: dict[str, NDArray[np.int64]],
) -> dict[str, Any]:
    """Clones matched to truth clones by spot overlap (Hungarian); per bin, (major, minor) called exactly or not.
    `calls` maps a label to its (G, 2) (major, minor)."""
    t_codes = list(truth)
    l_codes = list(calls)
    C = np.array(
        [
            [np.sum((truth_labels == tc) & (labels == lc)) for tc in t_codes]
            for lc in l_codes
        ]
    )
    rows, cols = linear_sum_assignment(-C)
    # NB Python ints, numpy integers once a match is added.
    exact: Any = 0
    altered_hit: Any = 0
    altered_n: Any = 0
    normal_hit: Any = 0
    normal_n: Any = 0
    for r_, c_ in zip(rows, cols, strict=True):
        call, tr = calls[l_codes[r_]], truth[t_codes[c_]]
        same = np.all(call == tr, axis=1)
        alt = ~np.all(tr == (1, 1), axis=1)
        exact += same.sum()
        altered_hit += same[alt].sum()
        altered_n += alt.sum()
        normal_hit += same[~alt].sum()
        normal_n += (~alt).sum()
    G = len(next(iter(truth.values())))
    return {
        "exact": exact / (len(rows) * G),
        "altered_recall": altered_hit / max(altered_n, 1),
        "diploid_specificity": normal_hit / max(normal_n, 1),
        "matched": {l_codes[r]: t_codes[c] for r, c in zip(rows, cols, strict=True)},
    }


def main() -> None:
    """Run EM on a draw and write its log, labels, barcodes, paths and phase beside `out`."""
    ap = argparse.ArgumentParser()
    ap.add_argument("draw", type=Path)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--clones", type=int, default=6)
    ap.add_argument("--beta", type=float, default=20.0)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--theta_steps", type=int, default=10)
    ap.add_argument("--theta_warmup", type=int, default=30)
    ap.add_argument("--sweeps", type=int, default=10)
    ap.add_argument("--damping", type=float, default=0.5)
    ap.add_argument("--cnn_steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--free_omega", action="store_true")
    # NB 1 - hmm.t, the copy chain's switch probability
    ap.add_argument("--switch", type=float, default=1.0 - S.T_STAY)
    a = ap.parse_args()
    t0 = time.perf_counter()

    def tick(m: str) -> None:
        print(f"[{time.perf_counter() - t0:7.1f}s] {m}", flush=True)

    d = S.load(a.draw, a.run_dir)
    N, G = d["u"].shape
    M = a.clones
    known = pd.notna(d["truth"])
    D = JI.Data(d["u"], d["base"], d["b"], d["n"])
    starts = jnp.asarray(d["starts"])
    log_Tc, log_Tp = JI.copy_transitions(a.switch), JI.phase_transitions(T_PHASE)
    log_entry = JI.copy_entry(a.switch)
    i_e, j_e = (np.asarray(x) for x in d["edges"])
    log: dict[str, Any] = {
        "switch": a.switch,
        "N": N,
        "G": G,
        "copy_states": len(JI.COPY),
        "M": M,
        "beta": a.beta,
        "seed": a.seed,
        "rounds": [],
    }

    # NB the CNN warm-started to the data-only partition, as study.py; r starts at its prior.
    init = S.init_labels(d, M, a.seed)
    psi: Psi = {"cnn": J.init_cnn(jax.random.PRNGKey(a.seed), M)}
    onehot = jax.nn.one_hot(jnp.asarray(init), M)

    def log_prior(p: Psi) -> jax.Array:
        logits = jnp.concatenate(
            [
                J.cnn(p["cnn"], img)[px[:, 0] // J.STRIDE, px[:, 1] // J.STRIDE]
                for img, px in zip(d["images"], d["pixels"], strict=True)
            ]
        )
        return jax.nn.log_softmax(logits.astype(jnp.float64), axis=1)

    def cross_entropy(p: Psi, target: jax.Array) -> jax.Array:
        return jnp.sum(target * log_prior(p)) - 1e-3 * sum(
            jnp.sum(layer["w"] ** 2) for layer in p["cnn"]
        )

    ce = jax.jit(jax.value_and_grad(cross_entropy))
    st = J.adam_init(psi)
    for _ in range(300):
        _, g = ce(psi, onehot)
        psi, st = J.adam_step(psi, g, st, 0.01)
    lp = log_prior(psi)
    r = jnp.exp(lp)
    log["init_ari_vs_truth"] = float(
        adjusted_rand_score(d["truth"][known], init[known])
    )
    log["prior_ari_vs_truth"] = float(
        adjusted_rand_score(d["truth"][known], np.asarray(r.argmax(1))[known])
    )
    tick(f"warm start; prior ARI {log['prior_ari_vs_truth']:.3f}")

    # NB purity starts at 0.9 (logit over [RHO_MIN, 1]); the dispersions at the earlier per-spot fits.
    theta: Theta = {
        "omega": jnp.asarray(0.0),
        "logit_rho": jnp.log((0.9 - JI.RHO_MIN) / (1.0 - 0.9)),
        "log_alpha": jnp.log(1.0),
        "log_tau": jnp.log(200.0),
    }
    st_t = J.adam_init(theta)

    qs = jnp.full(
        (G, 2), 0.5
    )  # NB the global phase's posterior, kept warm across every pass

    def posteriors(
        theta: Theta,
        r: jax.Array,
        qs: jax.Array,
        passes: int = 2,
        W: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        e = D.e(theta, r, W)
        for _ in range(passes):
            v, gamma, qs = JI.mean_field(e, qs, log_Tc, log_Tp, starts, log_entry)
        return v, gamma, qs

    def theta_steps(
        theta: Theta, st_t: dict[str, Any], r: jax.Array, qs: jax.Array, steps: int
    ) -> tuple[Theta, dict[str, Any], float, jax.Array, jax.Array]:
        W = D.pair_counts(
            r
        )  # NB r is fixed across theta's steps, so its (b, n) counts are too
        for _ in range(steps):
            v, gamma, qs = posteriors(theta, r, qs, W=W)
            g = D.theta_grad(theta, r, gamma, qs, W)
            # NB omega is held at 0 unless freed: the baseline is normal coverage, so a normal spot's diploid reads at
            #    mu = 1, as cnamaste's normal clone fixes the scale at s = 0. Freed, it fell to -0.55 and the genome
            #    doubled ((2, 2) in place of (1, 1)), the tie (2a, 2b) at omega - ln 2 against (a, b) at omega.
            g["omega"] = (
                g["omega"] - 10.0 * theta["omega"]
                if a.free_omega
                else jnp.zeros_like(g["omega"])
            )
            theta, st_t = J.adam_step(theta, g, st_t, 0.05)
        return theta, st_t, float(v), gamma, qs

    theta, st_t, v, gamma, qs = theta_steps(theta, st_t, r, qs, a.theta_warmup)
    tick(f"theta warm-up ELBO {v:.1f}")

    for rnd in range(a.rounds):
        theta, st_t, v, gamma, qs = theta_steps(theta, st_t, r, qs, a.theta_steps)
        for _ in range(a.sweeps):
            _, gamma, qs = posteriors(theta, r, qs, passes=1)
            s = D.scores(theta, gamma, qs)
            nb = J.neighbour_sum(r, i_e, j_e)
            r = a.damping * r + (1 - a.damping) * jax.nn.softmax(
                lp + s + a.beta * nb, axis=1
            )
        for _ in range(a.cnn_steps):
            _, g = ce(psi, r)
            psi, st = J.adam_step(psi, g, st, 0.01)
        lp = log_prior(psi)
        lab = np.asarray(r.argmax(1))
        log["rounds"].append(
            {
                "round": rnd,
                "elbo": v,
                "phase_flipped": float(jnp.mean(qs[:, 1] > 0.5)),
                "clones_used": len(np.unique(lab)),
                "ari_vs_truth": float(
                    adjusted_rand_score(d["truth"][known], lab[known])
                ),
                "prior_ari_vs_truth": float(
                    adjusted_rand_score(
                        d["truth"][known], np.asarray(lp.argmax(1))[known]
                    )
                ),
                "omega": float(theta["omega"]),
                "alpha": float(jnp.exp(theta["log_alpha"])),
                "tau": float(jnp.exp(theta["log_tau"])),
                "rho": float(JI.purity(theta)),
                "seconds": round(time.perf_counter() - t0, 1),
            }
        )
        print(json.dumps(log["rounds"][-1]), flush=True)

    # NB hard labels, each clone's MAP integer path, then clones whose (major, minor) paths agree on >= 0.99 of bins
    #    are one (int_clone_agreement).
    lab = np.asarray(r.argmax(1))
    used = np.unique(lab)
    hard = jax.nn.one_hot(jnp.asarray(lab), M)
    _, gamma, qs = posteriors(theta, hard, qs, passes=3)
    ebar = jnp.einsum("mgcs,gs->mgc", D.e(theta, hard), qs) + JI.parsimony()
    paths = JI.viterbi(ebar, log_Tc, d["starts"], log_entry)
    copies = JI.COPY[paths]  # (M, G, 2) major, minor
    merged = S.merge_by_paths(
        lab, {m: copies[m, :, 0] * 10 + copies[m, :, 1] for m in range(M)}, 0.99
    )
    seg = pd.read_csv(a.run_dir / "cnv_diploid_seglevel.tsv", sep="\t")[
        ["CHR", "START", "END"]
    ]
    truth = truth_per_bin(a.draw, seg)
    t_lab = np.asarray(pd.Series(d["truth"]).fillna("na"))
    joint_score = score_copies(
        t_lab[known], merged[known], {m: copies[m] for m in np.unique(merged)}, truth
    )

    # NB the baseline on the same scorer: its clones and its integer copies.
    cl = (
        pd.read_csv(a.run_dir / "clone_labels.tsv.gz", sep="\t")
        .set_index("barcode")["clone_label"]
        .reindex(d["barcodes"])
        .to_numpy()
    )
    sl = pd.read_csv(a.run_dir / "cnv_diploid_seglevel.tsv", sep="\t")
    base_calls = {
        int(c): np.stack(
            [
                np.maximum(sl[f"clone{c} A"], sl[f"clone{c} B"]),
                np.minimum(sl[f"clone{c} A"], sl[f"clone{c} B"]),
            ],
            1,
        ).astype(int)
        for c in np.unique(cl)
    }
    base_score = score_copies(t_lab[known], cl[known], base_calls, truth)

    log["final"] = {
        "clones": len(used),
        "clones_merged": len(np.unique(merged)),
        "ari_vs_truth": float(adjusted_rand_score(d["truth"][known], lab[known])),
        "ari_vs_truth_merged": float(
            adjusted_rand_score(d["truth"][known], merged[known])
        ),
        "copies": {k: v for k, v in joint_score.items() if k != "matched"},
        "baseline_ari": float(adjusted_rand_score(d["truth"][known], cl[known])),
        "baseline_copies": {k: v for k, v in base_score.items() if k != "matched"},
        "seconds": round(time.perf_counter() - t0, 1),
    }
    np.save(a.out.with_suffix(".labels.npy"), merged)
    np.save(a.out.with_suffix(".barcodes.npy"), d["barcodes"])
    np.save(a.out.with_suffix(".paths.npy"), paths)
    np.save(a.out.with_suffix(".phase.npy"), np.asarray(qs))
    a.out.write_text(json.dumps(log, indent=1, default=float))
    print(json.dumps(log["final"], default=float))


if __name__ == "__main__":
    main()
