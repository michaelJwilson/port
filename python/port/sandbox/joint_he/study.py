"""Study: joint differentiable inference (joint.py) on a draw with simulated H&E, against the baseline cnamaste run.

Inputs are the baseline run's own preprocessing (binned_data.npz: per-spot counts per bin, the normal baseline, the
phased B counts, the contig lengths), so the two differ only in inference. Scored by clone ARI against the planted
clones, and by agreement with the baseline's labels.

Ticket: PR- #943.
Measurement: PR- #943.
Exit: with the package (`port.sandbox.joint_he`).

Usage: python -m port.sandbox.joint_he.study <draw> <baseline run dir> <out json>
    [--arm cnn|cnn_delta|free|constant|meanfield] [--clones M] [--beta B]
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from port.sandbox.joint_he import joint as J
from port.sandbox.joint_he.joint import Psi, Theta


class _Counts:
    """single_X's two count matrices."""

    def __init__(self, rdr: Any, baf: Any) -> None:
        self.rdr, self.baf = rdr, baf


def from_npz(data: Any, key: str) -> Any:
    """binned_data.npz's arrays: a CSR matrix under key__{data,indices,indptr,shape}, a pair of them (rdr, baf) for
    single_X, or a dense array under the bare key."""

    def csr(k: str) -> Any:
        return sp.csr_matrix(
            (data[k + "__data"], data[k + "__indices"], data[k + "__indptr"]),
            shape=tuple(data[k + "__shape"]),
        )

    if key in data.files:
        return np.asarray(data[key])
    if key + "__rdr__data" in data.files:
        return _Counts(csr(key + "__rdr"), csr(key + "__baf"))
    return csr(key)


T_STAY = 0.9999999
"""The chain's stay probability, fixed: the baseline config's hmm.t, not learned (a learned t fell to 0.967 and let every
clone's path follow noise)."""

TAU0 = 1000.0
"""The BAF concentration's start, the baseline config's betabinom.start_disp."""

DOWNSAMPLE = 4
"""The H&E is read at 1/4 of its 24 pixels per spot: 6 pixels per spot, a nucleus a pixel or less."""


def dense(x: Any) -> NDArray[np.float64]:
    """A sparse or dense matrix as a float64 array."""
    a = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    return np.asarray(a, dtype=np.float64)


def load(draw: Path, run_dir: Path) -> dict[str, Any]:
    """The run's counts, reordered to the network's spot order, with the H&E, spot graph and planted labels."""
    data = np.load(run_dir / "binned_data.npz", allow_pickle=True)
    X = from_npz(data, "single_X")
    u = dense(X.rdr).T  # (N, G) read counts
    b = dense(X.baf).T  # (N, G) phased B counts
    base = dense(from_npz(data, "single_base_nb_mean")).T
    n = dense(from_npz(data, "single_total_bb_RD")).T
    lengths = np.asarray(data["lengths"])
    barcodes = np.asarray(data["barcodes"]).astype(str)
    starts = np.zeros(u.shape[1], bool)
    starts[np.concatenate([[0], np.cumsum(lengths)[:-1]])] = True

    truth = pd.read_csv(draw / "truth_clone_labels.tsv", sep="\t").set_index("barcode")[
        "labels"
    ]
    sheet = pd.read_csv(draw / "sample_sheet.tsv", sep="\t")
    pos: list[NDArray[Any]] = []
    images: list[jax.Array] = []
    pixels: list[jax.Array] = []
    slice_of = np.zeros(len(barcodes), int)
    index = {bc: i for i, bc in enumerate(barcodes)}
    order_by_slice: list[NDArray[Any]] = []
    for s, sid in enumerate(sheet["sample_id"].astype(str)):
        p = pd.read_csv(
            draw / sid / "spatial" / "tissue_positions_list.csv",
            header=None,
            names=["barcode", "in_tissue", "row", "col", "px", "py"],
        )
        p = p[p["barcode"].isin(index)]
        img = mpimg.imread(draw / sid / "spatial" / "tissue_hires_image.png")[..., :3]
        H, W = img.shape[0] // DOWNSAMPLE, img.shape[1] // DOWNSAMPLE
        img = (
            img[: H * DOWNSAMPLE, : W * DOWNSAMPLE]
            .reshape(H, DOWNSAMPLE, W, DOWNSAMPLE, 3)
            .mean(axis=(1, 3))
        )
        # NB optical density, standardized: the stain's own linear space.
        od = -np.log(np.clip(img, 1e-3, 1.0))
        od = (od - od.mean(axis=(0, 1))) / (od.std(axis=(0, 1)) + 1e-6)
        scale = 24 / DOWNSAMPLE
        px = np.stack(
            [
                np.clip((p["col"].to_numpy() * scale).astype(int), 0, H - 1),
                np.clip((p["row"].to_numpy() * scale).astype(int), 0, W - 1),
            ],
            axis=1,
        )
        images.append(jnp.asarray(od))
        pixels.append(jnp.asarray(px))
        idx = p["barcode"].map(index).to_numpy()
        order_by_slice.append(idx)
        slice_of[idx] = s
        pos.append(np.stack([p["row"].to_numpy(), p["col"].to_numpy()], axis=1))
    order = np.concatenate(order_by_slice)
    # NB the network's spots come slice by slice in tissue_positions order; reorder every array to match.
    u, b, base, n, barcodes = u[order], b[order], base[order], n[order], barcodes[order]
    coords = np.concatenate(pos)
    # NB the spot graph: each spot's 8 nearest within its slice, as the baseline's (lattice_type unset) is.
    edge_blocks = []
    offset = 0
    for pp in pos:
        _, nn = cKDTree(pp).query(pp, k=9)
        i = np.repeat(np.arange(len(pp)), 8) + offset
        j = nn[:, 1:].ravel() + offset
        keep = i < j
        edge_blocks.append(np.stack([i[keep], j[keep]]))
        offset += len(pp)
    edges = np.unique(np.concatenate(edge_blocks, axis=1), axis=1)
    labels = truth.reindex(barcodes).to_numpy()
    return {
        "u": u,
        "b": b,
        "base": base,
        "n": n,
        "starts": starts,
        "barcodes": barcodes,
        "images": images,
        "pixels": pixels,
        "edges": jnp.asarray(edges),
        "coords": coords,
        "truth": labels,
    }


def init_labels(d: dict[str, Any], M: int, seed: int = 0) -> NDArray[np.int32]:
    """k-means on each spot's coarse, neighbour-smoothed RDR and BAF: the starting partition (data only)."""
    G = d["u"].shape[1]
    w = 50
    groups = np.arange(G) // w

    def agg(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.stack([x[:, groups == k].sum(1) for k in np.unique(groups)], axis=1)

    rdr = np.log((agg(d["u"]) + 1) / (agg(d["base"]) + 1))
    baf = np.abs((agg(d["b"]) + 1) / (agg(d["n"]) + 2) - 0.5)
    f = np.concatenate([rdr, baf], axis=1)
    i, j = np.asarray(d["edges"])
    for _ in range(2):
        s = f.copy()
        c = np.ones(len(f))
        np.add.at(s, i, f[j])
        np.add.at(s, j, f[i])
        np.add.at(c, i, 1)
        np.add.at(c, j, 1)
        f = s / c[:, None]
    f = (f - f.mean(0)) / (f.std(0) + 1e-9)
    z = PCA(10, random_state=seed).fit_transform(f)
    labels: NDArray[np.int32] = KMeans(M, n_init=10, random_state=seed).fit_predict(z)
    return labels


def init_theta(d: dict[str, Any], labels: NDArray[np.integer], K: int) -> Theta:
    """States from k-means on the init clones' pseudobulk windows (initialization only): mu, p per state."""
    rows = []
    for m in np.unique(labels):
        sel = labels == m
        for g0 in range(0, d["u"].shape[1], 25):
            sl = slice(g0, g0 + 25)
            rows.append(
                [
                    d["u"][sel, sl].sum() / max(d["base"][sel, sl].sum(), 1e-9),
                    (d["b"][sel, sl].sum() + 1) / (d["n"][sel, sl].sum() + 2),
                ]
            )
    pts = np.asarray(rows)
    # NB standardized: unscaled, BAF's range (~0.25) is lost beside log RDR's and no state sits at the LOH windows.
    x = np.column_stack([np.log(pts[:, 0] + 1e-3), pts[:, 1]])
    mu, sd = x.mean(0), x.std(0) + 1e-9
    c = (
        KMeans(K, n_init=10, random_state=0).fit((x - mu) / sd).cluster_centers_ * sd
        + mu
    )
    return {
        "log_mu": jnp.asarray(c[:, 0]),
        "logit_p": jnp.asarray(
            np.log(np.clip(c[:, 1], 0.02, 0.98) / (1 - np.clip(c[:, 1], 0.02, 0.98)))
        ),
        "log_alpha": jnp.asarray(np.log(0.05)),
        "log_tau": jnp.asarray(np.log(TAU0)),
        "logit_t": jnp.asarray(np.log(T_STAY / (1 - T_STAY))),
    }


def merge_by_paths(
    labels: NDArray[Any],
    paths: NDArray[Any] | Mapping[Any, NDArray[Any]],
    agreement: float = 0.99,
) -> NDArray[Any]:
    """Clones whose MAP copy paths agree on >= `agreement` of bins are one clone (as int_clone_agreement)."""
    used = list(np.unique(labels))
    rep = {m: m for m in used}
    for a_i, a in enumerate(used):
        for bb in used[:a_i]:
            if rep[bb] == bb and np.mean(paths[a] == paths[bb]) >= agreement:
                rep[a] = bb
                break
    return np.array([rep[m] for m in labels])


def main() -> None:
    """Run one arm on a draw and write its log, labels and barcodes beside `out`."""
    ap = argparse.ArgumentParser()
    ap.add_argument("draw", type=Path)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument(
        "--arm",
        default="cnn",
        choices=["cnn", "cnn_delta", "free", "constant", "meanfield"],
    )
    ap.add_argument("--clones", type=int, default=6)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--label_steps", type=int, default=150)
    ap.add_argument("--theta_steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweeps", type=int, default=20)
    ap.add_argument("--damping", type=float, default=0.5)
    ap.add_argument("--theta_warmup", type=int, default=60)
    ap.add_argument("--theta_lr", type=float, default=0.05)
    a = ap.parse_args()
    t0 = time.perf_counter()
    d = load(a.draw, a.run_dir)

    def tick(m: str) -> None:
        print(f"[{time.perf_counter() - t0:7.1f}s] {m}", flush=True)

    tick(f"loaded {[img.shape for img in d['images']]}")
    N, G = d["u"].shape
    K = 7
    M = a.clones
    known = pd.notna(d["truth"])
    log: dict[str, Any] = {
        "arm": a.arm,
        "N": N,
        "G": G,
        "K": K,
        "M": M,
        "beta": a.beta,
        "rounds": [],
    }

    init = init_labels(d, M, a.seed)
    theta = init_theta(d, init, K)
    tick("init")
    u, b, base, n = (jnp.asarray(d[k]) for k in ("u", "b", "base", "n"))
    starts = jnp.asarray(d["starts"])
    if a.arm == "constant":
        d["images"] = [jnp.zeros_like(img) for img in d["images"]]
    key = jax.random.PRNGKey(a.seed)
    psi: Psi = {"cnn": J.init_cnn(key, M), "delta": jnp.zeros((N, M))}
    use_delta = a.arm in ("cnn_delta", "free")
    if a.arm == "free":
        psi["cnn"] = jax.tree_util.tree_map(jnp.zeros_like, psi["cnn"])

    # NB warm start: the assignment fit to the k-means partition by cross-entropy (the network, or the free logits).
    onehot = jax.nn.one_hot(jnp.asarray(init), M)

    def warm_loss(p: Psi) -> jax.Array:
        q = J.assignment(
            p, d["images"], d["pixels"], 1.0, p["delta"] if use_delta else None
        )
        return jnp.sum(onehot * jnp.log(q + 1e-9)) - 1e-3 * sum(
            jnp.sum(layer["w"] ** 2) for layer in p["cnn"]
        )

    warm_grad = jax.jit(jax.value_and_grad(warm_loss))
    st = J.adam_init(psi)
    for _ in range(300 if a.arm != "meanfield" else 0):
        v, g = warm_grad(psi)
        if a.arm == "free":
            g["cnn"] = jax.tree_util.tree_map(jnp.zeros_like, g["cnn"])
        psi, st = J.adam_step(psi, g, st, 0.05 if a.arm == "free" else 0.01)
    tick("warm start")
    q0 = J.assignment(
        psi, d["images"], d["pixels"], 1.0, psi["delta"] if use_delta else None
    )
    log["warm_start_ari_vs_init"] = float(
        adjusted_rand_score(init, np.asarray(q0.argmax(1)))
    )
    log["init_ari_vs_truth"] = float(
        adjusted_rand_score(d["truth"][known], init[known])
    )

    def theta_steps(
        theta: Theta, st_t: dict[str, Any], q: jax.Array, steps: int
    ) -> tuple[Theta, dict[str, Any], jax.Array]:
        for _ in range(steps):
            vt, gt = J.theta_value_and_grad(theta, q, u, base, b, n, starts)
            gt["logit_t"] = jnp.zeros_like(gt["logit_t"])  # NB t is fixed (T_STAY)
            theta, st_t = J.adam_step(theta, gt, st_t, a.theta_lr)
        return theta, st_t, vt

    st_t = J.adam_init(theta)
    if a.arm == "meanfield":
        # NB no H&E in this arm: theta warms up on the data-only partition
        q0 = jax.nn.one_hot(jnp.asarray(init), M)
    theta, st_t, vt = theta_steps(theta, st_t, q0, a.theta_warmup)
    tick(f"theta warm-up logF {float(vt):.1f}")
    i_e, j_e = d["edges"]

    @jax.jit
    def mf_field(
        q: jax.Array, L: jax.Array, lt: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """The block maximizer's scores: s_nm = dlogF/dq_nm + beta sum_{n'~n} q_n'm (the Potts term's gradient)."""
        e = jnp.einsum("nm,ngk->mgk", q, L)
        gamma = jax.grad(lambda e: jnp.sum(J.log_forward(e, lt, starts)))(e)
        s = jnp.einsum("mgk,ngk->nm", gamma, L)
        nb = J.neighbour_sum(q, i_e, j_e)
        return s + a.beta * nb, jnp.sum(J.log_forward(e, lt, starts))

    def meanfield(
        q: jax.Array,
        L: jax.Array,
        lt: jax.Array,
        eta: float,
        sweeps: int,
        damping: float,
    ) -> tuple[jax.Array, jax.Array]:
        """J's exact block ascent in q at fixed theta: q_n = softmax(s_n / eta), damped (synchronous updates)."""
        for _ in range(sweeps):
            s, v = mf_field(q, L, lt)
            q = damping * q + (1 - damping) * jax.nn.softmax(s / eta, axis=1)
        return q, v

    q = q0 if a.arm != "meanfield" else jax.nn.one_hot(jnp.asarray(init), M)
    lab_grad = jax.jit(jax.value_and_grad(J.objective_labels), static_argnums=(11,))
    L_fn = J.spot_loglik
    st_q = J.adam_init(psi)
    for r in range(a.rounds):
        frac = r / max(a.rounds - 1, 1)
        # NB annealed: towards a one-hot assignment
        eps, eta = 1.0 - 0.8 * frac, 1.0 - 0.95 * frac
        L = L_fn(theta, u, base, b, n)
        L.block_until_ready()
        tick("L")
        t = jax.nn.sigmoid(theta["logit_t"])
        lt = J.log_transitions(t, K)
        if a.arm == "meanfield":
            q, v = meanfield(q, L, lt, eta, a.sweeps, a.damping)
            tick("labels")
        for _ in range(a.label_steps if a.arm != "meanfield" else 0):
            v, g = lab_grad(
                psi,
                L,
                lt,
                starts,
                d["images"],
                d["pixels"],
                d["edges"],
                a.beta,
                eta,
                1e-3,
                eps,
                use_delta,
            )
            if a.arm == "free":
                g["cnn"] = jax.tree_util.tree_map(jnp.zeros_like, g["cnn"])
            psi, st_q = J.adam_step(psi, g, st_q, 0.05 if a.arm == "free" else 0.01)
        if a.arm != "meanfield":
            tick("labels")
            q = J.assignment(
                psi, d["images"], d["pixels"], eps, psi["delta"] if use_delta else None
            )
        theta, st_t, vt = theta_steps(theta, st_t, q, a.theta_steps)
        lab = np.asarray(q.argmax(1))
        log["rounds"].append(
            {
                "round": r,
                "J_labels": float(v),
                "logF": float(vt),
                "eps": eps,
                "eta": eta,
                "clones_used": len(np.unique(lab)),
                "ari_vs_truth": float(
                    adjusted_rand_score(d["truth"][known], lab[known])
                ),
                "seconds": round(time.perf_counter() - t0, 1),
            }
        )
        print(json.dumps(log["rounds"][-1]), flush=True)

    # NB final labels, copy paths and the merge of clones with the same path.
    if a.arm != "meanfield":
        q = J.assignment(
            psi, d["images"], d["pixels"], 0.2, psi["delta"] if use_delta else None
        )
    lab = np.asarray(q.argmax(1))
    L = L_fn(theta, u, base, b, n)
    e = jnp.einsum("nm,ngk->mgk", jax.nn.one_hot(jnp.asarray(lab), M), L)
    t = jax.nn.sigmoid(theta["logit_t"])
    paths = J.viterbi(e, np.asarray(J.log_transitions(t, K)), d["starts"])
    merged = merge_by_paths(lab, paths)
    # NB "theta" is filled below; `J.unpack` returns a plain tuple, so it starts as None.
    log["final"] = {
        "clones": len(np.unique(lab)),
        "clones_merged": len(np.unique(merged)),
        "ari_vs_truth": float(adjusted_rand_score(d["truth"][known], lab[known])),
        "ari_vs_truth_merged": float(
            adjusted_rand_score(d["truth"][known], merged[known])
        ),
        "theta": None,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    mu, p, alpha, tau, tt = J.unpack(theta)
    log["final"]["theta"] = {
        "mu": np.asarray(mu).round(3).tolist(),
        "p": np.asarray(p).round(3).tolist(),
        "alpha": float(alpha),
        "tau": float(tau),
        "t": float(tt),
    }
    np.save(a.out.with_suffix(".labels.npy"), merged)
    np.save(a.out.with_suffix(".barcodes.npy"), d["barcodes"])
    a.out.write_text(json.dumps(log, indent=1))
    print(json.dumps(log["final"]))


if __name__ == "__main__":
    main()
