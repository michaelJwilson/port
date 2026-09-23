"""Integer copies decoded from the error bars, against realizations (#353).

Run as `python -m tests.copy_audit [--realizations 8] [--output PATH]`.

`tests.realizations`' genome, planted on `COPY_LATTICE` so every state is an
integer `(A, B)`, is realized `N` times; each realization is run through
`run_cnaster_port` and its final fit decoded twice:

- **from the error bars** (`port.extensions.copy_errors`): every `(A, B)`
  inside each state's 95 per cent credible region, in the coordinates the
  neutral pin fixes (#299), through the `logmu_shift` Jacobian (#276);
- **by `cnaster`'s MILP**, what `run_cnaster_port` writes to
  `cnv_seglevel.tsv` by default.

What is scored, per realization, against the planted truth:

- **coverage**: whether each planted state's `(A, B)` is in its fitted
  state's set, which a calibrated 95 per cent region does 95 times in 100;
- **set size**: how many pairs each set holds; one is a decode, more is not;
- **ARIs** over matched clone-bins, against the planted state: the
  continuous state, the MILP's pair, the sets' argmin, the sets read as the
  planted pair where they admit it (**consistent**, an upper bound), and the
  sets over the bins whose set is a single pair (**unique**, with its
  share); and the clone ARI before and after merging clones of one integer
  profile (#344).

The figure is one panel per planted state: the lattice down, realizations
across, each set's members filled, its argmin starred, the MILP's pair for
that state circled, and the planted pair's row shaded.
"""

from __future__ import annotations

import argparse
import tempfile
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tests.fixtures import COPY_LATTICE

Pair = tuple[int, int]


def unphased(a: int, b: int) -> Pair:
    """`(major, minor)`: phasing makes the allele label arbitrary."""
    return (max(int(a), int(b)), min(int(a), int(b)))


def _code(pairs: Sequence[Pair]) -> np.ndarray:
    return np.array([a * 1_000 + b for a, b in pairs], dtype=np.int64)


def _match(confusion: np.ndarray) -> dict[int, int]:
    from scipy.optimize import linear_sum_assignment

    rows, columns = linear_sum_assignment(-confusion)
    return dict(zip(rows.tolist(), columns.tolist(), strict=True))


def _merge(profiles: dict[int, bytes]) -> dict[int, int]:
    """Each clone to the smallest clone of an identical profile (#344)."""
    seen: dict[bytes, int] = {}
    return {c: seen.setdefault(profiles[c], c) for c in sorted(profiles)}


def decode_one(index: int, root: Path) -> dict[str, Any]:
    """Realize, run, decode both ways, and score one realization."""
    from port.extensions.copy_errors import copy_sets, pinned_errors
    from sklearn.metrics import adjusted_rand_score

    from tests.realizations import match_states, planted_genome, realize, run

    truth = planted_genome(lattice=True)
    realization = realize(truth, index)
    captured = run(realization, root / f"realization-{index}")

    errors = pinned_errors(captured)
    sets = copy_sets(errors)
    order = match_states(realization, captured)

    n_states = truth.log_mu.size
    planted_pairs = [unphased(*COPY_LATTICE[k]) for k in range(n_states)]

    result = captured.result
    path = np.asarray(result["pred_cnv"], dtype=np.int64) % n_states
    assignment = np.asarray(result["new_assignment"], dtype=np.int64)
    fitted_clones = np.unique(assignment)

    run_dir = next((root / f"realization-{index}").rglob("cnv_seglevel.tsv")).parent
    seglevel = pd.read_csv(run_dir / "cnv_seglevel.tsv", sep="\t")

    # NB clones matched to planted ones by spot overlap; a clone-bin is scored
    #    against the planted state of its matched planted clone.
    n_planted = int(truth.labels.max()) + 1
    overlap = np.zeros((n_planted, int(assignment.max()) + 1), dtype=np.int64)
    np.add.at(overlap, (truth.labels, assignment), 1)
    clone_of = _match(overlap)

    planted_state = np.concatenate([truth.states[c] for c in clone_of])
    fitted_state = np.concatenate([path[:, clone_of[c]] for c in clone_of])
    milp = np.concatenate(
        [
            _code(
                [
                    unphased(a, b)
                    for a, b in zip(
                        seglevel[f"clone{clone_of[c]} A"],
                        seglevel[f"clone{clone_of[c]} B"],
                        strict=True,
                    )
                ]
            )
            for c in clone_of
        ]
    )
    planted_code = _code([planted_pairs[k] for k in planted_state])

    best = _code([unphased(*sets[s].best) for s in fitted_state])
    members = [{unphased(*pair) for pair in sets[s].consistent} for s in fitted_state]
    admitted = np.array(
        [planted_pairs[k] in m for k, m in zip(planted_state, members, strict=True)]
    )
    consistent = np.where(admitted, planted_code, best)
    unique = np.array([len(m) == 1 for m in members])

    def ari(labels: np.ndarray, mask: np.ndarray | None = None) -> float:
        keep = slice(None) if mask is None else mask
        return float(adjusted_rand_score(planted_state[keep], labels[keep]))

    # NB clone ARI before and after merging clones of one integer profile,
    #    by the MILP's per-bin pairs and by the sets' argmins.
    milp_profile = {
        int(c): _code(
            [
                unphased(a, b)
                for a, b in zip(
                    seglevel[f"clone{c} A"], seglevel[f"clone{c} B"], strict=True
                )
            ]
        ).tobytes()
        for c in fitted_clones
    }
    best_profile = {
        int(c): _code([unphased(*sets[s].best) for s in path[:, c]]).tobytes()
        for c in fitted_clones
    }
    milp_merge, best_merge = _merge(milp_profile), _merge(best_profile)

    return {
        "index": index,
        "neutral": errors.neutral,
        "decrement": errors.decrement,
        "order": order.tolist(),
        "mu": errors.mu.tolist(),
        "minor": errors.minor.tolist(),
        "sigma": np.sqrt(
            np.stack([errors.covariance[:, 0, 0], errors.covariance[:, 1, 1]], axis=1)
        ).tolist(),
        "sets": [[unphased(*p) for p in s.consistent] for s in sets],
        "best": [unphased(*s.best) for s in sets],
        "distance": [s.distance for s in sets],
        "covered": [
            planted_pairs[k] in {unphased(*p) for p in sets[order[k]].consistent}
            for k in range(n_states)
        ],
        "set_size": [len(sets[order[k]].consistent) for k in range(n_states)],
        "best_is_planted": [
            unphased(*sets[order[k]].best) == planted_pairs[k] for k in range(n_states)
        ],
        "milp_modal": [_modal(milp[fitted_state == order[k]]) for k in range(n_states)],
        "milp_in_set": float(
            np.mean(
                [
                    (code // 1_000, code % 1_000) in m
                    for code, m in zip(milp, members, strict=True)
                ]
            )
        ),
        "ari_clone": float(adjusted_rand_score(truth.labels, assignment)),
        "ari_clone_milp": float(
            adjusted_rand_score(
                truth.labels, np.array([milp_merge[int(c)] for c in assignment])
            )
        ),
        "ari_clone_best": float(
            adjusted_rand_score(
                truth.labels, np.array([best_merge[int(c)] for c in assignment])
            )
        ),
        "ari_state": ari(fitted_state),
        "ari_milp": ari(milp),
        "ari_best": ari(best),
        "ari_consistent": ari(consistent),
        "ari_unique": ari(best, unique) if unique.any() else float("nan"),
        "unique_share": float(unique.mean()),
    }


def _modal(codes: np.ndarray) -> Pair | None:
    if codes.size == 0:
        return None
    values, counts = np.unique(codes, return_counts=True)
    code = int(values[np.argmax(counts)])
    return (code // 1_000, code % 1_000)


def audit(n_realizations: int, root: Path, jobs: int = 1) -> list[dict[str, Any]]:
    """Every realization, one fresh process each (`tests.realizations`' reason)."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    context = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=jobs, mp_context=context, max_tasks_per_child=1
    ) as pool:
        futures = [
            pool.submit(decode_one, index, root) for index in range(n_realizations)
        ]
        return [future.result() for future in futures]


def plot(scores: Sequence[dict[str, Any]], output: Path) -> None:
    """One panel per planted state: sets down the lattice, realizations across."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    n_states = len(scores[0]["order"])
    planted = [unphased(*COPY_LATTICE[k]) for k in range(n_states)]
    rows = sorted(
        {pair for s in scores for sets in s["sets"] for pair in sets}
        | set(planted)
        | {p for s in scores for p in s["milp_modal"] if p is not None},
        key=lambda pair: (sum(pair), pair),
    )
    y = {pair: i for i, pair in enumerate(rows)}

    figure, axes = plt.subplots(
        1, n_states, figsize=(3.2 * n_states, 0.3 * len(rows) + 1.4), sharey=True
    )

    for k, axis in enumerate(np.atleast_1d(axes)):
        axis.axhspan(y[planted[k]] - 0.45, y[planted[k]] + 0.45, color="0.9")

        for x, s in enumerate(scores):
            state = s["order"][k]
            for pair in s["sets"][state]:
                axis.plot(x, y[pair], "s", color="C0", markersize=7)
            axis.plot(x, y[s["best"][state]], "*", color="C3", markersize=9)
            if s["milp_modal"][k] is not None:
                axis.plot(
                    x, y[s["milp_modal"][k]], "o", mfc="none", color="k", markersize=11
                )

        covered = np.mean([s["covered"][k] for s in scores])
        axis.set_title(f"planted {planted[k]}: covered {covered:.2f}", fontsize=9)
        axis.set_xticks(range(len(scores)))
        axis.set_xlabel("realization")
        axis.set_yticks(range(len(rows)), [f"{a},{b}" for a, b in rows])
        axis.set_ylim(-0.6, len(rows) - 0.4)

    np.atleast_1d(axes)[0].set_ylabel("(A, B), unphased")
    figure.suptitle(
        "95% credible sets (squares), their argmin (star), MILP (circle)", fontsize=9
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)


def summary(scores: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The table the pull request carries: means over realizations."""
    keys = [
        "ari_clone",
        "ari_clone_milp",
        "ari_clone_best",
        "ari_state",
        "ari_milp",
        "ari_best",
        "ari_consistent",
        "ari_unique",
        "unique_share",
        "milp_in_set",
    ]
    covered = np.array([s["covered"] for s in scores], dtype=float)
    sizes = np.array([s["set_size"] for s in scores], dtype=float)
    return {
        "realizations": len(scores),
        "coverage": covered.mean(),
        "coverage_per_state": covered.mean(axis=0).round(3).tolist(),
        "set_size_per_state": sizes.mean(axis=0).round(2).tolist(),
        "best_is_planted": float(np.mean([s["best_is_planted"] for s in scores])),
        **{key: float(np.nanmean([s[key] for s in scores])) for key in keys},
    }


def main(argv: Sequence[str] | None = None) -> int:
    import json

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--realizations", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/plots/realizations_copies.png")
    )
    arguments = parser.parse_args(argv)

    with warnings.catch_warnings(), tempfile.TemporaryDirectory() as scratch:
        warnings.simplefilter("ignore")
        scores = audit(arguments.realizations, Path(scratch), arguments.jobs)

    plot(scores, arguments.output)
    arguments.output.with_suffix(".json").write_text(
        json.dumps({"summary": summary(scores), "realizations": scores}, indent=1)
        + "\n"
    )
    print("COPY_AUDIT " + json.dumps(summary(scores)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
