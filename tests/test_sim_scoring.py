"""`port.qa.scoring` against brute force, and one draw scored end to end (T- #670 PR0).

The series T- #670 scores every `cnamaste` stage with `port.qa.scoring` and
`port.qa.audit.score_sample` on samples `port.sim.draw` writes, so both are
refereed here before anything is scored with them:

- `matched` against every injective labelling on tiny overlap matrices, the
  exhaustive answer `linear_sum_assignment` claims to find;
- `integer_clones` and `exact_by_class` against per-element loops written
  from their docstrings;
- a 20 x 20 `dev_tree_1s_easy` draw, scored by `score_sample` against a run
  written from its own planted truth under a permuted labelling: every ARI
  and exact share is 1 and the matching inverts the permutation.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from port.qa.audit import score_sample
from port.qa.scoring import CLASSES, exact_by_class, integer_clones, matched
from port.sim.draw import draw, extended, from_document, merged_tables
from port.sim.fixtures import SIM_ROOT, load_simulated, references

CASES = 300
"""Random overlap matrices per shape; every shape up to 4 x 5 is drawn."""


def _brute_force(counts: np.ndarray) -> int:
    """The largest total overlap of any one-to-one labelling, by enumeration."""
    rows, columns = counts.shape
    if rows <= columns:
        return max(
            int(sum(counts[r, c] for r, c in enumerate(chosen)))
            for chosen in itertools.permutations(range(columns), rows)
        )
    return _brute_force(counts.T)


@pytest.mark.oracle
def test_matching_attains_the_enumerated_optimum_on_tiny_problems() -> None:
    """Total overlap equals the exhaustive optimum, exactly; where that
    optimum is attained once, the labelling is the enumerated one."""
    rng = np.random.default_rng(670)
    unique = 0

    for _ in range(CASES):
        shape = (int(rng.integers(1, 5)), int(rng.integers(1, 6)))
        counts = rng.integers(0, 12, size=shape)
        found = matched(counts)

        assert len(found) == min(shape)
        assert len(set(found.values())) == len(found)
        total = int(sum(counts[r, c] for r, c in found.items()))
        assert total == _brute_force(counts)

        rows, columns = shape
        if rows <= columns:
            optima = [
                chosen
                for chosen in itertools.permutations(range(columns), rows)
                if sum(counts[r, c] for r, c in enumerate(chosen)) == total
            ]
            if len(optima) == 1:
                unique += 1
                assert tuple(found[r] for r in range(rows)) == optima[0]

    # NB the unique-optimum branch is exercised, not vacuous
    assert unique > CASES // 4


@pytest.mark.oracle
def test_integer_clones_is_the_smallest_clone_of_each_profile() -> None:
    """Each clone maps to the smallest clone equal to it at every bin, by loop."""
    rng = np.random.default_rng(671)

    for _ in range(CASES):
        n_bins, n_clones = int(rng.integers(1, 6)), int(rng.integers(1, 7))
        base_a = rng.integers(0, 3, size=(n_bins, 3))
        base_b = rng.integers(0, 3, size=(n_bins, 3))
        pick = rng.integers(0, 3, size=n_clones)
        a, b = base_a[:, pick], base_b[:, pick]

        expected = [
            min(
                other
                for other in range(n_clones)
                if all(
                    a[i, other] == a[i, clone] and b[i, other] == b[i, clone]
                    for i in range(n_bins)
                )
            )
            for clone in range(n_clones)
        ]
        assert integer_clones(a, b).tolist() == expected


def _class(major: int, minor: int) -> str | None:
    """The planted class of one pair, from `planted_classes`' docstring."""
    if min(major, minor) == 0:
        return "loh"
    if (major, minor) == (1, 1):
        return "neutral"
    if major + minor > 2:
        return "balanced_gain" if major == minor else "unbalanced_gain"
    return None


@pytest.mark.oracle
def test_exact_by_class_is_the_per_bin_count() -> None:
    """Exact and phase-free shares and the bin count per class, by loop."""
    rng = np.random.default_rng(672)

    for _ in range(CASES):
        n = int(rng.integers(1, 40))
        t = rng.integers(0, 4, size=n) * 1_000 + rng.integers(0, 4, size=n)
        ab = rng.integers(0, 4, size=n) * 1_000 + rng.integers(0, 4, size=n)
        found = exact_by_class(t, ab)

        for name in CLASSES:
            bins = [i for i in range(n) if _class(t[i] // 1_000, t[i] % 1_000) == name]
            exact, either, count = found[name]
            assert count == len(bins)
            if not bins:
                assert np.isnan(exact)
                assert np.isnan(either)
                continue
            swap = [(ab[i] % 1_000) * 1_000 + ab[i] // 1_000 for i in bins]
            assert exact == sum(t[i] == ab[i] for i in bins) / len(bins)
            assert either == sum(
                t[i] in (ab[i], s) for i, s in zip(bins, swap, strict=True)
            ) / len(bins)


def _write_run(sample_path: Path, output: Path, permutation: np.ndarray) -> None:
    """A run `read_run` reads, written from the sample's own planted truth.

    One `cnv_seglevel` row per planted segment; fitted clone `permutation[k]`
    carries planted clone `k`'s `(A, B)` and its spots; the decoded state `Z`
    is the index of the planted pair.
    """
    sample = load_simulated(sample_path.name, sample_path.parent)
    profile = sample.profile
    run = output / "run"
    run.mkdir(parents=True)

    seglevel = pd.DataFrame(
        {
            "CHR": profile["chr"].to_numpy(),
            "START": profile["start"].to_numpy(),
            "END": profile["end"].to_numpy(),
        }
    )
    codes = []
    for planted, clone in enumerate(sample.clones):
        a = profile[f"{clone}_A_copy"].to_numpy()
        b = profile[f"{clone}_B_copy"].to_numpy()
        seglevel[f"clone{permutation[planted]} A"] = a
        seglevel[f"clone{permutation[planted]} B"] = b
        codes.append(a * 1_000 + b)
    # NB columns in fitted order, as `cnaster` writes them
    seglevel = seglevel[
        ["CHR", "START", "END"]
        + [f"clone{c} {x}" for c in range(sample.n_clones) for x in ("A", "B")]
    ]
    seglevel.to_csv(run / "cnv_seglevel.tsv", sep="\t", index=False)

    states, z = np.unique(np.stack(codes, 1), return_inverse=True)
    pred = np.empty((len(seglevel), sample.n_clones), dtype=np.int64)
    pred[:, permutation] = z.reshape(len(seglevel), -1)
    np.savez(
        run / f"rdrbaf_final_nstates{states.size}_smp.npz",
        new_log_mu=np.zeros((states.size, 1)),
        pred_cnv=pred.ravel(),
    )
    pd.DataFrame(
        {"barcode": sample.barcodes, "clone_label": permutation[sample.labels]}
    ).to_csv(run / "clone_labels.tsv", sep="\t", index=False)


@pytest.mark.analytic
def test_a_drawn_sample_scored_against_its_own_truth_scores_one(
    tmp_path: Path,
) -> None:
    """Draw, load, write the truth as a run under a permutation, score it.

    The referee is a property: a fit equal to the planted truth up to
    labelling scores 1 on every ARI and exact share, and `clone_of` is the
    permutation. `analytic` rather than T- #670's `end2end`: no `cnaster`
    runs, so the judged coverage guard would credit statements nobody fit.
    """
    resources = references()
    if resources is None:
        pytest.skip("CalicoST's GRCh38_resources not found; set $PORT_GRCH38")

    manifests = SIM_ROOT / "manifests"
    document = merged_tables(
        extended(manifests / "dev_tree_1s_easy.toml"),
        {"array": {"rows": 20, "columns": 20}, "sample": {"realizations": 1}},
    )
    drawn = draw(from_document(document, manifests), tmp_path, resources=resources)
    path = drawn.realizations[0]
    sample = load_simulated(path.name, path.parent)
    assert sample.n_clones == len(drawn.clones) >= 3

    permutation = np.roll(np.arange(sample.n_clones), 1)
    _write_run(path, tmp_path / "out", permutation)
    score = score_sample(sample, tmp_path / "out", "truth", 0.0)

    assert score.clone_of == {k: int(permutation[k]) for k in range(sample.n_clones)}
    for name in ("ari", "state_ari", "copy_ari", "copy_ari_pf", "exact"):
        assert getattr(score, name) == 1.0, name
    assert score.exact_altered == 1.0
    assert score.n_clones == sample.n_clones
    assert score.bins == len(sample.profile)
