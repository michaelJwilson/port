"""Overlapping events, the altered share and `dev_tree_1s_dense` (#698).

Referees: independent recomputation from the event list, the LOH rule's properties, and
a `bug` pin.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from port.sim.draw import (
    CloneTree,
    DrawManifest,
    admissible,
    altered_share,
    draw_tree,
    extended,
    from_document,
    lineage_states,
    read_manifest,
    truth_profile,
)
from port.sim.fixtures import SIM_ROOT
from port.sim.laws import Event

MANIFESTS = SIM_ROOT / "manifests"
DENSE = MANIFESTS / "dev_tree_1s_dense.toml"
SEEDS = 100
"""Seeded trees per test; seed `k` draws `draw`'s tree stream for `[sample] seed = k`."""
GRID = 100_000
"""The brute-force grid, in bp: one point per `GRID` of every chromosome."""


def _tree(manifest: DrawManifest, seed: int) -> CloneTree:
    """The tree `port.sim.draw.draw` draws at `[sample] seed = seed`."""
    stream = np.random.SeedSequence(seed).spawn(3)[0]
    return draw_tree(manifest, np.random.default_rng(stream))


def _composed(tree: CloneTree, clone: str, chromosome: str, at: int) -> tuple[int, int]:
    """A clone's state at one locus, by walking its path's events in order."""
    state = (1, 1)
    for node in tree.path(clone):
        for event in tree.edge_events[node]:
            if event.chromosome == chromosome and event.start <= at < event.end:
                state = (event.a, event.b)
    return state


def _union(tree: CloneTree, lengths: list[int]) -> int:
    """bp covered by any event on any edge."""
    covered = 0
    for index in range(1, len(lengths) + 1):
        spans = sorted(
            (e.start, e.end)
            for events in tree.edge_events.values()
            for e in events
            if e.chromosome == str(index)
        )
        end = -1
        for left, right in spans:
            covered += max(0, right - max(left, end))
            end = max(end, right)
    return covered


@pytest.fixture(scope="module")
def dense() -> DrawManifest:
    return read_manifest(DENSE)


@pytest.mark.oracle
def test_the_altered_share_is_the_union_of_event_spans_and_exceeds_four_fifths(
    dense: DrawManifest,
) -> None:
    """`altered_share` equals the merged union of event spans on 100 trees; seed 0 exceeds 0.80."""
    lengths = [int(n) for n in dense.genome["chromosome_lengths"]]
    assert [1, 1] not in dense.cna["states"]
    overlapped = 0

    for seed in range(SEEDS):
        tree = _tree(dense, seed)
        share = altered_share(tree, lengths)
        assert share * sum(lengths) == pytest.approx(_union(tree, lengths), abs=0.5)
        total = sum(e.end - e.start for es in tree.edge_events.values() for e in es)
        overlapped += total > _union(tree, lengths)

    assert overlapped == SEEDS
    assert altered_share(_tree(dense, int(dense.seed)), lengths) > 0.80


@pytest.mark.oracle
def test_the_altered_share_agrees_with_a_grid_of_loci(dense: DrawManifest) -> None:
    """`altered_share` agrees with a 100 kb grid walk within `GRID` bp per boundary."""
    lengths = [int(n) for n in dense.genome["chromosome_lengths"]]
    for seed in range(3):
        tree = _tree(dense, seed)
        share = altered_share(tree, lengths)
        hit = 0
        for index, length in enumerate(lengths, start=1):
            for at in range(0, length, GRID):
                if any(
                    _composed(tree, c, str(index), at) != (1, 1) for c in tree.leaves
                ):
                    hit += min(GRID, length - at)

        profile = truth_profile(tree, tree.leaves, lengths)
        altered = np.any(profile.filter(regex=r"_[AB]_copy$").to_numpy() != 1, axis=1)
        same_chr = profile["chr"].to_numpy()[1:] == profile["chr"].to_numpy()[:-1]
        boundaries = int(np.sum(same_chr & (altered[1:] != altered[:-1])))
        assert 0 < share < 1
        assert boundaries > 0
        assert abs(hit / sum(lengths) - share) <= boundaries * GRID / sum(lengths)


@pytest.mark.oracle
@pytest.mark.parametrize("name", ["dev_tree_1s_dense", "dev_tree_1s_easy", "dev_tree"])
def test_the_truth_is_the_composition_of_each_lineages_events(name: str) -> None:
    """`truth_profile` equals a walk of each clone's path on 100 seeded trees."""
    manifest = read_manifest(MANIFESTS / f"{name}.toml")
    lengths = [int(n) for n in manifest.genome["chromosome_lengths"]]

    for seed in range(SEEDS):
        tree = _tree(manifest, seed)
        profile = truth_profile(tree, tree.leaves, lengths)
        for index, length in enumerate(lengths, start=1):
            rows = profile[profile["chr"] == index]
            starts, ends = rows["start"].to_numpy(), rows["end"].to_numpy()
            assert starts[0] == 0
            assert ends[-1] == length
            assert np.array_equal(starts[1:], ends[:-1])

        for row in profile.itertuples(index=False):
            for clone in tree.leaves:
                planted = (
                    getattr(row, f"{clone}_A_copy"),
                    getattr(row, f"{clone}_B_copy"),
                )
                for at in (row.start, row.end - 1):
                    assert planted == _composed(tree, clone, str(row.chr), at)


@pytest.mark.analytic
def test_an_irreversible_lineage_never_regains_a_lost_haplotype(
    dense: DrawManifest,
) -> None:
    """Under irreversible LOH no lineage regains a lost haplotype, on 100 dense trees."""
    states = {tuple(s) for s in dense.cna["states"]}
    stacked = 0

    for seed in range(SEEDS):
        tree = _tree(dense, seed)
        for node in tree.parent:
            if tree.parent[node] is None:
                continue
            before = tuple(
                e for up in tree.path(node)[:-1] for e in tree.edge_events[up]
            )
            for k, event in enumerate(tree.edge_events[node]):
                lineage = (*before, *tree.edge_events[node][:k])
                pieces = lineage_states(
                    lineage, event.chromosome, event.start, event.end
                )
                assert (event.a, event.b) in states
                assert min(event.a, event.b) >= 0
                assert all(a > 0 or event.a == 0 for a, _ in pieces)
                assert all(b > 0 or event.b == 0 for _, b in pieces)
                assert any((event.a, event.b) != p for p in pieces)
                stacked += any(p != (1, 1) for p in pieces)

    assert stacked >= SEEDS


@pytest.mark.analytic
def test_admissible_states_on_hand_built_lineages() -> None:
    """Nested and abutting events on one chromosome, against states read off by hand."""
    states = [(1, 0), (0, 1), (2, 1), (1, 2), (3, 1), (2, 2), (0, 2)]
    lineage = (
        Event("1", 0, 100, 1, 0),
        Event("1", 40, 60, 2, 2),
        Event("1", 100, 200, 0, 2),
    )

    assert lineage_states(lineage, "1", 20, 80) == [(1, 0), (2, 2), (1, 0)]
    assert lineage_states(lineage, "1", 90, 110) == [(1, 0), (0, 2)]
    assert lineage_states(lineage, "1", 200, 300) == [(1, 1)]
    assert lineage_states(lineage, "2", 0, 10) == [(1, 1)]
    # NB B lost over [0, 40) and [60, 100): only (1, 0) changes something, at [40, 60).
    assert admissible(states, [(1, 0), (2, 2), (1, 0)]) == [(1, 0)]
    # NB A lost on one piece and B on the other: nothing keeps both lost.
    assert admissible(states, [(1, 0), (0, 2)]) == []
    assert admissible(states, [(1, 0)]) == []
    assert admissible(states, [(1, 1)]) == states


@pytest.mark.bug
def test_a_reversible_lineage_regains_lost_haplotypes() -> None:
    """Without `[cna] loh`, `dev_tree_1s_easy` regains lost haplotypes; fails when fixed."""
    manifest = from_document(extended(MANIFESTS / "dev_tree_1s_easy.toml"))
    assert "loh" not in manifest.cna
    regained = events = 0

    for seed in range(SEEDS):
        tree = _tree(manifest, seed)
        for node in tree.parent:
            if tree.parent[node] is None:
                continue
            before = tuple(
                e for up in tree.path(node)[:-1] for e in tree.edge_events[up]
            )
            for k, event in enumerate(tree.edge_events[node]):
                lineage = (*before, *tree.edge_events[node][:k])
                pieces = lineage_states(
                    lineage, event.chromosome, event.start, event.end
                )
                events += 1
                regained += any(
                    (a == 0 and event.a > 0) or (b == 0 and event.b > 0)
                    for a, b in pieces
                )

    assert (regained, events) == (11, 679)


@pytest.mark.infra
def test_an_unknown_loh_rule_is_refused() -> None:
    document = extended(DENSE)
    document["cna"]["loh"] = "sometimes"
    with pytest.raises(ValueError, match="loh"):
        from_document(document)


def _pairs(tree: CloneTree) -> int:
    """Cross-branch overlaps: events on two leaf edges covering a common locus."""
    leaves = [tree.edge_events[c] for c in tree.leaves]
    return sum(
        e.chromosome == f.chromosome and e.start < f.end and f.start < e.end
        for x, y in itertools.combinations(leaves, 2)
        for e in x
        for f in y
    )


@pytest.mark.analytic
def test_branches_overlap_without_touching_each_other(dense: DrawManifest) -> None:
    """An event on one leaf edge leaves its siblings' profiles unchanged."""
    lengths = [int(n) for n in dense.genome["chromosome_lengths"]]
    crossing = 0
    for seed in range(20):
        tree = _tree(dense, seed)
        crossing += _pairs(tree) > 0
        full = truth_profile(tree, tree.leaves, lengths)
        for clone in tree.leaves:
            alone = truth_profile(
                CloneTree(
                    tree.parent,
                    {
                        n: tree.edge_events[n] if n in tree.path(clone) else ()
                        for n in tree.parent
                    },
                    tree.leaves,
                ),
                (clone,),
                lengths,
            )
            for row in full.itertuples(index=False):
                mid = (row.start + row.end) // 2
                hit = alone[
                    (alone["chr"] == row.chr)
                    & (alone["start"] <= mid)
                    & (alone["end"] > mid)
                ]
                assert (
                    hit[f"{clone}_A_copy"].item(),
                    hit[f"{clone}_B_copy"].item(),
                ) == (
                    getattr(row, f"{clone}_A_copy"),
                    getattr(row, f"{clone}_B_copy"),
                )
    assert crossing == 20
