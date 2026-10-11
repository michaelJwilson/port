# Fixture truth figures

**TL;DR:** each drawn fixture's planted truth, as drawn by the simulator: one
directory per fixture hash, two PNGs each.

| Hash | Fixture | Drawn from |
| --- | --- | --- |
| `23acef50` | `dev_tree_easy_square` r0: two slices of standard Visium spots (55 um) on a square lattice | `sim/manifests/dev_tree_easy_square.toml` |
| `e4bbe03f` | `dev_tree_easy` r0: two slices of Visium spots on a hexagonal lattice | `sim/manifests/dev_tree_easy.toml` |

- `truth_combined.png`: the clone tree, the planted copy-number profile, and
  RDR and BAF per tumour clone, with the phase switches.
- `truth_combined_spatial.png`: the same, with each spot's planted clone per
  slice in place of RDR and BAF.

CalicoST easy (`2d4ce9a9`) and hard (`8797710b`) have no figure here: their
committed samples carry the truth tables but not the tree, phase and manifest
the figure draws from.

## Generation

From the repository root, with the repository's own environment, for each
hash: draw realization 0 of the manifest (`[sample] realizations = 1`; its
`r0_hash` is the directory name), then draw its truth with the simulator's
`truth_figure.truth_combined_figure(r, metric=True, spatial=...)`, `spatial`
false and true, each saved at 300 dpi on white under the page style, with no
`Software` metadata. The capture's inputs (`--sim-inputs DIR`, `DIR/<hash>/`)
are the same realization.
