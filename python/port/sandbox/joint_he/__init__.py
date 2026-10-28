"""Joint inference of clones and copy paths from single spots, with an H&E-conditioned clone prior.

A study, set aside: nothing installs it and no row names it. Each spot carries
p(x_n | l_n) p(l_n | H&E); a clone is a copy path, and the prior is a small
untrained CNN on the paired slide. Spots enter a clone's chain through their
own emissions, never through a pseudobulk of their counts.

- `realistic_he`: a simulated H&E over a planted clone map (Beer--Lambert
  stain, stroma against tumour nests, subclones by nuclear grade).
- `make_he_draw`: a `port` draw with that H&E written into each slice.
- `joint`: per-spot emissions, the copy chain's forward pass and Viterbi, the
  CNN, and the continuous-state objective (free (mu_k, p_k) per state).
- `study`: that objective's arms (`cnn`, `cnn_delta`, `free`, `constant`) on a
  draw against its baseline `cnamaste` run.
- `joint_int`: the states tied to integer pairs (A, B), cnamaste's pair
  parameters at one purity, a phase path global to every clone, structured
  mean field over the chains; the emission cached by distinct (b, n) and u.
- `he_prior`: the H&E prior alone, the CNN fit to a label set (hard or soft) by
  cross-entropy.
- `study_int`: EM on that bound, scored by clone ARI and integer copies;
  `--labels truth` holds the clones at the planted partition.
- `combined_int`: its result on `cnamaste`'s combined page.
- `cnamaste_he_unary`, `run_cnamaste_he`: `cnamaste` itself with the H&E CNN as a
  further unary in its label step, refit to its labels each iteration.

Departure from `cnamaste`: per-spot rather than pseudobulk emission, so a
deviation a clone's spots share at a segment is counted once per spot; global
rather than per-clone shift and purity; no clone held at (1, 1).

Exit: promoted if, across seeds and at stress sizes, it matches `cnamaste`'s
clone ARI and integer copies on planted truth; removed otherwise. Measured in
PR- #943.
"""
