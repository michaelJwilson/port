# Audit: the paper against `cnaster`

Issue #28. Does the code implement what the methods state?

Paper at `0478c74`; `cnaster` at `4adad4d` (`finish_annotation_mushift3`).

`CLAUDE.md` sets the posture: **neither side is automatically right.** Each
finding names which should move. A disagreement with no direction is a note,
not a finding. Nothing is fixed here -- the dependency is read only.

**Fourteen claims checked. Five implemented, two implemented in effect, one
implemented and unreachable, six not implemented.** The sharpest finding is
that `cnaster`'s integer copy-number decoder is the CalicoST method the paper
explicitly contrasts itself with.

---

## Verdict table

| # | claim | section | `cnaster` | verdict |
| ---: | --- | --- | --- | --- |
| 1 | likelihood and Potts prior | `likelihood.tex` | `calc_assignment_cost`, `icm_sweep_deque` | **implemented** |
| 2 | inverse temperature `β` scales the energy | `likelihood.tex` | `spatial_temp_factor = spatial_weight / temp` | **partial** — scales the prior only |
| 3 | `J_nn' ≥ 0` | `likelihood.tex` | `adjacency_mat.data` | **unchecked** — assumed, not enforced |
| 4 | the inertia term is deprecated | `likelihood.tex` `\todo` | `hmrf.py:551`, `inertia: 0` | **implemented in effect** |
| 5 | circulant genomic prior, `t` / `(1-t)/(K-1)` | `likelihood.tex` | `get_log_transmat` | **implemented** |
| 6 | `t` and `Π_m` are subject to optimization | `likelihood.tex` | `params="stmp"` | **not implemented** |
| 7 | sufficient statistics `B^k_i`, `U^k_i` | `efficient_likelihood.tex` | `CountEncoder` | **not implemented** — different method |
| 8 | Wolff cluster moves | `wolff.tex` | `wolff.py` | **implemented, unreachable** |
| 8b | per-edge bond probability `1-e^{-βJ_nn'}` | `wolff.tex` | `p_add_base`, uniform | **partial** |
| 9 | α-expansion complements ICM | `wolff.tex` | — | **not implemented** |
| 10 | H&E gray-channel percentile segmentation | `he_clone_initialization.tex` | `he.py:107` | **implemented** |
| 11 | Wolff ground state at `H=0` as the initializer | `he_clone_initialization.tex` | `initialize_clones`, rectangular | **not implemented** |
| 12 | integer copy from the Hessian's 95% credible region | `integer_copy_numbers.tex` | `hill_climbing_integer_copynumber_*` | **not implemented** — CalicoST's method |
| 13 | model selection by Pareto front over `J` | `prior_model_selection.tex` | — | **not implemented** |
| 14 | `mubar_k = (a_k + b_k)/2` | `emission.tex` | `frac_rdr = total_copies / 2.0` | **implemented** |

---

## 1. The likelihood and its Potts prior — implemented

Paper:

> `ln P(ℓ) ≅ -β Σ_{n, n' ∈ N(n)} J_nn' 1(ℓ_n ≠ ℓ_n')` ... defined up to an
> additive constant

and the ground state

> `ℓ* = argmin [ β Σ J_nn' 1(ℓ_n ≠ ℓ_n') + β Σ_{n,m} 1(ℓ_n = m) H_nm ]`

with `H_nm` a **negative** log-likelihood (eq. `external_field`).

`cnaster` (`icm.py:130`, `calc_assignment_cost`) maximises

```
Σ_i llf[i, s_i] + spatial_weight Σ_(ij) J_ij 1(s_i == s_j)
```

These are the same objective. `1(≠) = 1 - 1(=)`, so the paper's pairwise term
is `-β Σ J + β Σ J 1(=)`, and the leading term is the additive constant the
paper says it works up to. Negating throughout turns the paper's `argmin`
into `cnaster`'s maximisation.

**Measured rather than argued.** PR #44 scored `calc_assignment_cost` against
`snakes_and_ladders.search.alpha_expansion.energy`, which is
`-Σ h_i[s_i] - Σ J_ij 1(s_i == s_j)` — the paper's form with the constant
dropped. `cost + energy = 0` to `7.1e-15` across every labelling tried, at
three couplings, with `spatial_weight` held away from one.

**No action.**

## 2. `β` scales the whole energy in the paper and only the prior in `cnaster`

The paper's `β` multiplies both terms of `E(ℓ)`, so `β → 0` gives "uniformly
flat", which is what licenses the annealing argument in `wolff.tex`.

`cnaster` forms `spatial_temp_factor = spatial_weight / temp` (`icm.py:548`)
and applies it to the pairwise term alone; `single_llf` is never scaled.

So `temp` changes **the ratio of prior to likelihood**, not the sharpness of
the surface. The limits differ: the paper's `β → 0` flattens everything and
admits any configuration; `cnaster`'s `temp → ∞` removes the prior and leaves
the unary argmax — a perfectly sharp surface with no spatial coherence.

**Consequence:** an annealing schedule written against the paper does not do
what the paper says when run through `cnaster`. `wolff.py` takes `beta`
directly and is therefore consistent with the paper; the live ICM path is
not, so the two solvers anneal in different senses.

**Direction: `cnaster` should move**, by scaling the unary term too, or by
renaming `temp` to something that does not read as the paper's `β`.

## 3. `J_nn' ≥ 0` is assumed, not enforced

The paper requires non-negative couplings, and says why: it is what makes the
prior ferromagnetic, and it is the metric condition α-expansion's bound rests
on. Nothing in `adjacency.py` or `icm.py` checks the sign of
`adjacency_mat.data`.

Harmless while `multislice_adjacency` only emits non-negative weights.
**Direction: `cnaster`**, as an assertion rather than a change.

## 4. The inertia term — deprecated in the paper, shipped and off

`likelihood.tex`'s `\todo`:

> This differs from CalicoST in deprecating an inertia term that retains
> memory of the initial conditions

`cnaster` keeps it, config-gated:

```python
inertia = bool(get_global_config().hmrf.inertia)      # hmrf.py:551
log_persample_weights = np.ones(...) * (-np.log(n_clones)) if inertia else None
```

and the shipped `config.yaml` sets `inertia : 0`.

**Verdict: implemented in effect** — the default behaviour matches the paper.
The code is retained for a comparison the paper no longer makes.

**Direction: neither, but say so.** The `\todo` should become prose naming
the switch, or the switch should go.

## 5. The genomic Markov prior — implemented

Paper: `ln t` on the diagonal, `ln((1-t)/(K-1))` off, plus `Π_m`. That is
exactly `cnaster.hmm_nophasing.get_log_transmat`, and PR #22 pinned the
matrix `cnaster` builds against an independent construction at `2e-17`.

**No action.**

## 6. `t` and `Π_m` are not optimized, though the paper says they are

> `t` represents the transition rate between copy states; both are subject to
> optimization as in CalicoST.

`hmm_nophasing.__init__(self, params="stmp", t=1 - 1e-4)`. `pack_params` and
`unpack_params` branch on `"s"`, `"m"` and `"p"`. **Nothing anywhere reads
`"t"`** — grepped across `hmm_nophasing.py`, `hmm_phased.py` and `hmm.py`.
So `t` is fixed at `1 - 1e-4` for the whole fit, and the `"t"` in the default
parameter string has no effect.

`Π_m` is worse than fixed — it is *inconsistently* fitted. In `mode="marginal"`
`cost_fn` unpacks `this_log_startprob` and uses it; in `mode="em"` the same
call discards it (`_, this_log_mu, ... = self.unpack_params(...)`,
`hmm_nophasing.py:910`) and scores against the fixed `self.log_startprob`.

**Direction: both could move.** If the paper's claim is the intent, `cnaster`
should read `"t"` and unify the two modes. If a near-unity fixed `t` is a
deliberate simplification — and `1 - 1e-4` reads like one — the paper should
say so, because a fitted transition rate and a hard-coded one are different
models.

## 7. The efficient evaluation is a different optimization entirely

`efficient_likelihood.tex` derives **pre-computable sufficient statistics**:
expand the beta-binomial into Gamma functions, apply
`ln Γ(z+c) = ln Γ(z) + Σ_{i<c} ln(z+i)`, and accumulate posterior-weighted
cumulative counts

```
B^k_i = Σ_{m,n,g} 1(ℓ_n = m) w^k_gm 1(b_gn > i)
```

reducing transcendental evaluations from `O(ngk)` to `O(max(c)·k)` per step.

`cnaster` does something else. `CountEncoder` rounds `(count, total)` pairs to
`compression_decimals` and deduplicates, scoring only the unique set and
scattering back. Searched for the paper's form — cumulative counts, a loop to
`max(b)`, any `1(b_gn > i)` accumulation — and there is none, in
`hmm_emission.py`, `hmm_nophasing.py` or `count_encoder.py`.

Both reduce the same cost and neither implements the other. They are not even
the same *kind* of reduction: the paper's wins when `max(c) ≪ ng`, i.e. at
low coverage, and is independent of duplication; `CountEncoder`'s wins with
the duplication factor and is independent of coverage.

**This is a substantive gap, not a naming difference.** The paper's
complexity claim is the basis of its efficiency argument, and the shipped
code does not have that complexity. The paper's own note that the formulation
is "particularly useful for modern optimizers (with automatic
differentiation), given the greatly simplified gradients" has no counterpart
either — the differentiable likelihood is in `sandbox/` (#6).

**Direction: unclear, and that is the finding.** Either the derivation was
superseded by deduplication and the paper should say so with the measurement
that justified it, or it was never implemented. #9 measures what the
deduplicated path costs against the dense one; it should also measure against
the paper's form before either is called the efficient one.

## 8. Wolff is implemented faithfully and cannot be reached

`wolff.py:47` and `:176`:

```python
p_add_base = 1.0 - np.exp(-beta * base_J)
```

which is the paper's Edwards–Sokal bond probability `1 - e^{-β J_nn'}`
exactly, inside a BFS cluster build with a Metropolis acceptance — Algorithm
`wolff` as written.

It is unreachable:

- `hmrf.py:12` — `# from cnaster.wolff import wolff_sweep`, commented.
- `scripts/run_cnaster.py:62` — `# from cnaster.wolff import initialize_clones_wolff`, commented.
- `scripts/run_cnaster.py:257–272` — the `initialize_clones_wolff(...)` call
  sits inside a `"""` block. It is dead text, not a live call.

So 159 statements implementing the paper's central methodological argument
ship in the wheel and execute never.

**Direction: `cnaster`**, and the paper is what says it matters — the
argument in `wolff.tex` is that ICM's failure mode "directly mimic[s]
metastasis", which is a clinical claim about the solver that is live.

### 8b. The bond probability ignores the per-edge weight

`wolff.py:74` carries the paper's per-edge form commented out:

```python
# p_add = 1.0 - np.exp(-beta * J)
if np.random.rand() <= p_add_base:
```

so every edge uses one `base_J`. The paper's `J_nn'` is per-edge, and
`likelihood.tex` introduces it as per-edge precisely so a weighted adjacency
can express spatial structure.

**Direction: `cnaster`**, when it wires Wolff up. A uniform bond probability
on a weighted graph samples a different distribution.

## 9. α-expansion is claimed and absent

`wolff.tex`:

> This complements ICM, minimum-cut-based α-expansion
> \citep{BoykovVekslerZabih, boykov_experimental_2004} in solving the ground
> state problem.

`cnaster` has no α-expansion, no max-flow and no min-cut. `icm.py` carries
three ICM variants; one is live.

Note the citation `BoykovVekslerZabih` does not resolve in `cnamaste.bib`
(#27 §4), so the claim is unsupported on both sides.

**Direction: the paper, or `snakes_and_ladders`.** Upstream implements
α-expansion with a stated bound and a Rust `max_flow`; #29 and #8 are where
that is measured. The paper should either drop the claim or cite the
implementation that backs it.

## 10. The H&E segmentation is implemented as described

> a straightforward segmentation based on intensity percentiles in a
> \todo{constructed gray channel}

`he.py:107–113`:

```python
gray = 0.2125 * rgb[:, 0] + 0.7154 * rgb[:, 1] + 0.0721 * rgb[:, 2]
percentiles = np.linspace(0.0, 100.0, 1 + num_labels)
bins = np.percentile(np.sort(cropped_gray.flatten()), percentiles)
labels = np.digitize(cropped_gray, bins=bins)
```

Luminance weights, percentile bins, `num_labels` classes. Reached live from
`io.py:19` and `scripts/run_cnaster.py:12`.

**No action**, except that the paper's `\todo{constructed gray channel}`
should name the weights, since they are a choice.

## 11. The initializer is a rectangular partition, not a Wolff ground state

The paper:

> we generate initial configurations corresponding to solutions of the Markov
> Random Field ground state problem for `H = 0` with the Wolff algorithm ...
> we iteratively increase `J` until desired thresholds, e.g. the median
> number of transcripts per (initial) clone are satisfied.

The live path (`scripts/run_cnaster.py:250`):

```python
initial_clone_for_phasing = initialize_clones(
    coords, sample_ids,
    x_part=config.phasing.npart_phasing,
    y_part=config.phasing.npart_phasing, config=config)
```

a rectangular partition, with a comment saying it is "equivalent to
parse_visium::perform_partition" — CalicoST's. The Wolff version with exactly
the paper's `J`-sweep signature (`wolff_num_temps`, `min_spots`) is the dead
text in §8.

**Direction: `cnaster`.** This and §8 are one change, and §12 is a third of
the same shape: the paper describes the method that replaces CalicoST's, and
the live code is CalicoST's.

## 12. Integer copy number is CalicoST's method, which the paper contrasts itself with

`integer_copy_numbers.tex` specifies the Laplace approximation, the inverse
Hessian, marginal errors `σ_α = √((H⁻¹)_αα)`, and a **one-to-many map into a
95% credible region**:

> `Ẑ_k = { (A_k, B_k) | ((A_k+B_k)/2, B_k/(A_k+B_k)) ∈ C_0.95(...) }`

and then says, of the alternative:

> This contrasts with CalicoST, which introduced ad-hoc factors in a custom
> objective, e.g. for down-weighting the read-depth ratio relative to
> `b`-allele frequency (irrespective of the inferred dispersions), neglected
> parameter uncertainty and assumed ill-posed regularisation with respect to
> ploidy.

`cnaster` implements the thing being contrasted with, point for point:

| the paper's criticism | `cnaster` |
| --- | --- |
| ad-hoc factors in a custom objective | `hill_climbing_integer_copynumber_oneclone`, an L1 relative residual |
| down-weighting RDR against BAF | `rdr_weight`, `integer_copy.py:144`, read from config in a commented block |
| irrespective of the inferred dispersions | weights are `points_per_state = bincount(pred_cnv)` — occupancy |
| neglected parameter uncertainty | no Hessian on any live path; the only one is `sandbox/` (#6) |
| ill-posed ploidy regularisation | `max_medploidy` and a ploidy penalty |

There is no credible region, no marginal error, and nothing returns a **set**
of integer assignments — `hill_climbing_*` returns one.

**Direction: `cnaster`.** The paper is unambiguous about which it intends,
and #6 and #25 are the two halves of getting there: #6 supplies the Hessian
and the errors, #25 supplies the exhaustive referee that makes a hill climb's
answer checkable.

## 13. Model selection by Pareto front is not implemented

> we leverage our computational efficiency to systematically evaluate a
> Pareto front across a range of strengths for the spatio-genomic coherence
> priors. We define the optimal prior as that which achieves the point of
> maximum curvature on the Pareto front.

Grepped `cnaster` for `pareto`, `curvature`, `model_selection`: nothing.
`spatial_weight` is a single configured scalar.

The paper's own argument says why this matters: maximum likelihood favours
`J → 0`, so without model selection the prior strength is set by hand at a
value the likelihood would not choose.

**Direction: `cnaster`**, and the paper's `\S\todo{...}` for the results
should be filled once it exists.

## 14. `frac_rdr = total_copies / 2.0` is correct

`integer_copy.py:176` carries `# TODO HACK` and a commented alternative
`denom = weight_per_state.dot(total_copies)`. Against `emission.tex`:

> `mubar_k = (a_k + b_k) / 2` and `p_k = b_k / (a_k + b_k)`

the `/ 2.0` **is the paper's formula**, correct for a de-biased `mubar`. The
commented alternative is the normalizer, and applying it in the denominator
as well would apply the correction twice.

Adjudicated in #5; recorded here so the `# TODO HACK` is not acted on.

**Direction: `cnaster`**, to delete the `# TODO HACK` — and to apply the
de-biasing, which is the actual missing piece (#5).

---

## The pattern

Six of the eight not-implemented or partial findings are the same shape:
**the paper describes the method that replaces CalicoST's, and the live code
is CalicoST's.**

| | paper | `cnaster` live |
| --- | --- | --- |
| label solver | Wolff + annealing + α-expansion | ICM (§8, §9) |
| clone initialization | Wolff ground state at `H = 0` | rectangular partition (§11) |
| integer copy | Hessian credible region | ad-hoc weighted hill climb (§12) |
| prior strength | Pareto front over `J` | a configured scalar (§13) |
| emission evaluation | sufficient statistics | dedup (§7) |
| inertia | deprecated | retained, defaulted off (§4) |

Only the last is resolved. The paper is a specification for work that is, in
these six places, partly unbuilt — and `wolff.py`, at 159 statements
implementing the bond probability exactly, shows that the gap is wiring in at
least one case rather than absence.

**That is the finding to act on first**, because it is cheap and because
`wolff.tex`'s argument — that ICM's artifacts mimic metastasis — is a claim
about the code that runs today.

## What this feeds

- **#5** — §6 and §14; the de-biasing is the missing piece, not the divisor.
- **#6, #25** — §12; the two halves of the paper's integer-copy method.
- **#8, #40** — §2, §8, §8b, §9; and §2 means an annealing comparison must
  say which `β` it used.
- **#9** — §7; measure against the paper's form, not only the dense one.
- **#10** — §11.
- **#29** — §9; α-expansion has no `cnaster` implementation to referee, so
  upstream's is the only one.
- **New** — §13 has no ticket.
