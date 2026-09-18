# Audit: integer copy-number decoding across the paper, CalicoST and `cnaster`

Issue #131. Companion to `docs/audit-paper-cnaster.md`, which records that
`cnaster`'s decoder is the method the paper contrasts itself with; this one
says how far it has moved since, and in which direction.

Paper at `0478c74`; `cnaster` at `4adad4d`; CalicoST at `c1abcae`.

**`cnaster` has changed the solver and kept the objective.** The MILP variant
is a real capability gain over CalicoST -- globally optimal where hill climbing
is not -- and it optimizes the same ad-hoc cost the paper names as the reason
not to do this at all. Two of the paper's three criticisms therefore still
stand, and the third is now inconsistent with itself: the same magic weight
has two different values in one module.

---

## What the paper asks for

`integer_copy_numbers.tex` defines a **one-to-many** map, not an optimization:

> `Ẑ_k = { (A_k, B_k) | ((A_k + B_k)/2, B_k/(A_k + B_k)) ∈ C_0.95(...) }`

the set of integer pairs whose `(μ̄, p)` lands inside the 95 per cent credible
region of the posterior marginalized over everything but state `k`, with the
region taken from the inverse Hessian under a Laplace approximation.

It then names what it is contrasting with, and CalicoST is named in the text:

> This contrasts with CalicoST, which introduced ad-hoc factors in a custom
> objective, e.g. for down-weighting the read-depth ratio relative to
> `b-allele' frequency (irrespective of the inferred dispersions), neglected
> parameter uncertainty and assumed ill-posed regularisation with respect to
> ploidy.

Three distinct charges: **ad-hoc weights**, **no uncertainty**, and
**ill-posed ploidy regularisation**. They are separable and each is checkable.

Note also that the paper's identity is `μ̄_k = (a_k + b_k)/2` -- the
**de-biased** rate. `docs/audit-logmu-shift-calicost.md` records that neither
implementation produces `μ̄`, so both decoders read an axis whose zero point is
unset. That is upstream of everything below.

---

## CalicoST's objective

`find_integer_copynumber.py:172-195`, the `oneclone` variant:

```python
return np.square(0.3 * (mu - frac_rdr)).dot(points_per_state) \
     + np.square(new_p_binom - frac_baf).dot(points_per_state) \
     + np.sum(crucial_ordered_pairs_1) * len(pred_cnv) \
     + np.sum(crucial_ordered_pairs_2) * len(pred_cnv) \
     + np.sum(derived_ploidy > ploidy + 0.5) * len(pred_cnv) \
     + unbalanced_penalty * len(pred_cnv)
```

- `0.3` is the down-weighting the paper names. It is a literal, and it is not
  derived from `alphas` or `taus` -- the dispersions the fit already estimated
  and which are exactly what would set the relative weight if it were derived.
- Four penalties are hard, each scaled by `len(pred_cnv)`, which makes them
  barriers rather than priors. `derived_ploidy > ploidy + 0.5` is the ploidy
  regularisation the paper calls ill-posed: a step function on a quantity that
  the parameters themselves determine.
- The block is labelled `### temp penalty ###` in CalicoST's own source, with
  the simpler `abs(mu - frac_rdr) + 5 * abs(p - frac_baf)` preserved beneath
  it as a comment. The weight was 5 and became 0.3; neither is explained.

Search is `hill_climb`, `max_iter=10`, from an initial guess per candidate
ploidy. It is a local method on a non-convex integer objective.

## `cnaster`'s objective

`integer_copy.py:417-437`:

```python
result = (
    np.square(rdr_relative_weight * (mu - frac_rdr)).dot(points_per_state)  # MAGIC
    + np.square(new_p_binom - frac_baf).dot(points_per_state)
    + np.sum(derived_ploidy > ploidy + 0.5) * len(pred_cnv)  # MAGIC
)
if enforce_order:
    result += np.sum(crucial_ordered_pairs_1) * len(pred_cnv)
    result += np.sum(crucial_ordered_pairs_2) * len(pred_cnv)
```

Same shape. What moved:

| | CalicoST | `cnaster` |
| --- | --- | --- |
| RDR weight | `0.3`, literal, applied consistently | **three costs: `1.0`, `0.3`, and none** |
| ordering penalties | always on | behind `enforce_order`, default `False` |
| ploidy penalty | always on | always on, unchanged |
| the weights' provenance | undocumented | `# MAGIC`, `# TODO HACK` |

**It is worse than an inconsistent weight: the three decoders optimize three
different costs, and a ploidy config key picks between them.** #134.

| decoder | RDR term | BAF term | ploidy penalty |
| --- | --- | --- | --- |
| `..._oneclone` (`:101`) | `\|1 - frac_rdr/mu\|`, **no weight** | `\|1 - frac_baf/p\|` | `\|1 - ratio\|` when `>` ploidy |
| `..._fixdiploid` (`:330`) | `(w (mu - frac_rdr))^2`, **w = 1.0** | `(p - frac_baf)^2` | step at `ploidy + 0.5` |
| `..._fixdiploid_milp` (`:571`) | `(w (mu - frac_rdr))^2`, **w = 0.3** | `(p - frac_baf)^2` | step at `ploidy + 0.5` |

So the RDR channel is weighted 1.0, 0.3, or not at all, and the norm is L2 in
two of three and relative L1 in the other. In `oneclone` the weighted squared
form survives commented out directly above the live relative-L1 return.

`run_cnaster.py:1391-1410` selects on `if max_medploidy is not None`, taking
`oneclone` when a ploidy ceiling is configured and the MILP otherwise. **One
run takes one branch**, so this is not two objectives per run -- it is a
config key that reads as a ploidy cap silently choosing the cost function.
`fixdiploid` is reachable from neither branch.

Marking a constant `MAGIC` is an improvement on leaving it a literal -- it is
findable. It does not answer the paper's charge, which is that the constant
should not exist.

## What `cnaster` added: the MILP

`hill_climbing_integer_copynumber_fixdiploid_milp` (`:571`) reformulates the
same cost as a mixed-integer linear program and solves it with
`scipy.optimize.milp` (HiGHS). Its docstring is accurate: *"Returns the
globally optimal integer copy states, the best objective, and the best
ploidy."*

This is the one place `cnaster` is unambiguously ahead of CalicoST. Hill
climbing with `max_iter=10` over a non-convex integer objective returns a
local optimum whose quality depends on the initial guess; a MILP returns the
optimum of the stated objective or proves none exists. `max_samples=20` is
retained in the signature "for compatibility, but ignored by MILP", which is
the sampling that the hill climber needed and the exact solver does not.

**It also sharpens the criticism rather than softening it.** Solving an ad-hoc
objective exactly means the answer is now determined entirely by the weights.
Under hill climbing a bad weight could be masked by the search failing to
exploit it; under a MILP it cannot.

## What `cnaster` dropped

CalicoST carries five decoding paths; `cnaster` carries three.

| CalicoST | `cnaster` |
| --- | --- |
| `hill_climbing_..._oneclone` | kept |
| `hill_climbing_..._fixdiploid` | kept |
| `hill_climbing_..._fixdiploid_milp` | **added** |
| `hill_climbing_..._joint` | dropped |
| `composite_hmm_optimize_integer_copynumber` | dropped |
| `optimize_integer_copynumber_oneclone` (+`_v2`) | dropped |
| `get_genelevel_cnv_oneclone`, `convert_copy_to_states`, `eval_objective` | dropped |

`joint` and `composite_hmm_*` are the multi-clone decoders -- they solve all
clones' states together under a shared scale factor. Dropping them makes
`cnaster` per-clone only, which is a capability regression and is not recorded
anywhere in `cnaster`. Whether it matters depends on whether clones are
expected to share a ploidy; the paper's formulation is per state `k` and does
not obviously require the joint form, so this may be a simplification rather
than a loss. It is listed here because it is unstated, which `CLAUDE.md` makes
a defect by default.

## A behavioural change worth stating: the diploid selector

Both implementations take candidates by occupancy and BAF balance, then choose
among them. They choose differently:

```python
# CalicoST find_integer_copynumber.py:81
return candidate[np.argmin(new_log_mu[candidate])]          # the smallest rate

# cnaster integer_copy.py:84
normal_candidate_idx = np.argmin(np.abs(1.0 - np.exp(new_log_mu[candidate])))   # the rate nearest 1
```

CalicoST's line survives in `cnaster` above the replacement, marked
`# DEPRECATE`. `cnaster` also warns when the chosen state's rate falls outside
`[0.9, 1.1]`.

**`cnaster`'s is the better definition and should be stated as a difference
rather than left as a diff.** "Diploid" means RDR 1, not "lowest of whatever
was fitted"; CalicoST's proxy picks a deletion over a true diploid whenever
one is present and balanced. The change has a consequence the fixture already
found: #106 and #120 are both about which state clears this selector, and
#122's `phasing.py` deadband is the same 0.1 threshold in a different module.

`min_prop_threshold` diverges too: `0.1` in both `find_diploid_balanced_state`
implementations, but `0.0  # MAGIC, previously 0.1` in `cnaster`'s MILP
signature. So the MILP will accept a candidate the selector that feeds it
would have rejected.

---

## Roadmap

1. **Reconcile the three objectives, and make the selection explicit.** #134.
   This is a `cnaster` defect independent of the paper: which cost decodes the
   copy numbers currently depends on whether `max_medploidy` is set.
2. **Derive the weight from the dispersions.** The paper's charge is that
   `0.3` is irrespective of `alphas` and `taus`. Both are fitted and available
   at the call site. A weight formed from the estimated variances of the RDR
   and BAF channels is not a credible region, but it removes the first of the
   three charges and is a much smaller change than the third.
3. **Return the set, not the argmin.** The paper's `Ẑ_k` is one-to-many.
   `hill_climbing_*` and the MILP both return a single `(A, B)` per state, so
   the interface -- not just the method -- has to change before a credible set
   is expressible. This is the data-structure work #131 asks for on this side:
   an `(n_states,)` array of pairs becomes an `(n_states,)` array of *sets*,
   and everything downstream that indexes it needs a representative.
4. **The Hessian is the real dependency.** `C_0.95` needs the inverse Hessian
   of the emission log-likelihood at the MLE, marginalized to `(μ_k, p_k)`.
   `cnaster` does not compute one. #6 is the ticket; this audit is its
   motivation, and the MILP is what makes the gap visible, because with an
   exact solver the only remaining error is the objective's.
5. **Decide the joint decoder's fate explicitly.** Either record why per-clone
   is sufficient, or restore the joint form. Silence reads as an oversight.

## Upstream correspondence

Little of this belongs in `snakes_and_ladders`. The integer-copy map is
application specific -- it is about allele-specific copy number, not about
HMMs or count emissions. The one exception is item 4: a family that can report
the observed information at its MLE would let the credible region be built
from upstream quantities, and that is `EmissionFamily`'s natural surface
rather than `cnaster`'s. That is the same hook `docs/audit-logmu-shift-calicost.md`
asks for, from a different direction.
