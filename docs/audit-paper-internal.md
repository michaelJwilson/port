# Audit: the paper against itself

Issue #27. Whether [`cna-maste-paper`](https://github.com/michaelJwilson/cna-maste-paper)
agrees with itself, before #28 and #29 compare it outward.

Audited at `0478c74`, 26 `.tex` files, 1,677 lines.

**Two defects change what the model says. Three more change what it claims.**
The emission mean, as written, is independent of the copy state. The
transition matrix is defined as the transpose of its own usage. The runtime
is out by 90.6x from its own derivation, and by four orders from the
protocol the abstract says it enables. Five citations do not resolve. Twelve
files never reach the compiled document, four of them substantive.

Nothing here is a claim about `cnaster`. That is #28, and it is unsafe to
start while the text disagrees with itself.

---

## 1. The emission mean does not depend on the copy state

`emission.tex`, the negative binomial:

> `<u_gn>_k = lambda_g T_n . mu_k / (sum_g lambda_g mu_k)`

`mu_k` carries no `g`, so it comes out of the sum:

```
sum_g lambda_g mu_k  =  mu_k sum_g lambda_g
```

and the `mu_k` cancels top and bottom:

```
<u_gn>_k  =  lambda_g T_n / sum_g lambda_g
```

**The mean is then the same for every copy state.** A model whose read-depth
channel cannot distinguish `k` cannot detect a copy-number change from
coverage at all, which is the opposite of what the section is for.

The same index error is in the constraint two clauses later:

> we define de-biased equivalents, `mubar_k`, that satisfy the further
> constraint `sum_g lambda_g mubar_k = 1`

which factors to `mubar_k sum_g lambda_g = 1`, making every `mubar_k` equal
to one another.

**Severity: the equation as written contradicts the paper's purpose.** It is
a notation defect rather than a modelling one -- the reading that binds is
`sum_g lambda_g mu_{k(g)}`, the sum over a clone's *profile*, where the state
varies with position. That is what `cnaster.hmm_nophasing.compute_logmu_shifts`
computes, which is corroboration that the intended object is the profile sum.

**Action:** bind the index. Write `k(g)` or `k_g` explicitly, or move the
normalizer onto a per-spot quantity as `efficient_likelihood.tex` does.
Whichever is canonical has to be stated, because #5 cannot assert an
invariant until it knows which of the two forms the code is meant to match.

---

## 2. The transition matrix is defined as the transpose of its own usage

`hidden_markov.tex` defines

> `T_kl = P(z_i = k | z_{i-1} = l)`

so the **first** index is the current state and the second the previous:
column-stochastic, `T[dest, source]`.

Three lines later the joint is

> `P(x, z) = prod_i E_{z_i}(x_i) . T_{0, z_1} prod_i T_{z_i, z_{i+1}}`

where `T_{z_i, z_{i+1}}` has the **earlier** state first: row-stochastic,
`T[source, dest]`. The Viterbi recursion agrees with the usage and not the
definition:

> `f(z_{i+1} = l) = E_l(x_{i+1}) max_k (T_kl P(z_i = k | ...))`

Here `k` is the previous state and `l` the next, so `T_kl = T[source, dest]`.

**So the definition is used nowhere and contradicted twice.** Every recursion
in the section is row-stochastic; only the defining equation is not.

**Severity: silent.** A symmetric transition -- which the circulant form
`cnaster` builds is -- scores identically under either convention, so the
error is invisible on exactly the matrix the implementation uses by default.
`port` hit the same blind spot in its own tests and had to add
`tests.fixtures.drift_transition` to break the symmetry before a transposed
matrix became detectable; measured there, the transpose shifts a total
log-likelihood by 2.32 against an asymmetric fixture and by `1.4e-12`
against a symmetric one.

**Action:** fix the defining equation to `T_kl = P(z_{i} = l | z_{i-1} = k)`,
which is what the rest of the section uses and what Durbin §3.2 writes.

---

## 3. The runtime is wrong twice, and the paper flags only the smaller error

`expected_runtime.tex`, in full:

> Given a mappable genome length of ≃2.9 billion base pairs ... and a typical
> segment size of 10 kilo base pairs, we anticipate 3.2 × 10³ segments along
> the genome. For approx. 5,000 pixels and an emission evaluation ... of
> ≃1 μs, the expected runtime scaling is
>
> `Time / num. slices ≃ 1 min . (num. segments / 3.2e3) . (emission [s] / 1 μs) . (K / 4)`
>
> `\todo{1 min. assumes a number of segments inconsistent with above, fix this.}`

### 3a. The segment count contradicts its own derivation, by 90.6x

```
2.9e9 bp / 1e4 bp per segment = 2.9e5 segments
paper states                    3.2e3
ratio                           90.6
```

`3.2e3` is what `3.2` *billion* base pairs at `1e6` per segment would give --
the full genome length from the parenthetical, at a segment size a hundred
times the one stated.

The `1 min` constant is self-consistent with the wrong figure:

```
5,000 px . 3.2e3 segments . K=4 . 1 us = 64 s
```

So the formula is internally coherent and its input is not. Corrected on the
paper's own numbers, the per-slice runtime is

```
5,000 px . 2.9e5 segments . K=4 . 1 us = 96.7 min
```

**~97 minutes per slice, not one.** The `\todo` records that something is
inconsistent; it does not record the direction or the factor, and the
headline number in the equation is the wrong one.

### 3b. The pixel count is for the wrong protocol, by a further 132x

The abstract claims the work "facilitate[s] the analysis of the new
VisiumHD-3' protocol at its native 8 μm resolution". The runtime section
assumes 5,000 pixels, which is Visium, not VisiumHD.

```
VisiumHD 8 um bins on a 6.5 mm capture area = (6500/8)^2 ~ 660,000
ratio to 5,000                              = 132
```

The formula has **no pixel term**, so it cannot be rescaled without
recomputing the constant. Combined with 3a:

```
90.6 . 132 ~ 1.2e4  ->  1 min becomes ~8.3 days per slice
```

**Severity: this is the paper's central efficiency claim.** The abstract
promises ">10x efficiency gains" precisely to make the 8 μm protocol
tractable, and the only quantitative runtime statement in the paper is
computed at 1/132 of that protocol's resolution and 1/90 of its own segment
count.

**Action:** recompute the constant with a pixel term in the formula, at the
resolution the abstract claims. Until then #7, #11, #16 and #40 have nothing
to measure against -- an audit cannot compare a measurement to a number the
text disowns, and this one is wrong in a direction the `\todo` does not name.

**Also:** the section has a math-mode error, `... S_{gn}, \xi) of \simeq 1
\mu s$`, with the word "of" inside the math. It has never been caught because
the file does not reach the compiled document (§5).

---

## 4. Five citations do not resolve

All `\ref` and `\eqref` resolve -- 14 labels, 8 referenced, no duplicates, no
dangling. Citations do not:

| key | cited in | what it is |
| --- | --- | --- |
| `BoykovVekslerZabih` | `wolff.tex` | the α-expansion approximation bound |
| `YedidiaFreemanWeiss` | `forney.tex` | belief propagation / free energy |
| `NewmanBarkema` | `wolff.tex` | Monte Carlo methods, the Wolff algorithm |
| `szeliski_benchmark` | `wolff.tex` | the MRF energy-minimization comparison |
| `Yang2026.02.04.703493` | `intro.tex` | a preprint |

33 keys are cited, 269 are in `cnamaste.bib`, and these five are in neither
the bib nor anywhere else.

**Severity: the first four are the solver argument's entire evidentiary
base.** `wolff.tex` argues that ICM freezes into spurious clusters and that
Wolff plus α-expansion resolves it; three of its four supporting citations do
not exist in the bibliography. #29 is where that argument gets a referee, and
it cannot cite the paper's sources if the paper does not.

---

## 5. Twelve of twenty-six files never reach the compiled document

Computed as the `\input`/`\include` closure from `main.tex`, ignoring
commented lines.

**Reachable (14):** `main`, `intro`, `forney`, `phasing`, `likelihood`,
`emission` (via `likelihood`), `wolff`, `cna_maste++`,
`he_clone_initialization`, `efficient_likelihood`, `prior_model_selection`,
`integer_copy_numbers`, `results`, `background`.

**Unreachable (12):**

| file | lines | words | status |
| --- | ---: | ---: | --- |
| `sample_tikz.tex` | 484 | 1,945 | figure scratch |
| `minka.tex` | 104 | 520 | **substantive** -- message passing |
| `initialize.tex` | 103 | 1,197 | **substantive** -- initialization |
| `old_tikz.tex` | 100 | 399 | superseded figure source |
| `hidden_markov.tex` | 85 | 793 | **substantive** -- defines `T`, `E`, Viterbi, forward/backward |
| `example_theorems.tex` | 38 | 169 | template; commented out in `main.tex` |
| `supplementary.tex` | 35 | 483 | **substantive** |
| `vi.tex` | 25 | 543 | **substantive** -- variational inference |
| `validation.tex` | 19 | 112 | **substantive** -- validation |
| `expected_runtime.tex` | 7 | 115 | **substantive** -- §3 |
| `abstract.tex` | 0 | 0 | empty |
| `tikz_cuboid.tex` | 0 | 0 | empty |

Seven are substantive. Two consequences:

- **`hidden_markov.tex` defines notation the rest of the paper uses.** `T_kl`
  and `E_k(b)` appear in the reachable sections without definition, so a
  reader of the compiled document meets undefined symbols and a reviewer
  comparing code against the paper cannot find the recursion the code
  implements. It is also where defect §2 lives, unseen.
- **`expected_runtime.tex` is the only quantitative runtime claim** and it is
  not in the paper. Four `port` tickets are written against it.

**Action:** for each file, either wire it in or record it as deliberately
out. A section that defines notation the document uses is not optional. Until
this is settled, #28 and #29 cannot know whether they may cite these files at
all -- and three of the five findings above live in files a reader never sees.

---

## 6. `\todo` inventory

**53 markers across 15 files**, 44 of them in files that reach the document.

| file | markers | | file | markers |
| --- | ---: | --- | --- | ---: |
| `main.tex` | 14 | | `intro.tex` | 4 |
| `phasing.tex` | 6 | | `validation.tex` | 4 |
| `background.tex` | 5 | | `vi.tex` | 4 |
| `he_clone_initialization.tex` | 4 | | `emission.tex` | 3 |
| `likelihood.tex` | 2 | | `prior_model_selection.tex` | 2 |
| others (5 files) | 1 each | | | |

They are not one kind of thing, and the distinction matters:

- **Standing in for a citation or cross-reference** — `\todo{[REF]}`,
  `\todo{REF}`, `\todo{Eagle2}`, `\todo{copy typing}`, `\todo{\S{...}}`.
  These are the ones that make a claim unverifiable.
- **Standing in for a result** — 8 bare `\todo{...}`, plus `\todo{tissue}`
  twice and `\todo{2–3x}` in the abstract. **Three of the abstract's
  quantitative claims are `\todo` markers**: `>10x` efficiency, `2–3x`
  sensitivity and specificity, and `8 μm`.
- **Standing in for prose that is already written inside the marker** — the
  longest are complete paragraphs, e.g. `emission.tex`'s explanation of the
  de-biasing, and `wolff.tex`'s comparison with SpaCNA. These are finished
  text wearing a marker.

**Action:** separate the three. The first class blocks #28, which cannot
verify a claim whose source is `\todo{REF}`. The third can be promoted
in place at no cost.

---

## 7. `results.tex` is a placeholder

```latex
\includegraphics[width=0.5\linewidth]{}   % empty path
\caption{Caption}
\label{fig:placeholder}
```

Three empty subsections follow: simulation validation, VisiumHD-3' ovarian,
VisiumHD-3' pancreatic.

Stated rather than treated as a finding -- a results section is written last.
It matters here only because it bears on §5: `validation.tex` exists, is
unreachable, and is the section `results.tex` would draw on.

---

## Summary

| § | finding | severity |
| --- | --- | --- |
| 1 | the emission mean is independent of the copy state, as written | changes the model |
| 2 | `T_kl` is defined as the transpose of its own usage | changes the model, silently |
| 3a | 3.2e3 segments contradicts the derivation by 90.6x; true per-slice runtime ~97 min | changes the claim |
| 3b | the runtime assumes 5,000 pixels; the abstract claims 660,000 | changes the claim |
| 4 | 5 citations unresolved, 4 of them the solver argument's basis | changes what is evidenced |
| 5 | 12 files unreachable, 7 substantive, incl. the notation section | changes what a reader sees |
| 6 | 53 `\todo`, incl. 3 of the abstract's quantitative claims | inventory |
| 7 | `results.tex` is a placeholder | expected |

**Blocking #28 and #29:** §1 and §2, because a code-vs-paper verdict against
an equation that does not say what it means is not a verdict. §3 and §5,
because #7, #11, #16 and #40 measure against a runtime claim that is neither
correct nor in the document.

**Feeding #5 directly:** §1 is the index ambiguity #5 records as a raised
difference, now quantified -- the literal reading does not merely underspecify
the normalizer, it cancels the state dependence entirely.
