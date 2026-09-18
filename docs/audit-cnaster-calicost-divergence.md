# Audit: `cnaster` against CalicoST, beyond `logmu_shift` and integer copy

Issue #131. The third of three: what else has diverged, and how the two
codebases compare as code.

Paper at `0478c74`; `cnaster` at `4adad4d`; CalicoST at `c1abcae`.
**CalicoST was read, never run** (#131).

**`cnaster` is a rewrite, not a fork with patches.** It is better engineered
on every axis that can be counted -- typing, logging, configuration, compiled
kernels, a real encoder on the emission -- and it has dropped capabilities
CalicoST ships without recording that it did. The sharpest of those is the
tumour-proportion mixture: CalicoST supports a **per-site** proportion in the
emission, and `cnaster`'s pipeline signature carries `tumor_prop=None, # TODO`.

---

## 1. Per-site tumour proportion

CalicoST's `compute_emission_probability_nb_betabinom_mix`
(`hmm_NB_BB_nophasing_v2.py:117-151`) mixes a tumour and a normal population
per observation:

```python
# nb_mean = base_nb_mean[idx,s] * (tumor_prop[s] * np.exp(log_mu[i,s]) + 1 - tumor_prop[s])
nb_mean = base_nb_mean[idx, s] * (tumor_prop[idx, s] * np.exp(log_mu[i, s]) + 1 - tumor_prop[idx, s])
...
mix_p_A = p_binom[i, s] * this_weighted_tp[idx] + 0.5 * (1 - this_weighted_tp[idx])
```

The commented line above it is the **per-spot scalar** form; the live line
indexes `tumor_prop` by observation. So CalicoST moved from a scalar to a
per-site proportion and kept the history in place. A dedicated module,
`estimate_tumor_proportion.py` (120 lines), estimates it.

`cnaster` has no `_mix` emission at all. `hmm_nophasing` defines
`compute_emission_probability_nb_betabinom` and its `_coded` variant, both
unmixed; `optimize` at `hmm_nophasing.py:797` reads `tumor_prop=None, # TODO`
and **the name occurs nowhere else in the function** -- while `hmm.py:110`
threads a caller's proportion straight into it. #135. What survives is the M step: `hmm_emission.py:258`
`Weighted_BetaBinom_mix` takes `tumor_prop` and uses it, and
`pseudobulk.merge_pseudobulk_by_index_mix` threads a proportion through
aggregation.

**So the mixture is half-present.** The M step can fit under a tumour
proportion the E step cannot score under. Any run with a non-trivial
proportion optimizes a different objective from the one it evaluates. The
round trip in `port` passes `None` throughout, so no test here has been in the
regime where this bites -- which is why it is recorded rather than
demonstrated.

`normal_spot.py` is 1,047 lines against CalicoST's 120, so `cnaster` has
invested heavily on the *identification* side while dropping the *application*
side. That asymmetry is the finding.

## 2. Phase-switch HMM

CalicoST ships `hmm_NB_BB_phaseswitch.py` as a first-class module with its own
Baum-Welch, and `hmm_NB_BB_nophasing_v2.py` beside it. `cnaster` has
`hmm_phased.py` and `hmm_nophasing.py`. The correspondence is direct, but
`hmm_phased`'s default path is unusable: it indexes `log_mu[i, s]` over the
encoder's spot count after overwriting `n_spots`, so it raises on any instance
with more than one spot (#80, pinned in
`tests/test_core_inference_end_to_end.py`). CalicoST's phase-switch module has
no equivalent defect -- the loop bounds come from `log_mu.shape` throughout.

This is a regression introduced by the rewrite, and it is the reason every
`port` fixture passes `hmmclass=hmm_nophasing` explicitly.

## 3. Modules CalicoST ships that `cnaster` does not

`phylogeography.py`, `phylogeny_startle.py`, `calicost_supervised.py`,
`hmm_gaussian.py`, `hmm_NB_sharedstates.py`, `hmrf_normalmixture.py`.

The phylogeny pair is downstream analysis rather than inference and its
absence is a scope choice. `hmm_gaussian` and `hmm_NB_sharedstates` are
alternative emission models; `hmrf_normalmixture` is a normal-mixture variant
of the spatial step. None of the six is referenced in `cnaster`, and none of
the absences is recorded there.

## 4. Packaging: CalicoST's version pins are dead metadata

Worth recording because it decided how this pin was added. `setup.py` pins
`numpy==1.24.4`, `scipy==1.11.3`, `numba==0.60.0`, `matplotlib==3.7.3` and ten
more exactly. Its `pyproject.toml` declares the same packages **unpinned**
under a PEP 621 `[project]` table, which setuptools treats as authoritative,
so the exact pins are never applied.

Measured: `uv lock --upgrade-package calicost` resolved in **12.5 s** and added
**two** packages, `calicost` and `ete3`. The environment kept `numpy 2.5.3`,
`numba 0.67.0` and `matplotlib 3.11.2` -- a stack CalicoST's `setup.py` claims
to forbid. Whether CalicoST *runs* against it is untested and untestable here,
since #131 says not to run it; what is established is that nothing stops it
being installed.

For a reader of CalicoST this is the more useful finding: its declared
environment (`environment.yml`, conda, Python 3.10, numpy 1.24) and its
installable environment are different, and only the first is reproducible.

## 5. Configuration

CalicoST parses arguments per script (`arg_parse.py`) and carries
`config.yaml` plus four `configuration_*` directories of examples. `cnaster`
has a single `config.py` with a `YAMLConfig` object and a global accessor,
93 keys, three ways of reading them (#83).

`cnaster`'s is better structured and worse audited: a global that any module
may read at any time is what makes #83's "no static check can tell dead from
live" true, and it is what `port`'s fixtures have to install and restore by
hand around every test.

---

## Pros and cons, as code

Counted over `src/calicost/*.py` (27 modules, 12,901 lines) against
`site-packages/cnaster/*.py` (37 modules, 15,134 lines).

| axis | CalicoST | `cnaster` | verdict |
| --- | ---: | ---: | --- |
| modules / lines | 27 / 12,901 | 37 / 15,134 | `cnaster` larger, more separated |
| `@njit` kernels | 30 | 30 | level |
| annotated returns (`def ... ->`) | **0** | 29 | `cnaster`, thinly |
| `logger.` calls | 6 | **364** | `cnaster` |
| bare `print(` | **80** | 2 | `cnaster` |
| test files shipped | 0 | 0 | level, and the reason `port` exists |

### Code style

**`cnaster` wins, and not narrowly.** 364 logger calls against 6, and 2 bare
prints against 80, is the difference between a library and a script
collection: CalicoST's output cannot be silenced, redirected or levelled by a
caller. `cnaster` also separates concerns into 37 modules where CalicoST's
`hmrf.py` and `utils_hmrf.py` carry the spatial step, the clone initialization
and the pseudobulk together.

Against that, `cnaster` carries its history in the source. `# DEPRECATE`,
`# MAGIC`, `# TODO HACK`, `# TODO PATCH` and commented-out call sites are
throughout, and two of this audit's findings are exactly such a comment
(`logmu_shifts` at `hmm_nophasing.py:280`, the diploid selector at
`integer_copy.py:84`). CalicoST does the same -- `### temp penalty ###`,
`oldcode.py` -- so neither is clean, but `cnaster`'s volume of annotation
makes it the one where a reader cannot tell intent from residue.

Neither is typed to a standard a checker could enforce: 29 annotated returns
over 15,134 lines is about one module's worth.

### Runtime

**`cnaster` should be faster, structurally, and the evidence is indirect.**

- CalicoST's emission is a Python double loop over `(n_states, n_spots)`
  calling `scipy.stats.nbinom.logpmf` and `scipy.stats.betabinom.logpmf` per
  state and spot. `cnaster` calls `@njit` kernels `_nb_logpmf_1d` and
  `_bb_logpmf_1d` over the same loop.
- `cnaster` adds `count_encoder.CountEncoder`, which scores each **distinct**
  `(count, exposure)` pair once and decodes back to the full array
  (`get_unique_obs`, `decode_array`). CalicoST has no equivalent -- no
  `np.unique` appears in its emission path. On count data with a small support
  this is a large constant factor, and it is the mechanism behind
  `compute_emission_probability_nb_betabinom_coded`.
- `cnaster` reformulates the integer-copy search as a MILP solved by HiGHS
  where CalicoST hill climbs (see `docs/audit-integer-copy-calicost.md`).

Two caveats, both measured here rather than assumed. First, the emission is
still **68 per cent** of a `run_cnaster` round trip at the configured
iteration count (#104), so the encoder has not removed the bottleneck.
Second, `port`'s own measurement puts `cnaster`'s two-channel emission at
**228.2 ms** against upstream's vectorized `snakes_and_ladders` families at
**67.2 ms** on the same 200-bin dev instance -- so whatever `cnaster` gains on
CalicoST, it gives back against a properly vectorized implementation.

**No ratio against CalicoST is claimed.** #131 says not to run it, so the
runtime column above is a structural argument, not a measurement. Establishing
one would need CalicoST importable and a shared fixture, which is what adding
it as a dependency makes possible.

### Capability

| capability | CalicoST | `cnaster` |
| --- | --- | --- |
| per-site tumour proportion in the emission | **yes** | no (`# TODO`) |
| joint / composite multi-clone integer copy | **yes** | no |
| phylogeny and phylogeography | **yes** | no |
| alternative emissions (Gaussian, shared-state NB) | **yes** | no |
| `logmu_shift` applied | **yes**, to BAF | no |
| globally optimal integer copy (MILP) | no | **yes** |
| distinct-count encoder on the emission | no | **yes** |
| structured configuration and logging | no | **yes** |
| credible-set decoding (the paper's method) | no | no |

**CalicoST is the more capable program; `cnaster` is the better-built one.**
Every capability in `cnaster`'s column is an engineering improvement to
something CalicoST already did. Every capability in CalicoST's column is
absent from `cnaster` without a record of the decision.

That asymmetry is the summary of this audit, and it is what makes the
dependency worth pinning: the divergences are findable only by reading the two
side by side, and there was no way to do that until now.

---

## Roadmap

1. **Restore the mixed emission, or state that the proportion is out of
   scope.** The half-present mixture (M step yes, E step no) is the only
   finding here that makes a run silently inconsistent rather than merely less
   capable. Fixing it is porting one function; deciding not to is one
   sentence. Leaving it is neither.
2. **Record the six dropped modules.** One line each in `cnaster` saying
   whether the absence is scope or backlog.
3. **Fix `hmm_phased`'s spot indexing (#80).** CalicoST's phase-switch module
   is a working reference for the loop bounds, which is a use for the pin that
   needs no measurement.
4. **Measure, once the pin lands.** The runtime column above is structural.
   A shared fixture through both emissions would turn it into a ratio, and
   `port` already has the fixture -- `core_inference_truth` plants exactly the
   arrays both signatures take.

## Upstream correspondence

Most of this is application specific and does not reach
`snakes_and_ladders`. The exception is the mixture: a two-population mixture
at a per-observation weight is a property of an emission family, not of a
caller, and upstream's covariate work (#57, #65, #77) is the surface it would
attach to. If that landed, both CalicoST's per-site form and `cnaster`'s
missing one would be the same upstream feature rather than two application
decisions.
