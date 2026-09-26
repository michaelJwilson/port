# Figures

**What `run_cnaster_port` draws for the dev instance, committed so a change to
the pipeline shows up as a change to a picture.** Nineteen of them, written at
every stage of the run rather than at the end, and `genomic.pdf` and
`spatial.pdf`, the final four as two figures.

**CI regenerates every figure here on every pull request and commits the
result** to the pull request's branch (`.github/workflows/figures.yml`,
#296): the twenty-one below, and `realizations.png`, `realizations_truth.png`
and `realizations.npz`, eight runs of one planted genome with the
likelihood's errors on one run and on the truth (#291). A branch another
open pull request is based on is the exception: its figures are uploaded as a
workflow artifact rather than committed, because every PDF carries a
timestamp and a commit there would re-conflict the pull request above it
(#350). By hand:
`python -m tests.generate_plots` (`--cnaster` for plain `cnaster`) and
`python -m tests.realizations`.

## The instance

`M = 4` clones, `K = 10` planted states, `G = 1,000` bins over ten unequal
chromosomes, `S = 1,000` spots. **Five** states are fitted, not the ten
planted: ten does not fit in 15 GB (#90), and asking for fewer states than the
data carries is what a real run does anyway.

State zero is planted diploid and balanced, `mu = 1` and `p = 0.5`. Without
one the run does not reach these figures at all (#106).

Through `run_cnaster_port` with its defaults, the shift included: 54 s end to
end and a peak of 4.25 GB (#304). Plain `cnaster` measured 31 s and 5.89 GB
before #298's normal clone, which made it loop forever (#304).

## What they are, and are not

A figure here is a **result**, not a referee. Nothing compares one against a
previous one, and a matplotlib PDF carries its creation time, so two
regenerations differ byte for byte with nothing having changed. Byte
reproduction is the standard `CLAUDE.md` names for a figure and reaching it
needs that timestamp pinned first: #103.

The run they come from is a completion claim, not a correctness one. No number
in these plots has been compared against the planted truth; that is the
component-wise work around `tests/test_run_cnaster_round_trip.py`.

## The order the run draws them

| stage | figures |
| --- | --- |
| before phasing | `pseudobulk_clones_genomic`, `prephasing_clones_genomic`, `prephasing_clones_spatial` |
| after phasing | `postphasing_clones_genomic`, `postphasing_aggr_clones_genomic`, `postphasing_pseudobulk_clones_genomic` |
| initial clones | `initial_clones_spatial` |
| BAF only | `bafonly_clones_genomic`, `bafonly_clones_spatial`, `merged_bafonly_clones_genomic`, `merged_bafonly_clones_spatial` |
| RDR and BAF | `rdr_baf_clones_genomic`, `rdr_baf_clones_spatial`, `merged_rdr_baf_clones_genomic`, `merged_rdr_baf_clones_spatial` |
| final | `real_clones_genomic`, `clones_genomic`, `clones_spatial`, `copy_number_profile` |
| after the run | `genomic`, `spatial` |

## `genomic.pdf` and `spatial.pdf`

The final four as two figures at a text column (#309, #280, #339).
`genomic.pdf` fills `llncs`'s text block, 122 by 193 mm: (a)
`clones_genomic` over (b) `copy_number_profile`, (b)'s axis spanning (a)'s
tracks so their chromosome boundaries line up. `spatial.pdf` is 122 mm wide
and about a quarter of the block tall: (a) an H&E slide and (b) `clones_spatial`,
square and as large as fit across, (b)'s clones keyed on the right edge. No
captions. `port.extensions.combined_figure` redraws the run's own calls and
writes each page at exactly its size, so it is included at
`width=\linewidth` unscaled.

The slide is **mocked** from the planted labels (`tests/he_slide.py`) and
read back through `cnaster.he.get_he_image`, as `run_cnaster` reads a slide.
It is
written beside the run's inputs, not into them: in them, `load_input_data`
would pick it up and refine the initial clones by it, and the other figures
would change.
