"""Set aside (T- #831): integer copies from the fit's error bars, every `(A, B)` they admit (#353).

Ticket: #353 -- the paper's set-valued decoding; T- #831 set aside what no
  `--sal` run executes, with `run_cnaster_port --copy-errors`.
Measurement: #705's population arms `errors`, `flat` and `shared` against
  `sal` (`port.studies.population`); not part of the `--sal` run.
Exit: graduate to `extensions/` with an opt-in flag if credible sets beat the
  lattice point decode's CNA recall on the population study; else retire
  with `parameter_errors` and `jax_hmm`.

Integer copies from the fit's error bars: every `(A, B)` they admit (#353).

`port.extensions.integer_copy.decode_copy_state` is the paper's decoding: a
state's `(mubar, p)` and its covariance give the **set** of integer pairs
inside the credible region, not one winner. `port.qa.parameter_errors`
computes the covariance from the observed information of the objective the
HMM maximized. Neither was reached by a run. This joins them, and
`run_cnaster_port --copy-errors` calls it on the final fit.

## The scale comes from the pin, and the neutral state is (1, 1)

`mubar = (A + B) / 2` holds only on the de-biased scale. With the per-clone
`logmu_shift` folded in (`port.patch.hmm_nophasing`, #276), the likelihood is
flat along `mu -> c mu`, and `port.patch.hmrf.core_inference.pin_neutral`
fixes `c` by setting the normal clone's dominant balanced state (#299's
`neutral_state`) to `mu = 1`. So:

- the covariance is taken **in the pinned coordinates**: the neutral `log mu`
  is held at 0 and the rest differentiated, through the shift, whose
  normalizer is a function of every rate (`parameter_errors`' Jacobian);
- the neutral state's `mu` is 1 exactly and carries no error, so its
  decoding conditions on `mubar = 1`: the pairs of total 2 its allele
  fraction admits, `(1, 1)` or `(2, 0)`, by a one-dimensional test at the
  same level.

Without the shift the fit's scale is the baseline's, not the pin's, and the
decode would compare `(A + B) / 2` against a rate with an unknown per-clone
factor; `run_cnaster_port` refuses `--copy-errors` without it.

## Allele fractions are folded

Phasing makes the allele label arbitrary, so `p` is folded to the minor
fraction and the lattice is the unphased one (`acn_lattice(phased=False)`,
`A >= B`, `p = B / (A + B)`). Folding flips the sign of the `(mu, p)`
covariance where it applies and leaves the variances alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from port.extensions.copy_likelihood import Captured
from port.qa.errors import VARIANCE_FLOOR, PinnedErrors, pinned_errors


def _neutral_set(minor: float, variance: float, level: float) -> Any:
    """The neutral state: `mubar = 1` exactly, so the total is 2 and `p` decides."""
    from scipy.stats import chi2

    from port.extensions.integer_copy import IntegerCopyResult

    pairs = ((1, 1), (2, 0))
    implied = np.array([0.5, 0.0])
    distances = (implied - minor) ** 2 / variance
    order = np.argsort(distances, kind="stable")
    threshold = float(chi2.ppf(level, 1))

    return IntegerCopyResult(
        best=pairs[int(order[0])],
        consistent=tuple(pairs[int(i)] for i in order if distances[i] <= threshold),
        distance=float(distances[order[0]]),
        threshold=threshold,
        level=level,
    )


def copy_sets(
    errors: PinnedErrors,
    *,
    level: float = 0.95,
    max_allele_copy: int | None = None,
    max_total_copy: int | None = None,
) -> list[Any]:
    """Every `(A, B)` each state's error bars admit, at `level`.

    The caps default to the configured ones (`int_copy_num.max_total_copy`,
    read by `port.patch.integer_copy.configured_caps`, else `cnaster`'s 5 and
    6), so the set is drawn from the lattice the run's own integer decoder
    searches.
    """
    from port.extensions.integer_copy import acn_lattice, decode_copy_state
    from port.patch.integer_copy import configured_caps

    allele, total = configured_caps()
    lattice = acn_lattice(
        max_allele_copy=allele if max_allele_copy is None else max_allele_copy,
        max_total_copy=total if max_total_copy is None else max_total_copy,
        phased=False,
    )

    decoded = []

    for state in range(errors.mu.size):
        if state == errors.neutral:
            decoded.append(
                _neutral_set(
                    float(errors.minor[state]),
                    float(errors.covariance[state, 1, 1]),
                    level,
                )
            )
            continue

        # NB a held parameter (`pinned_errors`) is known exactly; its variance
        #    is floored so the region is defined and admits only its value.
        covariance = errors.covariance[state] + np.eye(2) * VARIANCE_FLOOR
        decoded.append(
            decode_copy_state(
                [errors.mu[state], errors.minor[state]],
                covariance,
                level=level,
                lattice=lattice,
            )
        )

    return decoded


def copy_set_table(errors: PinnedErrors, decoded: list[Any]) -> pd.DataFrame:
    """One row per `(state, A, B)` in a state's set; a state whose set is empty
    has one row with `A` and `B` empty, so it is reported rather than dropped."""
    rows = []

    for state, result in enumerate(decoded):
        base = {
            "state": state,
            "neutral": state == errors.neutral,
            "mu": float(errors.mu[state]),
            "p_minor": float(errors.minor[state]),
            "sigma_mu": float(np.sqrt(errors.covariance[state, 0, 0])),
            "sigma_p": float(np.sqrt(errors.covariance[state, 1, 1])),
            "best_A": result.best[0],
            "best_B": result.best[1],
            "best_distance": result.distance,
            "threshold": result.threshold,
            "level": result.level,
            "set_size": len(result.consistent),
        }

        if not result.consistent:
            rows.append({**base, "A": pd.NA, "B": pd.NA})

        for a, b in result.consistent:
            rows.append({**base, "A": a, "B": b})

    return pd.DataFrame(rows)


def write_copy_sets(run: Path, captured: Captured, *, level: float = 0.95) -> Path:
    """Write `cnv_copy_sets.tsv` into `run`, and return its path."""
    errors = pinned_errors(captured)
    table = copy_set_table(errors, copy_sets(errors, level=level))
    path = Path(run) / "cnv_copy_sets.tsv"

    with path.open("w") as handle:
        handle.write(
            f"# every (A, B) inside the {level:.2%} credible region of each "
            f"fitted state (#353); state {errors.neutral} is pinned to mu = 1; "
            f"Newton decrement {errors.decrement:.3e}\n"
        )
        table.to_csv(handle, sep="\t", index=False)

    return path


def write_beside_final_fit(
    config: str, kept: list[Any], *, since: float | None = None
) -> None:
    """Write the credible sets beside the final fit this run wrote.

    The fit is the one of the configured `hmm.n_states` written at or after
    `since`, not the newest under `output_dir`, which may be another
    configuration's (T- #617). With none, nothing is written, and it says so,
    as for a fit `pinned_errors` refuses (#705).
    """
    import sys

    import yaml

    if not kept:
        print("run_cnaster_port: --copy-errors kept no fit", file=sys.stderr)
        return

    stated = yaml.safe_load(Path(config).read_text())
    output = Path(stated["paths"]["output_dir"])
    n_states = (stated.get("hmm") or {}).get("n_states")
    pattern = (
        "rdrbaf_final_nstates*_smp.npz"
        if n_states is None
        else f"rdrbaf_final_nstates{int(n_states)}_smp.npz"
    )
    fits = [
        fit
        for fit in output.rglob(pattern)
        if since is None or fit.stat().st_mtime >= since
    ]

    if len(fits) != 1:
        print(
            f"run_cnaster_port: --copy-errors found {len(fits)} fits this run "
            f"wrote under {output}; cnv_copy_sets.tsv not written",
            file=sys.stderr,
        )
        return

    # NB a refused fit (T- #599's large tau) leaves no sets and says so; the
    #    run it follows completed, and its outputs stand (#705).
    try:
        path = write_copy_sets(fits[0].parent, kept[-1])
    except ValueError as refused:
        print(
            f"run_cnaster_port: {refused}; cnv_copy_sets.tsv not written",
            file=sys.stderr,
        )
        return

    print(f"run_cnaster_port: wrote {path}", file=sys.stderr)
