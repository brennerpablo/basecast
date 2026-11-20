"""Landmark-date cohorts for the generation-queue survival curves (exploration X21).

Phase 0 Q6 fits every landmark on the cohort first listed on or after 2018-08 (the first report of the GINR
layout). For the FIS approved and IA signed landmarks that leaves out projects first listed in the old layout
that reached the milestone later, although everything after the milestone is recorded in the GINR layout.
This module defines the cohort by the **landmark date** instead, and fits the curves with the Q6 / X2 code as
is (:mod:`survival` and :mod:`queue_adjusted` are imported, never changed):

- ``q6``: first listed on or after ``landmark_start`` (Q6's cohort; delayed entry at the first listing);
- ``landmark``: milestone on or after ``landmark_start`` and first listed on or after ``listed_start``
  (2017-04, when cancellations start to be recorded); delayed entry where the project was listed after its
  milestone;
- ``union``: ``landmark`` plus ``q6`` (keeps Q6's projects listed after a pre-2018-08 milestone);
- ``period``: every project still at risk on ``landmark_start`` (calendar left truncation): a project listed
  before that date whose milestone is older enters the risk set on ``landmark_start``, not at the milestone.
  FIS approval dates only exist in the GINR layout, so a project that left the queue before 2018-08 has no
  FIS date at all; truncating at 2018-08 is what makes the older milestones usable without survivor bias.

The cohort is applied by nulling the milestone of the projects outside it (and, for ``period``, moving
``first_seen_month`` up to ``landmark_start``), so :func:`queue_adjusted.stage_frame` and
:func:`queue_adjusted.fit_stage_curves` run unchanged on the result. Only :func:`load_events_all` touches the
database.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date

import numpy as np
import polars as pl

from basecast_pipelines.models import queue_adjusted as qa
from basecast_pipelines.models import survival as sv

LANDMARK_START = sv.COHORT_START  # 2018-08-01: first GINR report; milestones after it are fully observed
LISTED_START = date(2017, 4, 1)  # cancellations are recorded from the 2017-04 report on (docs/decisions.md)
COHORTS: tuple[str, ...] = ("q6", "landmark", "union", "period")


def cohort_keep(
    stage: str,
    cohort: str,
    *,
    landmark_start: date = LANDMARK_START,
    listed_start: date | None = LISTED_START,
) -> pl.Expr:
    """Boolean expression on an outcomes frame: the project belongs to ``cohort`` at ``stage``.

    Needs ``first_seen_month``, the stage's milestone column and ``event_date`` (an outcomes frame, see
    :func:`survival.outcomes` or :func:`queue_adjusted.truncate_at`). A null milestone gives false.
    """
    if cohort not in COHORTS:
        raise ValueError(f"unknown cohort {cohort!r}; expected one of {COHORTS}")
    col = pl.col(qa.STAGE_DATES[stage])
    first = pl.col("first_seen_month")
    late_listed = first >= pl.lit(landmark_start)
    by_landmark = col >= pl.lit(landmark_start)
    if listed_start is not None:
        by_landmark &= first >= pl.lit(listed_start)
    if cohort == "q6":
        keep = late_listed
    elif cohort == "landmark":
        keep = by_landmark
    elif cohort == "union":
        keep = late_listed | by_landmark
    else:  # period: still at risk on landmark_start (the event, or censoring, comes after it)
        keep = late_listed | (pl.col("event_date") >= pl.lit(landmark_start))
    return col.is_not_null() & keep.fill_null(False)


def restrict(
    outcomes: pl.DataFrame,
    stage: str,
    cohort: str,
    *,
    landmark_start: date = LANDMARK_START,
    listed_start: date | None = LISTED_START,
) -> pl.DataFrame:
    """The projects of ``cohort`` at ``stage``, ready for :func:`queue_adjusted.stage_frame` /
    :func:`queue_adjusted.fit_stage_curves` with that one stage.

    For ``period``, ``first_seen_month`` is moved up to ``landmark_start`` so the delayed entry is the
    later of the first listing and ``landmark_start``. Only use the result for ``stage``.
    """
    keep = cohort_keep(stage, cohort, landmark_start=landmark_start, listed_start=listed_start)
    out = outcomes.filter(keep)
    if cohort == "period":
        out = out.with_columns(
            first_seen_month=pl.max_horizontal(pl.col("first_seen_month"), pl.lit(landmark_start))
        )
    return out


def cohort_frame(outcomes: pl.DataFrame, stage: str, cohort: str, **kw) -> pl.DataFrame:
    """Landmark frame (``origin``, ``entry``, ``time``, ``event``) of ``cohort`` at ``stage``."""
    return qa.stage_frame(restrict(outcomes, stage, cohort, **kw), stage)


def membership(
    outcomes: pl.DataFrame,
    stage: str,
    *,
    landmark_start: date = LANDMARK_START,
    listed_start: date | None = LISTED_START,
) -> pl.DataFrame:
    """One row per project that reached ``stage``: ``in_<cohort>`` flags plus ``why`` for the landmark vs q6
    difference (``both``, ``landmark only``, ``q6 only``, ``neither``)."""
    kw = {"landmark_start": landmark_start, "listed_start": listed_start}
    col = qa.STAGE_DATES[stage]
    flags = {f"in_{c}": cohort_keep(stage, c, **kw) for c in COHORTS}
    out = outcomes.filter(pl.col(col).is_not_null()).with_columns(**flags)
    return out.with_columns(
        why=pl.when(pl.col("in_q6") & pl.col("in_landmark")).then(pl.lit("both"))
        .when(pl.col("in_landmark")).then(pl.lit("landmark only"))
        .when(pl.col("in_q6")).then(pl.lit("q6 only"))
        .otherwise(pl.lit("neither"))
    )


def fit_cohort_curves(
    outcomes: pl.DataFrame,
    stage: str,
    cohort: str,
    *,
    weight: str | None = "capacity_mw",
    **kw,
) -> dict[tuple[str, str], qa.StageCurve]:
    """:func:`queue_adjusted.fit_stage_curves` for one stage on ``cohort`` (keys ``(stage, stratum)``)."""
    return qa.fit_stage_curves(restrict(outcomes, stage, cohort, **kw), (stage,), weight=weight)


def semi_markov_curves(
    pre_ia_outcomes: pl.DataFrame,
    ia_outcomes: pl.DataFrame,
    ia_cohort: str,
    *,
    weight: str | None = "capacity_mw",
    **kw,
) -> dict[tuple[str, str], qa.StageCurve]:
    """X2's ``entry_ia_sm`` curves with the IA stage fit on ``ia_cohort``.

    The pre-IA sojourn (clock from entry) is fit on ``pre_ia_outcomes`` exactly as in X2; the IA-stage curves,
    which also feed the semi-Markov convolution, come from ``ia_outcomes`` restricted to ``ia_cohort``.
    """
    curves = qa.fit_stage_curves(pre_ia_outcomes, (qa.PRE_IA,), weight=weight)
    return curves | fit_cohort_curves(ia_outcomes, "ia_signed", ia_cohort, weight=weight, **kw)


def ordering(
    cif36: Mapping[str, Mapping[str, float]],
    order: Sequence[str] = ("entry", "fis_approved", "ia_signed"),
) -> dict[str, bool]:
    """Per stratum: is the CIF strictly increasing along ``order`` (landmark -> value)?

    ``cif36`` is stratum -> {landmark: CIF}; a stratum missing a landmark is not ordered.
    """
    out = {}
    for stratum, by_lm in cif36.items():
        vals = [by_lm.get(lm) for lm in order]
        out[stratum] = all(v is not None for v in vals) and all(a < b for a, b in zip(vals, vals[1:]))
    return out


def bootstrap_difference(
    a: pl.DataFrame,
    b: pl.DataFrame,
    horizon: float,
    *,
    n_boot: int = 500,
    seed: int = 0,
    key: str = "inr",
    weight: str | None = None,
) -> tuple[float, float, float]:
    """CIF of COD at ``horizon``: point estimate of ``a - b`` and a 95% percentile interval.

    ``a`` and ``b`` are landmark frames that may share projects (FIS approved and IA signed); projects are
    resampled jointly by ``key`` so the correlation between the two curves is kept.
    """
    def cif(frame: pl.DataFrame) -> float:
        if frame.is_empty():
            return float("nan")
        w = None if weight is None else frame[weight].to_numpy()
        fit = sv.fit_cif(frame["time"].to_numpy(), frame["event"].to_numpy(), frame["entry"].to_numpy(), w)
        return float(fit.at([horizon], sv.COD)[0])

    point = cif(a) - cif(b)
    ids = np.unique(np.concatenate([a[key].to_numpy(), b[key].to_numpy()]))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        pick = pl.DataFrame({key: rng.choice(ids, size=ids.size, replace=True)})
        diffs[i] = cif(pick.join(a, on=key, how="inner")) - cif(pick.join(b, on=key, how="inner"))
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return point, float(lo), float(hi)


# --- loader ---------------------------------------------------------------------------------------------


def load_events_all() -> pl.DataFrame:
    """Every INR of ``gis_project_events`` (no cohort filter), with ``fuel_group`` and ``stratum``."""
    return sv.load_events(cohort_start=None).with_columns(stratum=qa.stratum())
