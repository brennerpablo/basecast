"""Uncertainty band for the adjusted generation queue, and a size split for large new gas (exploration X19).

Builds on :mod:`basecast_pipelines.models.queue_adjusted` (X2) and :mod:`basecast_pipelines.models.survival`
(Q6) by import only. What this module adds:

- **bootstrap band** (:func:`bootstrap`): resample the fitting cohort by project (with replacement), refit the
  stage curves, rescore the fixed active queue, then draw which projects reach COD (one Bernoulli per project
  and replicate, the same uniform for every horizon so "COD by 2027" implies "COD by 2028"). The spread of the
  simulated totals carries both curve (parameter) and outcome (Bernoulli) uncertainty; the spread of the
  expected totals carries the curve part only.
- **backtest-error band** (:func:`log_errors`, :func:`error_band`): the point forecast times
  ``exp(± z * RMSE)`` of the log errors ``ln(actual / predicted)`` of past backtests.
- **size split for large gas** (:func:`size_stratum`, :func:`split_gas_curves`): gas+other projects of at
  least ``threshold`` MW get their own stratum. Where that stratum is too thin to fit (Q6's rule: n >= 30 and
  >= 10 events), the IA-stage curve is the small-gas curve with its COD cause-specific hazard scaled by the
  observed/expected ratio of the large projects (:func:`observed_expected`, :func:`scale_cause_hazard`), and
  the pre-IA curve falls back to the combined gas+other one.
- **COD floor** (:func:`floor_delay`, :func:`rescore_with_floor`): a project cannot reach COD before a floor
  date (the developer's own projected COD, or a lead-time cap). The project's clock is delayed by the months
  from the as-of date to the floor, i.e. it gets the same curve over a window shortened by that delay.

Everything here is pure (no database): the analysis script loads the data with the ``queue_adjusted`` loaders.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date

import numpy as np
import polars as pl

from basecast_pipelines.models import queue_adjusted as qa
from basecast_pipelines.models import survival as sv

GAS = "gas+other"
BIG_MW = 500.0
Z80 = 1.2815515655446004  # standard normal quantile of 0.9: a P10-P90 band is +- Z80 standard deviations
QUANTILES = (0.1, 0.5, 0.9)

Curves = dict[tuple[str, str], qa.StageCurve]


def big_label(threshold: float = BIG_MW, fuel: str = GAS) -> str:
    return f"{fuel}>={threshold:g}"


def small_label(threshold: float = BIG_MW, fuel: str = GAS) -> str:
    return f"{fuel}<{threshold:g}"


def size_stratum(threshold: float = BIG_MW, fuel: str = GAS, *, col: str = "stratum", mw: str = "capacity_mw") -> pl.Expr:
    """``col`` with the ``fuel`` stratum split at ``threshold`` MW (null capacity counts as small)."""
    is_fuel = pl.col(col) == fuel
    return (
        pl.when(is_fuel & (pl.col(mw).fill_null(0.0) >= threshold)).then(pl.lit(big_label(threshold, fuel)))
        .when(is_fuel).then(pl.lit(small_label(threshold, fuel)))
        .otherwise(pl.col(col))
        .alias(col)
    )


# --- hazards --------------------------------------------------------------------------------------------


def cause_hazard(fit: sv.CifFit, cause: int = sv.COD) -> np.ndarray:
    """Cause-specific hazard increment at each event time of ``fit`` (events / at risk; 0 without the cause)."""
    if cause not in fit.causes or fit.times.size == 0:
        return np.zeros(fit.times.size)
    ev = fit.events[:, fit.causes.index(cause)]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(fit.at_risk > 0, ev / fit.at_risk, 0.0)


def cumulative_hazard(fit: sv.CifFit, t: Sequence[float] | np.ndarray, cause: int = sv.COD) -> np.ndarray:
    """Nelson–Aalen cumulative cause-specific hazard at ``t`` (a right-continuous step function)."""
    curve = np.concatenate([[0.0], np.cumsum(cause_hazard(fit, cause))])
    return curve[np.searchsorted(fit.times, np.asarray(t, dtype=float), side="right")]


def observed_expected(
    time: Sequence[float],
    event: Sequence[int],
    entry: Sequence[float],
    fit: sv.CifFit,
    *,
    cause: int = sv.COD,
    weights: Sequence[float] | None = None,
) -> tuple[float, float]:
    """Observed and expected (weighted) events of ``cause`` for subjects at risk on ``(entry, time]``.

    Expected = sum of the reference ``fit``'s cumulative cause-specific hazard over each subject's own
    at-risk window. ``observed / expected`` is the maximum-likelihood hazard ratio of the subjects against the
    reference curve under proportional cause-specific hazards (a standardized incidence ratio).
    """
    t, s = np.asarray(time, dtype=float), np.asarray(entry, dtype=float)
    e = np.asarray(event, dtype=int)
    w = np.ones_like(t) if weights is None else np.asarray(weights, dtype=float)
    observed = float((w * (e == cause)).sum())
    expected = float((w * (cumulative_hazard(fit, t, cause) - cumulative_hazard(fit, s, cause))).sum())
    return observed, expected


def scale_cause_hazard(fit: sv.CifFit, theta: float, cause: int = sv.COD) -> sv.CifFit:
    """The fit with the ``cause``-specific hazard multiplied by ``theta`` (the other causes unchanged).

    Survival and every CIF are recomputed from the scaled hazards (Aalen–Johansen form); the total hazard at a
    time is capped at 1. ``at_risk`` is kept, so :func:`queue_adjusted.support_end` of a count fit is unchanged.
    """
    if theta < 0:
        raise ValueError("theta must be >= 0")
    if fit.times.size == 0 or cause not in fit.causes:
        return fit
    with np.errstate(divide="ignore", invalid="ignore"):
        hazard = np.where(fit.at_risk[:, None] > 0, fit.events / fit.at_risk[:, None], 0.0)
    hazard[:, fit.causes.index(cause)] *= theta
    total = hazard.sum(axis=1)
    over = total > 1.0
    hazard[over] /= total[over, None]
    survival = np.cumprod(1.0 - hazard.sum(axis=1))
    before = np.concatenate([[1.0], survival[:-1]])
    cif = np.cumsum(before[:, None] * hazard, axis=0)
    return sv.CifFit(fit.times, fit.at_risk, hazard * fit.at_risk[:, None], survival, cif, fit.causes, fit.n)


# --- size split -----------------------------------------------------------------------------------------


def split_gas_curves(
    outcomes: pl.DataFrame,
    stage_set: str = "entry_ia_sm",
    *,
    threshold: float = BIG_MW,
    weight: str | None = "capacity_mw",
    min_n: int = sv.MIN_N,
    min_events: int = sv.MIN_EVENTS,
) -> tuple[Curves, dict[str, float]]:
    """Stage curves with gas+other split at ``threshold`` MW, and a dict describing the large-gas IA curve.

    ``outcomes`` is a (truncated) outcomes frame with the plain ``stratum`` column. Queues scored with these
    curves need ``stratum`` replaced by :func:`size_stratum`. Rules, per stage:

    - a size stratum that passes the size rule gets its own fit;
    - large gas at the IA stage, when too thin: the small-gas IA curve (own fit, else the combined gas+other
      one) with its COD hazard scaled by ``theta = O / E`` of the large projects (:func:`observed_expected`);
    - any other size stratum that fails falls back to the combined gas+other curve of that stage.

    The dict holds ``n``, ``events``, ``observed``, ``expected`` and ``theta`` (``nan`` when the large IA
    stratum was fit on its own or has no exposure) plus ``own_fit`` (1.0 / 0.0).
    """
    big, small = big_label(threshold), small_label(threshold)
    stages = qa.fit_stages(stage_set)
    combined = qa.fit_stage_curves(outcomes, stages, weight=weight, min_n=min_n, min_events=min_events)
    split = outcomes.with_columns(stratum=size_stratum(threshold))
    # solar / storage / wind and the pooled curves are the same fits as in ``combined``
    curves = qa.fit_stage_curves(split, stages, weight=weight, min_n=min_n, min_events=min_events)
    for stage in stages:
        for label in (big, small):
            if (stage, label) in curves or (stage, GAS) not in combined:
                continue
            if stage == "ia_signed" and label == big:
                continue  # scaled below
            curves[(stage, label)] = combined[(stage, GAS)]

    frame = qa.stage_frame(split, "ia_signed")
    if weight is not None:
        frame = frame.filter(pl.col(weight).is_not_null() & (pl.col(weight) > 0))
    g = frame.filter(pl.col("stratum") == big)
    info = {"n": float(g.height), "events": float((g["event"] == sv.COD).sum()), "observed": np.nan,
            "expected": np.nan, "theta": np.nan, "own_fit": float(("ia_signed", big) in curves)}
    if ("ia_signed", big) in curves:
        return curves, info
    ref = curves.get(("ia_signed", small)) or combined.get(("ia_signed", GAS))
    if ref is None or g.is_empty():
        return curves, info
    o, e = observed_expected(g["time"].to_numpy(), g["event"].to_numpy(), g["entry"].to_numpy(), ref.fit,
                             weights=None if weight is None else g[weight].to_numpy())
    theta = o / e if e > 0 else np.nan
    info |= {"observed": o, "expected": e, "theta": theta}
    if np.isfinite(theta):
        curves[("ia_signed", big)] = qa.StageCurve(
            "ia_signed", big, scale_cause_hazard(ref.fit, theta), ref.end, g.height, int(info["events"])
        )
    return curves, info


# --- COD floor ------------------------------------------------------------------------------------------


def floor_delay(as_of: date, floor: pl.Expr, *, slack_months: float = 0.0) -> pl.Expr:
    """Months from ``as_of`` to ``floor`` minus ``slack_months``, at least 0 (0 when the floor is null)."""
    d = qa.months(floor, pl.lit(as_of)) - slack_months
    return d.clip(lower_bound=0.0).fill_null(0.0)


def rescore_with_floor(
    scored: pl.DataFrame,
    curves: Mapping[tuple[str, str], qa.StageCurve],
    horizons: Mapping[str, float],
    *,
    delay: str = "delay",
    mask: pl.Expr | None = None,
    resolution: float = 0.5,
) -> pl.DataFrame:
    """Re-score the rows of ``scored`` selected by ``mask`` with their clock delayed by ``delay`` months.

    Each selected project gets ``P(COD within (e, e + h - d])`` from the same curves (its window shortened by
    the delay ``d``; 0 when ``d >= h``). ``d`` is rounded to ``resolution`` months so that projects share
    :func:`queue_adjusted.score` calls. ``scored`` is a :func:`queue_adjusted.score` output with a ``delay``
    column; ``p_*`` and ``mw_*`` are replaced for the selected rows, the rest is unchanged.
    """
    sel = pl.col(delay) > 0 if mask is None else (mask & (pl.col(delay) > 0))
    out = scored.with_row_index("_row")
    rows = out.filter(sel)
    if rows.is_empty():
        return scored
    d = (rows[delay].to_numpy() / resolution).round() * resolution
    base = rows.select("_row", "stage", "stratum", "elapsed", "capacity_mw")
    p_new: dict[str, np.ndarray] = {}
    for label, h in horizons.items():
        h_eff = np.clip(h - d, 0.0, h)
        p = np.zeros(rows.height)
        for value in np.unique(h_eff):
            idx = np.flatnonzero(h_eff == value)
            if value <= 0:
                continue
            sub = qa.score(base[idx], curves, {"x": float(value)})
            p[idx] = np.nan_to_num(sub["p_x"].to_numpy())
        p_new[label] = p
    patch = pl.DataFrame({"_row": rows["_row"], **{f"_p_{k}": v for k, v in p_new.items()}})
    out = out.join(patch, on="_row", how="left")
    out = out.with_columns(**{
        f"p_{k}": pl.coalesce(pl.col(f"_p_{k}"), pl.col(f"p_{k}")) for k in horizons
    }).with_columns(**{f"mw_{k}": pl.col("capacity_mw") * pl.col(f"p_{k}") for k in horizons})
    return out.drop("_row", *[f"_p_{k}" for k in horizons])


# --- bootstrap ------------------------------------------------------------------------------------------


def bootstrap(
    train: pl.DataFrame,
    queue: pl.DataFrame,
    fit_curves: Callable[[pl.DataFrame], Mapping[tuple[str, str], qa.StageCurve]],
    horizons: Mapping[str, float],
    *,
    n_boot: int = 500,
    seed: int = 0,
    groups: Sequence[str] = (),
    adjust: Callable[[pl.DataFrame, Mapping[tuple[str, str], qa.StageCurve]], pl.DataFrame] | None = None,
    resample: bool = True,
) -> pl.DataFrame:
    """Replicates of the adjusted queue: ``rep``, ``horizon``, ``level``, ``group``, ``expected_mw``, ``simulated_mw``.

    Each replicate resamples ``train`` by row (project) with replacement (``resample=False`` keeps it: a
    Bernoulli-only band), refits with ``fit_curves``, scores ``queue`` (:func:`queue_adjusted.score`, then
    ``adjust`` if given) and draws one uniform per project: it reaches COD by a horizon when the uniform is below
    its probability. ``level`` is ``"total"`` (group ``"all"``) or a column of ``groups`` (null -> ``"none"``).
    """
    rng = np.random.default_rng(seed)
    mw = queue["capacity_mw"].fill_null(0.0).to_numpy()
    codes: list[tuple[str, list[str], np.ndarray]] = [("total", ["all"], np.zeros(queue.height, dtype=np.int64))]
    for col in groups:
        labels = queue[col].cast(pl.String).fill_null("none")
        uniq = sorted(set(labels.to_list()))
        index = {v: i for i, v in enumerate(uniq)}
        codes.append((col, uniq, np.array([index[v] for v in labels.to_list()], dtype=np.int64)))
    fixed = fit_curves(train)
    out: list[pl.DataFrame] = []
    for b in range(n_boot):
        curves = fit_curves(train[rng.integers(0, train.height, train.height)]) if resample else fixed
        scored = qa.score(queue, curves, horizons)
        if adjust is not None:
            scored = adjust(scored, curves)
        u = rng.random(queue.height)
        for label in horizons:
            p = np.nan_to_num(scored[f"p_{label}"].to_numpy())
            exp_mw, sim_mw = mw * p, mw * (u < p)
            for level, uniq, c in codes:
                out.append(pl.DataFrame({
                    "rep": np.full(len(uniq), b, dtype=np.int32),
                    "horizon": [label] * len(uniq),
                    "level": [level] * len(uniq),
                    "group": uniq,
                    "expected_mw": np.bincount(c, weights=exp_mw, minlength=len(uniq)),
                    "simulated_mw": np.bincount(c, weights=sim_mw, minlength=len(uniq)),
                }))
    return pl.concat(out)


def band(reps: pl.DataFrame, value: str = "simulated_mw", qs: Sequence[float] = QUANTILES) -> pl.DataFrame:
    """P10/P50/P90 (``p10``, ``p50``, ``p90`` for the default ``qs``) and ``mean`` of ``value`` per horizon × level × group."""
    return reps.group_by("horizon", "level", "group").agg(
        mean=pl.col(value).mean(),
        **{f"p{round(q * 100)}": pl.col(value).quantile(q, interpolation="linear") for q in qs},
    ).sort("horizon", "level", "group")


def percentile_of(draws: Sequence[float] | np.ndarray, actual: float) -> float:
    """Mid-rank percentile of ``actual`` among ``draws`` (the PIT value; 0.5 = the median)."""
    d = np.asarray(draws, dtype=float)
    return float((d < actual).mean() + 0.5 * (d == actual).mean())


# --- backtest-error band --------------------------------------------------------------------------------


def log_errors(pred: Sequence[float], actual: Sequence[float]) -> np.ndarray:
    """``ln(actual / predicted)`` per backtest (positive = the model was low)."""
    p, a = np.asarray(pred, dtype=float), np.asarray(actual, dtype=float)
    if (p <= 0).any() or (a <= 0).any():
        raise ValueError("predicted and actual must be > 0")
    return np.log(a / p)


def error_band(point: float | np.ndarray, errors: Sequence[float], *, z: float = Z80) -> tuple[np.ndarray, np.ndarray]:
    """``point * exp(-+ z * RMSE(errors))``: a band centred on the model (no bias correction)."""
    rmse = float(np.sqrt(np.mean(np.square(np.asarray(errors, dtype=float)))))
    pt = np.asarray(point, dtype=float)
    return pt * np.exp(-z * rmse), pt * np.exp(z * rmse)
