"""X21: a landmark-date cohort for the generation-queue survival curves (FIS approved and IA signed).

Q6 fits every landmark on the projects first listed on or after 2018-08. Here the FIS approved and IA signed
curves are refit on cohorts defined by the milestone date (``landmark``: milestone >= 2018-08 and first listed
>= 2017-04; plus the ``union`` and ``period`` sensitivities), compared with Q6 at 12/24/36/48 months, and fed
to X2's backtest (semi-Markov entry step, MW-weighted). Model code: ``basecast_pipelines/models/
survival_landmark.py`` on top of ``survival.py`` and ``queue_adjusted.py`` (imported, unchanged).

Run from the repo root: ``uv run --group analysis python analysis/x21_landmark_cohort.py``. Numbers go to
``docs/analysis/x21_landmark_cohort.md``; CSVs to ``analysis/out/x21_*``.
"""

# %% setup
from __future__ import annotations

import dataclasses
from datetime import date

import polars as pl
from _common import out_path, read_sql

from basecast_pipelines.models import queue_adjusted as qa
from basecast_pipelines.models import survival as sv
from basecast_pipelines.models import survival_landmark as slm

events_all = slm.load_events_all()
outcomes_all = sv.outcomes(events_all)
LATEST = events_all["latest_report_month"].max()
STAGES = ("fis_approved", "ia_signed")
H = sv.HORIZONS
print(f"all INRs {events_all.height}; latest report {LATEST}; landmark start {slm.LANDMARK_START}, "
      f"listed start {slm.LISTED_START}")

# %% 1. cohorts: who is in which, and n / events / censoring by stratum
for stage in STAGES:
    m = slm.membership(outcomes_all, stage)
    print(f"\n{stage}: reached by {m.height}")
    print(" ", {c: int(m[f"in_{c}"].sum()) for c in slm.COHORTS})
    print("  landmark vs q6:", m.group_by("why").len().sort("why").rows())
    col = qa.STAGE_DATES[stage]
    print("  landmark only, by first-listed year and outcome:",
          m.filter(pl.col("why") == "landmark only").group_by(pl.col("first_seen_month").dt.year().alias("y"), "exit_status")
          .len().sort("y", "exit_status").rows())
    print("  q6 only (listed >= 2018-08 after a pre-2018-08 milestone), by stratum:",
          m.filter(pl.col("why") == "q6 only").group_by("stratum").len().sort("stratum").rows())
    print("  neither, milestone >= 2018-08 but first listed before 2017-04:",
          m.filter((pl.col("why") == "neither") & (pl.col(col) >= pl.lit(slm.LANDMARK_START))).height,
          "| neither, milestone before 2018-08 and listed before 2018-08:",
          m.filter((pl.col("why") == "neither") & (pl.col(col) < pl.lit(slm.LANDMARK_START))).height)

frames = {(c, s): slm.cohort_frame(outcomes_all, s, c) for c in slm.COHORTS for s in STAGES}
# the entry landmark is unchanged (Q6's cohort); it only enters the ordering check
q6_events = outcomes_all.filter(pl.col("first_seen_month") >= pl.lit(sv.COHORT_START))
frames[("q6", "entry")] = qa.stage_frame(q6_events, "entry")

rows = []
for (c, s), f in frames.items():
    d = f.filter(pl.col("entry") > 0)
    print(f"{c:>8} {s:>12}: n={f.height}, COD={int((f['event'] == sv.COD).sum())}, "
          f"withdrawn={int((f['event'] == sv.WITHDRAWN).sum())}, censored={int((f['event'] == sv.CENSORED).sum())}, "
          f"delayed entry={d.height} ({d.height / f.height:.0%}, median {d['entry'].median() or 0:.1f} mo), "
          f"clipped={int(f['clipped'].sum())}, median origin {f['origin'].median()}")
    for key, g in [("all", f), *[(k[0], g) for k, g in f.group_by("stratum")]]:
        rows.append({"cohort": c, "landmark": s, "stratum": key, "n": g.height,
                     "cod": int((g["event"] == sv.COD).sum()), "withdrawn": int((g["event"] == sv.WITHDRAWN).sum()),
                     "censored": int((g["event"] == sv.CENSORED).sum()), "delayed": int((g["entry"] > 0).sum())})
counts = pl.DataFrame(rows).sort("landmark", "stratum", "cohort")
counts.write_csv(out_path("x21_cohort_counts.csv"))
print(counts.filter(pl.col("landmark") != "entry"))

# fuel groups before the gas+other merge: does "other" (or gas) pass n >= 30 / >= 10 COD on its own now?
fuel_rows = []
for (c, s), f in frames.items():
    for k, g in f.group_by("fuel_group"):
        fuel_rows.append({"cohort": c, "landmark": s, "fuel_group": k[0], "n": g.height,
                          "cod": int((g["event"] == sv.COD).sum())})
fuel_counts = pl.DataFrame(fuel_rows).filter(pl.col("fuel_group").is_in(["gas", "other"])).sort("fuel_group", "landmark", "cohort")
print(fuel_counts)
for c in slm.COHORTS:
    merged = {}
    for s in STAGES:
        f = frames[(c, s)]
        cnt = {k[0]: (g.height, int((g["event"] == sv.COD).sum())) for k, g in f.group_by("fuel_group")}
        merged[s] = sv.merge_small_strata(cnt)
    print(f"size rule on fuel groups, {c}:", {s: {k: v for k, v in m.items() if k != v} for s, m in merged.items()})

# %% 2. CIF of COD from FIS approved and from IA signed, per cohort x stratum (count and MW-weighted)
def table(weight: str | None = None) -> pl.DataFrame:
    parts = []
    for (c, s), f in frames.items():
        for by in ("stratum", None):
            parts.append(sv.cif_table(f, by, weight=weight).with_columns(cohort=pl.lit(c), landmark=pl.lit(s)))
    return pl.concat(parts, how="diagonal").select("cohort", "landmark", pl.exclude("cohort", "landmark"))


cif = table()
cif_mw = table("capacity_mw")
cif.write_csv(out_path("x21_cif_table.csv"))
cif_mw.write_csv(out_path("x21_cif_table_mw.csv"))
show = ["cohort", "landmark", "stratum", "n", "cod", "withdrawn", *[f"cif_{h}" for h in H], "wd_36", "at_risk_36", "at_risk_48"]
pct = [pl.col("^cif_.*$", "wd_36").mul(100).round(1)]
print(cif.filter(pl.col("landmark") != "entry").select(show).with_columns(pct).sort("landmark", "stratum", "cohort"))
print("MW-weighted:\n", cif_mw.filter(pl.col("landmark") != "entry").select("cohort", "landmark", "stratum", "n", "cif_36", "cif_48")
      .with_columns(pl.col("cif_36", "cif_48").mul(100).round(1)).sort("landmark", "stratum", "cohort"))

# ordering at 36 months: entry (q6) < FIS approved < IA signed, per cohort
entry36 = {r["stratum"]: r["cif_36"] for r in cif.filter(pl.col("landmark") == "entry").iter_rows(named=True)}
for c in slm.COHORTS:
    sub = cif.filter((pl.col("cohort") == c) & (pl.col("landmark") != "entry"))
    by = {st: {"entry": entry36.get(st)} for st in sub["stratum"].unique().to_list()}
    for r in sub.iter_rows(named=True):
        by[r["stratum"]][r["landmark"]] = r["cif_36"]
    print(f"ordering {c}:", slm.ordering(by), {st: {k: round(v * 100, 1) for k, v in d.items() if v is not None}
                                               for st, d in sorted(by.items())})

# wind: is FIS approved > IA signed at 36 months beyond sampling noise? (joint bootstrap over projects)
for c in slm.COHORTS:
    for st in ("wind", "all"):
        a, b = (frames[(c, s)] for s in STAGES)
        if st != "all":
            a, b = a.filter(pl.col("stratum") == st), b.filter(pl.col("stratum") == st)
        point, lo, hi = slm.bootstrap_difference(a, b, 36, n_boot=400, seed=0)
        print(f"CIF36 FIS approved - IA signed, {c} {st}: {point * 100:+.1f} pp [{lo * 100:+.1f}, {hi * 100:+.1f}]")

# nesting: share of projects with both milestones that signed the IA first (landmark cohort at IA)
both = frames[("landmark", "ia_signed")].filter(pl.col("fis_approved").is_not_null())
print("IA before FIS approval (landmark IA cohort, projects with both):", both.group_by("stratum").agg(
    n=pl.len(), ia_first=(pl.col("ia_signed") < pl.col("fis_approved")).mean().round(3)).sort("stratum").rows())

# step curves for review
curves = []
for (c, s), f in frames.items():
    for key, g in [("all", f), *[(k[0], g) for k, g in f.group_by("stratum")]]:
        fit = sv.fit_cif(g["time"].to_numpy(), g["event"].to_numpy(), g["entry"].to_numpy())
        curves.append(pl.DataFrame({"cohort": c, "landmark": s, "stratum": key, "month": fit.times,
                                    "at_risk": fit.at_risk, "cif_cod": fit.cif[:, fit.causes.index(sv.COD)]}))
pl.concat(curves).write_csv(out_path("x21_cif_curves.csv"))

# %% 3. X2's backtest with the IA-stage curves refit per cohort (pre-IA sojourn as in X2: Q6's cohort)
BACKTEST_MONTHS = [date(2022, 6, 1), date(2023, 1, 1), date(2024, 8, 1)]
WINDOW = 24
X2_ERR = {date(2022, 6, 1): -0.125, date(2023, 1, 1): 0.094, date(2024, 8, 1): 0.018}
stages_sm = qa.STAGE_SETS["entry_ia_sm"]
q6_cohort = events_all.filter(pl.col("first_seen_month") >= pl.lit(sv.COHORT_START))


def add_month(d: date, n: int) -> date:
    m = d.month - 1 + n
    return date(d.year + m // 12, m % 12 + 1, 1)


bt_parts, curve_sizes = [], []
for rm in BACKTEST_MONTHS:
    as_of = add_month(rm, 1)
    until = add_month(as_of, WINDOW)
    train_q6 = qa.truncate_at(q6_cohort, as_of)
    train_all = qa.truncate_at(events_all, as_of)
    q = qa.build_queue(qa.load_snapshot(rm), events_all, as_of, stages_sm).filter(pl.col("capacity_mw") > 0)
    # X2 exactly (reference): both stages fit on Q6's cohort
    variants = {"x2": qa.fit_stage_curves(train_q6, qa.fit_stages("entry_ia_sm"), weight="capacity_mw")}
    for c in slm.COHORTS:
        variants[c] = slm.semi_markov_curves(train_q6, train_all, c)
    for name, curves_v in variants.items():
        ia = {k[1]: v for k, v in curves_v.items() if k[0] == "ia_signed"}
        curve_sizes.append({"report_month": rm, "variant": name,
                            **{f"ia_n_{k}": v.n for k, v in ia.items()}, **{f"ia_cod_{k}": v.cod for k, v in ia.items()},
                            **{f"ia_end_{k}": round(v.end, 1) for k, v in ia.items()}})
        scored = qa.score(q, curves_v, {"w": qa.months_between(until, as_of)}).with_columns(actual=qa.actual_cod_mw(until))
        bt_parts.append(scored.select("inr", "stratum", "stage", "curve", "capacity_mw", "p_w", "mw_w", "clamped_w", "actual")
                        .with_columns(report_month=pl.lit(rm), variant=pl.lit(name)))
bt = pl.concat(bt_parts)
bt.write_csv(out_path("x21_backtest_projects.csv"))
print(pl.DataFrame(curve_sizes).sort("report_month", "variant"))

state = (
    bt.group_by("report_month", "variant")
    .agg(n=pl.len(), raw=pl.col("capacity_mw").sum(), pred=pl.col("mw_w").sum(), actual=pl.col("actual").sum())
    .with_columns(err=pl.col("pred") / pl.col("actual") - 1)
    .sort("report_month", "variant")
)
state.write_csv(out_path("x21_backtest_statewide.csv"))
print(state.with_columns(pl.col("raw", "pred", "actual").round(0), pl.col("err").mul(100).round(1)))
for r in state.filter(pl.col("variant") == "x2").iter_rows(named=True):
    print(f"X2 replication {r['report_month']}: {r['err']:+.3f} (doc {X2_ERR[r['report_month']]:+.3f})")
print(state.group_by("variant").agg(mean_abs_err=pl.col("err").abs().mean().mul(100).round(1),
                                    mean_err=pl.col("err").mean().mul(100).round(1),
                                    max_abs_err=pl.col("err").abs().max().mul(100).round(1)).sort("variant"))

by_stage = (
    bt.group_by("report_month", "variant", "stage")
    .agg(pred=pl.col("mw_w").sum(), actual=pl.col("actual").sum())
    .with_columns(err=pl.col("pred") / pl.col("actual") - 1)
    .sort("stage", "report_month", "variant")
)
print(by_stage.with_columns(pl.col("pred", "actual").round(0), pl.col("err").mul(100).round(1)))
by_fuel = (
    bt.group_by("report_month", "variant", "stratum")
    .agg(pred=pl.col("mw_w").sum(), actual=pl.col("actual").sum())
    .with_columns(err=pl.col("pred") / pl.col("actual") - 1)
    .sort("stratum", "report_month", "variant")
)
by_fuel.write_csv(out_path("x21_backtest_by_fuel.csv"))
print(by_fuel.with_columns(pl.col("pred", "actual").round(0), pl.col("err").mul(100).round(1)))
print(by_fuel.group_by("variant", "stratum").agg(mean_abs_err=pl.col("err").abs().mean().mul(100).round(1))
      .pivot(on="stratum", index="variant", values="mean_abs_err").sort("variant"))

# %% 4. today's adjusted queue with each cohort's IA curves (statewide, Dec 2027 / Dec 2028), for context
AS_OF = add_month(LATEST, 1)
HZ = {"2027": qa.months_between(date(2028, 1, 1), AS_OF), "2028": qa.months_between(date(2029, 1, 1), AS_OF)}
queue = qa.build_queue(qa.load_snapshot(LATEST), events_all, AS_OF, stages_sm)
tr_q6, tr_all = qa.truncate_at(q6_cohort, AS_OF), qa.truncate_at(events_all, AS_OF)
today = {"x2": qa.fit_stage_curves(tr_q6, qa.fit_stages("entry_ia_sm"), weight="capacity_mw")}
today |= {c: slm.semi_markov_curves(tr_q6, tr_all, c) for c in slm.COHORTS}
for name, cv in today.items():
    s = qa.score(queue, cv, HZ)
    tot = {h: s[f"mw_{h}"].sum() for h in HZ}
    ia_mw = {h: s.filter(pl.col("stage") == "ia_signed")[f"mw_{h}"].sum() for h in HZ}
    print(f"today {name:>8}: adjusted MW Dec 2027 {tot['2027']:,.0f}, Dec 2028 {tot['2028']:,.0f} "
          f"(IA stage {ia_mw['2027']:,.0f} / {ia_mw['2028']:,.0f}) of raw {s['capacity_mw'].sum():,.0f}")

# %% 5. why the landmark cohort overpredicts: IA vintage (calendar drift) and the support clamp
# CIF of COD from the IA by the year the IA was signed (landmark cohort, today's data; later vintages censored)
ia_lm = frames[("landmark", "ia_signed")].with_columns(
    vintage=pl.when(pl.col("origin") < pl.lit(date(2020, 1, 1))).then(pl.lit("2018-08..2019"))
    .when(pl.col("origin") < pl.lit(date(2021, 1, 1))).then(pl.lit("2020"))
    .when(pl.col("origin") < pl.lit(date(2022, 1, 1))).then(pl.lit("2021"))
    .when(pl.col("origin") < pl.lit(date(2023, 1, 1))).then(pl.lit("2022"))
    .when(pl.col("origin") < pl.lit(date(2024, 1, 1))).then(pl.lit("2023"))
    .otherwise(pl.lit("2024+")),
    added=pl.col("first_seen_month") < pl.lit(sv.COHORT_START),
)
vint = []
for strat in ("all", "solar"):
    sub = ia_lm if strat == "all" else ia_lm.filter(pl.col("stratum") == strat)
    for w in (None, "capacity_mw"):
        t = sv.cif_table(sub, "vintage", (12, 24, 36), weight=w).rename({"stratum": "vintage"}).with_columns(
            stratum=pl.lit(strat), weight=pl.lit(w or "count"))
        vint.append(t)
vint = pl.concat(vint, how="diagonal").select("stratum", "weight", "vintage", "n", "cod", "cif_12", "cif_24", "cif_36", "at_risk_24", "at_risk_36")
vint.write_csv(out_path("x21_ia_vintage.csv"))
print(vint.with_columns(pl.col("^cif_.*$").mul(100).round(1)).sort("stratum", "weight", "vintage"))
print("IA-landmark projects added by the landmark cohort, by IA vintage:",
      ia_lm.filter(pl.col("added")).group_by("vintage").agg(n=pl.len(), cod=(pl.col("event") == sv.COD).sum()).sort("vintage").rows())

# decomposition on the backtest: landmark curves with Q6's supported end (clamp) vs Q6 curves with the landmark end
dec_rows = []
for rm in BACKTEST_MONTHS:
    as_of = add_month(rm, 1)
    until = add_month(as_of, WINDOW)
    train_q6 = qa.truncate_at(q6_cohort, as_of)
    train_all = qa.truncate_at(events_all, as_of)
    q = qa.build_queue(qa.load_snapshot(rm), events_all, as_of, stages_sm).filter(pl.col("capacity_mw") > 0)
    c_q6 = slm.semi_markov_curves(train_q6, train_all, "q6")
    c_lm = slm.semi_markov_curves(train_q6, train_all, "landmark")
    swap_end = {k: dataclasses.replace(v, end=c_q6[k].end) if k in c_q6 and k[0] == "ia_signed" else v for k, v in c_lm.items()}
    swap_lvl = {k: dataclasses.replace(v, end=c_lm[k].end) if k in c_lm and k[0] == "ia_signed" else v for k, v in c_q6.items()}
    for name, cv in {"q6": c_q6, "landmark": c_lm, "landmark curves, q6 end": swap_end, "q6 curves, landmark end": swap_lvl}.items():
        s = qa.score(q, cv, {"w": qa.months_between(until, as_of)}).with_columns(actual=qa.actual_cod_mw(until))
        ia_s = s.filter(pl.col("stage") == "ia_signed")
        dec_rows.append({"report_month": rm, "variant": name, "pred": s["mw_w"].sum(), "actual": s["actual"].sum(),
                         "ia_clamped_mw_share": (ia_s["capacity_mw"] * ia_s["clamped_w"]).sum() / ia_s["capacity_mw"].sum()})
dec = pl.DataFrame(dec_rows).with_columns(err=pl.col("pred") / pl.col("actual") - 1)
print(dec.with_columns(pl.col("pred", "actual").round(0), pl.col("err", "ia_clamped_mw_share").mul(100).round(1)))

# %% 6. the clamp today vs in the backtest: IA-stage MW share whose window runs past the supported end
for name in ("q6", "landmark"):
    cv = today[name]
    ends = {k[1]: round(v.end, 1) for k, v in cv.items() if k[0] == "ia_signed"}
    s = qa.score(queue, cv, {"h24": 24.0, **HZ})
    ia_s = s.filter(pl.col("stage") == "ia_signed")
    share = {h: round(float((ia_s["capacity_mw"] * ia_s[f"clamped_{h}"]).sum() / ia_s["capacity_mw"].sum()) * 100, 1)
             for h in ("h24", *HZ)}
    print(f"today {name}: IA supported end by stratum {ends}; IA-stage clamped MW share (%) {share}")

# %% 7. side check on X2's truncation: IA dates before the as-of date that were first reported after it
# (sv.load_events does not load ia_first_month, so qa.truncate_at cannot apply its "first reported" rule)
first_ia = read_sql("SELECT inr, ia_first_month FROM gis_project_events")
for rm in BACKTEST_MONTHS:
    as_of = add_month(rm, 1)
    known = q6_cohort.join(first_ia, on="inr", how="left").filter(pl.col("ia_signed") < pl.lit(as_of))
    late = known.filter(pl.col("ia_first_month").is_null() | (pl.col("ia_first_month") > pl.lit(rm)))
    print(f"{rm}: Q6-cohort projects with an IA dated before the as-of date {known.height}, "
          f"first reported after the as-of report {late.height} ({late['capacity_mw'].sum():,.0f} MW)")

# X2 and the landmark variant again with ia_first_month joined, so truncate_at hides IAs not yet reported
ev_f = events_all.join(first_ia, on="inr", how="left")
q6_f = ev_f.filter(pl.col("first_seen_month") >= pl.lit(sv.COHORT_START))
for rm in BACKTEST_MONTHS:
    as_of = add_month(rm, 1)
    until = add_month(as_of, WINDOW)
    tr_q6_f, tr_all_f = qa.truncate_at(q6_f, as_of), qa.truncate_at(ev_f, as_of)
    q = qa.build_queue(qa.load_snapshot(rm), events_all, as_of, stages_sm).filter(pl.col("capacity_mw") > 0)
    out_row = []
    for name in ("q6", "landmark"):
        cv = slm.semi_markov_curves(tr_q6_f, tr_all_f, name)
        s = qa.score(q, cv, {"w": qa.months_between(until, as_of)}).with_columns(actual=qa.actual_cod_mw(until))
        out_row.append(f"{name} {s['mw_w'].sum() / s['actual'].sum() - 1:+.1%}")
    print(f"{rm} with the first-reported rule: {', '.join(out_row)}")
