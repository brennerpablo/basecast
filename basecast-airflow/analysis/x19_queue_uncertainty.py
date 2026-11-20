"""X19: an uncertainty band for the adjusted generation queue, and a size split for large new gas.

Task 1: a P10/P50/P90 band for X2's adjusted queue from a bootstrap over projects (refit the curves) plus the
Bernoulli draw of which projects make it, vs a band from X2's three backtest errors; coverage of the bootstrap
band on the three backtests (as-of fits only). Task 2: gas+other >= 500 MW vs smaller (strata sizes, a hazard
ratio when the stratum is too thin, the developer's projected COD as a floor, a lead-time cap as an assumption),
with the totals, the county ranking and the backtests re-run.

Model code: ``basecast_pipelines/models/queue_uncertainty.py`` on top of ``queue_adjusted.py`` and
``survival.py`` (imported, unchanged). Run from the repo root:
``uv run --group analysis python analysis/x19_queue_uncertainty.py``. Numbers go to
``docs/analysis/x19_queue_uncertainty.md``; CSVs to ``analysis/out/x19_*``.
"""

# %% setup
from __future__ import annotations

import os
import time
from datetime import date

import numpy as np
import polars as pl
from _common import out_path, read_sql
from scipy.stats import chi2, spearmanr

from basecast_pipelines.models import queue_adjusted as qa
from basecast_pipelines.models import queue_uncertainty as qu
from basecast_pipelines.models import survival as sv

N_BOOT_TODAY = int(os.environ.get("X19_N_BOOT", 1000))  # X19_N_BOOT=20 for a smoke run
N_BOOT_BT, SEED = max(N_BOOT_TODAY // 2, 10), 19
STAGE_SET, WEIGHT = "entry_ia_sm", "capacity_mw"  # X2's primary variant
STAGES = qa.STAGE_SETS[STAGE_SET]
BIG, SMALL = qu.big_label(), qu.small_label()

events_all = sv.load_events(cohort_start=None).with_columns(stratum=qa.stratum())
cohort = events_all.filter(pl.col("first_seen_month") >= pl.lit(sv.COHORT_START))
geo = qa.load_geography()
LATEST = events_all["latest_report_month"].max()
AS_OF = date(LATEST.year + (LATEST.month == 12), LATEST.month % 12 + 1, 1)
HORIZONS = {"2027": date(2028, 1, 1), "2028": date(2029, 1, 1)}
H_MONTHS = {k: qa.months_between(v, AS_OF) for k, v in HORIZONS.items()}
BACKTEST_MONTHS = [date(2022, 6, 1), date(2023, 1, 1), date(2024, 8, 1)]
WINDOW = 24
print(f"latest report {LATEST}, as of {AS_OF}, horizons {H_MONTHS}")


def add_month(d: date, n: int) -> date:
    m = d.month - 1 + n
    return date(d.year + m // 12, m % 12 + 1, 1)


def with_geo(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(county_key=qa.norm_county("county")).join(
        geo.select("county_key", "county_fips", "county_name"), on="county_key", how="left"
    )


def make_queue(report_month: date, as_of: date) -> pl.DataFrame:
    """Active queue of a report month with geography, the plain fuel stratum (``fuel``) and the size stratum."""
    q = qa.build_queue(qa.load_snapshot(report_month), events_all, as_of, STAGES).filter(pl.col("capacity_mw") > 0)
    return with_geo(q).with_columns(fuel=pl.col("stratum"), size_stratum=qu.size_stratum())


BIG_GAS = (pl.col("fuel") == qu.GAS) & (pl.col("capacity_mw") >= qu.BIG_MW)


# model variants: (name, split curves?, floor rule or None); the floor rule returns the floor date expression
def dev_floor(q: pl.DataFrame, as_of: date, slack: float = 1.0) -> pl.Expr:
    return qu.floor_delay(as_of, pl.col("projected_cod"), slack_months=slack)


def lead_cap(months_after_ia: int):
    def rule(q: pl.DataFrame, as_of: date, slack: float = 0.0) -> pl.Expr:
        ia_plus = pl.col("ia_signed").dt.offset_by(f"{months_after_ia}mo")
        floor = pl.when(pl.col("stage") == "ia_signed").then(ia_plus).otherwise(pl.lit(add_month(as_of, months_after_ia)))
        return qu.floor_delay(as_of, floor)
    return rule


VARIANTS = {
    "baseline (X2)": (False, None, None),
    "split: hazard ratio": (True, None, None),
    "floor: dev. COD, big gas": (False, dev_floor, BIG_GAS),
    "split + floor (big gas)": (True, dev_floor, BIG_GAS),
    "sens: floor, all gas+other": (False, dev_floor, pl.col("fuel") == qu.GAS),
    "sens: floor, all fuels": (False, dev_floor, pl.lit(True)),
    "sens: cap IA+24 mo, big gas": (False, lead_cap(24), BIG_GAS),
    "sens: cap IA+36 mo, big gas": (False, lead_cap(36), BIG_GAS),
}


def fitter(split: bool):
    if split:
        return lambda train: qu.split_gas_curves(train, STAGE_SET, weight=WEIGHT)[0]
    return lambda train: qa.fit_stage_curves(train, qa.fit_stages(STAGE_SET), weight=WEIGHT)


def prepare(q: pl.DataFrame, as_of: date, variant: str) -> tuple[pl.DataFrame, callable, callable | None]:
    """Queue (stratum column set for the variant, ``delay`` added), curve fitter and the post-score adjustment."""
    split, rule, mask = VARIANTS[variant]
    q = q.with_columns(stratum=pl.col("size_stratum") if split else pl.col("fuel"))
    if rule is None:
        return q, fitter(split), None
    q = q.with_columns(delay=pl.when(mask).then(rule(q, as_of)).otherwise(0.0))
    return q, fitter(split), lambda s, c, h: qu.rescore_with_floor(s, c, h, mask=mask)


def point(q: pl.DataFrame, train: pl.DataFrame, as_of: date, variant: str, horizons: dict[str, float]) -> pl.DataFrame:
    qv, fit, adj = prepare(q, as_of, variant)
    curves = fit(train)
    s = qa.score(qv, curves, horizons)
    return adj(s, curves, horizons) if adj else s


# %% 0. reproduce X2 today, and the queue with both strata
train_today = qa.truncate_at(cohort, AS_OF)
queue = make_queue(LATEST, AS_OF)
base = point(queue, train_today, AS_OF, "baseline (X2)", H_MONTHS)
print("X2 reproduced:", {h: round(base[f"mw_{h}"].sum()) for h in H_MONTHS}, "projects", base.height,
      "raw MW", round(base["capacity_mw"].sum()))

# %% 1a. today's band: Bernoulli only, curves only (bootstrap, expected MW), both (bootstrap + Bernoulli)
t0 = time.time()
_, fit_base, _ = prepare(queue, AS_OF, "baseline (X2)")
qb = queue.with_columns(stratum=pl.col("fuel"))
reps = qu.bootstrap(train_today, qb, fit_base, H_MONTHS, n_boot=N_BOOT_TODAY, seed=SEED, groups=("county_fips", "fuel"))
bern = qu.bootstrap(train_today, qb, fit_base, H_MONTHS, n_boot=N_BOOT_TODAY, seed=SEED, groups=(), resample=False)
print(f"bootstrap today: {time.time() - t0:.0f} s")
reps.write_parquet(out_path("x19_reps_today.parquet"))

band_both = qu.band(reps, "simulated_mw")
band_curve = qu.band(reps, "expected_mw")
band_bern = qu.band(bern, "simulated_mw")

# %% 1b. backtests: as-of fits, bootstrap band, coverage (statewide, fuel, county) and the backtest errors
bt_rows, cov_rows, bt_scored = [], [], {}
for rm in BACKTEST_MONTHS:
    as_of = add_month(rm, 1)
    until = add_month(as_of, WINDOW)
    h = {"w": qa.months_between(until, as_of)}
    train = qa.truncate_at(cohort, as_of)
    q = make_queue(rm, as_of).with_columns(actual=qa.actual_cod_mw(until))
    bt_scored[rm] = (q, train, as_of, h)
    qbase = q.with_columns(stratum=pl.col("fuel"))
    t0 = time.time()
    r = qu.bootstrap(train, qbase, fit_base, h, n_boot=N_BOOT_BT, seed=SEED, groups=("county_fips", "fuel"))
    rb = qu.bootstrap(train, qbase, fit_base, h, n_boot=N_BOOT_BT, seed=SEED, groups=("county_fips", "fuel"), resample=False)
    print(f"{rm}: bootstrap {time.time() - t0:.0f} s")
    s = qa.score(qbase, fit_base(train), h)
    actual = {"total": {"all": float(q["actual"].sum())}}
    for col in ("county_fips", "fuel"):
        a = q.group_by(pl.col(col).cast(pl.String).fill_null("none").alias("group")).agg(a=pl.col("actual").sum())
        actual[col] = dict(a.iter_rows())
    pred = {"total": {"all": float(s["mw_w"].sum())}}
    for col in ("county_fips", "fuel"):
        p = s.group_by(pl.col(col).cast(pl.String).fill_null("none").alias("group")).agg(p=pl.col("mw_w").sum())
        pred[col] = dict(p.iter_rows())
    for kind, rr in (("bootstrap+Bernoulli", r), ("Bernoulli only", rb)):
        bd = qu.band(rr, "simulated_mw")
        for row in bd.iter_rows(named=True):
            a = actual[row["level"]].get(row["group"], 0.0)
            pr = pred[row["level"]].get(row["group"], 0.0)
            draws = rr.filter((pl.col("level") == row["level"]) & (pl.col("group") == row["group"]))["simulated_mw"]
            cov_rows.append({"report_month": rm, "band": kind, "level": row["level"], "group": row["group"],
                             "pred": pr, "actual": a, "p10": row["p10"], "p50": row["p50"], "p90": row["p90"],
                             "covered": row["p10"] <= a <= row["p90"], "pit": qu.percentile_of(draws.to_numpy(), a)})
    bt_rows.append({"report_month": rm, "pred": pred["total"]["all"], "actual": actual["total"]["all"]})

bt = pl.DataFrame(bt_rows).with_columns(err=pl.col("pred") / pl.col("actual") - 1)
cov = pl.DataFrame(cov_rows)
cov.write_csv(out_path("x19_backtest_coverage.csv"))
print(bt.with_columns(pl.col("pred", "actual").round(0), pl.col("err").round(3)))
errs = qu.log_errors(bt["pred"].to_numpy(), bt["actual"].to_numpy())
rmse = float(np.sqrt(np.mean(errs**2)))
print("log errors ln(actual/pred):", errs.round(4), "RMSE", round(rmse, 4),
      f"-> P10/P90 multipliers {np.exp(-qu.Z80 * rmse):.3f} / {np.exp(qu.Z80 * rmse):.3f}")
# leave-one-out: each date's band from the other two errors
for i, rm in enumerate(BACKTEST_MONTHS):
    others = np.delete(errs, i)
    lo, hi = qu.error_band(bt["pred"][i], others)
    print(f"LOO {rm}: band {float(lo):,.0f}-{float(hi):,.0f}, actual {bt['actual'][i]:,.0f}, "
          f"covered {float(lo) <= bt['actual'][i] <= float(hi)}")

# coverage summary: statewide / fuel / counties (counties with a prediction >= 50 MW or any actual)
cov_c = cov.with_columns(width=(pl.col("p90") - pl.col("p10")) / pl.col("pred").clip(lower_bound=1.0))
counties = cov_c.filter((pl.col("level") == "county_fips") & ((pl.col("pred") >= 50) | (pl.col("actual") > 0)))
print("statewide:\n", cov_c.filter(pl.col("level") == "total").select(
    "report_month", "band", pl.col("pred", "actual", "p10", "p50", "p90").round(0), "covered", pl.col("pit").round(3)))
print("by fuel:\n", cov_c.filter(pl.col("level") == "fuel").select(
    "report_month", "band", "group", pl.col("pred", "actual", "p10", "p90").round(0), "covered", pl.col("pit").round(3))
    .sort("band", "group", "report_month"))
fuel_cov = cov_c.filter(pl.col("level") == "fuel").group_by("band").agg(cells=pl.len(), covered=pl.col("covered").sum())
print("fuel coverage:", fuel_cov.rows())
cty_cov = counties.group_by("band", "report_month").agg(
    counties=pl.len(), coverage=pl.col("covered").mean().round(3),
    below=(pl.col("actual") < pl.col("p10")).mean().round(3), above=(pl.col("actual") > pl.col("p90")).mean().round(3),
).sort("band", "report_month")
print("county coverage (nominal 80%):\n", cty_cov)
print("county coverage pooled:", counties.group_by("band").agg(n=pl.len(), coverage=pl.col("covered").mean().round(3),
      below=(pl.col("actual") < pl.col("p10")).mean().round(3), above=(pl.col("actual") > pl.col("p90")).mean().round(3)).rows())
# bootstrap band width relative to the point at the backtests
print("statewide band width at the backtests (P90-P10)/pred:",
      cov_c.filter(pl.col("level") == "total").select("report_month", "band", pl.col("width").round(3)).rows())

# %% 1c. today's statewide band: the four versions side by side, and the top-10 counties
state = []
for hz in H_MONTHS:
    pt = float(base[f"mw_{hz}"].sum())
    lo, hi = qu.error_band(pt, errs)
    row = {"horizon": hz, "point": pt, "backtest_p10": float(lo), "backtest_p90": float(hi)}
    for name, b in (("bernoulli", band_bern), ("curves", band_curve), ("both", band_both)):
        r = b.filter((pl.col("horizon") == hz) & (pl.col("level") == "total")).row(0, named=True)
        row |= {f"{name}_p10": r["p10"], f"{name}_p50": r["p50"], f"{name}_p90": r["p90"]}
    state.append(row)
state = pl.DataFrame(state)
state.write_csv(out_path("x19_statewide_band.csv"))
print(state.with_columns(pl.exclude("horizon").round(0)))
for row in state.iter_rows(named=True):
    print(f"{row['horizon']}: width (P90-P10)/point  Bernoulli {(row['bernoulli_p90'] - row['bernoulli_p10']) / row['point']:.1%}, "
          f"curves {(row['curves_p90'] - row['curves_p10']) / row['point']:.1%}, both {(row['both_p90'] - row['both_p10']) / row['point']:.1%}, "
          f"backtest {(row['backtest_p90'] - row['backtest_p10']) / row['point']:.1%}")

county_pt = base.group_by("county_fips", "county_name").agg(
    raw_mw=pl.col("capacity_mw").sum(), **{f"point_{h}": pl.col(f"mw_{h}").sum() for h in H_MONTHS})
cb = band_both.filter(pl.col("level") == "county_fips").rename({"group": "county_fips"})
county_band = county_pt.with_columns(pl.col("county_fips").cast(pl.String).fill_null("none")).join(
    cb.pivot(on="horizon", index="county_fips", values=["p10", "p50", "p90"]), on="county_fips", how="left"
).sort("point_2028", descending=True)
county_band.write_csv(out_path("x19_county_band.csv"))
print("top-10 counties (point = X2 expected MW):\n", county_band.head(10).select(
    "county_name", pl.col("raw_mw").round(0), pl.col("point_2027").round(0), pl.col("p10_2027").round(0),
    pl.col("p50_2027").round(0), pl.col("p90_2027").round(0), pl.col("point_2028").round(0),
    pl.col("p10_2028").round(0), pl.col("p50_2028").round(0), pl.col("p90_2028").round(0)))

# %% 2a. large gas: stratum sizes by stage, the hazard ratio, and the developer's projected COD as a floor
split_train = train_today.with_columns(stratum=qu.size_stratum())
ia_frame = qa.stage_frame(split_train, "ia_signed").filter(pl.col("stratum").str.starts_with(qu.GAS))
pre_frame = qa.pre_ia_frame(split_train).filter(pl.col("stratum").str.starts_with(qu.GAS))
print("IA stage:", ia_frame.group_by("stratum").agg(n=pl.len(), cod=(pl.col("event") == sv.COD).sum(),
      withdrawn=(pl.col("event") == sv.WITHDRAWN).sum(), censored=(pl.col("event") == sv.CENSORED).sum(),
      med_followup=pl.col("time").median().round(1), med_mw=pl.col("capacity_mw").median()).sort("stratum").rows())
print("pre-IA sojourn:", pre_frame.group_by("stratum").agg(n=pl.len(), ia=(pl.col("event") == qa.IA_EVENT).sum(),
      cod=(pl.col("event") == sv.COD).sum(), withdrawn=(pl.col("event") == sv.WITHDRAWN).sum()).sort("stratum").rows())
curves_split, info = qu.split_gas_curves(train_today, STAGE_SET, weight=WEIGHT)
curves_split_n, info_n = qu.split_gas_curves(train_today, STAGE_SET, weight=None)
o, e = info_n["observed"], info_n["expected"]
ci = (chi2.ppf(0.025, 2 * o) / 2 / e if o > 0 else 0.0, chi2.ppf(0.975, 2 * o + 2) / 2 / e)
print(f"hazard ratio, MW-weighted: O {info['observed']:,.0f} MW / E {info['expected']:,.0f} MW = {info['theta']:.2f}; "
      f"count: O {o:.0f} / E {e:.2f} = {info_n['theta']:.2f} (exact Poisson 95% CI {ci[0]:.2f}-{ci[1]:.2f})")
for k in sorted(curves_split):
    if k[1].startswith(qu.GAS):
        c = curves_split[k]
        print(f"curve {k}: n={c.n}, events={c.cod}, CIF12/24/36 {np.round(c.fit.at([12, 24, 36]), 3)}")

# does anything reach COD before the developer's projected COD? (IA signed at the snapshot, COD 3-27 months later)
cod = events_all.filter(pl.col("exit_status") == "operational").select(
    "inr", "stratum", "capacity_mw", cod=pl.coalesce("commercial_operation_date", "cod_first_month"))
snaps = read_sql("SELECT inr, report_month, projected_cod, ia_signed FROM gis_snapshots "
                 "WHERE status = 'active' AND projected_cod IS NOT NULL AND ia_signed IS NOT NULL")
slip = snaps.join(cod, on="inr").with_columns(lead=qa.months(pl.col("cod"), pl.col("report_month")),
                                              slip=qa.months(pl.col("cod"), pl.col("projected_cod")))
slip_rows = []
for lo, hi in [(3, 9), (9, 15), (15, 27)]:
    g = slip.filter(pl.col("lead").is_between(lo, hi)).sort("inr", "report_month").group_by("inr").agg(pl.all().last())
    for gas, gg in g.group_by(gas=pl.col("stratum") == qu.GAS):
        slip_rows.append({"lead_months": f"{lo}-{hi}", "gas": gas[0], "projects": gg.height,
                          "early_gt_1mo": float((gg["slip"] < -1).mean()), "early_gt_3mo": float((gg["slip"] < -3).mean()),
                          "median_slip_mo": float(gg["slip"].median())})
slip_tab = pl.DataFrame(slip_rows).sort("lead_months", "gas")
print("COD vs the projected COD printed 3-27 months before (IA-stage projects that reached COD):\n",
      slip_tab.with_columns(pl.col(pl.Float64).round(3)))

# today's large gas and its floor
bg = queue.filter(BIG_GAS).with_columns(dev_delay=dev_floor(queue, AS_OF))
print("big gas today:", bg.group_by("stage").agg(n=pl.len(), mw=pl.col("capacity_mw").sum(),
      proj_after_2027=(pl.col("projected_cod") >= pl.lit(HORIZONS["2027"])).sum(),
      proj_after_2028=(pl.col("projected_cod") >= pl.lit(HORIZONS["2028"])).sum(),
      proj_null=pl.col("projected_cod").is_null().sum()).rows())

# %% 2b. the variants today: totals, gas+other, counties
var_scored = {v: point(queue, train_today, AS_OF, v, H_MONTHS) for v in VARIANTS}
tot_rows = []
for v, s in var_scored.items():
    r = {"variant": v}
    for hz in H_MONTHS:
        r[f"total_{hz}"] = float(s[f"mw_{hz}"].sum())
        r[f"gas_{hz}"] = float(s.filter(pl.col("fuel") == qu.GAS)[f"mw_{hz}"].sum())
        r[f"big_gas_{hz}"] = float(s.filter(BIG_GAS)[f"mw_{hz}"].sum())
    tot_rows.append(r)
totals = pl.DataFrame(tot_rows)
print(totals.with_columns(pl.exclude("variant").round(0)))

NAMED = ["Bastrop", "Ward", "Wheeler", "Wilbarger", "Freestone", "Caldwell", "Chambers", "Grayson", "Brazoria", "Pecos"]
rank_frames = []
for v, s in var_scored.items():
    c = s.group_by("county_fips", "county_name").agg(**{f"mw_{h}": pl.col(f"mw_{h}").sum() for h in H_MONTHS})
    c = c.with_columns(**{f"rank_{h}": pl.col(f"mw_{h}").rank("ordinal", descending=True) for h in H_MONTHS})
    rank_frames.append(c.with_columns(variant=pl.lit(v)))
ranks = pl.concat(rank_frames)
ranks.write_csv(out_path("x19_county_variants.csv"))
show = ranks.filter(pl.col("county_name").is_in(NAMED) & pl.col("variant").is_in(list(VARIANTS)[:4]))
print(show.pivot(on="variant", index="county_name", values="mw_2028").with_columns(pl.exclude("county_name").round(0)))
print(show.pivot(on="variant", index="county_name", values="rank_2028"))
print(show.pivot(on="variant", index="county_name", values="rank_2027"))
b0 = ranks.filter(pl.col("variant") == "baseline (X2)")
for v in list(VARIANTS)[1:4]:
    b1 = ranks.filter(pl.col("variant") == v)
    j = b0.join(b1, on="county_fips", suffix="_v")
    top0 = set(b0.sort("mw_2028", descending=True).head(10)["county_name"])
    top1 = set(b1.sort("mw_2028", descending=True).head(10)["county_name"])
    print(f"{v}: Spearman vs baseline (Dec 2028) {spearmanr(j['mw_2028'], j['mw_2028_v']).statistic:.3f}; "
          f"top-10 out {sorted(top0 - top1)}, in {sorted(top1 - top0)}")
    print("   new top 10:", b1.sort("mw_2028", descending=True).head(10)["county_name"].to_list())

# %% 2c. backtests with the variants (as-of fits, the snapshot's own projected COD)
bt_var = []
for rm, (q, train, as_of, h) in bt_scored.items():
    for v in VARIANTS:
        s = point(q, train, as_of, v, h)
        for fuel, g in [("all", s), *[(k[0], gg) for k, gg in s.group_by("fuel")]]:
            bt_var.append({"report_month": rm, "variant": v, "fuel": fuel, "pred": float(g["mw_w"].sum()),
                           "actual": float(g["actual"].sum())})
        if v == "split: hazard ratio":
            _, inf = qu.split_gas_curves(train, STAGE_SET, weight=WEIGHT)
            print(f"{rm}: large-gas IA stratum n={inf['n']:.0f}, COD {inf['events']:.0f}, theta {inf['theta']:.2f}; "
                  f"big-gas queue {q.filter(BIG_GAS).height} projects, {q.filter(BIG_GAS)['capacity_mw'].sum():,.0f} MW, "
                  f"actual {q.filter(BIG_GAS)['actual'].sum():,.0f} MW")
btv = pl.DataFrame(bt_var).with_columns(err=pl.col("pred") / pl.col("actual") - 1)
btv.write_csv(out_path("x19_backtest_variants.csv"))
print(btv.filter(pl.col("fuel") == "all").pivot(on="report_month", index="variant", values="err").with_columns(
    pl.exclude("variant").round(3)))
print(btv.filter(pl.col("fuel") == qu.GAS).pivot(on="report_month", index="variant", values="pred").with_columns(
    pl.exclude("variant").round(0)))
print("gas+other actual:", btv.filter((pl.col("fuel") == qu.GAS) & (pl.col("variant") == "baseline (X2)"))
      .select("report_month", pl.col("actual").round(0)).rows())
print(btv.group_by("variant").agg(
    mean_abs_err=pl.col("err").filter(pl.col("fuel") == "all").abs().mean().round(3),
    mean_err=pl.col("err").filter(pl.col("fuel") == "all").mean().round(3),
    gas_mean_abs_err=pl.col("err").filter(pl.col("fuel") == qu.GAS).abs().mean().round(3),
).sort("mean_abs_err"))

# %% 2d. the band for the recommended variant: today (statewide + top 10) and coverage at the backtests
REC = "split + floor (big gas)"
qr, fit_r, adj_r = prepare(queue, AS_OF, REC)
t0 = time.time()
reps_r = qu.bootstrap(train_today, qr, fit_r, H_MONTHS, n_boot=N_BOOT_TODAY, seed=SEED, groups=("county_fips", "fuel"),
                      adjust=lambda s, c: adj_r(s, c, H_MONTHS))
print(f"bootstrap ({REC}) today: {time.time() - t0:.0f} s")
band_r = qu.band(reps_r, "simulated_mw")
pt_r = var_scored[REC]
rec_state = []
for hz in H_MONTHS:
    ptv = float(pt_r[f"mw_{hz}"].sum())
    lo, hi = qu.error_band(ptv, errs)
    r = band_r.filter((pl.col("horizon") == hz) & (pl.col("level") == "total")).row(0, named=True)
    rc = qu.band(reps_r, "expected_mw").filter((pl.col("horizon") == hz) & (pl.col("level") == "total")).row(0, named=True)
    rec_state.append({"horizon": hz, "point": ptv, "p10": r["p10"], "p50": r["p50"], "p90": r["p90"],
                      "curves_p10": rc["p10"], "curves_p90": rc["p90"], "backtest_p10": float(lo), "backtest_p90": float(hi)})
rec_state = pl.DataFrame(rec_state)
print(rec_state.with_columns(pl.exclude("horizon").round(0)))
rec_county = pt_r.group_by("county_fips", "county_name").agg(
    raw_mw=pl.col("capacity_mw").sum(), **{f"point_{h}": pl.col(f"mw_{h}").sum() for h in H_MONTHS}
).with_columns(pl.col("county_fips").cast(pl.String).fill_null("none")).join(
    band_r.filter(pl.col("level") == "county_fips").rename({"group": "county_fips"})
    .pivot(on="horizon", index="county_fips", values=["p10", "p50", "p90"]), on="county_fips", how="left",
).sort("point_2028", descending=True)
rec_county.write_csv(out_path("x19_county_band_recommended.csv"))
print("top-10 counties, recommended:\n", rec_county.head(10).select(
    "county_name", pl.col("raw_mw").round(0), pl.col("point_2027").round(0), pl.col("p10_2027").round(0),
    pl.col("p50_2027").round(0), pl.col("p90_2027").round(0), pl.col("point_2028").round(0),
    pl.col("p10_2028").round(0), pl.col("p50_2028").round(0), pl.col("p90_2028").round(0)))

rec_cov = []
for rm, (q, train, as_of, h) in bt_scored.items():
    qv, fit_v, adj_v = prepare(q, as_of, REC)
    r = qu.bootstrap(train, qv, fit_v, h, n_boot=N_BOOT_BT, seed=SEED, groups=("county_fips", "fuel"),
                     adjust=lambda s, c, h=h, adj_v=adj_v: adj_v(s, c, h))
    s = point(q, train, as_of, REC, h)
    for level in ("total", "fuel", "county_fips"):
        col = None if level == "total" else level
        if col is None:
            act = {"all": float(q["actual"].sum())}
            prd = {"all": float(s["mw_w"].sum())}
        else:
            act = dict(q.group_by(pl.col(col).cast(pl.String).fill_null("none")).agg(pl.col("actual").sum()).iter_rows())
            prd = dict(s.group_by(pl.col(col).cast(pl.String).fill_null("none")).agg(pl.col("mw_w").sum()).iter_rows())
        for row in qu.band(r.filter(pl.col("level") == level), "simulated_mw").iter_rows(named=True):
            a, p = act.get(row["group"], 0.0), prd.get(row["group"], 0.0)
            rec_cov.append({"report_month": rm, "level": level, "group": row["group"], "pred": p, "actual": a,
                            "p10": row["p10"], "p90": row["p90"], "covered": row["p10"] <= a <= row["p90"]})
rec_cov = pl.DataFrame(rec_cov)
rec_cov.write_csv(out_path("x19_backtest_coverage_recommended.csv"))
print("recommended, statewide:\n", rec_cov.filter(pl.col("level") == "total").with_columns(pl.col(pl.Float64).round(0)))
print("recommended, fuel coverage:", rec_cov.filter(pl.col("level") == "fuel")["covered"].sum(), "of",
      rec_cov.filter(pl.col("level") == "fuel").height)
print(rec_cov.filter(pl.col("level") == "fuel").select("report_month", "group", pl.col("pred", "actual", "p10", "p90").round(0), "covered"))
rc = rec_cov.filter((pl.col("level") == "county_fips") & ((pl.col("pred") >= 50) | (pl.col("actual") > 0)))
print("recommended, county coverage:", rc.group_by("report_month").agg(n=pl.len(), coverage=pl.col("covered").mean().round(3),
      below=(pl.col("actual") < pl.col("p10")).mean().round(3), above=(pl.col("actual") > pl.col("p90")).mean().round(3))
      .sort("report_month").rows(), "pooled", round(rc["covered"].mean(), 3))
