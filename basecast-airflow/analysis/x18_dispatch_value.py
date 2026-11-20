# %% [markdown]
# # X18 — A battery's evening dispatch vs. its 4CP value, in dollars per MW
#
# Exploration X18. Questions and results in `docs/analysis/x18_dispatch_value.md`; logic in
# `basecast_pipelines/models/dispatch_value.py` (tests in `tests/models/test_dispatch_value.py`). Reuses
# `basecast_pipelines/models/four_cp.py` by import for the 4CP intervals and X3's dispatch rule. Read-only: Postgres
# through the Cloud SQL proxy (load, weather, monthly peaks) plus the local raw lake for real-time prices
# (`ercot_spp_hist`, parsed with the repo's parser and cached under `analysis/out/`, gitignored).
#
# Run from the repo root: `uv run --group analysis python analysis/x18_dispatch_value.py`.

# %%
from __future__ import annotations

from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from analysis._common import ROOT, out_path
from basecast_pipelines.models import dispatch_value as dv
from basecast_pipelines.models import four_cp as fc
from basecast_pipelines.models import weather_load as wl

YEARS = range(2021, 2026)  # full calendar years of prices used for the annual values
PRICE_YEARS = range(2021, 2027)  # 2026 runs through the latest lake snapshot (summer 2026 partial)
SUMMER = (6, 7, 8, 9)
ETA = 0.85  # round-trip efficiency, AC to AC: an assumption (Base's battery spec is not verified)
ETA_HIGH = 0.90
X_RULE = 0.96  # X3's headline 4CP alert rule: forecast peak ≥ 96% of the month-to-date max
URI = (date(2021, 2, 13), date(2021, 2, 20))  # Winter Storm Uri price days, excluded in a sensitivity
# The simple rule: fixed windows every summer day (chosen from the 2023–2025 summer price profile: in-sample)
RULES = {
    2: {"rule_charge": (9 * 60, 120), "rule_discharge": (19 * 60, 120)},  # charge 09–11, discharge 19–21 (HE 20–21)
    4: {"rule_charge": (8 * 60, 240), "rule_discharge": (18 * 60, 240)},  # charge 08–12, discharge 18–22 (HE 19–22)
}

# %% Prices (local raw lake, cached)
cache = out_path("x18_rtm_prices.parquet")
if not cache.exists():
    dv.load_rtm_prices(ROOT / "data" / "raw", PRICE_YEARS).write_parquet(cache)
prices = dv.with_local_day(pl.read_parquet(cache))
print("prices", prices.height, prices["interval_start_utc"].min(), "→", prices["interval_start_utc"].max())
print("points:", sorted(prices["settlement_point"].unique().to_list()))
slots = prices.group_by("settlement_point", "operating_date").agg(pl.len().alias("n"))
print("slots per day:", slots["n"].value_counts().sort("n").rows())
last_day = prices["operating_date"].max()
prices = prices.filter(pl.col("operating_date") < last_day)  # drop a possibly incomplete last day
print("last full day:", prices["operating_date"].max())

# Summer price profile by hour (HB_HUBAVG): where the rule windows come from
prof = (
    prices.filter((pl.col("settlement_point") == "HB_HUBAVG") & pl.col("operating_date").dt.month().is_in(SUMMER))
    .with_columns((pl.col("start_min") // 60).alias("hour"), pl.col("operating_date").dt.year().alias("year"))
    .group_by("year", "hour").agg(pl.col("price_usd_mwh").mean().round(1))
    .pivot(on="year", index="hour", values="price_usd_mwh", sort_columns=True).sort("hour")
)
print(prof)

# %% Daily values per settlement point: perfect foresight, the rule, 4CP-committed
daily = {}
for hours in (2, 4):
    for eta in (ETA, ETA_HIGH):
        daily[(hours, eta)] = dv.daily_values(prices, hours, eta, **RULES[hours])
        print(hours, eta, daily[(hours, eta)].height)
d2, d4 = daily[(2, ETA)], daily[(4, ETA)]

# %% [markdown]
# ## Q1 — Arbitrage value per MW-year, 2021–2025

# %%
def _year_table(d: pl.DataFrame, value: str, months=None, exclude_uri: bool = False) -> pl.DataFrame:
    if exclude_uri:
        d = d.filter(~pl.col("operating_date").is_between(*URI))
    t = dv.annual_values(d, value, months).filter(pl.col("year").is_in(list(YEARS)))
    return t.pivot(on="year", index="settlement_point", values=value, sort_columns=True).sort("settlement_point")


tables = {}
for hours in (2, 4):
    for eta in (ETA, ETA_HIGH):
        d = daily[(hours, eta)]
        tables[(hours, eta, "pf_year")] = _year_table(d, "pf")
        tables[(hours, eta, "pf_year_ex_uri")] = _year_table(d, "pf", exclude_uri=True)
        tables[(hours, eta, "pf_summer")] = _year_table(d, "pf", SUMMER)
        tables[(hours, eta, "rule_summer")] = _year_table(d, "rule", SUMMER)
        tables[(hours, eta, "rule_year")] = _year_table(d, "rule")
for key in [(2, ETA, "pf_year"), (2, ETA, "pf_year_ex_uri"), (2, ETA, "pf_summer"), (2, ETA, "rule_summer"),
            (4, ETA, "pf_year"), (4, ETA, "pf_year_ex_uri"), (4, ETA, "pf_summer"), (4, ETA, "rule_summer"),
            (2, ETA_HIGH, "pf_year"), (4, ETA_HIGH, "pf_year"), (2, ETA, "rule_year"), (4, ETA, "rule_year")]:
    print(f"--- {key[0]} h, eta {key[1]}, {key[2]} ($/MW)")
    print(tables[key].with_columns(pl.exclude("settlement_point").round(0)))

rows = []
for (hours, eta, kind), t in tables.items():
    for r in t.iter_rows(named=True):
        for y in YEARS:
            rows.append({"hours": hours, "eta": eta, "measure": kind, "settlement_point": r["settlement_point"],
                         "year": y, "usd_per_mw": r.get(str(y))})
annual = pl.DataFrame(rows)
annual.write_csv(out_path("x18_annual_values.csv"))

# Share of the year's perfect-foresight value earned in June–September, and in the top 10 days
for hours, d in ((2, d2), (4, d4)):
    s = d.with_columns(pl.col("operating_date").dt.year().alias("year")).filter(pl.col("year").is_in(list(YEARS)))
    share = s.group_by("settlement_point", "year").agg(
        (pl.col("pf").filter(pl.col("operating_date").dt.month().is_in(SUMMER)).sum() / pl.col("pf").sum()).alias("summer"),
        (pl.col("pf").sort(descending=True).head(10).sum() / pl.col("pf").sum()).alias("top10"),
    )
    print(f"{hours} h: share of the year in Jun–Sep and in its 10 best days (median over points, by year)")
    print(share.group_by("year").agg(pl.col("summer").median().round(2), pl.col("top10").median().round(2)).sort("year"))

# Where the perfect-foresight discharge starts in summer (HB_HUBAVG), by year
when = d2.filter((pl.col("settlement_point") == "HB_HUBAVG") & pl.col("operating_date").dt.month().is_in(SUMMER)
                 & pl.col("pf_discharge_min").is_not_null()).with_columns(
    pl.col("operating_date").dt.year().alias("year"), (pl.col("pf_discharge_min") // 60).alias("start_hour"))
print("2 h pf discharge start hour, summer HB_HUBAVG (median, share starting 17:00 or later):")
print(when.group_by("year").agg(pl.col("start_hour").median(),
                                (pl.col("start_hour") >= 17).mean().round(2).alias("share_ge_17"),
                                pl.col("pf_charge_min").median().alias("charge_min_median")).sort("year"))

# %% [markdown]
# ## Q2 — The 4CP days: X3's dispatch days and what they cost in arbitrage

# %% X3's rule on the same inputs (read-only DB)
monthly = fc.load_monthly_peaks()
load = wl.load_hourly_load()
weather = wl.load_hourly_weather()
cps = fc.cp_calendar(fc.cp_intervals(monthly), fc.hourly_cps(load))
fdaily = fc.with_reference(fc.daily_peaks(load))
sys_wx = wl.system_weather_daily(wl.weather_daily(weather), wl.zone_weights(load, range(2003, 2023))).select(
    "date", "t_max", "t_mean")
fdaily = fc.weather_forecast(fdaily, sys_wx, range(2018, 2027))
dispatch = fc.threshold_dispatch(fdaily, "forecast_wx", X_RULE).filter(pl.col("year").is_between(2021, 2026))
per_year = fc.evaluate_dispatch(dispatch, cps.filter(pl.col("year").is_between(2021, 2026)),
                                dv.CP_START_MIN, dv.CP_LENGTH_MIN)
print(per_year)
dispatch.filter(pl.col("dispatch")).write_csv(out_path("x18_cp_dispatch_days.csv"))

# %% 4CP value per summer: the billing year's postage-stamp rate × the share of the four intervals caught
cp_value = per_year.select("year", "n_cp", "caught", "dispatch_days").with_columns(
    pl.col("year").map_elements(lambda y: dv.four_cp_value(y), return_dtype=pl.Float64).alias("usd_all_four"),
    pl.struct("year", "caught", "n_cp").map_elements(
        lambda r: dv.four_cp_value(r["year"], r["caught"] / 4), return_dtype=pl.Float64).alias("usd_caught"),
    (pl.col("year") + 1).alias("billing_year"),
)
print(cp_value)

# %% Opportunity cost on the dispatch days, per settlement point and summer
opp_rows = []
for hours, d, schedules in ((2, d2, ["cp"]), (4, d4, ["cp", "cp_split"])):
    for sched in schedules:
        o = dv.opportunity_cost(d, dispatch, sched).rename({sched: "committed"}).with_columns(
            pl.lit(hours).alias("hours"), pl.lit(sched).alias("schedule"))
        opp_rows.append(o)
opp = pl.concat(opp_rows).filter(pl.col("year").is_between(2021, 2026))
opp.write_csv(out_path("x18_opportunity_cost.csv"))
for (hours, sched), g in opp.group_by("hours", "schedule", maintain_order=True):
    print(f"--- {hours} h, {sched}: arbitrage lost on 4CP dispatch days ($/MW per summer)")
    print(g.pivot(on="year", index="settlement_point", values="lost", sort_columns=True).with_columns(
        pl.exclude("settlement_point").round(0)).sort("settlement_point"))
    print(g.group_by("year").agg(pl.col("days").first(), pl.col("pf").median().round(0),
                                 pl.col("committed").median().round(0), pl.col("lost").median().round(0),
                                 pl.col("lost").min().round(0).alias("lost_min"),
                                 pl.col("lost").max().round(0).alias("lost_max")).sort("year"))

# Same days, the price in the 4CP window vs. the best 2 h block (mean over dispatch days, HB_HUBAVG and LZs)
on_days = d2.join(dispatch.filter(pl.col("dispatch")).select("operating_date"), on="operating_date").with_columns(
    pl.col("operating_date").dt.year().alias("year"))
print("4CP days: mean price 15:45–17:45 vs the pf discharge block ($/MWh), and the share of days whose best 2 h "
      "block starts at 17:45 or later:")
print(on_days.group_by("year").agg(
    pl.col("cp_price").mean().round(1), pl.col("pf_discharge_price").mean().round(1),
    (pl.col("pf_discharge_min") >= dv.CP_START_MIN + dv.CP_LENGTH_MIN).mean().round(2).alias("pf_after_1745"),
).sort("year"))
# the CP days themselves: price at the CP window vs the day's best 2 h
cp_days = cps.filter(pl.col("year").is_between(2021, 2026)).select(pl.col("cp_date").alias("operating_date"), "year",
                                                                     "month")
print(d2.join(cp_days, on="operating_date").filter(pl.col("settlement_point") == "HB_HUBAVG").select(
    "operating_date", "cp_price", "pf_discharge_price", "pf_discharge_min", "pf", "cp").sort("operating_date"))

# eta sensitivity of the lost value (median over points)
for hours, sched in ((2, "cp"), (4, "cp"), (4, "cp_split")):
    o = dv.opportunity_cost(daily[(hours, ETA_HIGH)], dispatch, sched)
    print(f"eta {ETA_HIGH}, {hours} h {sched} lost (median over points):",
          o.group_by("year").agg(pl.col("lost").median().round(0)).sort("year").rows())

# %% Net per summer at the median load zone: 4CP value − lost arbitrage, and the free-day arbitrage
lz = opp.filter(pl.col("settlement_point").str.starts_with("LZ_"))
net = lz.group_by("hours", "schedule", "year").agg(
    pl.col("lost").median().alias("lost_median"), pl.col("lost").min().alias("lost_min"),
    pl.col("lost").max().alias("lost_max"),
).join(cp_value.select("year", "usd_caught", "usd_all_four"), on="year", how="left").with_columns(
    (pl.col("usd_caught") - pl.col("lost_median")).alias("net_median")
).sort("hours", "schedule", "year")
print(net)

# Whole-year arbitrage with and without the 4CP commitment (median over load zones)
year_pf = {h: dv.annual_values(daily[(h, ETA)], "pf").filter(pl.col("settlement_point").str.starts_with("LZ_"))
           for h in (2, 4)}
for h in (2, 4):
    print(f"{h} h: whole-year pf per load zone ($/MW-yr), min / median / max:")
    print(year_pf[h].filter(pl.col("year").is_in(list(YEARS))).group_by("year").agg(
        pl.col("pf").min().round(0).alias("min"), pl.col("pf").median().round(0).alias("median"),
        pl.col("pf").max().round(0).alias("max")).sort("year"))

# %% Figure: per summer, 4CP value vs. arbitrage lost on the dispatch days (median load zone, range as whiskers)
fig, ax = plt.subplots(figsize=(8.5, 4.2))
series = [("4CP value (rate of billing year)", None, "#2a78d6"),
          ("2 h: lost to 15:45–17:45", (2, "cp"), "#eb6834"),
          ("4 h: lost to 15:45–19:45", (4, "cp"), "#1baf7a"),
          ("4 h: lost to 15:45–17:45 + best evening 2 h", (4, "cp_split"), "#eda100")]
years = list(YEARS)
width = 0.2
for i, (label, key, color) in enumerate(series):
    xs = [y + (i - 1.5) * width for y in years]
    if key is None:
        vals = [cp_value.filter(pl.col("year") == y)["usd_caught"].item() for y in years]
        ys = [v / 1000 if v is not None else 0 for v in vals]
        ax.bar(xs, ys, width * 0.9, color=color, label=label)
        for x, v in zip(xs, vals, strict=True):
            if v is None:
                ax.text(x, 1, "rate not\nsourced", ha="center", va="bottom", fontsize=6, color="#555", rotation=90)
    else:
        g = net.filter((pl.col("hours") == key[0]) & (pl.col("schedule") == key[1])).sort("year")
        med = [g.filter(pl.col("year") == y)["lost_median"].item() / 1000 for y in years]
        lo = [g.filter(pl.col("year") == y)["lost_min"].item() / 1000 for y in years]
        hi = [g.filter(pl.col("year") == y)["lost_max"].item() / 1000 for y in years]
        ax.bar(xs, med, width * 0.9, color=color, label=label,
               yerr=[[m - a for m, a in zip(med, lo, strict=True)], [b - m for m, b in zip(med, hi, strict=True)]],
               error_kw={"elinewidth": 0.8, "ecolor": "#444", "capsize": 2})
ax.set_xticks(years)
ax.set_ylabel("$ thousand per MW per summer")
ax.set_title("4CP value vs. arbitrage given up on X3 alert days (RT load zones, η = 85%)", fontsize=10)
ax.legend(fontsize=7, frameon=False, loc="upper left")
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(out_path("x18_cp_vs_arbitrage.png"), dpi=140)
plt.close(fig)
print("wrote", out_path("x18_cp_vs_arbitrage.png"))
