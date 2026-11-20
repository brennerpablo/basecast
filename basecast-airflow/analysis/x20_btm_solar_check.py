"""X20 — is X1's "flat" load excess real, or partly behind-the-meter solar shaving the daytime peak?

Run from the repo root: ``uv run --group analysis python analysis/x20_btm_solar_check.py``.
Reads ``ercot_load_hourly_wz`` and ``weather_hourly_wz`` (read-only, through ``weather_load``'s loaders), the
EIA-861 zips in the local lake (``data/raw/source=eia_861``) and the ERCOT fuel-mix solar series (X3's cache of
``data/raw/source=ercot_fuel_mix``, rebuilt with the repo's parser when missing). Logic:
``basecast_pipelines/models/flatness.py`` on top of ``peak_excess.py`` (X1) and ``weather_normalized.py`` (X12),
both unchanged. Numbers go to ``docs/analysis/x20_btm_solar_check.md``.
"""

# %% Setup
from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from analysis._common import ROOT, out_path
from basecast_pipelines.models import flatness as fl
from basecast_pipelines.models import peak_excess as px
from basecast_pipelines.models import weather_load as wl
from basecast_pipelines.models import weather_normalized as wn

SERIES = (*wl.WEATHER_ZONES, wl.TOTAL)
BASE_YEAR = 2019
YEARS = (2022, 2023, 2024, 2025, 2026)
LAST_SEP_DAY = 19  # 2026's last day with complete weather is Sep 19 (X12); same window every year
BLOCKS = fl.HOUR_BLOCKS
LATE_BLOCKS = {"overnight": (1, 5), "midday": (11, 15), "afternoon": (16, 19), "evening": (22, 24)}
SPEC = wn.Spec(level_knots_per_year=2)  # X12's chosen variant (lag + dew point + 6-month level knots)

load = wl.load_hourly_load()
weather = wl.load_hourly_weather()
print(load.shape, weather.shape, "last operating day:", load["operating_date"].max(),
      "| last weather hour:", weather["ts_utc"].max())


def wide(frame: pl.DataFrame, value: str, index: str = "year", on: str = "weather_zone",
         cols: tuple[str, ...] = SERIES) -> pl.DataFrame:
    w = frame.pivot(on=on, index=index, values=value).sort(index)
    return w.select(index, *[c for c in cols if c in w.columns])


# %% 1a. Raw summer hour-block means and their growth since 2019 (Jun 1 - Sep 19)
blocks = fl.block_daily(load, BLOCKS)
summer = fl.summer_block_means(blocks, last_sep_day=LAST_SEP_DAY)
growth = fl.growth_vs_base(summer, BASE_YEAR).filter(pl.col("year").is_in(YEARS))
growth.write_csv(out_path("x20_block_growth_raw.csv"))
print(f"summer block mean, {BASE_YEAR} (MW):")
print(wide(summer.filter(pl.col("year") == BASE_YEAR), "mw", index="block").with_columns(pl.exclude("block").round(0)))
for blk in ("overnight", "evening", "midday", "afternoon"):
    print(f"raw growth {BASE_YEAR} -> year, {blk} (MW):")
    print(wide(growth.filter(pl.col("block") == blk), "delta_mw").with_columns(pl.exclude("year").round(0)))

split_raw = fl.solar_free_split(growth)
split_raw.write_csv(out_path("x20_split_raw.csv"))
print("raw solar-free split (overnight vs evening): flat F, proportional g, dark ratio, midday gap")
print(split_raw.filter(pl.col("year").is_in([2023, 2025, 2026])).select(
    "weather_zone", "year", "overnight_delta_mw", "evening_delta_mw", "dark_ratio", "proportional_ratio", "flat_mw",
    "proportional_pct", "midday_delta_mw", "midday_pred_mw", "midday_gap_mw", "afternoon_gap_mw")
    .with_columns(pl.selectors.float().round(2)))

# Sensitivity: evening = HE 22-24 only (HE 21 still has a low sun in late June).
g_late = fl.growth_vs_base(fl.summer_block_means(fl.block_daily(load, LATE_BLOCKS), last_sep_day=LAST_SEP_DAY),
                           BASE_YEAR).filter(pl.col("year").is_in(YEARS))
split_late = fl.solar_free_split(g_late)
print("sensitivity, evening = HE 22-24: flat F and dark ratio")
print(split_late.filter(pl.col("year").is_in([2025, 2026])).select(
    "weather_zone", "year", "dark_ratio", "flat_mw", "midday_gap_mw").with_columns(pl.selectors.float().round(2)))

# Where is the summer daily maximum? (hour ending of each day's max, ERCOT)
dmax_hour = (load.filter(fl.in_summer_window("operating_date", last_sep_day=LAST_SEP_DAY))
             .group_by("weather_zone", "operating_date")
             .agg(pl.col("hour_ending").sort_by("mw").last().alias("he_max"),
                  pl.col("hour_ending").sort_by("mw").first().alias("he_min")))
he_tab = (dmax_hour.with_columns(pl.col("operating_date").dt.year().alias("year"))
          .filter(pl.col("year").is_in([2015, BASE_YEAR, *YEARS]))
          .group_by("weather_zone", "year")
          .agg(pl.col("he_max").mode().first().alias("he_max_mode"), pl.col("he_max").mean().alias("he_max_mean"),
               pl.col("he_min").mode().first().alias("he_min_mode")))
print("summer daily max: modal and mean hour ending; daily min modal hour ending")
print(he_tab.filter(pl.col("weather_zone") == wl.TOTAL).sort("year").with_columns(pl.selectors.float().round(2)))
print(wide(he_tab, "he_max_mean").with_columns(pl.exclude("year").round(1)))

# %% 1b. X1-style excess over a pre-break model, per block (target ~ year + summer mean temperature, fit <= 2019)
daily_w = wl.weather_daily(weather)
weights = wl.zone_weights(load, range(2003, px.PRE_BREAK_UNTIL + 1))
daily_w_all = pl.concat([daily_w.drop("n_hours"), wl.system_weather_daily(daily_w, weights).drop("n_hours")])
tsum = px.summer_mean_weather(daily_w_all)
bpanel = (summer.pivot(on="block", index=["weather_zone", "year"], values="mw")
          .join(tsum, on=["weather_zone", "year"], how="inner").filter(pl.col("year") >= 2003))
ex_blocks = []
for blk in BLOCKS:
    e = px.excess_vs_prebreak(bpanel, blk, "t_summer")
    ex_blocks.append(e.select("weather_zone", "year", pl.lit(blk).alias("block"), "excess_mw", "predicted_mw",
                              "train_resid_sd_mw"))
ex_blocks = pl.concat(ex_blocks)
ex_blocks.write_csv(out_path("x20_block_excess_prebreak.csv"))
exw = ex_blocks.pivot(on="block", index=["weather_zone", "year"], values="excess_mw")
pdw = ex_blocks.pivot(on="block", index=["weather_zone", "year"], values="predicted_mw")
sdw = ex_blocks.filter(pl.col("year") == 2019).pivot(on="block", index="weather_zone", values="train_resid_sd_mw")
pre = (exw.join(pdw, on=["weather_zone", "year"], suffix="_pred")
       .with_columns((pl.col("overnight") / pl.col("evening")).alias("dark_flatness"),
                     (pl.col("overnight_pred") / pl.col("evening_pred")).alias("proportional_ratio"),
                     (pl.col("midday") / pl.col("evening")).alias("midday_over_evening")))
print("excess over the pre-break model (MW) by block; dark flatness = overnight / evening excess")
print(pre.filter((pl.col("weather_zone") == wl.TOTAL) & (pl.col("year") >= 2020)).select(
    "year", "overnight", "evening", "midday", "afternoon", "dark_flatness", "proportional_ratio",
    "midday_over_evening").with_columns(pl.selectors.float().round(2)))
print("training residual SD (MW):", sdw.filter(pl.col("weather_zone") == wl.TOTAL))
print(pre.filter(pl.col("year").is_in([2025, 2026])).select(
    "weather_zone", "year", "overnight", "evening", "midday", "afternoon", "dark_flatness", "proportional_ratio")
    .sort("weather_zone", "year").with_columns(pl.selectors.float().round(2)))
pre.write_csv(out_path("x20_block_excess_prebreak_wide.csv"))

# %% 2. Weather-adjusted blocks: X12's daily model and normalization applied to each block's daily mean
panel, feats = wn.build_panel(load, weather)
normals = wn.normal_weather(feats, wn.NORMAL_YEARS)
bp = fl.add_blocks_to_panel(panel, blocks)
norm_rows = []
for z in SERIES:
    for blk in BLOCKS:
        target = f"b_{blk}"
        sub = bp.filter(pl.col("weather_zone") == z).drop_nulls(target)
        d, _, _ = wn.normalize_zone(sub, z, target, normals, SPEC, years=list(range(2017, 2027)))
        norm_rows.append(d.select("weather_zone", "date", pl.lit(blk).alias("block"),
                                  pl.col(target).alias("mw"), "normalized", "w_actual", "w_normal"))
normd = pl.concat(norm_rows)
normd.write_parquet(out_path("x20_block_daily_normalized.parquet"))
summer_n = fl.summer_block_means(normd.rename({"mw": "raw_mw", "normalized": "mw"}), last_sep_day=LAST_SEP_DAY)
summer_raw_chk = fl.summer_block_means(normd.select("weather_zone", "date", "block", "mw"),
                                       last_sep_day=LAST_SEP_DAY)
adj = (summer_n.join(summer_raw_chk, on=["weather_zone", "year", "block"], suffix="_raw")
       .with_columns((pl.col("mw") - pl.col("mw_raw")).alias("weather_adj_mw")))
print("weather adjustment of the summer block means (normalized - actual, MW), ERCOT:")
print(adj.filter(pl.col("weather_zone") == wl.TOTAL).pivot(on="block", index="year", values="weather_adj_mw")
      .sort("year").with_columns(pl.exclude("year").round(0)))
growth_n = fl.growth_vs_base(summer_n, BASE_YEAR).filter(pl.col("year").is_in(YEARS))
growth_n.write_csv(out_path("x20_block_growth_normalized.csv"))
for blk in ("overnight", "evening", "midday", "afternoon"):
    print(f"weather-normalized growth {BASE_YEAR} -> year, {blk} (MW):")
    print(wide(growth_n.filter(pl.col("block") == blk), "delta_mw").with_columns(pl.exclude("year").round(0)))
split_n = fl.solar_free_split(growth_n)
split_n.write_csv(out_path("x20_split_normalized.csv"))
print("weather-normalized solar-free split")
print(split_n.filter(pl.col("year").is_in([2023, 2024, 2025, 2026])).select(
    "weather_zone", "year", "overnight_delta_mw", "evening_delta_mw", "dark_ratio", "proportional_ratio", "flat_mw",
    "proportional_pct", "midday_delta_mw", "midday_pred_mw", "midday_gap_mw", "afternoon_gap_mw")
    .sort("weather_zone", "year").with_columns(pl.selectors.float().round(2)))

# %% 3a. EIA-861 distributed PV reported by Texas utilities in the ERCOT BA (net-metered + non-net-metered)
eia = fl.load_eia861_pv(ROOT / "data" / "raw" / "source=eia_861" / "dt=2026-09-25")
pv = fl.btm_pv_by_year(eia)
pv_w = (pv.pivot(on="kind", index="year", values="pv_mw").fill_null(0.0)
        .with_columns((pl.col("net_metering") + pl.col("non_net_metering")).alias("total_mw")).sort("year"))
pv.write_csv(out_path("x20_eia861_pv_tx_ercot.csv"))
print("EIA-861, TX utilities with BA code ERCO, distributed PV capacity (MW; types as reported; 2025 = early release)")
print(pv.with_columns(pl.selectors.float().round(1)))
print(pv_w.with_columns(pl.selectors.float().round(1)))
tx_all = (eia.filter(pl.col("state") == "TX").group_by("year", "kind").agg(pl.col("pv_mw").sum())
          .pivot(on="kind", index="year", values="pv_mw").sort("year"))
print("all TX utilities (any BA), MW:", tx_all.with_columns(pl.selectors.float().round(1)))
top = (eia.filter((pl.col("state") == "TX") & (pl.col("ba_code") == fl.ERCOT_BA) & pl.col("year").is_in([2019, 2024]))
       .group_by("year", "utility_name").agg(pl.col("pv_mw").sum()).sort("pv_mw", descending=True)
       .group_by("year").head(6).sort("year", "pv_mw", descending=[False, True]))
print(top.with_columns(pl.col("pv_mw").round(1)))

# %% 3b. Solar output shape at the hour of the daily load maximum (ERCOT utility-scale solar, fuel mix)
fuel_cache = out_path("x3_fuel_wind_solar.parquet")
if not fuel_cache.exists():
    from basecast_pipelines.parsers.ercot._market_common import workbooks
    from basecast_pipelines.parsers.ercot.fuel_mix import _workbook

    frames = []
    for p in sorted((ROOT / "data" / "raw" / "source=ercot_fuel_mix").glob("dt=*/*")):
        if p.suffix.lower() not in {".zip", ".xlsx", ".xls"}:
            continue
        f = SimpleNamespace(name=p.name, suffix=p.suffix.lower(), key=str(p), read_bytes=p.read_bytes)
        for name, data in workbooks(f):
            if int("".join(ch for ch in name if ch.isdigit())[:4]) < 2011:
                continue
            wb = _workbook(name, data, "x20")
            frames.append(wb.filter(pl.col("fuel").cast(pl.String).is_in(["wind", "solar"])).select(
                "interval_start_utc", pl.col("fuel").cast(pl.String), "generation_mwh"))
    pl.concat(frames).unique(["interval_start_utc", "fuel"], keep="last").sort("interval_start_utc").write_parquet(
        fuel_cache)
fuel = pl.read_parquet(fuel_cache)
solar = fl.solar_hourly_local(fuel)
print("solar hourly rows", solar.height, solar["date"].min(), "->", solar["date"].max())
max_hours = (dmax_hour.filter(pl.col("weather_zone") == wl.TOTAL)
             .select(pl.col("operating_date").alias("date"), pl.col("he_max").alias("hour_ending")))
shares = fl.solar_share_at(solar, max_hours)
share_y = (shares.with_columns(pl.col("date").dt.year().alias("year")).filter(pl.col("year") >= 2019)
           .group_by("year").agg(pl.col("share_of_max").mean(), pl.col("share_of_midday").mean(),
                                 pl.col("hour_ending").mean().alias("he_max_mean"), pl.len().alias("days"))
           .sort("year"))
print("utility-scale solar at the hour of ERCOT's summer daily max, as a share of the day's max hour and of the"
      " HE 11-15 mean:")
print(share_y.with_columns(pl.selectors.float().round(3)))
# Same shape at fixed clock hours (for rooftop, which is mostly fixed-tilt: tracking plants run flatter).
prof = (solar.filter(fl.in_summer_window("date", last_sep_day=LAST_SEP_DAY) & (pl.col("date").dt.year() >= 2024))
        .join(solar.group_by("date").agg(pl.col("solar_mw").max().alias("mx")), on="date")
        .group_by("hour_ending").agg((pl.col("solar_mw") / pl.col("mx")).mean().alias("share_of_day_max"))
        .sort("hour_ending"))
print("summer 2024-2026 utility solar by hour ending, share of the day's max hour:")
print(prof.filter(pl.col("hour_ending").is_between(7, 22)).with_columns(pl.col("share_of_day_max").round(3)))

# %% 3c. How much could BTM PV have shaved the daily maximum? Redo X1's flatness with the max corrected
daily = px.daily_load_stats(load)
shape = px.summer_shape(daily)
spanel = shape.join(tsum, on=["weather_zone", "year"], how="inner").filter(pl.col("year") >= 2003)
ex_min = px.excess_vs_prebreak(spanel, "dmin_mean", "t_summer")
ex_max = px.excess_vs_prebreak(spanel, "dmax_mean", "t_summer")
x1 = (ex_max.filter(pl.col("weather_zone") == wl.TOTAL).select("year", pl.col("excess_mw").alias("dmax_excess"))
      .join(ex_min.filter(pl.col("weather_zone") == wl.TOTAL).select("year", pl.col("excess_mw").alias("dmin_excess")),
            on="year").filter(pl.col("year").is_between(2021, 2026)).sort("year"))
pv_base = pv_w.filter(pl.col("year") == BASE_YEAR)["total_mw"].item()
pv_last = pv_w.filter(pl.col("year") == pv_w["year"].max())
share_max = share_y.filter(pl.col("year") >= 2023)["share_of_max"].mean()
share_mid = share_y.filter(pl.col("year") >= 2023)["share_of_midday"].mean()
rows = []
for r in x1.iter_rows(named=True):
    y = r["year"]
    cap = pv_w.filter(pl.col("year") == min(y, pv_w["year"].max()))["total_mw"].item()
    d_cap = cap - pv_base
    # Upper bound: every added MW of reported PV at full nameplate at its best hour, times the solar shape at the
    # hour of the daily max (tracking plants; fixed rooftop falls off faster in the late afternoon).
    shave_cap = d_cap * share_max
    gap_row = split_n.filter((pl.col("weather_zone") == wl.TOTAL) & (pl.col("year") == y))
    mid_gap = gap_row["midday_gap_mw"].item() if gap_row.height else None
    shave_gap = max(-mid_gap, 0.0) * share_mid if mid_gap is not None else None
    rows.append({
        "year": y, "dmax_excess": r["dmax_excess"], "dmin_excess": r["dmin_excess"],
        "flatness_x1": r["dmin_excess"] / r["dmax_excess"],
        "pv_cap_used_year": min(y, pv_w["year"].max()), "pv_added_since_2019_mw": d_cap,
        "shave_upper_capacity_mw": shave_cap,
        "flatness_corrected_capacity": fl.corrected_flatness(r["dmin_excess"], r["dmax_excess"], shave_cap),
        "midday_gap_norm_mw": mid_gap, "shave_from_midday_gap_mw": shave_gap,
        "flatness_corrected_gap": (fl.corrected_flatness(r["dmin_excess"], r["dmax_excess"], shave_gap)
                                   if shave_gap is not None else None),
        "shave_to_reach_0_61": r["dmin_excess"] / 0.61 - r["dmax_excess"],
    })
corr = pl.DataFrame(rows)
corr.write_csv(out_path("x20_flatness_corrected.csv"))
print(f"share of max used: {share_max:.3f}; share of midday used: {share_mid:.3f}; PV 2019 {pv_base:.0f} MW")
print("X1 flatness with the daily max corrected for BTM PV (ERCOT):")
print(corr.with_columns(pl.selectors.float().round(2)))

# %% 4. Flat MW by zone (weather-normalized, solar-free split) and a figure
flat_tab = (split_n.filter(pl.col("year").is_in([2023, 2024, 2025, 2026]))
            .pivot(on="year", index="weather_zone", values="flat_mw"))
print("flat block F by zone, weather-normalized, vs 2019 (MW):")
print(flat_tab.with_columns(pl.selectors.float().round(0)))
dark_tab = (split_n.filter(pl.col("year").is_in([2025, 2026]))
            .pivot(on="year", index="weather_zone", values="dark_ratio"))
print("dark ratio (overnight / evening growth), normalized:")
print(dark_tab.with_columns(pl.selectors.float().round(2)))

fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True)
order = ["overnight", "midday", "afternoon", "evening"]
colors = {"overnight": "#1f3b73", "midday": "#e0a100", "afternoon": "#c0392b", "evening": "#6c5ce7"}
for ax, z in zip(axes.flat, SERIES, strict=True):
    sub = fl.growth_vs_base(summer_n, BASE_YEAR).filter((pl.col("weather_zone") == z) & (pl.col("year") >= 2017))
    for blk in order:
        s = sub.filter(pl.col("block") == blk).sort("year")
        ax.plot(s["year"], s["delta_mw"], marker="o", ms=3, lw=1.6, color=colors[blk], label=blk)
    ax.axhline(0, color="#999", lw=0.6)
    ax.set_title(z, fontsize=10)
    ax.tick_params(labelsize=8)
axes.flat[0].legend(fontsize=7, frameon=False)
fig.suptitle("Weather-normalized summer load by hour block, change vs 2019 (MW; Jun 1 - Sep 19)", fontsize=11)
fig.tight_layout()
fig.savefig(out_path("x20_block_growth_by_zone.png"), dpi=140)
plt.close(fig)
print("wrote", out_path("x20_block_growth_by_zone.png"))
