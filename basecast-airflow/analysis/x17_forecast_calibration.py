"""X17 — calibrate the peak forecast's P10/P90 band and bring X11's zone allocation into the forecast.

Run from the repo root: ``uv run --group analysis python analysis/x17_forecast_calibration.py`` (~1 min).
Reads the same tables as X7 (read-only) plus X11's share tables in ``analysis/out/`` (``x11_stock_allocation.csv``,
``x11_pipeline_shares.csv``, ``x11_x1_check.csv``; run ``analysis/x11_large_load_geography.py`` first) and X7's
``x7_backtest.csv`` / ``x7_forecast_zones.csv`` (run ``analysis/x7_peak_forecast.py`` first) for the checks.
Logic: ``basecast_pipelines/models/peak_forecast.py`` (X17 section, additive). Numbers go to
``docs/analysis/x17_forecast_calibration.md``.
"""

# %% Setup (same inputs and code path as analysis/x7_peak_forecast.py)
from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from analysis._common import ROOT, out_path
from basecast_pipelines.models import backtest as bt
from basecast_pipelines.models import large_load as ll
from basecast_pipelines.models import peak_excess as px
from basecast_pipelines.models import peak_forecast as pf
from basecast_pipelines.models import weather_load as wl

FIRST, FEATURE = 2003, "t_mean_3d"
TODAY = date(2026, 9, 26)
FORECAST_YEARS = [2027, 2028, 2029, 2030, 2031]
AS_OF_DATES = [date(2023, 5, 31), date(2023, 12, 31), date(2024, 5, 31), date(2024, 7, 31), date(2024, 12, 31),
               date(2025, 5, 31), date(2025, 12, 31), date(2026, 5, 31)]
CALIB_BEFORE = date(2025, 1, 1)  # time split: choose on as-of dates before 2025, check on the rest
LAST_ACTUAL = 2026
BATCH_ZERO = date(2026, 4, 1)
N = 10_000
ZONES = wl.WEATHER_ZONES

load = wl.load_hourly_load()
weather = wl.load_hourly_weather()
cv = ll.load_chart_values()
headlines = ll.load_headlines()
manifests = ll.load_manifests(ROOT / "data")
actual_tbl = bt.actual_summer_peaks()
actual = dict(zip(actual_tbl["year"].to_list(), actual_tbl["actual_mw"].to_list(), strict=True))

daily_w = wl.weather_daily(weather)
weights = wl.zone_weights(load, range(FIRST, pf.PRE_BREAK_UNTIL + 1))
daily_all = pl.concat([daily_w.drop("n_hours"), wl.system_weather_daily(daily_w, weights).drop("n_hours")])
features = wl.summer_weather_features(daily_all)
peaks = wl.summer_peaks(load)
panel = (peaks.select("weather_zone", "year", "peak_mw", "peak_local")
         .join(features, on=["weather_zone", "year"], how="inner").filter(pl.col("year") >= FIRST))
ercot = panel.filter(pl.col("weather_zone") == wl.TOTAL).sort("year")
peak_months = {r["year"]: r["peak_local"].strftime("%Y-%m") for r in ercot.iter_rows(named=True)}
x_actual = dict(zip(ercot["year"].to_list(), ercot[FEATURE].to_list(), strict=True))
load_peak = dict(zip(ercot["year"].to_list(), ercot["peak_mw"].to_list(), strict=True))
fit = pf.fit_organic(ercot, FEATURE)
excess = {y: load_peak[y] - fit.mean(y, x_actual[y]) for y in load_peak if y > pf.PRE_BREAK_UNTIL}
roll = pf.rolling_origin_errors(ercot, FEATURE, 2011, pf.PRE_BREAK_UNTIL)
RESID_SD = float(np.sqrt((roll["error_mw"] ** 2).mean()))

wide = ll.in_service_wide(cv)
vintages = ll.pick_vintages(ll.check_headlines(ll.chart_checks(wide), headlines), manifests)
bars = wide.join(vintages.select("vintage", "document", "page", "chart_title",
                                 pl.col("a2e_stock_mw").alias("base_a2e_mw")),
                 on=["vintage", "document", "page", "chart_title"], how="inner")
realized = pf.realized_year_end(vintages)
print(f"RESID_SD (X7) {RESID_SD:.0f} MW; in-sample sigma {fit.sigma:.0f} MW")


def inputs_asof(as_of: date, *, vintage_before: date | None = None) -> dict:
    """X7's ``inputs_asof`` (copied, unchanged logic)."""
    last_summer = as_of.year if as_of.month >= 9 else as_of.year - 1
    points = pf.a2e_points(cv, vintages, bars, as_of)
    months = {y: m for y, m in peak_months.items() if y <= last_summer}
    fac = pf.observed_factor(cv, points, months, as_of)
    ratios = pf.incremental_ratios(bars, realized, as_of)
    if fac.is_empty():
        first = min(d for d in vintages["report_date"].to_list() if not pf.observed_factor(
            cv, pf.a2e_points(cv, vintages, bars, d), {y: m for y, m in peak_months.items() if y <= d.year - 1},
            d).is_empty())
        fac = pf.observed_factor(cv, pf.a2e_points(cv, vintages, bars, first),
                                 {y: m for y, m in peak_months.items() if y <= first.year - 1}, first)
    if ratios.is_empty():
        first = realized.filter(pl.col("known_from") > as_of)["known_from"].min()
        ratios = pf.incremental_ratios(bars, realized, first)
    factor = float(fac["factor"].mean())
    u = pf.unattributed({y: v for y, v in excess.items() if y <= last_summer}, points, months, factor)
    decks = pf.published_by(vintages, as_of).drop_nulls("a2e_stock_mw")
    if vintage_before is not None:
        decks = decks.filter(pl.col("vintage") < vintage_before)
    v = decks.sort("vintage").row(-1, named=True)
    vbars = bars.filter((pl.col("vintage") == v["vintage"]) & (pl.col("document") == v["document"]))
    return {"as_of": as_of, "points": points, "factor": factor, "ratios": ratios["ratio"].to_list(),
            "u": u["u_mw"].to_list(), "vintage": v["vintage"], "base": v["a2e_stock_mw"], "bars": vbars}


def last_summer_of(as_of: date) -> int:
    return as_of.year if as_of.month >= 9 else as_of.year - 1


# %% 1. Organic residual by horizon: h-year-ahead rolling-origin RMSE at median weather, pre-break targets only
# External to the backtest cells (targets <= 2019), so nothing here is fit on the 18 cells.
H_RMSE = pf.horizon_rmse(ercot, FEATURE, range(1, 6))
print(H_RMSE.with_columns(pl.selectors.float().round(1)))
H_RMSE.write_csv(out_path("x17_horizon_rmse.csv"))
assert abs(H_RMSE.filter(pl.col("horizon") == 1)["rmse_mw"].item() - RESID_SD) < 1e-6  # h=1 is X7's 1,814 MW

# %% 2. Band methods on X7's 18 backtest cells
# Structural options (no parameter fit on the cells) from X7's review item 2, plus the X7 method itself.
METHODS = {
    "x7": dict(sim="x7"),
    "x7_new_rng": dict(resid="flat", u_mode="resample", rho=0.0),
    "u_normal": dict(resid="flat", u_mode="normal", rho=0.0),
    "h_rmse": dict(resid="horizon", u_mode="resample", rho=0.0),
    "h_rmse+u_uniform": dict(resid="horizon", u_mode="uniform", rho=0.0),
    "h_rmse+u_normal": dict(resid="horizon", u_mode="normal", rho=0.0),
    "h_rmse+u_normal+rho0.5": dict(resid="horizon", u_mode="normal", rho=0.5),
}


def run_method(spec: dict, inp: dict, years: list[int], horizons: list[int], *, ll_paths=None, seed: int = 7):
    promised = pf.promised_by_year(inp["bars"], range(inp["vintage"].year, max(years) + 1))
    if spec.get("sim") == "x7":
        return pf.simulate(fit, years, base=inp["base"], vintage=inp["vintage"], promised=promised,
                           ratios=inp["ratios"], factor=inp["factor"], u_values=inp["u"], resid_sd=RESID_SD, n=N,
                           seed=seed)
    if ll_paths is None:
        ll_paths = pf.ratio_paths(inp["base"], inp["vintage"], promised, inp["ratios"], inp["factor"], years)
    resid = RESID_SD if spec["resid"] == "flat" else pf.resid_sd_by_year(H_RMSE, horizons)
    return pf.simulate_calibrated(fit, years, ll_paths=ll_paths, u_values=inp["u"], resid_sd=resid,
                                  u_mode=spec["u_mode"], u_sd_floor=fit.sigma, rho_ll_u=spec["rho"], n=N, seed=seed)


cells = []
for as_of in AS_OF_DATES:
    years = [y for y in range(last_summer_of(as_of) + 1, LAST_ACTUAL + 1) if bt.summers_ahead(as_of, y) >= 1]
    horizons = [bt.summers_ahead(as_of, y) for y in years]
    inp = inputs_asof(as_of)
    for name, spec in METHODS.items():
        d = run_method(spec, inp, years, horizons)
        q = np.quantile(d["total"], [0.1, 0.5, 0.9], axis=0)
        for j, y in enumerate(years):
            cells.append({"method": name, "as_of": as_of, "target_year": y, "horizon": horizons[j],
                          "p10_mw": q[0, j], "p50_mw": q[1, j], "p90_mw": q[2, j], "actual_mw": actual[y],
                          "n_u_values": len(inp["u"])})
cells = pl.DataFrame(cells).with_columns(
    (100 * (pl.col("p50_mw") / pl.col("actual_mw") - 1)).alias("error_pct"),
    pl.col("actual_mw").is_between(pl.col("p10_mw"), pl.col("p90_mw")).alias("in_band"),
    (pl.col("as_of") < CALIB_BEFORE).alias("calib"),
)

# Check: the "x7" rows reproduce X7's own backtest (method rows of x7_backtest.csv)
x7_bt = pl.read_csv(out_path("x7_backtest.csv"), try_parse_dates=True).filter(pl.col("variant") == "method")
chk = cells.filter(pl.col("method") == "x7").join(x7_bt.select("as_of", "target_year", pl.col("p10_mw").alias("x7_p10"),
                                                               pl.col("p90_mw").alias("x7_p90")),
                                                  on=["as_of", "target_year"])
assert chk.height == 18, chk.height
max_gap = float(chk.select(((pl.col("p10_mw") - pl.col("x7_p10")).abs().max()).alias("g"))["g"].item())
print(f"x7 method reproduced on {chk.height} cells, max |P10 gap| {max_gap:.6f} MW")


def band_stats(df: pl.DataFrame, k: float = 1.0) -> pl.DataFrame:
    """Scale each cell's band around its P50 by ``k`` (affine: the quantiles of the scaled draws) and score it."""
    return df.with_columns(
        (pl.col("p50_mw") + k * (pl.col("p10_mw") - pl.col("p50_mw"))).alias("lo"),
        (pl.col("p50_mw") + k * (pl.col("p90_mw") - pl.col("p50_mw"))).alias("hi"),
    ).with_columns(
        pl.col("actual_mw").is_between(pl.col("lo"), pl.col("hi")).alias("in_k"),
        ((pl.col("hi") - pl.col("lo")) / 2 / pl.col("p50_mw") * 100).alias("half_width_pct"),
        ((pl.col("hi") - pl.col("lo")) / 1000).alias("width_gw"),
        pl.struct("lo", "hi", "actual_mw").map_elements(
            lambda r: pf.interval_score(r["lo"], r["hi"], r["actual_mw"]) / 1000, return_dtype=pl.Float64
        ).alias("interval_score_gw"),
    )


def summarize_cells(df: pl.DataFrame, by: list[str], k: float = 1.0) -> pl.DataFrame:
    return (band_stats(df, k).group_by(by).agg(
        pl.len().alias("n"), pl.col("in_k").mean().alias("coverage"), pl.col("in_k").sum().alias("inside"),
        pl.col("half_width_pct").mean().alias("half_width_pct"), pl.col("width_gw").mean().alias("width_gw"),
        pl.col("interval_score_gw").mean().alias("interval_score_gw"), pl.col("error_pct").abs().mean().alias("mape"),
        ((pl.col("actual_mw") > pl.col("hi")).sum()).alias("above_p90"),
        ((pl.col("actual_mw") < pl.col("lo")).sum()).alias("below_p10"),
    ).sort(by))


methods_all = summarize_cells(cells, ["method"]).with_columns(pl.lit("all 18").alias("cells"))
methods_cal = summarize_cells(cells.filter(pl.col("calib")), ["method"]).with_columns(
    pl.lit("calibration: as-of < 2025 (14)").alias("cells"))
methods_test = summarize_cells(cells.filter(~pl.col("calib")), ["method"]).with_columns(
    pl.lit("test: as-of >= 2025 (4)").alias("cells"))
methods_tab = pl.concat([methods_all, methods_cal, methods_test])
print(methods_tab.with_columns(pl.selectors.float().round(3)))
methods_tab.write_csv(out_path("x17_band_methods.csv"))
print(summarize_cells(cells, ["method", "horizon"]).filter(pl.col("method").is_in(["x7", "h_rmse+u_normal"]))
      .with_columns(pl.selectors.float().round(3)))

# %% 3. Selection protocol
# (a) Time split: pick the structural method by mean interval score (80% interval) on the calibration cells, then a
#     band scale k on the same cells; check coverage on the as-of >= 2025 cells.
# (b) Leave one target summer out (the cells of one summer share its actual, so they are the dependent unit) and
#     leave one as-of date out: pick k without the held-out cells, score the held-out cells.
STRUCTURAL = [m for m in METHODS if m not in ("x7", "x7_new_rng")]
best = (methods_cal.filter(pl.col("method").is_in(STRUCTURAL)).sort("interval_score_gw").row(0, named=True))
BEST = best["method"]
print(f"structural method with the lowest calibration interval score: {BEST} ({best['interval_score_gw']:.3f} GW)")
K_GRID = [round(v, 2) for v in np.arange(0.8, 2.51, 0.05)]


def pick_k(df: pl.DataFrame) -> float:
    """k in K_GRID with the lowest mean interval score; ties go to the smallest k (the narrowest band)."""
    scores = [(float(band_stats(df, k)["interval_score_gw"].mean()), k) for k in K_GRID]
    return min(scores)[1]


best_cells = cells.filter(pl.col("method") == BEST)
k_time = pick_k(best_cells.filter(pl.col("calib")))
sel_rows = [{"protocol": "time split (choose on as-of < 2025)", "held_out": "as-of >= 2025", "k": k_time,
             **summarize_cells(best_cells.filter(~pl.col("calib")), ["method"], k_time).drop("method").row(0, named=True)}]
held = []
for unit, col in (("target summer", "target_year"), ("as-of date", "as_of")):
    for key in best_cells[col].unique().sort().to_list():
        k_u = pick_k(best_cells.filter(pl.col(col) != key))
        held.append(band_stats(best_cells.filter(pl.col(col) == key), k_u).with_columns(
            pl.lit(f"leave one {unit} out").alias("protocol"), pl.lit(k_u).alias("k"), pl.lit(str(key)).alias("held")))
held = pl.concat(held)
for proto, g in held.group_by("protocol", maintain_order=True):
    sel_rows.append({"protocol": proto[0], "held_out": "each in turn", "k": float("nan"), "n": g.height,
                     "coverage": g["in_k"].mean(), "inside": int(g["in_k"].sum()),
                     "half_width_pct": g["half_width_pct"].mean(), "width_gw": g["width_gw"].mean(),
                     "interval_score_gw": g["interval_score_gw"].mean(), "mape": g["error_pct"].abs().mean(),
                     "above_p90": int((g["actual_mw"] > g["hi"]).sum()), "below_p10": int((g["actual_mw"] < g["lo"]).sum())})
print(held.group_by("protocol", "held").agg(pl.col("k").first(), pl.len(), pl.col("in_k").sum()).sort("protocol", "held"))
k_all = pick_k(best_cells)
for label, k in (("structural only (k = 1)", 1.0), (f"k fit on all 18 cells (in-sample, k = {k_all})", k_all)):
    sel_rows.append({"protocol": label, "held_out": "none", "k": k,
                     **summarize_cells(best_cells, ["method"], k).drop("method").row(0, named=True)})
sel = pl.DataFrame(sel_rows)
print(sel.with_columns(pl.selectors.float().round(3)))
sel.write_csv(out_path("x17_selection.csv"))
held.select("protocol", "held", "k", "as_of", "target_year", "horizon", "lo", "p50_mw", "hi", "actual_mw", "in_k").write_csv(
    out_path("x17_heldout_cells.csv"))
band_stats(cells, 1.0).write_csv(out_path("x17_backtest_cells.csv"))
K_FINAL = k_time  # the band scale that goes to the forecast: chosen on the calibration cells only
print(f"K_FINAL = {K_FINAL} (time split); k on all 18 would be {k_all}; per-summer LOSO ks: "
      f"{sorted(set(held.filter(pl.col('protocol') == 'leave one target summer out')['k'].to_list()))}")

# %% 4. Forecast 2027-2031 as of today: Mar 2026 deck (default), Jun 2026 deck, approvals pace; before / after
now = inputs_asof(TODAY)
now_pre = inputs_asof(TODAY, vintage_before=BATCH_ZERO)
H_NOW = [bt.summers_ahead(TODAY, y) for y in FORECAST_YEARS]
dec = dict(zip(realized["target_year"].to_list(), realized["realized_a2e_mw"].to_list(), strict=True))
PACES = [dec[y + 1] - dec[y] for y in (2022, 2023, 2024)]
pts_now = list(zip(now["points"]["month_idx"].to_list(), now["points"]["a2e_mw"].to_list(), strict=True))
A2E_JUN26 = pf.interpolate(pts_now, pf.month_index("2026-06"))
pace_ll = pf.pace_paths(A2E_JUN26, pf.month_index("2026-06"), PACES, now["factor"], FORECAST_YEARS)
print(f"horizons {H_NOW}; paces {[round(p) for p in PACES]} MW/yr; A2E Jun 2026 {A2E_JUN26:.0f} MW")

VARIANTS = {"deck_pre_batch_zero_2026_03": (now_pre, None), "deck_latest_2026_06": (now, None),
            "approvals_pace": (now, pace_ll)}
fc_rows, draws_by_variant = [], {}
for vname, (inp, paths) in VARIANTS.items():
    specs = {"x7_band": METHODS["x7_new_rng"] if paths is not None else METHODS["x7"], "x17_band": METHODS[BEST]}
    for band, spec in specs.items():
        d = run_method(spec, inp, FORECAST_YEARS, H_NOW, ll_paths=paths)
        if band == "x17_band":
            d = {**d, "total": pf.scale_band(d["total"], K_FINAL)}
            draws_by_variant[vname] = d
        for layer in ("organic", "large_load", "unattributed", "total"):
            q = np.quantile(d[layer], [0.1, 0.5, 0.9], axis=0)
            for j, y in enumerate(FORECAST_YEARS):
                fc_rows.append({"variant": vname, "band": band, "layer": layer, "year": y, "horizon": H_NOW[j],
                                "p10_mw": q[0, j], "p50_mw": q[1, j], "p90_mw": q[2, j]})
fc = pl.DataFrame(fc_rows)
fc.write_csv(out_path("x17_forecast_ercot.csv"))
tot = (fc.filter(pl.col("layer") == "total").with_columns((pl.selectors.ends_with("_mw") / 1000).round(1))
       .pivot(on="band", index=["variant", "year"], values=["p10_mw", "p50_mw", "p90_mw"]).sort("variant", "year"))
print(tot)

# X7's own CSV for the deck variants must match the x7_band rows (same function, same seed)
x7_fc = pl.read_csv(out_path("x7_forecast_ercot.csv"))
x7_map = {"pre_batch_zero_deck_2026_03": "deck_pre_batch_zero_2026_03", "latest_deck_2026_06": "deck_latest_2026_06"}
cmp7 = (x7_fc.with_columns(pl.col("variant").replace_strict(x7_map)).join(
    fc.filter(pl.col("band") == "x7_band"), on=["variant", "layer", "year"], suffix="_x17"))
print("max |x7 CSV - x7_band| (MW):", float((cmp7["p50_mw"] - cmp7["p50_mw_x17"]).abs().max()),
      float((cmp7["p90_mw"] - cmp7["p90_mw_x17"]).abs().max()))

# %% 5. Zones: X11's three-part split (stock / new approvals / U) with Dirichlet uncertainty
stock_tab = pl.read_csv(out_path("x11_stock_allocation.csv"))
pipe_tab = pl.read_csv(out_path("x11_pipeline_shares.csv"))
chk_tab = pl.read_csv(out_path("x11_x1_check.csv"))


def col_split(tab: pl.DataFrame, col: str) -> dict[str, float]:
    return dict(zip(tab["weather_zone"].to_list(), tab[col].to_list(), strict=True))


S_STOCK = col_split(stock_tab, "central")
STOCK_ALTS = [col_split(stock_tab, c) for c in ("raked_f1_x1", "raked_f0", "bz_base_unraked")]
S_PIPE = col_split(pipe_tab, "central")
PIPE_ALTS = [col_split(pipe_tab, c) for c in ("bz_studied_2032", "bz_tracking_jun2026", "llis_studied_may2026",
                                              "tsp_rfi_2030_area", "tsp_rfi_2030_pop", "qsa_q4_2026")]
# U: X11's central is the mean 2023-26 zone excess minus the mean allocated LL (clipped); its alternatives are the
# same residual split summer by summer
mean_ex = {z: float(chk_tab.filter(pl.col("weather_zone") == z)["coincident_excess_mw"].mean()) for z in ZONES}
mean_ll = {z: float(chk_tab.filter(pl.col("weather_zone") == z)["allocated_ll_at_peak_mw"].mean()) for z in ZONES}


def clip_shares(v: dict[str, float]) -> dict[str, float]:
    pos = {z: max(v.get(z, 0.0), 0.0) for z in ZONES}
    s = sum(pos.values())
    return {z: pos[z] / s for z in ZONES}


S_U = clip_shares({z: mean_ex[z] - mean_ll[z] for z in ZONES})
U_ALTS = [clip_shares({r["weather_zone"]: r["coincident_excess_mw"] - r["allocated_ll_at_peak_mw"]
                       for r in chk_tab.filter(pl.col("year") == y).iter_rows(named=True)}) for y in (2023, 2024, 2025, 2026)]
CONC = {"stock": pf.dirichlet_concentration(S_STOCK, STOCK_ALTS, ZONES),
        "pipeline": pf.dirichlet_concentration(S_PIPE, PIPE_ALTS, ZONES),
        "unattributed": pf.dirichlet_concentration(S_U, U_ALTS, ZONES)}
print("Dirichlet concentration:", {k: round(v, 1) for k, v in CONC.items()})
rng_sh = np.random.default_rng(17)
SH = {"stock": pf.share_draws(S_STOCK, CONC["stock"], N, rng_sh, ZONES),
      "pipeline": pf.share_draws(S_PIPE, CONC["pipeline"], N, rng_sh, ZONES),
      "unattributed": pf.share_draws(S_U, CONC["unattributed"], N, rng_sh, ZONES)}
share_rows = []
for part, central, alts in (("stock", S_STOCK, STOCK_ALTS), ("pipeline", S_PIPE, PIPE_ALTS), ("unattributed", S_U, U_ALTS)):
    q = np.quantile(SH[part], [0.1, 0.5, 0.9], axis=0)
    for i, z in enumerate(ZONES):
        share_rows.append({"part": part, "weather_zone": z, "central": central[z], "concentration": CONC[part],
                           "draw_p10": q[0, i], "draw_p50": q[1, i], "draw_p90": q[2, i],
                           "x11_low": min([central[z]] + [a[z] for a in alts]),
                           "x11_high": max([central[z]] + [a[z] for a in alts])})
share_tab = pl.DataFrame(share_rows)
print(share_tab.with_columns(pl.selectors.float().round(3)))
share_tab.write_csv(out_path("x17_zone_shares.csv"))

# Zone organic: coincident contribution fits (X7) with the zone's own h-year-ahead RMSE at median weather
coinc = px.coincident_zone_loads(load)
cpanel = coinc.join(ercot.select("year", FEATURE), on="year").filter(pl.col("year") >= FIRST)
zpanels = {z: cpanel.filter(pl.col("weather_zone") == z).rename({"coincident_mw": "peak_mw"}) for z in ZONES}
zone_fits = {z: pf.fit_organic(zpanels[z], FEATURE) for z in ZONES}
zone_rmse = {z: pf.horizon_rmse(zpanels[z], FEATURE, range(1, 6)) for z in ZONES}
pl.concat([t.with_columns(pl.lit(z).alias("weather_zone")) for z, t in zone_rmse.items()]).write_csv(
    out_path("x17_zone_horizon_rmse.csv"))

zrows = []
for vname, (inp, _) in VARIANTS.items():
    d = draws_by_variant[vname]
    stock_ll = inp["factor"] * (inp["base"] if vname != "approvals_pace" else A2E_JUN26)
    extra = pf.allocate_extra(d["large_load"], d["unattributed"], stock_ll, SH["stock"], SH["pipeline"],
                              SH["unattributed"])
    assert np.allclose(extra.sum(axis=2), d["large_load"] + d["unattributed"])
    for i, z in enumerate(ZONES):
        org = pf.organic_draws(zone_fits[z], FORECAST_YEARS, N, np.random.default_rng(sum(map(ord, z))),
                               resid_sd=pf.resid_sd_by_year(zone_rmse[z], H_NOW))
        tot_z = pf.scale_band(org + extra[:, :, i], K_FINAL)
        qo, qe, qt = (np.quantile(a, [0.1, 0.5, 0.9], axis=0) for a in (org, extra[:, :, i], tot_z))
        for j, y in enumerate(FORECAST_YEARS):
            zrows.append({"variant": vname, "weather_zone": z, "year": y, "organic_p50_mw": qo[1, j],
                          "ll_u_p10_mw": qe[0, j], "ll_u_p50_mw": qe[1, j], "ll_u_p90_mw": qe[2, j],
                          "p10_mw": qt[0, j], "p50_mw": qt[1, j], "p90_mw": qt[2, j]})
zfc = pl.DataFrame(zrows)
zfc.write_csv(out_path("x17_forecast_zones.csv"))
ercot_p50 = fc.filter((pl.col("band") == "x17_band") & (pl.col("layer") == "total")).select("variant", "year", pl.col("p50_mw").alias("ercot_p50"))
zsum = (zfc.group_by("variant", "year").agg(pl.col("p50_mw").sum().alias("sum_zone_p50")).join(ercot_p50, on=["variant", "year"])
        .with_columns((100 * (pl.col("sum_zone_p50") / pl.col("ercot_p50") - 1)).alias("gap_pct")).sort("variant", "year"))
print(zsum.with_columns(pl.selectors.float().round(2)))

# Compare with X7's fixed share and X11's central re-run (Mar 2026 deck)
x7z = pl.read_csv(out_path("x7_forecast_zones.csv")).rename({"zone": "weather_zone"})
x11z = pl.read_csv(out_path("x11_zone_forecast_compare.csv"))
zcmp = (zfc.filter(pl.col("variant") == "deck_pre_batch_zero_2026_03")
        .select("weather_zone", "year", "organic_p50_mw", "ll_u_p50_mw", pl.col("p10_mw").alias("x17_p10"),
                pl.col("p50_mw").alias("x17_p50"), pl.col("p90_mw").alias("x17_p90"))
        .join(x7z.select("weather_zone", "year", pl.col("p10_mw").alias("x7_p10"), pl.col("p50_mw").alias("x7_p50"),
                         pl.col("p90_mw").alias("x7_p90")), on=["weather_zone", "year"])
        .join(x11z.select("weather_zone", "year", "x11_p50", "p50_alloc_low", "p50_alloc_high"), on=["weather_zone", "year"])
        .with_columns((pl.col("x17_p50") - pl.col("x7_p50")).alias("d_p50_vs_x7"),
                      (pl.col("x17_p50") - pl.col("x11_p50")).alias("d_p50_vs_x11"),
                      ((pl.col("x17_p90") - pl.col("x17_p10")) - (pl.col("x7_p90") - pl.col("x7_p10"))).alias("d_width_vs_x7"))
        .sort("year", "weather_zone"))
print(zcmp.filter(pl.col("year").is_in([2027, 2030])).with_columns(pl.selectors.float().round(0)))
zcmp.write_csv(out_path("x17_zone_compare.csv"))
print("figures and tables written to analysis/out/x17_*")
