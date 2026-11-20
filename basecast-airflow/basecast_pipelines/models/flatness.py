"""Exploration X20: is X1's "flat" load excess real, or partly behind-the-meter (BTM) solar shaving the daytime peak?

X1 read the post-2020 summer excess as flat 24-hour load because the average day's minimum rose as much as its
maximum. Rooftop solar lowers *metered* load in daylight, so it would hold the daily maximum down and make that
ratio look flatter than the load really is. The checks here compare hours rooftop solar cannot touch:

- **Hour blocks** by ERCOT hour ending (America/Chicago, the operating day's local clock): overnight HE 1-5 and
  evening HE 21-24 are dark in a Texas summer; midday HE 11-15 is where rooftop output is highest; HE 16-19 holds
  the summer daily maximum.
- **Solar-free split:** two dark blocks with different baseline levels solve X1's flat + proportional equations
  (:func:`peak_excess.flat_block`) without any daylight hour. The split then predicts the midday change; the
  midday shortfall against that prediction is the most daylight load that could be hidden (BTM solar, or anything
  else that grew less at midday).
- **EIA-861 distributed PV:** the net-metered and non-net-metered PV capacity reported by Texas utilities in the
  ERCOT balancing authority (``BA Code`` ``ERCO``), to size how much of the daily maximum rooftop solar could shave.

DST: blocks are sets of hour-ending labels, so a 23-hour spring day averages the hours it has (HE 3 is skipped) and
the 25-hour fall day counts HE 2 twice; summer never crosses a DST change. Pure functions on DataFrames; the
loaders at the bottom are thin (lake files and the ``weather_load`` DB loaders).
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import polars as pl

from basecast_pipelines.models import peak_excess as pe

# Hour-ending ranges, inclusive, on the operating day's local clock.
HOUR_BLOCKS: dict[str, tuple[int, int]] = {
    "overnight": (1, 5),
    "midday": (11, 15),
    "afternoon": (16, 19),
    "evening": (21, 24),
}
SOLAR_FREE = ("overnight", "evening")
SUMMER_MONTHS = pe.SHAPE_MONTHS
SUMMER_LAST_SEP_DAY = pe.SHAPE_LAST_DAY
ERCOT_BA = "ERCO"


# --- hour blocks ---------------------------------------------------------------------------------------------


def block_of(hour_ending: pl.Expr, blocks: Mapping[str, tuple[int, int]] = HOUR_BLOCKS) -> pl.Expr:
    """The block name of an hour-ending label (null outside every block). Blocks must not overlap."""
    expr = None
    for name, (lo, hi) in blocks.items():
        cond = hour_ending.is_between(lo, hi)
        expr = pl.when(cond).then(pl.lit(name)) if expr is None else expr.when(cond).then(pl.lit(name))
    return expr.otherwise(pl.lit(None, dtype=pl.String))


def block_daily(load: pl.DataFrame, blocks: Mapping[str, tuple[int, int]] = HOUR_BLOCKS) -> pl.DataFrame:
    """Hourly zone load (``weather_zone``, ``operating_date``, ``hour_ending``, ``mw``) -> one row per zone,
    operating day and block: the block's mean MW and its number of hourly values."""
    return (
        load.with_columns(block_of(pl.col("hour_ending"), blocks).alias("block"))
        .drop_nulls("block")
        .group_by("weather_zone", pl.col("operating_date").alias("date"), "block")
        .agg(pl.col("mw").mean().alias("mw"), pl.len().alias("n_hours"))
        .sort("weather_zone", "date", "block")
    )


def in_summer_window(date_col: str = "date", months: Sequence[int] = SUMMER_MONTHS,
                     last_sep_day: int = SUMMER_LAST_SEP_DAY) -> pl.Expr:
    """June 1 - September ``last_sep_day`` (X1's shape window, the same days every year)."""
    m = pl.col(date_col).dt.month()
    return m.is_in(list(months)) & ~((m == 9) & (pl.col(date_col).dt.day() > last_sep_day))


def summer_block_means(daily: pl.DataFrame, value: str = "mw", months: Sequence[int] = SUMMER_MONTHS,
                       last_sep_day: int = SUMMER_LAST_SEP_DAY) -> pl.DataFrame:
    """Per zone, year and block: the mean of ``value`` over the summer window's days, and the day count."""
    return (
        daily.filter(in_summer_window("date", months, last_sep_day))
        .group_by("weather_zone", pl.col("date").dt.year().alias("year"), "block")
        .agg(pl.col(value).mean().alias(value), pl.len().alias("days"))
        .sort("weather_zone", "year", "block")
    )


def growth_vs_base(summer: pl.DataFrame, base_year: int, value: str = "mw") -> pl.DataFrame:
    """Per zone, year and block: ``base_mw`` (the ``base_year`` value), ``delta_mw`` and ``delta_pct``."""
    base = summer.filter(pl.col("year") == base_year).select("weather_zone", "block", pl.col(value).alias("base_mw"))
    return (
        summer.join(base, on=["weather_zone", "block"], how="inner")
        .with_columns((pl.col(value) - pl.col("base_mw")).alias("delta_mw"))
        .with_columns((100 * pl.col("delta_mw") / pl.col("base_mw")).alias("delta_pct"))
        .sort("weather_zone", "block", "year")
    )


def solar_free_split(growth: pl.DataFrame, *, low: str = "overnight", high: str = "evening",
                     daylight: Sequence[str] = ("midday", "afternoon")) -> pl.DataFrame:
    """Split each zone-year's growth into a flat block ``F`` and a proportional scaling ``g`` using only the two
    dark blocks (``low`` has the lower baseline), then predict the ``daylight`` blocks' change from the split.

    ``growth`` has ``weather_zone``, ``year``, ``block``, ``base_mw`` and ``delta_mw`` (:func:`growth_vs_base`).
    Returns one row per zone and year with ``flat_mw``, ``proportional_pct``, ``dark_ratio`` (delta ``low`` /
    delta ``high``: 1 = flat, base ``low``/``high`` = proportional) and, per daylight block, the predicted change
    and ``<block>_gap_mw`` = actual - predicted (negative: that block grew less than the dark hours imply)."""
    wide_d = growth.pivot(on="block", index=["weather_zone", "year"], values="delta_mw")
    wide_b = growth.pivot(on="block", index=["weather_zone", "year"], values="base_mw")
    rows = []
    base = {(r["weather_zone"], r["year"]): r for r in wide_b.iter_rows(named=True)}
    for r in wide_d.iter_rows(named=True):
        b = base[(r["weather_zone"], r["year"])]
        if r.get(low) is None or r.get(high) is None or b[low] == b[high]:
            continue
        flat, g = pe.flat_block(r[high], r[low], b[high], b[low])
        out = {
            "weather_zone": r["weather_zone"], "year": r["year"],
            f"{low}_delta_mw": r[low], f"{high}_delta_mw": r[high],
            "flat_mw": flat, "proportional_pct": 100 * g,
            "dark_ratio": r[low] / r[high] if r[high] else None,
            "proportional_ratio": b[low] / b[high],
        }
        for blk in daylight:
            if r.get(blk) is None or b.get(blk) is None:
                continue
            pred = g * b[blk] + flat
            out[f"{blk}_delta_mw"] = r[blk]
            out[f"{blk}_pred_mw"] = pred
            out[f"{blk}_gap_mw"] = r[blk] - pred
        rows.append(out)
    return pl.DataFrame(rows).sort("weather_zone", "year") if rows else pl.DataFrame()


def corrected_flatness(dmin_excess: float, dmax_excess: float, shave_at_max: float) -> float:
    """X1's flatness (dmin excess / dmax excess) with the daily maximum's excess raised by ``shave_at_max`` MW
    of load that BTM solar may have hidden at the hour of the daily maximum."""
    return dmin_excess / (dmax_excess + shave_at_max)


# --- joining blocks into the X12 daily panel -----------------------------------------------------------------


def add_blocks_to_panel(panel: pl.DataFrame, blocks: pl.DataFrame, prefix: str = "b_") -> pl.DataFrame:
    """X12's daily panel (``weather_normalized.daily_panel``) plus one column per block (``b_overnight``...), so
    X12's model can normalize a block's daily mean exactly as it normalizes ``dmean``. Days missing a block get
    a null there."""
    wide = blocks.pivot(on="block", index=["weather_zone", "date"], values="mw")
    wide = wide.rename({c: f"{prefix}{c}" for c in wide.columns if c not in ("weather_zone", "date")})
    return panel.join(wide, on=["weather_zone", "date"], how="left")


# --- solar output shape (ERCOT fuel mix, utility-scale) ------------------------------------------------------


def solar_hourly_local(fuel: pl.DataFrame, tz: str = "America/Chicago") -> pl.DataFrame:
    """15-minute fuel-mix rows (``interval_start_utc``, ``fuel``, ``generation_mwh``) -> hourly solar MW with the
    local operating date and hour ending (the interval's local start hour + 1)."""
    return (
        fuel.filter(pl.col("fuel") == "solar")
        .with_columns(pl.col("interval_start_utc").dt.truncate("1h").alias("hour_start_utc"))
        .group_by("hour_start_utc")
        .agg(pl.col("generation_mwh").sum().alias("solar_mw"), pl.len().alias("n"))
        .filter(pl.col("n") == 4)
        .with_columns(pl.col("hour_start_utc").dt.convert_time_zone(tz).alias("_local"))
        .with_columns(pl.col("_local").dt.date().alias("date"), (pl.col("_local").dt.hour() + 1).alias("hour_ending"))
        .select("date", "hour_ending", "solar_mw")
        .sort("date", "hour_ending")
    )


def solar_share_at(solar: pl.DataFrame, hours: pl.DataFrame, ref_block: tuple[int, int] = (11, 15)) -> pl.DataFrame:
    """For each day in ``hours`` (``date``, ``hour_ending`` = the hour of interest, e.g. the daily load max), the
    solar output in that hour divided by the day's mean output over ``ref_block`` (midday) and by the day's
    maximum hourly output."""
    lo, hi = ref_block
    ref = solar.filter(pl.col("hour_ending").is_between(lo, hi)).group_by("date").agg(
        pl.col("solar_mw").mean().alias("midday_mw"))
    top = solar.group_by("date").agg(pl.col("solar_mw").max().alias("max_mw"))
    return (
        hours.join(solar, on=["date", "hour_ending"], how="inner")
        .join(ref, on="date").join(top, on="date")
        .with_columns((pl.col("solar_mw") / pl.col("midday_mw")).alias("share_of_midday"),
                      (pl.col("solar_mw") / pl.col("max_mw")).alias("share_of_max"))
        .sort("date")
    )


# --- EIA-861 distributed PV ----------------------------------------------------------------------------------


def _header_anchor(raw: pl.DataFrame, scan_rows: int = 8, scan_cols: int = 4) -> tuple[int, int, int]:
    """(first group-header row, leaf-header row, first column) of an EIA-861 sheet: the leaf row holds ``Year``
    (``Data Year`` in 2015) and the group rows start at ``Utility Characteristics`` in the same column. Title and
    note rows above, and blank or note columns on the left (the 2025 early release), are skipped."""
    for c, col in enumerate(raw.columns[:scan_cols]):
        cells = raw.get_column(col).cast(pl.String).to_list()[:scan_rows]
        norm = [(v or "").strip().lower() for v in cells]
        if "utility characteristics" in norm:
            top = norm.index("utility characteristics")
            leaf = next((i for i in range(top + 1, len(norm)) if norm[i] in ("year", "data year")), None)
            if leaf is not None:
                return top, leaf, c
    raise ValueError("EIA-861 sheet: no 'Utility Characteristics' / 'Year' header found")


def eia861_columns(raw: pl.DataFrame) -> tuple[list[str], int, int]:
    """Flatten an EIA-861 sheet read without a header: the group header rows are forward-filled across columns
    (a new group resets its sub-groups) and joined with the leaf row as "group|sub|leaf". Returns (names of the
    columns from the anchor column on, index of the first data row, index of the anchor column)."""
    top, leaf, first_col = _header_anchor(raw)
    levels = leaf - top
    names: list[str] = []
    carry = [""] * levels
    for col in raw.columns[first_col:]:
        cells = raw.get_column(col).cast(pl.String).to_list()[top: leaf + 1]
        parts = []
        for level in range(levels):
            v = cells[level]
            if v is not None and v.strip():
                carry[level] = v.strip()
                for deeper in range(level + 1, levels):
                    carry[deeper] = ""
            parts.append(carry[level])
        names.append("|".join([*parts, (cells[levels] or "").strip()]))
    return names, leaf + 1, first_col


def _num(expr: pl.Expr) -> pl.Expr:
    return expr.cast(pl.String).str.strip_chars().replace({".": None, "": None}).cast(pl.Float64, strict=False)


def eia861_pv(raw: pl.DataFrame, kind: str) -> pl.DataFrame:
    """Utility rows of an EIA-861 "Net_Metering" or "Non_Net_Metering_Distributed" sheet (read with
    ``has_header=False``) -> ``year``, ``state``, ``utility_number``, ``utility_name``, ``ba_code``, ``pv_mw``
    (the photovoltaic capacity total; for net metering the "Capacity MW" group), ``pv_type`` (AC/DC when
    reported) and ``kind``. Header names are matched, not positions, because the layout changes by year."""
    names, start, first_col = eia861_columns(raw)
    body = raw.select(raw.columns[first_col:]).slice(start)
    body.columns = [f"{i}:{n}" for i, n in enumerate(names)]

    def pick(pred) -> str | None:
        return next((c for c, n in zip(body.columns, names, strict=True) if pred(n.lower().split("|"))), None)

    col_year = pick(lambda p: p[-1] in ("year", "data year"))
    col_state = pick(lambda p: p[-1] == "state")
    col_num = pick(lambda p: p[-1] == "utility number")
    col_name = pick(lambda p: p[-1] == "utility name")
    col_ba = pick(lambda p: p[-1] in ("ba code", "ba_code"))

    def is_pv(p: list[str]) -> bool:
        return p[0].startswith("photovoltaic")

    if kind == "net_metering":
        col_pv = pick(lambda p: is_pv(p) and len(p) > 2 and p[1].startswith("capacity") and p[-1] == "total")
        col_type = pick(lambda p: is_pv(p) and p[-1] == "type")
    else:
        col_pv = pick(lambda p: is_pv(p) and p[-1] == "total")
        col_type = pick(lambda p: is_pv(p) and p[-1] == "type")
    if col_pv is None or col_state is None:
        raise ValueError(f"EIA-861 {kind}: photovoltaic capacity or state column not found")
    out = body.select(
        _num(pl.col(col_year)).cast(pl.Int32).alias("year") if col_year else pl.lit(None, pl.Int32).alias("year"),
        pl.col(col_state).cast(pl.String).str.strip_chars().alias("state"),
        pl.col(col_num).cast(pl.String).str.strip_chars().alias("utility_number") if col_num else pl.lit(None).alias(
            "utility_number"),
        pl.col(col_name).cast(pl.String).str.strip_chars().alias("utility_name") if col_name else pl.lit(None).alias(
            "utility_name"),
        pl.col(col_ba).cast(pl.String).str.strip_chars().alias("ba_code") if col_ba else pl.lit(None, pl.String).alias(
            "ba_code"),
        _num(pl.col(col_pv)).fill_null(0.0).alias("pv_mw"),
        (pl.col(col_type).cast(pl.String).str.strip_chars() if col_type else pl.lit(None, pl.String)).alias("pv_type"),
    )
    return out.filter(pl.col("state").str.len_chars() == 2).with_columns(pl.lit(kind).alias("kind"))


def btm_pv_by_year(rows: pl.DataFrame, state: str = "TX", ba_code: str = ERCOT_BA) -> pl.DataFrame:
    """Per year and kind: the PV MW of ``state`` utilities in ``ba_code`` (adjustment rows kept apart), the
    utility count and the AC/DC type(s) reported."""
    sub = rows.filter((pl.col("state") == state) & (pl.col("ba_code") == ba_code))
    adj = pl.col("utility_name").str.to_lowercase().str.starts_with("adjustment")
    return (
        sub.group_by("year", "kind")
        .agg(pl.col("pv_mw").filter(~adj).sum().alias("pv_mw"),
             pl.col("pv_mw").filter(adj).sum().alias("adjustment_mw"),
             (pl.col("pv_mw").filter(~adj) > 0).sum().alias("utilities_with_pv"),
             pl.col("pv_type").drop_nulls().unique().sort().str.join("/").alias("pv_type"))
        .sort("kind", "year")
    )


# --- ERCOT CDR rooftop PV ------------------------------------------------------------------------------------


def labeled_row(raw: pl.DataFrame, pattern: str) -> tuple[int, list[float]] | None:
    """The first row of a sheet read without a header whose text matches ``pattern`` (case-insensitive regex):
    (row index, the numeric cells after the matching cell). ``None`` when no row matches."""
    import re

    rx = re.compile(pattern, re.IGNORECASE)
    for i, row in enumerate(raw.iter_rows()):
        for j, v in enumerate(row):
            if isinstance(v, str) and rx.search(v):
                nums = []
                for w in row[j + 1:]:
                    try:
                        nums.append(float(w))
                    except (TypeError, ValueError):
                        continue
                return i, nums
    return None


def cdr_hourly_pv(raw: pl.DataFrame) -> pl.DataFrame:
    """The CDR's ``LongTermLoadForecast`` sheet (read without a header) -> ``year``, ``month``, ``day``, ``hour``
    (hour ending) and ``pv_mw`` (the ``ercot_pv`` column: ERCOT's incremental rooftop PV forecast, negative MW).
    The header row is found by its ``ercot_pv`` cell."""
    header = next(i for i, row in enumerate(raw.iter_rows())
                  if any(isinstance(v, str) and v.strip() == "ercot_pv" for v in row))
    names = [str(v).strip() if v is not None else f"_c{j}" for j, v in enumerate(raw.row(header))]
    body = raw.slice(header + 1)
    body.columns = [f"{j}|{n}" for j, n in enumerate(names)]
    pick = {n: c for c, n in zip(body.columns, names, strict=True) if n in ("year", "month", "day", "hour", "ercot_pv")}
    keys = ("year", "month", "day", "hour")
    return (
        body.select(*[pl.col(pick[k]).cast(pl.Float64, strict=False).alias(k) for k in keys],
                    pl.col(pick["ercot_pv"]).cast(pl.Float64, strict=False).alias("pv_mw"))
        .drop_nulls(["year", "hour"])
        .with_columns(pl.col(*keys).cast(pl.Int32))
    )


def hour_profile_ratio(hourly: pl.DataFrame, value: str, num: tuple[int, int], den: tuple[int, int],
                       hour_col: str = "hour") -> float:
    """Mean of ``value`` over hours ``num`` divided by its mean over hours ``den`` (inclusive hour-ending ranges),
    e.g. rooftop output at the afternoon peak relative to midday."""
    def mean_in(lo: int, hi: int) -> float:
        return float(hourly.filter(pl.col(hour_col).is_between(lo, hi))[value].mean())

    return mean_in(*num) / mean_in(*den)


# --- thin loaders --------------------------------------------------------------------------------------------


def _eia861_members(zip_path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(zip_path) as zf:
        return {n: zf.read(n) for n in zf.namelist()
                if n.lower().startswith(("net_metering", "non_net_metering_distributed")) and n.lower().endswith(".xlsx")}


def load_eia861_pv(raw_dir: Path) -> pl.DataFrame:
    """Every EIA-861 zip under ``raw_dir`` (``data/raw/source=eia_861/dt=*``) -> :func:`eia861_pv` rows for the
    utility-level sheet of both files (first sheet; ``TPO`` and territory sheets are skipped)."""
    frames = []
    for zp in sorted(raw_dir.glob("f861*.zip")):
        for name, data in _eia861_members(zp).items():
            kind = "net_metering" if name.lower().startswith("net_metering") else "non_net_metering"
            raw = pl.read_excel(io.BytesIO(data), sheet_id=1, has_header=False, engine="calamine")
            frames.append(eia861_pv(raw, kind).with_columns(pl.lit(name).alias("source_file")))
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
