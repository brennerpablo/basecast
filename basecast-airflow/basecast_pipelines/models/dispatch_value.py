"""Dollar value of one MW of battery: daily energy arbitrage on ERCOT real-time prices vs. the 4CP discharge.

Exploration X18 (``docs/analysis/x18_dispatch_value.md``). Everything is **per MW of discharge power** for a
battery of ``hours`` of storage (usable MWh = ``hours`` × MW), with a round-trip efficiency ``eta``: to deliver
E MWh the battery draws E / ``eta`` MWh from the grid. At most one cycle per operating day. No degradation cost,
no state of charge carried across days, no ancillary services, no backup reserve (Base's per-home kW, kWh and
reserve policy are not verified).

Prices: ERCOT real-time settlement point prices (``parsers/ercot/spp_hist.py``, NP6-785-ER), 15-minute intervals
indexed by UTC interval start. Load zones use the ``LZ`` settlement point type (each load zone is also published
as ``LZEW``, whose definition is not verified), hubs ``HU`` plus ``HB_HUBAVG`` (``AH``).

Time: an interval belongs to the America/Chicago date of its start (the operating day); within a day the
intervals are ordered by UTC, so the spring-forward day has 92 and the fall-back day 100 (the repeated hour
stays in time order). Fixed windows are local wall-clock minutes after midnight at the interval **start**,
``[start, start + length)``: 15:45–17:45 is ``(945, 120)`` and covers intervals starting 15:45 … 17:30. On the
fall-back day both passes of the repeated 01:00 hour match a window that contains 01:00; no window used here does.

Schedules per day (value in $ per MW of discharge power):

- ``pf`` (perfect foresight, the upper bound): the best contiguous charge block of ``hours`` followed, later the
  same day, by the best contiguous discharge block of ``hours``; skipped (0) when not positive.
- ``rule``: a fixed charge window and a fixed discharge window every day, no foresight, not skipped.
- ``cp``: committed to discharge from 15:45 for ``hours`` (the X3 4CP window when ``hours`` = 2; 15:45–19:45 for a
  4-hour battery), charging in the best block that ends before 15:45; not skipped (a 4CP alert overrides price).
- ``cp_split`` (``hours`` > 2): 2 hours at 15:45–17:45, then the remaining ``hours`` − 2 in the best contiguous
  block that starts at or after 17:45, charging the whole energy in the best block before 15:45.

Loaders are thin (the repo's own parser on the local raw lake); everything else is a pure function.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import polars as pl

LOCAL_TZ = "America/Chicago"
INTERVAL_H = 0.25  # hours per real-time settlement interval
SLOTS_PER_HOUR = 4
CP_START_MIN = 15 * 60 + 45  # the X3 4CP window, 15:45–17:45 CPT
CP_LENGTH_MIN = 120
LOAD_ZONES = ("LZ_AEN", "LZ_CPS", "LZ_HOUSTON", "LZ_LCRA", "LZ_NORTH", "LZ_RAYBN", "LZ_SOUTH", "LZ_WEST")
HUBS = ("HB_HOUSTON", "HB_NORTH", "HB_PAN", "HB_SOUTH", "HB_WEST", "HB_HUBAVG")

# ERCOT postage-stamp transmission rate, $/kW-yr, by the calendar year it bills (summer Y's 4CP bills Y+1).
# Source: docs/analysis/x15_source_checks.md Q2 (PUCT dockets 56050, 57491, 59080; 2026 order on remand pending).
POSTAGE_STAMP_USD_KW_YR = {2024: 66.756998, 2025: 68.547301, 2026: 75.527270}
POSTAGE_STAMP_STATUS = {2024: "final matrix", 2025: "final order", 2026: "pending (order on remand)"}


# --- loader (local raw lake, read-only) ----------------------------------------------------------------------


def load_rtm_prices(raw_root: Path, years: Iterable[int], points: Sequence[str] = (*LOAD_ZONES, *HUBS)) -> pl.DataFrame:
    """Real-time 15-minute prices for ``points`` in ``years`` from ``raw/source=ercot_spp_hist`` (the newest
    ``RTMLZHBSPP_<year>`` workbook per year, as the dataset's ``select`` does): ``interval_start_utc``,
    ``settlement_point``, ``price_usd_mwh``. Load zones keep only the ``LZ`` type."""
    from types import SimpleNamespace

    from basecast_pipelines.parsers.ercot.spp_hist import parse_rtm

    wanted = set(years)
    newest: dict[int, Path] = {}
    for p in sorted((raw_root / "source=ercot_spp_hist").glob("dt=*/*RTMLZHBSPP_*.zip")):
        year = int(p.stem.rsplit("_", 1)[-1])
        if year in wanted:
            newest[year] = p  # dt= folders sort oldest first
    frames = []
    for year, p in sorted(newest.items()):
        f = SimpleNamespace(name=p.name, suffix=".zip", key=str(p), meta={}, url=str(p), read_bytes=p.read_bytes)
        rtm = parse_rtm(f)  # type: ignore[arg-type]  # duck-typed RawFile, as analysis/x3_four_cp.py does
        if rtm is not None:
            frames.append(select_points(rtm, points))
    return pl.concat(frames).unique(["interval_start_utc", "settlement_point"], keep="last").sort(
        "settlement_point", "interval_start_utc"
    )


def select_points(rtm: pl.DataFrame, points: Sequence[str]) -> pl.DataFrame:
    """Keep ``points``; a load zone keeps its ``LZ`` row (not ``LZEW``)."""
    return rtm.filter(
        pl.col("settlement_point").is_in(list(points)) & (pl.col("settlement_point_type") != "LZEW")
    ).select("interval_start_utc", "settlement_point", "price_usd_mwh")


# --- time ---------------------------------------------------------------------------------------------------


def with_local_day(prices: pl.DataFrame, ts: str = "interval_start_utc") -> pl.DataFrame:
    """Adds ``operating_date`` (local date of the interval start), ``start_min`` (local wall-clock minutes after
    midnight at the start) and ``slot`` (0-based position within the day in UTC order)."""
    local = pl.col(ts).dt.convert_time_zone(LOCAL_TZ)
    return (
        prices.with_columns(
            local.dt.date().alias("operating_date"),
            (local.dt.hour().cast(pl.Int32) * 60 + local.dt.minute().cast(pl.Int32)).alias("start_min"),
        )
        .sort("settlement_point", ts)
        .with_columns(pl.int_range(pl.len()).over("settlement_point", "operating_date").cast(pl.Int32).alias("slot"))
    )


# --- one day, numpy -----------------------------------------------------------------------------------------


def _block_sums(p, k: int):
    """``s[i] = p[i:i+k].sum()`` for every start ``i``."""
    import numpy as np

    c = np.concatenate([[0.0], np.cumsum(p)])
    return c[k:] - c[:-k]


def best_cycle(p, k: int, eta: float) -> tuple[float, int | None, int | None]:
    """Best contiguous charge block of ``k`` slots followed (not overlapping) by the best discharge block of ``k``
    slots, per MW: ``0.25 × (Σ discharge prices − Σ charge prices / eta)``. Returns (value, charge start,
    discharge start); (0, None, None) when no cycle is positive or the day is too short."""
    import numpy as np

    p = np.asarray(p, dtype=float)
    if len(p) < 2 * k:
        return 0.0, None, None
    s = _block_sums(p, k)
    pref = np.minimum.accumulate(s)  # cheapest charge block starting at or before each slot
    d = np.arange(k, len(s))  # discharge starts that leave room for a charge block before them
    val = s[d] - pref[d - k] / eta
    j = int(np.argmax(val))
    best = INTERVAL_H * float(val[j])
    if best <= 0:
        return 0.0, None, None
    di = int(d[j])
    ci = int(np.argmin(s[: di - k + 1]))
    return best, ci, di


def committed_cycle(p, discharge_idx, k_charge: int, eta: float) -> float | None:
    """Value of discharging exactly at ``discharge_idx`` (slot indices), charging the same energy in the cheapest
    contiguous block of ``k_charge`` slots that ends before the first discharge slot. Not clipped: a committed
    discharge happens whatever the price. None when there is no room to charge before it."""
    import numpy as np

    p = np.asarray(p, dtype=float)
    idx = np.asarray(sorted(discharge_idx), dtype=int)
    first = int(idx[0])
    if first < k_charge or idx[-1] >= len(p):
        return None
    charge = float(_block_sums(p[:first], k_charge).min())
    return INTERVAL_H * (float(p[idx].sum()) - charge / eta)


def best_block_after(p, k: int, from_idx: int) -> list[int] | None:
    """Slots of the best contiguous ``k``-slot block starting at or after ``from_idx`` (highest price sum)."""
    import numpy as np

    p = np.asarray(p, dtype=float)
    if len(p) - from_idx < k:
        return None
    s = _block_sums(p[from_idx:], k)
    i = from_idx + int(np.argmax(s))
    return list(range(i, i + k))


def _window_idx(start_mins, start: int, length: int) -> list[int]:
    return [i for i, m in enumerate(start_mins) if start <= m < start + length]


def fixed_cycle(p, start_mins, charge: tuple[int, int], discharge: tuple[int, int], eta: float) -> float | None:
    """The rule: charge in a fixed wall-clock window, discharge in another, every day (no foresight). Windows
    are (start minute, length in minutes). None when a window is incomplete that day."""
    ci = _window_idx(start_mins, *charge)
    di = _window_idx(start_mins, *discharge)
    if len(ci) != charge[1] // 15 or len(di) != discharge[1] // 15:
        return None
    return INTERVAL_H * (sum(p[i] for i in di) - sum(p[i] for i in ci) / eta)


# --- all days -----------------------------------------------------------------------------------------------


def daily_values(
    prices: pl.DataFrame, hours: int, eta: float, *, rule_charge: tuple[int, int], rule_discharge: tuple[int, int],
    cp_start_min: int = CP_START_MIN,
) -> pl.DataFrame:
    """One row per settlement point and operating day with the $/MW value of each schedule (see the module
    docstring): ``pf`` (and its window starts ``pf_charge_min``, ``pf_discharge_min`` in local minutes), ``rule``,
    ``cp`` and, when ``hours`` > 2, ``cp_split``; plus ``n_slots``, ``cp_price`` (mean price 15:45–17:45) and
    ``pf_discharge_price`` (mean price of the pf discharge block)."""
    import numpy as np

    k = hours * SLOTS_PER_HOUR
    cp_k = CP_LENGTH_MIN // 15
    d = prices if "slot" in prices.columns else with_local_day(prices)
    grouped = (
        d.sort("settlement_point", "operating_date", "slot")
        .group_by("settlement_point", "operating_date", maintain_order=True)
        .agg(pl.col("price_usd_mwh"), pl.col("start_min"))
    )
    rows = []
    for point, day, p_list, m_list in grouped.iter_rows():
        p = np.asarray(p_list, dtype=float)
        mins = list(m_list)
        pf, ci, di = best_cycle(p, k, eta)
        cp_idx = _window_idx(mins, cp_start_min, hours * 60)
        cp = committed_cycle(p, cp_idx, k, eta) if len(cp_idx) == k else None
        row = {
            "settlement_point": point, "operating_date": day, "n_slots": len(p),
            "pf": pf, "pf_charge_min": mins[ci] if ci is not None else None,
            "pf_discharge_min": mins[di] if di is not None else None,
            "pf_discharge_price": float(p[di:di + k].mean()) if di is not None else None,
            "rule": fixed_cycle(p, mins, rule_charge, rule_discharge, eta),
            "cp": cp,
        }
        cp2 = _window_idx(mins, cp_start_min, CP_LENGTH_MIN)
        row["cp_price"] = float(p[cp2].mean()) if len(cp2) == cp_k else None
        if hours > 2:
            split = None
            if len(cp2) == cp_k:
                after = best_block_after(p, k - cp_k, cp2[-1] + 1)
                if after is not None:
                    split = committed_cycle(p, cp2 + after, k, eta)
            row["cp_split"] = split
        rows.append(row)
    schema = {"pf_charge_min": pl.Int32, "pf_discharge_min": pl.Int32}
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(
        [pl.col(c).cast(t) for c, t in schema.items()]
    )


def annual_values(daily: pl.DataFrame, value: str, months: Sequence[int] | None = None) -> pl.DataFrame:
    """Sum of ``value`` per settlement point and year ($/MW-yr, or $/MW per season with ``months``), with the
    number of days counted."""
    d = daily.with_columns(pl.col("operating_date").dt.year().alias("year"))
    if months is not None:
        d = d.filter(pl.col("operating_date").dt.month().is_in(list(months)))
    return (
        d.group_by("settlement_point", "year")
        .agg(pl.col(value).sum().alias(value), pl.col(value).is_not_null().sum().alias("days"))
        .sort("settlement_point", "year")
    )


def opportunity_cost(daily: pl.DataFrame, dispatch_days: pl.DataFrame, committed: str = "cp") -> pl.DataFrame:
    """Per settlement point and year, over the 4CP dispatch days (``dispatch_days``: ``operating_date`` of the days
    with ``dispatch`` true): the perfect-foresight value, the committed schedule's value and the difference
    (``lost`` = Σ (pf − committed), $/MW)."""
    days = dispatch_days.filter(pl.col("dispatch")).select("operating_date").unique()
    return (
        daily.join(days, on="operating_date", how="inner")
        .with_columns(pl.col("operating_date").dt.year().alias("year"))
        .group_by("settlement_point", "year")
        .agg(
            pl.len().alias("days"),
            pl.col("pf").sum().alias("pf"),
            pl.col(committed).sum().alias(committed),
            (pl.col("pf") - pl.col(committed)).sum().alias("lost"),
        )
        .sort("settlement_point", "year")
    )


def four_cp_value(summer_year: int, share_caught: float = 1.0, rates: dict[int, float] = POSTAGE_STAMP_USD_KW_YR) -> float | None:
    """$ per MW of 4CP load avoided in ``summer_year``: the postage-stamp rate of the billing year (Y+1) × 1,000 ×
    ``share_caught`` (the share of the four intervals the fleet discharged in; each is a quarter of the average).
    None when that billing year's rate is not sourced."""
    rate = rates.get(summer_year + 1)
    return None if rate is None else rate * 1000 * share_caught
