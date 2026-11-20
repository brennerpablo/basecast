"""Tiny synthetic checks for ``basecast_pipelines.models.dispatch_value`` (no database, no lake)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from basecast_pipelines.models import dispatch_value as dv

CHI = ZoneInfo(dv.LOCAL_TZ)


def _day_prices(day: date, price_at, point: str = "LZ_TEST") -> pl.DataFrame:
    """Every 15-minute interval of the local ``day`` (92/96/100 on DST days), priced by ``price_at(local start)``."""
    start = datetime(day.year, day.month, day.day, tzinfo=CHI).astimezone(UTC)
    end = (datetime(day.year, day.month, day.day, tzinfo=CHI) + timedelta(days=1)).astimezone(UTC)
    rows, t = [], start
    while t < end:
        rows.append({"interval_start_utc": t, "settlement_point": point,
                     "price_usd_mwh": float(price_at(t.astimezone(CHI)))})
        t += timedelta(minutes=15)
    return pl.DataFrame(rows).with_columns(pl.col("interval_start_utc").dt.convert_time_zone("UTC"))


def _evening_peak(local: datetime) -> float:
    """$10 all day, $0 from 10:00 to 12:00, $100 from 19:00 to 21:00, $50 from 15:45 to 17:45."""
    m = local.hour * 60 + local.minute
    if 600 <= m < 720:
        return 0.0
    if 1140 <= m < 1260:
        return 100.0
    if 945 <= m < 1065:
        return 50.0
    return 10.0


def test_best_cycle_orders_charge_before_discharge_and_applies_efficiency():
    p = [100.0] * 8 + [0.0] * 8 + [50.0] * 8  # the high block comes first: it cannot be used after charging
    value, ci, di = dv.best_cycle(p, 8, eta=0.8)
    assert (ci, di) == (8, 16)
    assert value == pytest.approx(0.25 * (8 * 50 - 0 / 0.8))
    value, _, _ = dv.best_cycle([10.0, 20.0, 30.0, 40.0], 1, eta=0.5)
    assert value == pytest.approx(0.25 * (40 - 10 / 0.5))


def test_best_cycle_skips_unprofitable_days_and_pays_negative_charging():
    assert dv.best_cycle([30.0] * 16, 4, eta=0.9) == (0.0, None, None)
    value, ci, _ = dv.best_cycle([-20.0] * 4 + [5.0] * 4, 4, eta=0.8)
    assert ci == 0
    assert value == pytest.approx(0.25 * (4 * 5 + 4 * 20 / 0.8))  # drawing more energy at a negative price pays


def test_committed_and_split_cycles():
    p = [0.0] * 4 + [50.0] * 2 + [10.0] * 2 + [100.0] * 2
    assert dv.committed_cycle(p, [4, 5], 2, eta=1.0) == pytest.approx(0.25 * 100)
    assert dv.committed_cycle(p, [0, 1], 2, eta=1.0) is None  # no room to charge before
    assert dv.best_block_after(p, 2, 6) == [8, 9]
    assert dv.committed_cycle(p, [4, 5, 8, 9], 4, eta=1.0) == pytest.approx(0.25 * 300)


@pytest.mark.parametrize("day, slots", [(date(2025, 7, 1), 96), (date(2025, 3, 9), 92), (date(2025, 11, 2), 100)])
def test_daily_values_on_normal_and_dst_days(day, slots):
    prices = _day_prices(day, _evening_peak)
    d2 = dv.daily_values(prices, 2, 1.0, rule_charge=(600, 120), rule_discharge=(1140, 120))
    row = d2.row(0, named=True)
    assert row["operating_date"] == day
    assert row["n_slots"] == slots
    assert row["pf"] == pytest.approx(0.25 * 8 * 100)  # charge 10–12 at $0, discharge 19–21 at $100
    assert (row["pf_charge_min"], row["pf_discharge_min"]) == (600, 1140)
    assert row["rule"] == pytest.approx(row["pf"])
    assert row["cp"] == pytest.approx(0.25 * 8 * 50)  # committed to 15:45–17:45, charged at $0 before it
    assert row["cp_price"] == pytest.approx(50.0)

    d4 = dv.daily_values(prices, 4, 1.0, rule_charge=(540, 240), rule_discharge=(1080, 240))
    row4 = d4.row(0, named=True)
    # best 4 h: charge 09:00–13:00 around the $0 block (2 h at $0, 2 h at $10), discharge 17:45–21:45 or similar
    assert row4["pf"] > row4["cp"]
    # split: 15:45–17:45 at $50, then the best 2 h after 17:45 (19:00–21:00 at $100); charge 4 h before 15:45
    expected_split = 0.25 * (8 * 50 + 8 * 100 - (8 * 0 + 8 * 10))
    assert row4["cp_split"] == pytest.approx(expected_split)
    # 15:45–19:45: 2 h at $50, 1 h 15 min at $10, 45 min at $100
    expected_fixed = 0.25 * (8 * 50 + 5 * 10 + 3 * 100 - (8 * 0 + 8 * 10))
    assert row4["cp"] == pytest.approx(expected_fixed)


def test_with_local_day_keeps_the_repeated_hour_in_time_order():
    d = dv.with_local_day(_day_prices(date(2025, 11, 2), lambda t: t.hour))
    assert d.height == 100
    assert d["slot"].to_list() == list(range(100))
    one_am = d.filter(pl.col("start_min") == 60)
    assert one_am["slot"].to_list() == [4, 8]  # 01:00 CDT, then 01:00 CST an hour later


def test_annual_values_and_opportunity_cost():
    prices = pl.concat([_day_prices(date(2025, 7, 1), _evening_peak), _day_prices(date(2025, 7, 2), _evening_peak),
                        _day_prices(date(2025, 12, 1), _evening_peak)])
    daily = dv.daily_values(prices, 2, 1.0, rule_charge=(600, 120), rule_discharge=(1140, 120))
    year = dv.annual_values(daily, "pf").row(0, named=True)
    assert (year["year"], year["days"]) == (2025, 3)
    assert year["pf"] == pytest.approx(3 * 200)
    summer = dv.annual_values(daily, "pf", months=(6, 7, 8, 9)).row(0, named=True)
    assert summer["days"] == 2
    dispatch = pl.DataFrame({"operating_date": [date(2025, 7, 1), date(2025, 7, 2)], "dispatch": [True, False]})
    opp = dv.opportunity_cost(daily, dispatch).row(0, named=True)
    assert opp["days"] == 1
    assert opp["lost"] == pytest.approx(200 - 100)


def test_four_cp_value_uses_the_next_years_rate():
    assert dv.four_cp_value(2024) == pytest.approx(68_547.301)
    assert dv.four_cp_value(2025, share_caught=0.75) == pytest.approx(0.75 * 75_527.270)
    assert dv.four_cp_value(2021) is None


def test_select_points_drops_lzew():
    rtm = pl.DataFrame({
        "interval_start_utc": [datetime(2025, 1, 1, 6, tzinfo=UTC)] * 3,
        "settlement_point": ["LZ_NORTH", "LZ_NORTH", "HB_NORTH"],
        "settlement_point_type": ["LZ", "LZEW", "HU"],
        "price_usd_mwh": [1.0, 2.0, 3.0],
    })
    out = dv.select_points(rtm, ["LZ_NORTH", "HB_NORTH"])
    assert out["price_usd_mwh"].to_list() == [1.0, 3.0]
