"""Landmark-date cohorts of ``models/survival_landmark.py`` on tiny synthetic fixtures (no DB)."""

from datetime import date

import pytest

np = pytest.importorskip("numpy")
pl = pytest.importorskip("polars")

from basecast_pipelines.models import queue_adjusted as qa  # noqa: E402
from basecast_pipelines.models import survival as sv  # noqa: E402
from basecast_pipelines.models import survival_landmark as slm  # noqa: E402

COD, WD, CENS = sv.COD, sv.WITHDRAWN, sv.CENSORED


def _outcomes(rows: list[tuple]) -> pl.DataFrame:
    cols = ["inr", "first_seen_month", "ia_signed", "event", "event_date"]
    df = pl.DataFrame(rows, schema=cols, orient="row")
    return df.with_columns(
        entry_date=pl.col("first_seen_month"), fis_approved=pl.col("ia_signed"),
        event=pl.col("event").cast(pl.Int8), stratum=pl.lit("solar"), fuel_group=pl.lit("solar"),
        capacity_mw=pl.lit(100.0),
    )


@pytest.fixture
def outcomes() -> pl.DataFrame:
    return _outcomes([
        ("A", date(2017, 6, 1), date(2019, 1, 1), COD, date(2020, 1, 1)),  # old layout, IA after 2018-08
        ("B", date(2019, 3, 1), date(2018, 10, 1), COD, date(2021, 1, 1)),  # listed after its (post-2018-08) IA
        ("C", date(2019, 1, 1), date(2017, 12, 1), WD, date(2020, 6, 1)),  # q6's delayed entry, pre-2018-08 IA
        ("D", date(2016, 1, 1), date(2019, 6, 1), CENS, date(2026, 9, 1)),  # listed before cancellations are recorded
        ("E", date(2017, 5, 1), date(2017, 9, 1), CENS, date(2026, 9, 1)),  # old IA, still listed on 2018-08
        ("F", date(2017, 5, 1), date(2017, 9, 1), WD, date(2018, 3, 1)),  # old IA, gone before 2018-08
        ("G", date(2019, 5, 1), None, CENS, date(2026, 9, 1)),  # no IA
    ])


def _members(outcomes: pl.DataFrame, cohort: str, **kw) -> set[str]:
    return set(outcomes.filter(slm.cohort_keep("ia_signed", cohort, **kw))["inr"])


def test_cohort_membership(outcomes):
    assert _members(outcomes, "q6") == {"B", "C"}
    assert _members(outcomes, "landmark") == {"A", "B"}
    assert _members(outcomes, "union") == {"A", "B", "C"}
    assert _members(outcomes, "period") == {"A", "B", "C", "D", "E"}
    # without the 2017-04 floor, D (listed in 2016, IA in 2019) joins the landmark cohort
    assert _members(outcomes, "landmark", listed_start=None) == {"A", "B", "D"}


def test_unknown_cohort_raises():
    with pytest.raises(ValueError, match="unknown cohort"):
        slm.cohort_keep("ia_signed", "nope")


def test_membership_labels(outcomes):
    m = slm.membership(outcomes, "ia_signed")
    why = dict(zip(m["inr"], m["why"], strict=True))
    assert why == {"A": "landmark only", "B": "both", "C": "q6 only", "D": "neither", "E": "neither", "F": "neither"}
    assert "G" not in why  # never reached the landmark


def test_delayed_entry_by_cohort(outcomes):
    lm = slm.cohort_frame(outcomes, "ia_signed", "landmark")
    entry = dict(zip(lm["inr"], lm["entry"], strict=True))
    assert entry["A"] == 0.0  # listed before its IA
    assert entry["B"] == pytest.approx((date(2019, 3, 1) - date(2018, 10, 1)).days / sv.DAYS_PER_MONTH)

    period = slm.cohort_frame(outcomes, "ia_signed", "period")
    pe = dict(zip(period["inr"], period["entry"], strict=True))
    # E signed in 2017-09 and is observed from 2018-08 on (calendar left truncation)
    assert pe["E"] == pytest.approx((date(2018, 8, 1) - date(2017, 9, 1)).days / sv.DAYS_PER_MONTH)
    assert pe["C"] == pytest.approx((date(2019, 1, 1) - date(2017, 12, 1)).days / sv.DAYS_PER_MONTH)
    assert pe["A"] == pytest.approx((date(2019, 1, 1) - date(2019, 1, 1)).days)  # IA after 2018-08: no truncation
    assert (period["time"] > period["entry"]).all()
    # the move of first_seen_month only applies to projects listed before 2018-08
    restricted = slm.restrict(outcomes, "ia_signed", "period")
    first = dict(zip(restricted["inr"], restricted["first_seen_month"], strict=True))
    assert first["B"] == date(2019, 3, 1) and first["E"] == date(2018, 8, 1)


def test_ordering():
    got = slm.ordering({
        "solar": {"entry": 0.02, "fis_approved": 0.19, "ia_signed": 0.22},
        "wind": {"entry": 0.07, "fis_approved": 0.49, "ia_signed": 0.33},
        "gas": {"entry": 0.2, "ia_signed": 0.6},
    })
    assert got == {"solar": True, "wind": False, "gas": False}


def test_bootstrap_difference_of_a_frame_with_itself_is_zero(outcomes):
    f = slm.cohort_frame(outcomes, "ia_signed", "period")
    point, lo, hi = slm.bootstrap_difference(f, f, 36, n_boot=20, seed=1)
    assert point == 0.0 and lo == 0.0 and hi == 0.0
    a = slm.bootstrap_difference(f, f.filter(pl.col("inr") != "A"), 36, n_boot=20, seed=3)
    b = slm.bootstrap_difference(f, f.filter(pl.col("inr") != "A"), 36, n_boot=20, seed=3)
    assert a == b  # seeded


def test_semi_markov_curves_take_the_ia_stage_from_the_cohort():
    # 40 projects: entry 2017-06 or 2019-02, IA 2019-06 (+ i days), half reach COD a year later
    rows = []
    for i in range(40):
        first = date(2017, 6, 1) if i % 4 == 0 else date(2019, 2, 1)
        ia = date(2019, 6, 1 + i % 28)
        ev, when = (COD, date(2020, 6, 1 + i % 28)) if i % 2 else (CENS, date(2026, 9, 1))
        rows.append((f"P{i}", first, ia, ev, when))
    out = _outcomes(rows)
    pre_q6 = out.filter(pl.col("first_seen_month") >= pl.lit(sv.COHORT_START))
    curves = slm.semi_markov_curves(pre_q6, out, "landmark", weight=None)
    assert {k[0] for k in curves} == {qa.PRE_IA, "ia_signed"}
    assert curves[("ia_signed", "all")].n == 40  # landmark: all 40 (IA >= 2018-08, listed >= 2017-04)
    assert curves[(qa.PRE_IA, "all")].n == 30  # pre-IA sojourn stays on Q6's cohort (listed >= 2018-08)
    q6 = slm.semi_markov_curves(pre_q6, out, "q6", weight=None)
    assert q6[("ia_signed", "all")].n == 30
