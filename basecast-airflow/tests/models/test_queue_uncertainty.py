"""Bootstrap band, hazard scaling, size split and COD floor of ``models/queue_uncertainty.py`` (tiny fixtures, no DB)."""

from datetime import date, timedelta

import pytest

np = pytest.importorskip("numpy")
pl = pytest.importorskip("polars")

from basecast_pipelines.models import queue_adjusted as qa  # noqa: E402
from basecast_pipelines.models import queue_uncertainty as qu  # noqa: E402
from basecast_pipelines.models import survival as sv  # noqa: E402

COD, WD, CENS = sv.COD, sv.WITHDRAWN, sv.CENSORED


@pytest.fixture
def fit():
    # q6's hand fit: at risk 4, 3, 2 at t = 1, 2, 3; COD at 1 and 3, WD at 2, censored at 4
    return sv.fit_cif([1, 2, 3, 4], [COD, WD, COD, CENS])


def test_size_stratum_splits_only_the_gas_stratum():
    df = pl.DataFrame({"stratum": ["gas+other", "gas+other", "gas+other", "solar"],
                       "capacity_mw": [600.0, 499.9, None, 900.0]})
    got = df.select(qu.size_stratum())["stratum"].to_list()
    assert got == ["gas+other>=500", "gas+other<500", "gas+other<500", "solar"]


def test_cumulative_hazard_by_hand(fit):
    np.testing.assert_allclose(qu.cause_hazard(fit), [1 / 4, 0, 1 / 2])
    np.testing.assert_allclose(qu.cumulative_hazard(fit, [0.5, 1, 2.5, 3, 9]), [0, 0.25, 0.25, 0.75, 0.75])
    np.testing.assert_allclose(qu.cumulative_hazard(fit, [9], WD), [1 / 3])


def test_observed_expected_uses_each_subjects_own_window(fit):
    # subject 1 at risk (0, 2]: E = 0.25; subject 2 at risk (1.5, 3.5]: E = 0.75 - 0.25 = 0.5
    o, e = qu.observed_expected([2.0, 3.5], [COD, CENS], [0.0, 1.5], fit)
    assert (o, e) == pytest.approx((1.0, 0.75))
    o, e = qu.observed_expected([2.0, 3.5], [COD, CENS], [0.0, 1.5], fit, weights=[2.0, 1.0])
    assert (o, e) == pytest.approx((2.0, 1.0))


def test_scale_cause_hazard_by_hand(fit):
    same = qu.scale_cause_hazard(fit, 1.0)
    np.testing.assert_allclose(same.cif, fit.cif)
    np.testing.assert_allclose(same.survival, fit.survival)
    # theta = 0: no COD, only the WD hazard 1/3 at t = 2
    zero = qu.scale_cause_hazard(fit, 0.0)
    np.testing.assert_allclose(zero.at([10], COD), [0.0])
    np.testing.assert_allclose(zero.at([10], WD), [1 / 3])
    np.testing.assert_allclose(zero.survival, [1, 2 / 3, 2 / 3])
    # theta = 2: COD hazard 1/2 at 1 and 1 at 3 -> F_cod = 1/2 + (1/2)(2/3)(1) = 5/6, F_wd = (1/2)(1/3) = 1/6
    two = qu.scale_cause_hazard(fit, 2.0)
    np.testing.assert_allclose(two.at([10], COD), [5 / 6])
    np.testing.assert_allclose(two.at([10], WD), [1 / 6])
    np.testing.assert_allclose(two.survival[-1], 0.0, atol=1e-12)
    # theta = 3 at t = 3 would be a hazard of 1.5: capped at 1, the CIFs still sum to at most 1
    three = qu.scale_cause_hazard(fit, 3.0)
    assert three.at([10], COD)[0] + three.at([10], WD)[0] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        qu.scale_cause_hazard(fit, -1.0)


def _outcomes() -> pl.DataFrame:
    """Gas: 8 small projects with an IA (5 COD) and 3 large (1 COD); 8 solar; all listed at entry."""
    rows = []
    start = date(2020, 1, 1)
    spec = [("gas+other", 100.0, COD, 6 + i) for i in range(5)] + [("gas+other", 100.0, CENS, 30 + i) for i in range(3)]
    spec += [("gas+other", 800.0, COD, 9), ("gas+other", 800.0, CENS, 4), ("gas+other", 900.0, WD, 5)]
    spec += [("solar", 50.0, COD if i < 4 else WD, 10 + i) for i in range(8)]
    for i, (stratum, mw, event, months_after_ia) in enumerate(spec):
        ia = start + timedelta(days=60 + i)
        rows.append({"inr": f"P{i}", "stratum": stratum, "capacity_mw": mw, "first_seen_month": start,
                     "entry_date": start, "ia_signed": ia, "event": event,
                     "event_date": ia + timedelta(days=round(months_after_ia * sv.DAYS_PER_MONTH))})
    return pl.DataFrame(rows).with_columns(pl.col("event").cast(pl.Int8))


def test_split_gas_curves_scales_the_small_curve_for_thin_large_gas():
    out = _outcomes()
    curves, info = qu.split_gas_curves(out, weight=None, min_n=5, min_events=2)
    big, small = qu.big_label(), qu.small_label()
    assert info["n"] == 3 and info["events"] == 1 and info["own_fit"] == 0.0
    ref = curves[("ia_signed", small)]
    frame = qa.stage_frame(out.with_columns(stratum=qu.size_stratum()), "ia_signed").filter(pl.col("stratum") == big)
    o, e = qu.observed_expected(frame["time"].to_numpy(), frame["event"].to_numpy(), frame["entry"].to_numpy(), ref.fit)
    assert info["observed"] == pytest.approx(o) and info["expected"] == pytest.approx(e)
    assert info["theta"] == pytest.approx(o / e)
    np.testing.assert_allclose(curves[("ia_signed", big)].fit.cif, qu.scale_cause_hazard(ref.fit, o / e).cif)
    assert curves[("ia_signed", big)].end == ref.end
    # the large pre-IA stratum (3 projects) falls back to the combined gas+other pre-IA curve
    combined = qa.fit_stage_curves(out, qa.fit_stages("entry_ia_sm"), weight=None, min_n=5, min_events=2)
    np.testing.assert_allclose(curves[(qa.PRE_IA, big)].fit.cif, combined[(qa.PRE_IA, "gas+other")].fit.cif)
    assert ("ia_signed", "solar") in curves and ("ia_signed", qa.POOLED) in curves


def test_floor_delay():
    df = pl.DataFrame({"floor": [date(2027, 9, 1), date(2026, 1, 1), None]})
    d = df.select(qu.floor_delay(date(2026, 9, 1), pl.col("floor"), slack_months=1.0))["floor"].to_numpy()
    np.testing.assert_allclose(d, [365 / sv.DAYS_PER_MONTH - 1, 0.0, 0.0])


def _curves(fit):
    return {("ia_signed", "x"): qa.StageCurve("ia_signed", "x", fit, 100.0, 4, 2),
            ("ia_signed", qa.POOLED): qa.StageCurve("ia_signed", qa.POOLED, fit, 100.0, 4, 2)}


def test_rescore_with_floor_shortens_the_window(fit):
    q = pl.DataFrame({"stage": ["ia_signed"] * 4, "stratum": ["x"] * 4, "elapsed": [0.0, 0.0, 1.5, 0.0],
                      "capacity_mw": [10.0, 10.0, 10.0, 10.0], "delay": [1.0, 5.0, 0.5, 2.0]})
    curves = _curves(fit)
    scored = qa.score(q, curves, {"h": 3.0})
    keep = pl.Series([True, True, True, False])
    out = qu.rescore_with_floor(scored, curves, {"h": 3.0}, mask=pl.lit(keep), resolution=0.5)
    want = [qa.conditional_cod(fit, [0.0], 2.0)[0][0], 0.0, qa.conditional_cod(fit, [1.5], 2.5)[0][0],
            scored["p_h"][3]]
    np.testing.assert_allclose(out["p_h"].to_numpy(), want)
    np.testing.assert_allclose(out["mw_h"].to_numpy(), np.array(want) * 10)
    assert out.columns == scored.columns


def test_bootstrap_with_certain_outcomes():
    sure = sv.fit_cif([1.0, 1.0, 1.0], [COD, COD, COD])  # F_cod = 1 from t = 1
    never = sv.fit_cif([1.0, 2.0, 3.0], [WD, WD, WD])  # no COD at all
    curves = {("ia_signed", "a"): qa.StageCurve("ia_signed", "a", sure, 10.0, 3, 3),
              ("ia_signed", "b"): qa.StageCurve("ia_signed", "b", never, 10.0, 3, 0),
              ("ia_signed", qa.POOLED): qa.StageCurve("ia_signed", qa.POOLED, sure, 10.0, 3, 3)}
    q = pl.DataFrame({"stage": ["ia_signed"] * 3, "stratum": ["a", "b", "a"], "elapsed": [0.0, 0.0, 0.0],
                      "capacity_mw": [100.0, 50.0, 7.0], "county_fips": ["48001", "48001", None]})
    train = pl.DataFrame({"x": [1, 2, 3]})
    seen = []

    def fit_curves(t):
        seen.append(t.height)
        return curves

    reps = qu.bootstrap(train, q, fit_curves, {"2027": 2.0, "2028": 5.0}, n_boot=4, groups=("county_fips",))
    assert set(seen) == {3}
    assert reps.height == 4 * 2 * 3  # reps x horizons x (total + 2 counties)
    tot = reps.filter(pl.col("level") == "total")
    assert (tot["expected_mw"] == 107.0).all() and (tot["simulated_mw"] == 107.0).all()
    cty = reps.filter((pl.col("level") == "county_fips") & (pl.col("group") == "none"))
    assert (cty["simulated_mw"] == 7.0).all()
    b = qu.band(reps)
    assert b.filter(pl.col("level") == "total")["p10"].to_list() == [107.0, 107.0]


def test_bootstrap_bernoulli_mean_and_same_uniform_across_horizons(fit):
    curves = _curves(fit)
    q = pl.DataFrame({"stage": ["ia_signed"] * 400, "stratum": ["x"] * 400, "elapsed": [0.0] * 400,
                      "capacity_mw": [1.0] * 400})
    reps = qu.bootstrap(pl.DataFrame({"x": [0]}), q, lambda t: curves, {"a": 1.5, "b": 3.5}, n_boot=50, resample=False)
    tot = reps.filter(pl.col("level") == "total").pivot(on="horizon", index="rep", values="simulated_mw")
    # P(COD by 1.5) = .25 and by 3.5 = .5; the same uniform per project keeps "by a" inside "by b"
    assert (tot["a"] <= tot["b"]).all()
    assert tot["a"].mean() == pytest.approx(100, rel=0.1)
    assert tot["b"].mean() == pytest.approx(200, rel=0.1)


def test_percentile_and_error_band():
    assert qu.percentile_of([1, 2, 3, 4], 2.5) == 0.5
    assert qu.percentile_of([1, 2, 3, 4], 2) == pytest.approx(0.375)
    errs = qu.log_errors([100.0, 110.0], [110.0, 100.0])
    np.testing.assert_allclose(errs, [np.log(1.1), -np.log(1.1)])
    lo, hi = qu.error_band(1000.0, errs)
    np.testing.assert_allclose([lo, hi], [1000 * 1.1 ** -qu.Z80, 1000 * 1.1 ** qu.Z80])
    with pytest.raises(ValueError):
        qu.log_errors([0.0], [1.0])
