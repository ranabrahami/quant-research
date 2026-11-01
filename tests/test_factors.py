"""Factor model, French parser and long-short portfolio tests on synthetic data (no network)."""

import numpy as np
import pandas as pd
import pytest

from quant_research.data.french import parse_monthly
from quant_research.data.prices import monthly_returns
from quant_research.factors import model
from quant_research.factors.portfolios import event_time, holdings, long_short_returns
from quant_research.stats import newey_west_lags, rolling_ols

MONTHS = pd.date_range("2000-01-31", periods=240, freq="ME")


@pytest.fixture
def factors():
    rng = np.random.default_rng(0)
    f = pd.DataFrame(rng.normal(0.005, 0.04, (len(MONTHS), 6)), index=MONTHS, columns=model.FF5_MOM)
    f["RF"] = 0.002
    return f


def synthetic(factors, alpha=0.004, betas=(1.1, -0.3, 0.5, 0.2, 0.0, -0.1), noise=0.01, seed=1):
    rng = np.random.default_rng(seed)
    y = alpha + factors[model.FF5_MOM].to_numpy() @ np.array(betas) + rng.normal(0, noise, len(factors))
    return pd.Series(y, index=factors.index, name="asset")


def test_fit_recovers_known_loadings(factors):
    betas = (1.1, -0.3, 0.5, 0.2, 0.0, -0.1)
    res = model.fit(synthetic(factors, betas=betas), factors)
    assert res.table.loc[model.FF5_MOM, "coef"].to_numpy() == pytest.approx(betas, abs=0.05)
    assert res.alpha_annual == pytest.approx(0.048, abs=0.02)
    assert res.r2 > 0.9 and res.n == len(MONTHS)
    assert res.lags == newey_west_lags(len(MONTHS))


def test_zero_alpha_is_not_significant(factors):
    res = model.fit(synthetic(factors, alpha=0.0), factors)
    assert abs(res.alpha_t) < 2.5


def test_attribution_adds_up_to_the_average_return(factors):
    y = synthetic(factors)
    res = model.fit(y, factors)
    parts = model.attribution(y, factors, res)
    total = parts.pop("Total (actual average excess return)")
    # OLS with an intercept fits the mean exactly: alpha + sum(beta * mean factor) = mean y
    assert parts.sum() == pytest.approx(total, abs=1e-12)


def test_summary_shows_alpha_per_year(factors):
    res = model.fit(synthetic(factors), factors)
    assert res.summary().at["const", "coef"] == pytest.approx(res.alpha_annual)
    assert res.summary().at["const", "factor"] == "Alpha"


def test_rolling_ols_last_window_equals_full_sample_fit(factors):
    y = synthetic(factors)
    roll = rolling_ols(y, factors[model.FF5_MOM], window=len(MONTHS))
    full = model.fit(y, factors).table["coef"]
    assert len(roll) == 1
    assert roll.iloc[0].to_numpy() == pytest.approx(full.to_numpy(), abs=1e-10)


def test_rolling_exposures_window_count(factors):
    out = model.rolling_exposures(synthetic(factors), factors, window=60)
    assert len(out) == len(MONTHS) - 59 and "alpha_annual" in out


def test_excess_subtracts_risk_free(factors):
    r = pd.Series(0.01, index=MONTHS)
    assert model.excess(r, factors).to_numpy() == pytest.approx(0.008)


def test_newey_west_lags_rule():
    assert newey_west_lags(100) == 4
    assert newey_west_lags(441) == 5


FRENCH = """This file was created using the 202609 CRSP database.
The 1-month TBill return is from Ibbotson and Associates Inc.

,Mkt-RF,SMB,HML,RF
196307,   -0.39,   -0.48,   -0.81,    0.27
196308,    5.07,   -0.80,    1.70,    0.25

 Annual Factors: January-December
,Mkt-RF,SMB,HML,RF
1964,   12.0,    0.1,    0.2,    3.0
"""


def test_parse_monthly_takes_the_monthly_block_in_decimals():
    df = parse_monthly(FRENCH)
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert len(df) == 2                                   # annual table excluded
    assert df.index[0] == pd.Timestamp("1963-07-31")      # month-end, matching price returns
    assert df.at[pd.Timestamp("1963-08-31"), "Mkt-RF"] == pytest.approx(0.0507)


def test_monthly_returns_are_month_end_to_month_end():
    days = pd.date_range("2024-01-01", "2024-03-31", freq="D")
    px = pd.Series(100.0, index=days)
    px[days >= "2024-02-01"] = 110.0
    r = monthly_returns(px)
    assert list(r.index) == [pd.Timestamp("2024-02-29"), pd.Timestamp("2024-03-31")]
    assert r.iloc[0] == pytest.approx(0.10) and r.iloc[1] == pytest.approx(0.0)


# --- long-short portfolios ---------------------------------------------------------------------

TICKERS = list("ABCDEF")


def signal_panel(dates=("2024-03-31", "2024-06-30")):
    # A is always the highest signal, F the lowest
    return pd.DataFrame([{"date": d, "ticker": t, "score": 6 - i} for d in dates for i, t in enumerate(TICKERS)])


def returns_panel():
    months = pd.date_range("2024-01-31", "2025-06-30", freq="ME")
    r = pd.DataFrame(0.0, index=months, columns=TICKERS)
    r["A"], r["B"] = 0.02, 0.01                           # longs
    r["E"], r["F"] = -0.01, -0.02                         # shorts
    return r


def test_holdings_take_top_and_bottom_thirds():
    h = holdings(signal_panel(), "score")
    first = h[h["date"] == "2024-03-31"]
    assert set(first.loc[first.side == 1, "ticker"]) == {"A", "B"}
    assert set(first.loc[first.side == -1, "ticker"]) == {"E", "F"}


def test_no_return_before_the_signal_date():
    r = returns_panel()
    r.loc["2024-03-31", "A"] = 10.0                       # a huge return in the signal month itself
    ls = long_short_returns(signal_panel(), r, "score")
    assert ls.index.min() == pd.Timestamp("2024-04-30")
    assert ls["long_short"].max() == pytest.approx(0.03)


def test_rebalance_holding_runs_to_next_signal_date():
    ls = long_short_returns(signal_panel(), returns_panel(), "score")
    assert (ls["cohorts"] == 1).all()
    assert ls["long_short"].to_numpy() == pytest.approx(0.03)
    assert ls.index.max() == returns_panel().index.max()


def test_overlapping_cohorts_average_live_portfolios():
    sig = signal_panel()
    # Flip the ranking on the second date: its cohort is long E,F and short A,B
    second = sig["date"] == "2024-06-30"
    sig.loc[second, "score"] = -sig.loc[second, "score"]
    ls = long_short_returns(sig, returns_panel(), "score", holding_months=12)
    assert ls.loc["2024-05-31", "cohorts"] == 1 and ls.loc["2024-05-31", "long_short"] == pytest.approx(0.03)
    assert ls.loc["2024-08-31", "cohorts"] == 2 and ls.loc["2024-08-31", "long_short"] == pytest.approx(0.0)
    assert ls.index.max() == pd.Timestamp("2025-06-30")   # second cohort ends 12 months after June 2024
    assert ls.loc["2025-04-30", "cohorts"] == 1           # first cohort expired after March 2025


def test_event_time_counts_months_after_each_signal():
    ev = event_time(signal_panel(), returns_panel(), "score", max_months=3)
    assert sorted(ev["k"].unique()) == [1, 2, 3]
    first = ev[ev["date"] == "2024-03-31"]
    assert list(first["month"]) == list(pd.date_range("2024-04-30", periods=3, freq="ME"))
    assert ev["long_short"].to_numpy() == pytest.approx(0.03)


def test_parse_monthly_marks_missing_codes_as_nan():
    text = ",A,B\n192607,  -99.99,   1.00\n192608,  -999,   2.00\n"
    df = parse_monthly("header line\n" + text)
    assert df["A"].isna().all()
    assert df["B"].tolist() == pytest.approx([0.01, 0.02])


def test_price_cache_serves_earlier_starts(monkeypatch, tmp_path):
    """Regression: a cache filled by a late-start call must not truncate a later early-start call."""
    from quant_research.data import prices as px

    calls = []

    class FakeTicker:
        def __init__(self, ticker):
            pass

        def history(self, start, auto_adjust):
            calls.append(start)
            idx = pd.date_range("2000-01-03", "2024-12-31", freq="B", tz="America/New_York")
            return pd.DataFrame({"Adj Close": np.linspace(1, 2, len(idx))}, index=idx)

    monkeypatch.setattr(px, "CACHE", tmp_path)
    monkeypatch.setattr(px.yf, "Ticker", FakeTicker)
    late = px.daily_prices("XYZ", start="2022-01-01")
    early = px.daily_prices("XYZ", start="2004-01-01")
    assert late.index.min() >= pd.Timestamp("2022-01-01")
    assert pd.Timestamp("2004-01-01") <= early.index.min() < pd.Timestamp("2004-01-10")
    assert len(calls) == 1                                    # second call served from the full cache
