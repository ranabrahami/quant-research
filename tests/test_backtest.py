"""Backtester, metrics and momentum-strategy tests on hand-checkable data (no network)."""

import numpy as np
import pandas as pd
import pytest

from quant_research.backtest import metrics
from quant_research.backtest.engine import overlapping, run
from quant_research.factors.portfolios import long_short_returns
from quant_research.strategies.momentum import long_short, trailing_return

MONTHS = pd.date_range("2020-01-31", periods=6, freq="ME")


def flat_returns(a=0.10, b=0.0):
    return pd.DataFrame({"A": a, "B": b}, index=MONTHS)


def test_weights_earn_only_later_months():
    r = flat_returns()
    r.loc[MONTHS[0], "A"] = 5.0                           # the signal month itself must not count
    w = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=[MONTHS[0]])
    bt = run(w, r)
    assert bt.returns.index[0] == MONTHS[1]
    assert bt.gross.to_numpy() == pytest.approx(0.10)


def test_buy_and_hold_drifts_without_trading():
    r = flat_returns(0.10, 0.0)
    bt = run(pd.DataFrame({"A": [0.5], "B": [0.5]}, index=[MONTHS[0]]), r)
    wealth = (1 + bt.gross).prod()
    assert wealth == pytest.approx(0.5 * 1.1 ** 5 + 0.5)   # 50/50 bought once, never rebalanced
    assert bt.returns["turnover"].iloc[0] == pytest.approx(1.0)   # initial purchase
    assert (bt.returns["turnover"].iloc[1:] == 0).all()


def test_rebalancing_pays_for_drift_and_costs_are_charged():
    r = flat_returns(0.10, 0.0)
    w = pd.DataFrame({"A": 0.5, "B": 0.5}, index=MONTHS[:-1])        # restore 50/50 every month
    bt = run(w, r, cost_bps=100)
    # after a month A is 0.55/1.05 of capital; restoring 50/50 trades 2 x (0.55/1.05 - 0.5)
    drift_trade = 2 * (0.55 / 1.05 - 0.5)
    assert bt.returns["turnover"].iloc[1] == pytest.approx(drift_trade)
    assert bt.returns["trading_cost"].iloc[1] == pytest.approx(drift_trade * 0.01)
    assert bt.net.iloc[1] == pytest.approx(0.05 - drift_trade * 0.01)


def test_long_short_return_and_borrow():
    r = flat_returns(0.03, 0.01)
    w = pd.DataFrame({"A": 1.0, "B": -1.0}, index=MONTHS[:-1])
    bt = run(w, r, borrow_bps=120)
    assert bt.gross.to_numpy() == pytest.approx(0.02)
    assert bt.returns["borrow_cost"].iloc[0] == pytest.approx(0.012 / 12)   # 1.2% a year on 1x short
    assert bt.returns["turnover"].iloc[0] == pytest.approx(2.0)


def test_missing_return_on_a_held_asset_raises():
    r = flat_returns()
    r.loc[MONTHS[2], "A"] = np.nan
    with pytest.raises(ValueError, match="Missing return"):
        run(pd.DataFrame({"A": [1.0]}, index=[MONTHS[0]]), r)


def test_weight_dates_must_be_return_dates():
    with pytest.raises(ValueError, match="not in the return index"):
        run(pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("2020-01-15")]), flat_returns())


def test_overlapping_cohorts_match_cohort_return_averaging():
    """Averaging cohort weights reproduces module 1's averaging of cohort returns."""
    rng = np.random.default_rng(3)
    months = pd.date_range("2022-12-31", periods=30, freq="ME")
    tickers = list("ABCDEFGHI")
    r = pd.DataFrame(rng.normal(0.01, 0.08, (len(months), len(tickers))), index=months, columns=tickers)
    dates = months[::3][:6]
    sig = pd.DataFrame([{"date": d.strftime("%Y-%m-%d"), "ticker": t, "score": rng.normal()}
                        for d in dates for t in tickers])
    ref = long_short_returns(sig, r, "score", holding_months=12)["long_short"]

    targets = {}
    for d, g in sig.assign(date=pd.to_datetime(sig["date"])).groupby("date"):
        g = g.sort_values("score")
        w = pd.Series(0.0, index=tickers)
        w[g["ticker"].iloc[-3:]], w[g["ticker"].iloc[:3]] = 1 / 3, -1 / 3
        targets[d] = w
    w = overlapping(pd.DataFrame(targets).T, 12, r.index)
    bt = run(w, r)
    common = ref.index.intersection(bt.gross.index)
    assert len(common) > 15
    assert bt.gross[common].to_numpy() == pytest.approx(ref[common].to_numpy(), abs=1e-12)


def test_overlapping_averages_live_targets_only():
    idx = pd.date_range("2020-01-31", periods=5, freq="ME")
    t = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]}, index=[idx[0], idx[1]])
    w = overlapping(t, 2, idx)
    assert w.loc[idx[0]].tolist() == [1.0, 0.0]
    assert w.loc[idx[1]].tolist() == [0.5, 0.5]
    assert w.loc[idx[2]].tolist() == [0.0, 1.0]          # first target expired
    assert idx[3] not in w.index


# --- metrics ------------------------------------------------------------------------------------

def test_max_drawdown():
    r = pd.Series([0.10, -0.50, 0.20, 0.0])
    assert metrics.max_drawdown(r) == pytest.approx(-0.50)


def test_summary_and_bootstrap_interval():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.01, 0.04, 600), index=pd.date_range("1970-01-31", periods=600, freq="ME"))
    s = metrics.summary(r)
    lo, hi = metrics.bootstrap_sharpe(r, n=2000)
    assert lo < s["Sharpe"] < hi
    assert s["t (mean)"] == pytest.approx(r.mean() / r.std() * 600 ** 0.5)


def test_probabilistic_sharpe_is_one_half_at_the_benchmark():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.01, 0.04, 400))
    assert metrics.probabilistic_sharpe(r, r.mean() / r.std()) == pytest.approx(0.5)
    assert metrics.probabilistic_sharpe(r) > 0.95


def test_more_trials_deflate_the_sharpe():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.006, 0.04, 400))
    few = metrics.deflated_sharpe(r, rng.normal(0, 0.05, 5))
    many = metrics.deflated_sharpe(r, rng.normal(0, 0.05, 500))
    assert many < few < metrics.probabilistic_sharpe(r)


def test_break_even_cost():
    gross, turnover = pd.Series([0.01, 0.01]), pd.Series([0.5, 0.5])
    assert metrics.break_even_cost_bps(gross, turnover) == pytest.approx(200)


# --- momentum -----------------------------------------------------------------------------------

def test_trailing_return_skips_the_latest_month():
    idx = pd.date_range("2020-01-31", periods=5, freq="ME")
    r = pd.DataFrame({"A": [0.1, 0.1, 0.1, 0.5, 0.0]}, index=idx)
    tr = trailing_return(r, lookback=3, skip=1)
    # at month 4 (index 3): months 2-3 compounded, month 4's +50% skipped
    assert tr["A"].iloc[3] == pytest.approx(1.1 * 1.1 - 1)
    assert tr["A"].iloc[:2].isna().all()


def test_long_short_weights_are_dollar_neutral():
    sig = pd.DataFrame([np.arange(10.0), [np.nan] * 5 + list(range(5))], columns=list("ABCDEFGHIJ"))
    w = long_short(sig, fraction=0.2, min_assets=10)
    assert len(w) == 1                                    # second row has only 5 assets
    row = w.iloc[0]
    assert row.sum() == pytest.approx(0) and row.abs().sum() == pytest.approx(2)
    assert row["J"] == 0.5 and row["A"] == -0.5
