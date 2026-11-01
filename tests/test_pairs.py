"""Pairs selection and trading on constructed prices with known outcomes (no network)."""

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import coint

from quant_research.pairs import selection as sel
from quant_research.pairs.trading import Spec, run, spread, spread_params, trade_pair


def random_walks(n=600, k=4, seed=0):
    rng = np.random.default_rng(seed)
    return np.exp(np.cumsum(rng.normal(0, 0.01, (n, k)), axis=0))


def test_engle_granger_matches_statsmodels():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(0, 0.01, 500))
    y = 0.3 + 0.8 * x + rng.normal(0, 0.01, 500)
    p, t, b, a = sel.engle_granger(y, x)
    t_sm, p_sm, _ = coint(y, x, trend="c", maxlag=1, autolag=None)
    assert t == pytest.approx(t_sm, rel=1e-6) and p == pytest.approx(p_sm, rel=1e-6)
    assert b == pytest.approx(0.8, abs=0.05) and p < 0.01


def test_independent_random_walks_are_rarely_cointegrated():
    ps = [sel.engle_granger(*np.log(random_walks(500, 2, seed=s)).T)[0] for s in range(200)]
    assert 0.01 < np.mean(np.array(ps) < 0.05) < 0.12            # about the 5% false-positive rate


def test_distance_ranks_co_moving_pair_first():
    base = random_walks(300, 3, seed=2)
    prices = pd.DataFrame({"A": base[:, 0], "B": base[:, 0] * 1.001, "C": base[:, 1], "D": base[:, 2]})
    d = sel.distances(prices, sel.candidate_pairs(None, list(prices.columns)))
    assert d.index[0] == ("A", "B")


def test_candidate_pairs_stay_within_groups():
    pairs = sel.candidate_pairs({"x": ["A", "B"], "y": ["C", "D", "E"]}, ["A", "B", "C", "D"])
    assert pairs == [("A", "B"), ("C", "D")]


def test_converging_trade_pnl_and_costs():
    idx = pd.bdate_range("2024-01-01", periods=5)
    pa = pd.Series([1.0, 1.1, 1.1, 1.0, 1.0], index=idx)       # A rises (rich), then comes back
    pb = pd.Series(1.0, index=idx)
    z = pd.Series([0.0, 3.0, 2.5, -0.1, 0.0], index=idx)       # open day 1, converge day 3
    pnl, trades = trade_pair(pa, pb, z, entry=2.0, cost_bps=10)
    # short A at 1.1, long B at 1.0: A falls to 1.0 -> short leg worth 1/1.1 -> gain 1 - 1/1.1
    gain = 1 - 1.0 / 1.1
    assert pnl.sum() == pytest.approx(gain - 0.001 * 2 - 0.001 * (1 + 1 / 1.1))
    assert len(trades) == 1 and trades[0]["converged"] and trades[0]["days"] == 2
    assert trades[0]["pnl"] == pytest.approx(pnl.sum())
    assert pnl.iloc[1] == pytest.approx(-0.002)                  # opening day: cost only, no price P&L yet


def test_open_trade_is_closed_at_window_end():
    idx = pd.bdate_range("2024-01-01", periods=4)
    pa, pb = pd.Series([1.0, 1.0, 1.2, 1.3], index=idx), pd.Series(1.0, index=idx)
    z = pd.Series([0.0, -2.5, -3.0, -4.0], index=idx)           # A cheap: long A, never converges
    pnl, trades = trade_pair(pa, pb, z, entry=2.0)
    assert pnl.sum() == pytest.approx(0.3) and not trades[0]["converged"]


def test_borrow_accrues_on_the_short_leg():
    idx = pd.bdate_range("2024-01-01", periods=3)
    pa, pb = pd.Series(1.0, index=idx), pd.Series(1.0, index=idx)
    z = pd.Series([3.0, 3.0, 3.0], index=idx)
    pnl, _ = trade_pair(pa, pb, z, entry=2.0, borrow_bps=252)    # 1 bp a day
    assert pnl.sum() == pytest.approx(-0.0002)                   # two days held


def test_spread_z_uses_formation_statistics():
    idx = pd.bdate_range("2024-01-01", periods=4)
    form = pd.DataFrame({"A": [1.0, 1.02, 0.98, 1.0], "B": [1.0, 1.0, 1.0, 1.0]}, index=idx)
    par = spread_params(form, "A", "B", "distance")
    z = spread(form["A"], form["B"], par)
    assert z.mean() == pytest.approx(0) and z.std() == pytest.approx(1)


def test_strategy_profits_from_a_mean_reverting_pair_and_has_no_lookahead():
    rng = np.random.default_rng(5)
    n = 900
    common = np.cumsum(rng.normal(0, 0.01, n))
    gap = np.zeros(n)
    for t in range(1, n):
        gap[t] = 0.9 * gap[t - 1] + rng.normal(0, 0.01)          # stationary spread
    noise = np.cumsum(rng.normal(0, 0.01, (n, 2)), axis=0)
    prices = pd.DataFrame(np.exp(np.column_stack([common + gap, common, noise])), columns=list("ABCD"),
                          index=pd.bdate_range("2015-01-01", periods=n))
    spec = Spec(method="cointegration", formation=252, trading=126, n_pairs=1, step=63)
    daily, picks, trades = run(prices, spec)
    assert (picks[["a", "b"]].apply(tuple, axis=1) == ("A", "B")).mean() > 0.8
    assert daily.sum() > 0 and trades["converged"].mean() > 0.7
    assert daily.index.min() == prices.index[252]               # nothing earned in the first formation window


def test_beta_hedged_legs_match_the_spread():
    """With legs sized 1 : beta, a trade that converges on log A - beta log B makes money."""
    idx = pd.bdate_range("2024-01-01", periods=3)
    beta = 2.0
    pb = pd.Series([1.0, 1.0, 1.10], index=idx)
    pa = pd.Series([1.0, 1.05, 1.10 ** beta], index=idx)          # log A - 2 log B: 0, +0.049 (rich), 0
    z = pd.Series([0.0, 3.0, -0.01], index=idx)
    dollar, _ = trade_pair(pa, pb, z, entry=2.0)
    hedged, trades = trade_pair(pa, pb, z, entry=2.0, wa=2 / 3, wb=4 / 3)
    assert dollar.sum() < 0                                     # $1/$1 loses although the spread converged
    assert hedged.sum() > 0 and trades[0]["converged"]
