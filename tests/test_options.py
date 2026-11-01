"""Black-Scholes, implied volatility, Monte Carlo and delta hedging against known answers (no network)."""

import numpy as np
import pandas as pd
import pytest

from quant_research.options import black_scholes as bs
from quant_research.options.hedging import hedge


def test_textbook_prices():
    # Hull, Options, Futures and Other Derivatives, Example 15.6: S 42, K 40, r 10%, sigma 20%, T 0.5
    assert bs.price(42, 40, 0.5, 0.10, 0.20, kind="call") == pytest.approx(4.76, abs=0.005)
    assert bs.price(42, 40, 0.5, 0.10, 0.20, kind="put") == pytest.approx(0.81, abs=0.005)


def test_put_call_parity_with_dividends():
    S, K, T, r, s, q = 100, 95, 0.75, 0.04, 0.3, 0.02
    c, p = bs.price(S, K, T, r, s, q, "call"), bs.price(S, K, T, r, s, q, "put")
    assert c - p == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-10)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_match_finite_differences(kind):
    S, K, T, r, s, q, h = 100.0, 105.0, 0.4, 0.03, 0.25, 0.01, 1e-4
    g = bs.greeks(S, K, T, r, s, q, kind)
    f = lambda **kw: bs.price(**{"S": S, "K": K, "T": T, "r": r, "sigma": s, "q": q, "kind": kind, **kw})
    assert g["delta"] == pytest.approx((f(S=S + h) - f(S=S - h)) / (2 * h), rel=1e-5)
    assert g["gamma"] == pytest.approx((f(S=S + 0.01) - 2 * f() + f(S=S - 0.01)) / 0.01 ** 2, rel=1e-4)
    assert g["vega"] == pytest.approx((f(sigma=s + h) - f(sigma=s - h)) / (2 * h), rel=1e-5)
    assert g["theta"] == pytest.approx(-(f(T=T + h) - f(T=T - h)) / (2 * h), rel=1e-5)
    assert g["rho"] == pytest.approx((f(r=r + h) - f(r=r - h)) / (2 * h), rel=1e-5)


@pytest.mark.parametrize("kind,K", [("call", 90), ("call", 120), ("put", 80), ("put", 110)])
def test_implied_vol_round_trip(kind, K):
    p = bs.price(100, K, 0.25, 0.04, 0.37, 0.01, kind)
    assert bs.implied_vol(p, 100, K, 0.25, 0.04, 0.01, kind) == pytest.approx(0.37, abs=1e-8)


def test_implied_vol_rejects_arbitrage_prices():
    assert np.isnan(bs.implied_vol(0.5, 100, 90, 0.25, 0.0, kind="call"))      # below intrinsic
    assert np.isnan(bs.implied_vol(101, 100, 90, 0.25, 0.0, kind="call"))      # above the stock price


def test_monte_carlo_converges_to_black_scholes():
    exact = bs.price(100, 100, 1.0, 0.05, 0.2, kind="put")
    est, se = bs.monte_carlo_price(100, 100, 1.0, 0.05, 0.2, kind="put", n_paths=400_000, seed=0)
    assert abs(est - exact) < 3 * se and se < 0.02


def gbm_path(sigma, days=63, S0=100.0, seed=0, mu=0.0):
    z = np.random.default_rng(seed).standard_normal(days)
    dt = 1 / 252
    return pd.Series(S0 * np.exp(np.cumsum(np.r_[0, (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z])))


def test_hedging_at_the_true_vol_roughly_breaks_even():
    pnls = [hedge(gbm_path(0.2, seed=s), 100, 0.2, 0.0)["pnl"] for s in range(300)]
    premium = hedge(gbm_path(0.2), 100, 0.2, 0.0)["premium"]
    assert abs(np.mean(pnls)) < 0.05 * premium               # discrete hedging error averages out
    assert np.std(pnls) < 0.25 * premium


def test_short_vol_wins_when_realized_is_below_implied_and_loses_above():
    calm = np.mean([hedge(gbm_path(0.10, seed=s), 100, 0.20, 0.0)["pnl"] for s in range(100)])
    wild = np.mean([hedge(gbm_path(0.35, seed=s), 100, 0.20, 0.0)["pnl"] for s in range(100)])
    assert calm > 0 > wild


def test_theta_gamma_approximation_tracks_the_pnl():
    res = [hedge(gbm_path(v, seed=s), 100, 0.2, 0.01) for s, v in enumerate(np.linspace(0.1, 0.35, 60))]
    pnl, approx = np.array([x["pnl"] for x in res]), np.array([x["approx_pnl"] for x in res])
    assert np.corrcoef(pnl, approx)[0, 1] > 0.95
    long = hedge(gbm_path(0.3, seed=1), 100, 0.2, 0.0, position=1)
    assert np.sign(long["pnl"]) == np.sign(long["approx_pnl"])


def test_long_and_short_hedged_positions_are_mirror_images():
    path = gbm_path(0.25, seed=9)
    short, long = hedge(path, 100, 0.2, 0.03, 0.01, position=-1), hedge(path, 100, 0.2, 0.03, 0.01, position=1)
    assert short["pnl"] == pytest.approx(-long["pnl"], abs=1e-10)
