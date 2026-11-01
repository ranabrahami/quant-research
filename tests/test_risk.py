"""VaR models, GARCH Monte Carlo and VaR backtests on simulated data with known answers (no network)."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from quant_research.backtest.engine import daily_returns, run
from quant_research.data.french import parse_daily
from quant_research.risk import backtest as vb
from quant_research.risk import garch, var

DAYS = pd.bdate_range("2000-01-03", periods=6000)


def normal_returns(sd=0.01, seed=0):
    return pd.Series(np.random.default_rng(seed).normal(0, sd, len(DAYS)), index=DAYS)


def test_normal_var_and_es_of_standard_normal():
    v, es = var.normal_var_es(0.0, 1.0, 0.99)
    assert v == pytest.approx(2.326, abs=1e-3)
    assert es == pytest.approx(2.665, abs=1e-3)


def test_ewma_recursion():
    r = pd.Series([0.0, 0.02, -0.01, 0.0])
    vol = var.ewma_vol(r, lam=0.9, seed_window=4)
    s2_0 = np.var([0.0, 0.02, -0.01, 0.0])
    assert vol.iloc[1] ** 2 == pytest.approx(0.9 * s2_0)
    assert vol.iloc[2] ** 2 == pytest.approx(0.9 * vol.iloc[1] ** 2 + 0.1 * 0.02 ** 2)


def test_forecasts_do_not_see_the_day_they_forecast():
    r = normal_returns()
    base = var.forecasts(r, window=250)
    shocked = r.copy()
    shocked.iloc[3000] = -0.5
    after = var.forecasts(shocked, window=250)
    assert after.iloc[3000].to_numpy() == pytest.approx(base.iloc[3000].to_numpy())
    assert (after.iloc[3001] > base.iloc[3001]).all()            # the shock enters the next day


def test_models_are_calibrated_on_normal_data():
    r = normal_returns()
    f = var.forecasts(r, window=500)
    for m in ("normal", "historical", "ewma", "fhs"):
        rep = vb.report(r, f[f"var_{m}"], f[f"es_{m}"])
        assert 0.006 < rep["rate"] < 0.015, m
        assert rep["Kupiec p"] > 0.01, m


def test_kupiec_accepts_the_right_rate_and_rejects_a_wrong_one():
    exact = pd.Series([True] * 10 + [False] * 990)
    assert vb.kupiec(exact, 0.01)[1] == pytest.approx(1.0)
    assert vb.kupiec(pd.Series([True] * 30 + [False] * 970), 0.01)[1] < 1e-4


def test_christoffersen_detects_clustering():
    spread = pd.Series(False, index=range(2000))
    spread.iloc[::100] = True
    clustered = pd.Series(False, index=range(2000))
    clustered.iloc[1000:1020] = True
    assert vb.christoffersen(spread)[1] > 0.1
    assert vb.christoffersen(clustered)[1] < 1e-6


def test_basel_zones():
    exc = pd.Series(False, index=range(750))
    exc.iloc[:3] = True                  # block 1: 3 -> green
    exc.iloc[250:257] = True             # block 2: 7 -> yellow
    exc.iloc[500:512] = True             # block 3: 12 -> red
    assert list(vb.basel_zones(exc)) == ["green", "yellow", "red"]


def test_es_z2_is_zero_when_es_is_right():
    r = normal_returns(sd=1.0, seed=4)
    v, es = var.normal_var_es(0.0, 1.0, 0.99)
    z2 = vb.es_z2(r, pd.Series(v, index=r.index), pd.Series(es, index=r.index))
    assert abs(z2) < 0.2
    z2_low = vb.es_z2(r, pd.Series(v, index=r.index), pd.Series(es / 2, index=r.index))
    assert z2_low > 0.5                  # ES understated -> positive


def test_unit_variance_t_quantile_and_es():
    rng = np.random.default_rng(5)
    nu = 5.0
    x = rng.standard_t(nu, 2_000_000) * np.sqrt((nu - 2) / nu)
    assert x.std() == pytest.approx(1.0, abs=0.01)
    q = np.quantile(x, 0.01)
    assert garch.t_quantile(0.01, nu) == pytest.approx(q, abs=0.02)
    assert garch.t_es(0.01, nu) == pytest.approx(x[x <= q].mean(), abs=0.05)


def test_monte_carlo_one_day_matches_the_analytic_quantile():
    p = garch.GarchParams(mu=0.0, omega=1e-6, alpha=0.08, beta=0.9, nu=6.0)
    sims = garch.simulate_losses(p, sigma2_next=1e-4, horizon=1, n=400_000, seed=1)
    assert np.quantile(sims, 0.01) == pytest.approx(0.01 * garch.t_quantile(0.01, 6.0), rel=0.03)


def test_monte_carlo_multi_day_tails():
    """Summing days thins fat tails (CLT); volatility clustering fattens them again."""
    long_run = 1e-4
    iid = garch.GarchParams(mu=0.0, omega=long_run, alpha=0.0, beta=0.0, nu=5.0)
    clustered = garch.GarchParams(mu=0.0, omega=long_run * 0.01, alpha=0.15, beta=0.84, nu=5.0)
    q = {name: -np.quantile(garch.simulate_losses(par, long_run, horizon=10, n=400_000, seed=2), 0.01)
         for name, par in (("iid", iid), ("clustered", clustered))}
    sqrt_rule = np.sqrt(10) * 0.01 * -garch.t_quantile(0.01, 5.0)
    assert q["iid"] < sqrt_rule
    assert q["clustered"] > q["iid"] * 1.05


def test_garch_recovers_simulated_parameters():
    p = garch.GarchParams(mu=0.0005, omega=2e-6, alpha=0.08, beta=0.9, nu=6.0)
    rng = np.random.default_rng(7)
    n = 4000
    e = rng.standard_t(p.nu, n) * np.sqrt((p.nu - 2) / p.nu)
    s2, r = p.omega / (1 - p.persistence), np.empty(n)
    for t in range(n):
        r[t] = p.mu + np.sqrt(s2) * e[t]
        s2 = p.omega + p.alpha * (r[t] - p.mu) ** 2 + p.beta * s2
    out, fits = garch.rolling_forecasts(pd.Series(r, index=pd.bdate_range("2000-01-03", periods=n)),
                                        window=3000, refit=500)
    assert fits["alpha"].iloc[0] == pytest.approx(0.08, abs=0.04)
    assert fits["beta"].iloc[0] == pytest.approx(0.9, abs=0.05)
    assert len(out) == n - 3000 and out["var_garch_t"].gt(0).all()


def test_daily_returns_compound_to_the_monthly_backtest():
    rng = np.random.default_rng(8)
    days = pd.bdate_range("2021-01-01", "2021-12-31")
    d = pd.DataFrame(rng.normal(0.0005, 0.01, (len(days), 3)), index=days, columns=list("ABC"))
    monthly = (1 + d).resample("ME").prod() - 1
    w = pd.DataFrame({"A": 1.0, "B": -0.5, "C": -0.5}, index=monthly.index[:-1])
    bt = run(w, monthly)
    daily = daily_returns(bt.held, d)
    compounded = (1 + daily).resample("ME").prod() - 1
    assert compounded.loc[bt.gross.index].to_numpy() == pytest.approx(bt.gross.to_numpy(), abs=1e-12)


def test_parse_daily():
    df = parse_daily("intro\n,Mkt-RF,RF\n19260701,   0.10,  0.01\n19260702,  -99.99,  0.01\n\n")
    assert df.index[0] == pd.Timestamp("1926-07-01")
    assert df.at[pd.Timestamp("1926-07-01"), "Mkt-RF"] == pytest.approx(0.001)
    assert np.isnan(df.at[pd.Timestamp("1926-07-02"), "Mkt-RF"])
