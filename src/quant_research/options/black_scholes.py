"""Black-Scholes-Merton pricing, Greeks, implied volatility and a Monte Carlo check.

European options on an asset paying a continuous dividend yield q:

    C = S e^{-qT} N(d1) - K e^{-rT} N(d2),   P = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)
    d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt T),   d2 = d1 - sigma sqrt T

Greeks are per unit of the underlying: vega per 1.00 of volatility (divide by
100 for a vol point), theta per year (divide by 365 for a calendar day).
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats

N, n = stats.norm.cdf, stats.norm.pdf


def _d1d2(S, K, T, r, sigma, q=0.0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def price(S, K, T, r, sigma, q=0.0, kind="call"):
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    if kind == "call":
        return S * np.exp(-q * T) * N(d1) - K * np.exp(-r * T) * N(d2)
    return K * np.exp(-r * T) * N(-d2) - S * np.exp(-q * T) * N(-d1)


def greeks(S, K, T, r, sigma, q=0.0, kind="call") -> dict:
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    disc_q, disc_r = np.exp(-q * T), np.exp(-r * T)
    gamma = disc_q * n(d1) / (S * sigma * np.sqrt(T))
    vega = S * disc_q * n(d1) * np.sqrt(T)
    common = -S * disc_q * n(d1) * sigma / (2 * np.sqrt(T))
    if kind == "call":
        delta = disc_q * N(d1)
        theta = common - r * K * disc_r * N(d2) + q * S * disc_q * N(d1)
        rho = K * T * disc_r * N(d2)
    else:
        delta = -disc_q * N(-d1)
        theta = common + r * K * disc_r * N(-d2) - q * S * disc_q * N(-d1)
        rho = -K * T * disc_r * N(-d2)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol(target, S, K, T, r, q=0.0, kind="call", lo=1e-4, hi=5.0) -> float:
    """Volatility that reproduces `target`; NaN if the price is outside the no-arbitrage bounds."""
    intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0) if kind == "call" \
        else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
    upper = S * np.exp(-q * T) if kind == "call" else K * np.exp(-r * T)
    if not intrinsic < target < upper:
        return np.nan
    f = lambda s: price(S, K, T, r, s, q, kind) - target
    if f(lo) * f(hi) > 0:
        return np.nan
    return float(optimize.brentq(f, lo, hi, xtol=1e-10))


def monte_carlo_price(S, K, T, r, sigma, q=0.0, kind="call", n_paths=100_000, seed=None) -> tuple[float, float]:
    """Price and standard error by simulating terminal prices under the risk-neutral measure (antithetic draws)."""
    z = np.random.default_rng(seed).standard_normal(n_paths // 2)
    z = np.concatenate([z, -z])
    ST = S * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)
    payoff = np.maximum(ST - K, 0) if kind == "call" else np.maximum(K - ST, 0)
    pv = np.exp(-r * T) * payoff
    pairs = 0.5 * (pv[: n_paths // 2] + pv[n_paths // 2:])        # antithetic pairs are the independent draws
    return float(pv.mean()), float(pairs.std(ddof=1) / np.sqrt(len(pairs)))
