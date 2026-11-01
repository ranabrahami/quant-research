# Quant Research

**Quantitative equity research studies, each built on tested code and published as an executed notebook.**

Every study asks its own question, checks the method against a known answer first, and reports what the data can't support as clearly as what it can. Studies build on each other and on my [SaaS Valuation Engine](https://github.com/ranabrahami/saas-valuation-engine), whose point-in-time valuation signal several of them test.

| # | Study | Question | Status |
|---|---|---|---|
| 01 | [Factor model](notebooks/01_factor_model.ipynb) | Is Berkshire's record alpha or factor exposure? Is my engine's valuation signal alpha or a disguised factor bet? | ✅ |
| 02 | [Backtester: industry momentum](notebooks/02_backtester.ipynb) | Does a published anomaly survive factor controls, publication, trading costs and crashes? | ✅ |
| 03 | [Risk: VaR, ES and Monte Carlo](notebooks/03_risk_var.ipynb) | Which VaR models keep their promise over 90 years of crashes, and where do they fail? | ✅ |
| 04 | Pairs trading | Does cointegration within software survive out of sample? | Planned |
| 05 | Options | Black-Scholes pricing and Greeks against market prices | Planned |

*Educational project, not investment advice.*

---

## 01 · Factor model

Regresses monthly excess returns on the Fama-French five factors plus momentum, with Newey-West standard errors.

### Part A: Berkshire Hathaway, 1990–2026 (441 months)

| | Market | Size | Value | Profitability | Alpha / yr |
|---|---|---|---|---|---|
| Loading | 0.73 | −0.36 | +0.52 | +0.24 | +3.6% |
| t (Newey-West) | 9.6 | −3.5 | 4.7 | 2.9 | 1.4 |

- **The tool reproduces the published result.** Berkshire is a low-beta, large-cap, value-and-quality portfolio, as *Buffett's Alpha* (Frazzini, Kabiller and Pedersen) found.
- **The factors absorb the alpha.** It is significant under CAPM (6.0%, t 2.1) but not after the factors are added (3.6%, t 1.4).
- **The alpha has decayed.** It fell from 30%+ a year in the late 1990s (rolling 5-year windows) to about zero over the last decade.

<img src="docs/images/01_berkshire_rolling.png" alt="Berkshire rolling 60-month factor loadings and alpha">

### Part B: does the engine's valuation signal earn alpha? (2023–26)

Each quarter from March 2023 the portfolio goes long the six companies the engine ranked cheapest for their growth and short the six most expensive. The signals are rebuilt using only data public on each date.

**Step 1: find out when the signal pays.** Measured month by month after each signal date, most of the return arrives 7–12 months later. A quarterly-rebalanced portfolio captures only the weak first three months.

<img src="docs/images/01_event_time.png" alt="Long-short return by months since the signal">

**Step 2: match the holding period to the signal.** The same rankings, held two ways:

| Design | Long − short / yr | Alpha, FF5 + Mom | t |
|---|---|---|---|
| Rebalanced quarterly | −1.0% | +13.3% | 1.2 |
| **12-month overlapping cohorts** (Jegadeesh-Titman) | **+5.7%** | **+17.5%** | **1.6** |

**Step 3: read the exposures.**
- **It isn't a disguised factor bet.** Alpha stays positive under CAPM, FF3 and FF5 + Mom.
- **The one significant tilt is to small caps.** Small caps, momentum and the market's rise all ran against the portfolio in 2023–26, so the factors make the alpha *bigger*, not smaller.
- **It is not significant.** With residual volatility of 23%, this alpha would need about 7 years of data to reach t = 2. There are 3.5 years.

**Conclusion.** The signal is consistent with the engine's backtest (12-month IC 0.21): encouraging, not proven. The limits are 18 stocks in one sector, one regime, today's survivors and gross returns. Market-wide factors also explain little of a spread *within* one sector.

---

## 02 · Backtester: industry momentum, 1927–2026

A monthly backtester (`backtest/engine.py`) takes target weights and returns net performance:
- **Timing:** a signal dated *t* only earns returns from *t + 1*.
- **Drift:** positions drift between trades, and turnover is measured against the drifted weights.
- **Costs:** trading costs are charged per unit of turnover, and borrow costs on the short side.
- **Statistics:** block-bootstrap confidence intervals and the deflated Sharpe ratio.

**The test case** is Moskowitz and Grinblatt's (1999) industry momentum strategy: long the 20% of French's 49 industries with the best past-year returns, short the worst 20%. Industries don't get delisted, so the test has no survivorship bias. The design was fixed before running it.

| Test | Result |
|---|---|
| Raw performance | 7.8% a year, Sharpe 0.46 (95% CI 0.22–0.71), t 4.5 |
| Just another factor? | **Mostly yes.** FF3 alpha 10% (t 5.3), but the loading on stock momentum is 0.90 (t 33) and the FF5 + Mom alpha falls to 1.7% (t 1.4) |
| Survived publication? | Sharpe 0.75 in the paper's 1963–95 sample, 0.33 since 1996 |
| Survives costs? | It turns over ~12x a year. Break-even is 67 bps per trade: it survives today's costs but not historical ones |
| Tail risk | Max drawdown −85%. All six worst months were market rebounds after a bear market (Aug 1932 −58%, Apr 2009 −35%) |
| Data snooping | Ranks 16th of 96 variants; deflated Sharpe 0.96 |

<img src="docs/images/02_industry_momentum.png" alt="Industry momentum growth of $1 since 1927, log scale">

**The crash pattern makes the risk partly predictable.** After the market has fallen over two years, the strategy loses 24% a year (annualized) in months the market rises and gains 35% in months it falls. After a crash, the losing industries are the riskiest ones, so the short side becomes a bet against a market rebound.

**The valuation engine's signal with costs:** with the 12-month design, about 5.7% gross becomes about 4.8% net at 10 bps per trade and 50 bps a year to borrow. Break-even is above 130 bps, so costs aren't the binding constraint; the 3.5-year sample is.

<img src="docs/images/02_variants.png" alt="Sharpe ratios of 96 variants and mean Sharpe by lookback and holding period">

---

## 03 · Risk: VaR, Expected Shortfall and Monte Carlo, 1926–2026

Five 99% VaR models are backtested on about 23,500 days each (`risk/`). The two portfolios are the US market and study 02's industry momentum strategy, valued daily. Every forecast uses only past data.

**One-day VaR:**

| Model | Exceptions (target 1%) | Clustered? | Red Basel years | ES Z2 (0 = right) |
|---|---|---|---|---|
| Normal | 2.2% | yes | 15–19 | +1.8 |
| Historical simulation | 1.4% | yes | 7–10 | +0.5 |
| EWMA (RiskMetrics) | 1.9–2.1% | yes | 0–7 | +1.1 to +1.6 |
| **Filtered historical** | **1.2%** | mildly | **0** | **+0.2** |
| GARCH(1,1)-t | 1.3–1.4% | market: no | 0–1 | +0.3 to +0.4 |

- **Fat tails:** each series has about 100 days beyond 5 standard deviations. A normal distribution expects 0.015.
- **No model foresees the crash day:** 19 October 1987 was a 17% loss against a best VaR of 5%. What separates the models is how fast they adapt. In March 2020, historical VaR was breached 9 times and filtered historical 3.
- **Ten days, by Monte Carlo:** a simulated GARCH-t first did *worse* than √10 scaling. With fat-tailed errors the maximum-likelihood mean behaves like the **median** (0.08% a day, against an actual mean of 0.04%), and over 10 days that takes ~0.8 pts off every VaR. With a zero drift the market's 10-day VaR passes both tests (1.4% exceptions).
- **Momentum's multi-day crashes beat every model** (≥ 2% exceptions at 10 days, apart from the very conservative historical method). They come from its exposure to a market rebound, not from volatility, which argues for scenario tests alongside VaR.
- **The models fail in opposite regimes:** volatility models miss in calm markets, historical simulation in turbulent ones.

<img src="docs/images/03_crises.png" alt="Daily returns against historical and filtered historical VaR in March 2020 and spring 2009">

---

## Layout

```
src/quant_research/
  data/french.py         Kenneth French factor library: download, parse, cache
  data/prices.py         Yahoo Finance monthly total returns, cached
  stats.py               OLS with Newey-West errors, rolling regressions
  factors/model.py       factor regressions, attribution, rolling exposures
  factors/portfolios.py  signal -> long-short portfolios (rebalance or overlapping cohorts), event-time returns
  backtest/engine.py     weights -> gross/net returns with drift, turnover, trading and borrow costs
  backtest/metrics.py    Sharpe, drawdown, block-bootstrap CI, probabilistic and deflated Sharpe, break-even cost
  strategies/momentum.py trailing-return signals and long-short weights
  risk/var.py            normal, historical, EWMA and filtered-historical VaR / ES forecasts
  risk/garch.py          rolling GARCH(1,1)-t forecasts and Monte Carlo multi-day losses
  risk/backtest.py       Kupiec, Christoffersen, Basel traffic light, Acerbi-Szekely ES test
notebooks/               one executed notebook per study
data/engine_signals.csv  the valuation engine's point-in-time signals (scripts/export_engine_signals.py)
tests/                   synthetic-data tests, no network
```

## Run it

```bash
git clone https://github.com/ranabrahami/quant-research.git
cd quant-research
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                   # 45 tests, no network
jupyter lab notebooks/01_factor_model.ipynb
```

Data downloads on first run (French library and Yahoo Finance, no keys needed) and is cached in `data/raw/`.
