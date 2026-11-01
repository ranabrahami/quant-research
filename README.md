# Quant Research

**Quantitative equity research studies, each built on tested code and published as an executed notebook.**

Every study asks its own question, checks the method against a known answer first, and reports what the data can't support as clearly as what it can. Studies build on each other and on my [SaaS Valuation Engine](https://github.com/ranabrahami/saas-valuation-engine), whose point-in-time valuation signal several of them test.

| # | Study | Question | Status |
|---|---|---|---|
| 01 | [Factor model](notebooks/01_factor_model.ipynb) | Is Berkshire's record alpha or factor exposure? Is my engine's valuation signal alpha or a disguised factor bet? | ✅ |
| 02 | Backtester | A reusable event-driven backtester with costs, turnover and capacity | Planned |
| 03 | Risk: Monte Carlo and VaR | How wrong are normal-distribution VaR estimates for a factor portfolio? | Planned |
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

## Layout

```
src/quant_research/
  data/french.py         Kenneth French factor library: download, parse, cache
  data/prices.py         Yahoo Finance monthly total returns, cached
  stats.py               OLS with Newey-West errors, rolling regressions
  factors/model.py       factor regressions, attribution, rolling exposures
  factors/portfolios.py  signal -> long-short portfolios (rebalance or overlapping cohorts), event-time returns
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
pytest                                   # 15 tests, no network
jupyter lab notebooks/01_factor_model.ipynb
```

Data downloads on first run (French library and Yahoo Finance, no keys needed) and is cached in `data/raw/`.
