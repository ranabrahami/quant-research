"""Copy the SaaS Valuation Engine's point-in-time signals into this repo.

    python scripts/export_engine_signals.py [path/to/saas-valuation-engine]

The engine's backtest (saas_valuation.research.backtest) rebuilds its valuation
signal at each quarter-end using only data public on that day. This keeps the
signal columns and writes data/engine_signals.csv, so the factor study runs
without the engine installed.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["date", "ticker", "score", "dcf_upside", "comps_upside", "cheap_ev_revenue", "momentum_12_1"]

engine = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "saas-valuation-engine"
panel = pd.read_csv(engine / "data" / "processed" / "backtest" / "panel.csv")
out = panel[COLUMNS].sort_values(["date", "ticker"])
out.to_csv(ROOT / "data" / "engine_signals.csv", index=False)
print(f"{len(out)} rows, {out['date'].nunique()} rebalance dates ({out['date'].min()} to {out['date'].max()}), "
      f"{out['ticker'].nunique()} companies -> data/engine_signals.csv")
