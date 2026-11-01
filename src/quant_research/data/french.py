"""Factor returns from Kenneth French's data library (free, no key).

https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

Files are zipped CSVs that hold several tables (monthly, then annual), in
percent. The parser takes the first block of rows whose first field is a
six-digit YYYYMM date, which is the monthly table (for the industry files,
value-weighted), and converts to decimals with missing values as NaN.
"""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
DATASETS = {
    "ff5": "F-F_Research_Data_5_Factors_2x3_CSV.zip",     # Mkt-RF, SMB, HML, RMW, CMA, RF
    "mom": "F-F_Momentum_Factor_CSV.zip",                 # Mom (UMD)
    "ff3": "F-F_Research_Data_Factors_CSV.zip",           # Mkt-RF, SMB, HML, RF
    "ind49": "49_Industry_Portfolios_CSV.zip",            # 49 value-weighted industry returns
    "strev": "F-F_ST_Reversal_Factor_CSV.zip",            # short-term reversal: last month's losers minus winners
    "ff3_daily": "F-F_Research_Data_Factors_daily_CSV.zip",
    "ind49_daily": "49_Industry_Portfolios_daily_CSV.zip",
}
MISSING = -99.99             # the library codes missing values as -99.99 or -999
CACHE = Path(__file__).resolve().parents[3] / "data" / "raw" / "french"
MAX_AGE_DAYS = 30            # the library updates monthly


def parse_monthly(text: str) -> pd.DataFrame:
    """The monthly table from a French CSV, in decimals, indexed by month-end."""
    df = parse_table(text, digits=6)
    df.index = pd.to_datetime(df.index, format="%Y%m") + pd.offsets.MonthEnd(0)
    df.index.name = "date"
    return df


def parse_daily(text: str) -> pd.DataFrame:
    """The first daily (YYYYMMDD) table from a French CSV, in decimals."""
    df = parse_table(text, digits=8)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    return df


def parse_table(text: str, digits: int) -> pd.DataFrame:
    """The first block of rows dated with `digits`-digit dates, values in decimals, index left as text."""
    lines = text.splitlines()
    is_row = lambda fields: fields[0].isdigit() and len(fields[0]) == digits
    start = next(i for i, ln in enumerate(lines) if is_row([f.strip() for f in ln.split(",")]))
    header = [h.strip() for h in lines[start - 1].split(",")]
    rows = []
    for ln in lines[start:]:
        fields = [f.strip() for f in ln.split(",")]
        if not is_row(fields):
            break                                   # end of the first block
        rows.append(fields)
    df = pd.DataFrame(rows, columns=["date"] + header[1:]).set_index("date")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.mask(df <= MISSING) / 100
    df.columns = [c.replace(" ", "") for c in df.columns]
    return df


def download(name: str) -> str:
    resp = requests.get(BASE + DATASETS[name], timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        member = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        return z.read(member).decode("latin-1")


def load(name: str, refresh: bool = False) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.csv"
    if not refresh and path.exists() and time.time() - path.stat().st_mtime < MAX_AGE_DAYS * 86400:
        return pd.read_csv(path, index_col="date", parse_dates=["date"])
    df = (parse_daily if name.endswith("_daily") else parse_monthly)(download(name))
    df.to_csv(path)
    return df


def factors(refresh: bool = False) -> pd.DataFrame:
    """Monthly Mkt-RF, SMB, HML, RMW, CMA, Mom and RF, in decimals, where all are available."""
    ff5 = load("ff5", refresh)
    mom = load("mom", refresh).rename(columns=lambda c: "Mom")
    return ff5.join(mom, how="inner")
