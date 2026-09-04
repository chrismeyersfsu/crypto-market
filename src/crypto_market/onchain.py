"""Average on-chain transaction size as a proxy for large ('whale') transfers.

blockchain.info publishes two free daily charts, no key, back to Bitcoin's
early years: total confirmed transaction count, and estimated transaction
*value* in USD (their heuristic that strips out change outputs, so it
approximates real economic transfers rather than raw output sum). Dividing
volume by count gives the average transaction size for the day; a robust
z-score against the trailing 90 days flags days where that average is
unusually large, which is what a burst of big transfers looks like without
needing per-transaction data or exchange address labels.

This is a real proxy, not the "large transaction count" on-chain-analytics
vendors sell — those need address clustering to say *where* coins moved
(toward an exchange vs. into cold storage), which isn't available for
free. This only sees that unusually large transfers happened, not why.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import httpx
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "data"
CACHE_FILE = CACHE_DIR / "onchain_avg_tx_size.csv"
STALE_S = 6 * 3600
CHARTS = {"vol_usd": "estimated-transaction-volume-usd", "n_tx": "n-transactions"}


def _fetch(chart: str) -> pd.Series:
    r = httpx.get(f"https://api.blockchain.info/charts/{chart}",
                   params={"timespan": "all", "format": "json", "sampled": "false"}, timeout=30)
    r.raise_for_status()
    v = pd.DataFrame(r.json()["values"])
    v["date"] = pd.to_datetime(v["x"], unit="s").dt.normalize()
    return v.set_index("date")["y"]


def load(refresh: bool = True) -> pd.DataFrame:
    """One row per day: vol_usd, n_tx, avg_tx_usd, and z (a 90-day robust
    z-score of avg_tx_usd computed only from data up to and including that
    day, so it's usable as a same-day signal without lookahead)."""
    CACHE_DIR.mkdir(exist_ok=True)
    stale = not CACHE_FILE.exists() or time.time() - CACHE_FILE.stat().st_mtime > STALE_S
    if refresh and stale:
        d = pd.concat({k: _fetch(v) for k, v in CHARTS.items()}, axis=1).dropna()
        d.index.name = "date"
        d.to_csv(CACHE_FILE)
    else:
        d = pd.read_csv(CACHE_FILE, index_col="date", parse_dates=["date"])

    d["avg_tx_usd"] = d["vol_usd"] / d["n_tx"]
    roll = d["avg_tx_usd"].rolling(90, min_periods=30)
    med = roll.median()
    mad = (d["avg_tx_usd"] - med).abs().rolling(90, min_periods=30).median()
    d["z"] = (d["avg_tx_usd"] - med) / (1.4826 * mad)
    return d.dropna(subset=["z"])


def _welch_t(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna(), b.dropna()
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se else float("nan")


def backtest(d: pd.DataFrame, close: pd.Series, z_thresh: float, hold_days: int) -> dict:
    """Buy at the close of any day where z crosses above `z_thresh`, sell
    `hold_days` calendar days later; legs never overlap (a spike inside an
    open leg is ignored). `close` must be daily and cover d's date range."""
    df = d.join(close.rename("close"), how="inner").sort_index()
    dates = df.index
    legs, i = [], 0
    while i < len(dates):
        if df["z"].iloc[i] > z_thresh:
            target = dates[i] + pd.Timedelta(days=hold_days)
            j = dates.searchsorted(target)
            if j < len(dates):
                legs.append({"entry": dates[i].date().isoformat(), "exit": dates[j].date().isoformat(),
                             "days": (dates[j] - dates[i]).days,
                             "ret": float(df["close"].iloc[j] / df["close"].iloc[i] - 1)})
                i = j + 1
                continue
        i += 1
    trades = pd.DataFrame(legs)

    held = pd.Series(False, index=dates)
    for leg in legs:
        held[(dates > leg["entry"]) & (dates <= leg["exit"])] = True
    ret = df["close"].pct_change().fillna(0)
    equity = (1 + ret.where(held, 0)).cumprod()

    baseline = df["close"].pct_change().shift(-hold_days).dropna()  # every day's own fwd-N return
    spike_days = df.index[df["z"] > z_thresh]
    return {
        "equity": equity, "trades": trades,
        "n_spikes": len(spike_days),
        "fwd_return_spike_mean": float(baseline.reindex(spike_days).dropna().mean()) if len(spike_days) else float("nan"),
        "fwd_return_base_mean": float(baseline.drop(spike_days, errors="ignore").mean()),
        "t_stat": _welch_t(baseline.reindex(spike_days).dropna(), baseline.drop(spike_days, errors="ignore")),
    }
