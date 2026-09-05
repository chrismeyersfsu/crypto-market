"""Hourly BTC-USD OHLCV from Bitfinex (public, no key), back to Apr 2013.

Bitfinex's candle endpoint returns up to 10,000 bars per call and is one
of the few free, keyless sources with real historical intraday depth (most
others — Kraken's public OHLC, Yahoo's intraday — cap out at a rolling
window of a few hundred bars or ~60 days). Prices are Bitfinex's own, a
single exchange rather than a volume-weighted index, so treat this as
representative of BTC's intraday shape, not as an execution-quality feed.
"""
from __future__ import annotations

import time
from pathlib import Path

import httpx
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "data"
CACHE_FILE = CACHE_DIR / "btc_hourly_bitfinex.csv"
API = "https://api-pub.bitfinex.com/v2/candles/trade:1h:tBTCUSD/hist"
EARLIEST = pd.Timestamp("2013-04-01")
STALE_S = 3600
PAGE = 10_000  # bars per call; ~416 days at 1h


def _fetch_page(start: pd.Timestamp) -> list[list]:
    r = httpx.get(API, params={"start": int(start.timestamp() * 1000), "limit": PAGE, "sort": 1}, timeout=30)
    r.raise_for_status()
    return r.json()


def _backfill(since: pd.Timestamp) -> pd.DataFrame:
    rows, cur, now = [], since, pd.Timestamp.utcnow().tz_localize(None)
    while cur < now:
        page = _fetch_page(cur)
        if not page:
            break
        rows += page
        last = pd.Timestamp(page[-1][0], unit="ms")
        if last <= cur:  # no forward progress — stop rather than loop forever
            break
        cur = last + pd.Timedelta(hours=1)
        time.sleep(0.2)
    if not rows:
        return pd.DataFrame(columns=["ts", "open", "close", "high", "low", "volume"])
    df = pd.DataFrame(rows, columns=["ts", "open", "close", "high", "low", "volume"])
    return df.drop_duplicates("ts").sort_values("ts")


def load(refresh: bool = True) -> pd.DataFrame:
    """Hourly bars indexed by UTC timestamp: open, high, low, close, volume."""
    CACHE_DIR.mkdir(exist_ok=True)
    if CACHE_FILE.exists():
        cached = pd.read_csv(CACHE_FILE)
        since = pd.to_datetime(cached["ts"].max(), unit="ms") if len(cached) else EARLIEST
        stale = not len(cached) or (time.time() * 1000 - cached["ts"].max()) / 1000 > STALE_S
    else:
        cached, since, stale = pd.DataFrame(columns=["ts", "open", "close", "high", "low", "volume"]), EARLIEST, True

    if refresh and stale:
        fresh = _backfill(since - pd.Timedelta(hours=1))
        merged = pd.concat([cached, fresh]).drop_duplicates("ts").sort_values("ts")
        merged.to_csv(CACHE_FILE, index=False)
    else:
        merged = cached

    df = merged.copy()
    df.index = pd.to_datetime(df["ts"], unit="ms")
    return df.drop(columns="ts")[["open", "high", "low", "close", "volume"]]
