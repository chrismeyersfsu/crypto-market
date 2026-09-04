"""Historical BTC perpetual funding rate from Deribit (public, no key).

Deribit settles funding continuously: `interest_1h` is the fraction of
notional actually paid that hour (positive = longs pay shorts), so summing
it over a period gives the exact carry a fully-hedged short-perp position
would have earned — no compounding-frequency assumption needed. Data starts
~2019-04-25; a request for anything wider than 31 days returns only the
most recent 31 days, so history is paged forward a month at a time and
cached to a flat file that's simply appended to on each refresh.
"""
from __future__ import annotations

import time
from pathlib import Path

import httpx
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "data"
CACHE_FILE = CACHE_DIR / "funding_btc_perpetual.csv"
API = "https://www.deribit.com/api/v2/public/get_funding_rate_history"
EARLIEST = pd.Timestamp("2019-04-25")
CHUNK_DAYS = 28
STALE_S = 3600  # refresh the tail if the cache is older than this


def _fetch_chunk(start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    r = httpx.get(API, params={
        "instrument_name": "BTC-PERPETUAL",
        "start_timestamp": int(start.timestamp() * 1000),
        "end_timestamp": int(end.timestamp() * 1000),
    }, timeout=30)
    r.raise_for_status()
    body = r.json()
    if body.get("error"):
        raise ValueError(body["error"])
    return body["result"]


def _backfill(since: pd.Timestamp) -> pd.DataFrame:
    rows, cur, now = [], since, pd.Timestamp.utcnow().tz_localize(None)
    while cur < now:
        nxt = min(cur + pd.Timedelta(days=CHUNK_DAYS), now)
        rows += _fetch_chunk(cur, nxt)
        cur = nxt
        time.sleep(0.15)
    if not rows:
        return pd.DataFrame(columns=["ts", "rate_1h", "index_price"])
    df = pd.DataFrame(rows)[["timestamp", "interest_1h", "index_price"]]
    df.columns = ["ts", "rate_1h", "index_price"]
    return df.drop_duplicates("ts").sort_values("ts")


def load(refresh: bool = True) -> pd.DataFrame:
    """Hourly rows indexed by UTC timestamp: rate_1h (fraction of notional
    paid that hour, short-perp-receives-positive) and index_price."""
    CACHE_DIR.mkdir(exist_ok=True)
    if CACHE_FILE.exists():
        cached = pd.read_csv(CACHE_FILE)
        since = pd.to_datetime(cached["ts"].max(), unit="ms") if len(cached) else EARLIEST
        stale = not len(cached) or (time.time() * 1000 - cached["ts"].max()) / 1000 > STALE_S
    else:
        cached, since, stale = pd.DataFrame(columns=["ts", "rate_1h", "index_price"]), EARLIEST, True

    if refresh and stale:
        fresh = _backfill(since)
        merged = pd.concat([cached, fresh]).drop_duplicates("ts").sort_values("ts")
        merged.to_csv(CACHE_FILE, index=False)
    else:
        merged = cached

    df = merged.copy()
    df.index = pd.to_datetime(df["ts"], unit="ms")
    return df.drop(columns="ts")


def daily(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per UTC day: summed funding rate and the day's close."""
    df = load() if df is None else df
    g = df.resample("D")
    return pd.DataFrame({
        "rate": g["rate_1h"].sum(),
        "close": g["index_price"].last(),
    }).dropna()
