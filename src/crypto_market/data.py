"""Daily OHLC from Yahoo Finance's chart endpoint, cached on disk.

No API key. One JSON file per ticker under data/, refreshed when older
than CACHE_TTL. Crypto spot tickers (BTC-USD, ETH-USD) include weekend
bars; ETFs and stocks only have trading days.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "data"
CACHE_TTL = 6 * 3600
UA = "Mozilla/5.0 (X11; Linux x86_64) crypto-market/0.1"


def _fetch(ticker: str) -> dict:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    # range=max silently coarsens to weekly bars; an explicit window keeps 1d.
    r = httpx.get(url, params={"period1": 0, "period2": int(time.time()) + 86400,
                               "interval": "1d", "events": "div"},
                  headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    body = r.json()["chart"]
    if body.get("error"):
        raise ValueError(body["error"].get("description", "unknown ticker"))
    return body["result"][0]


def _to_frame(result: dict) -> pd.DataFrame:
    q = result["indicators"]["quote"][0]
    df = pd.DataFrame({
        "ts": result["timestamp"],
        "open": q["open"], "high": q["high"], "low": q["low"],
        "close": q["close"], "volume": q["volume"],
    }).dropna(subset=["open", "close"])
    tz = result["meta"].get("exchangeTimezoneName", "UTC")
    # Bar date in the exchange's local day, so a 00:00 UTC crypto bar and a
    # 09:30 ET equity bar both land on the calendar day people expect.
    df["date"] = (pd.to_datetime(df["ts"], unit="s", utc=True)
                  .dt.tz_convert(tz).dt.normalize().dt.tz_localize(None))
    df = df.drop(columns="ts").drop_duplicates("date").set_index("date").sort_index()
    df.attrs["meta"] = {
        "symbol": result["meta"].get("symbol"),
        "name": result["meta"].get("longName") or result["meta"].get("shortName"),
        "type": result["meta"].get("instrumentType"),
        "currency": result["meta"].get("currency"),
    }
    return df


def load(ticker: str) -> pd.DataFrame:
    ticker = ticker.upper().strip()
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{ticker}.json"
    if path.exists() and time.time() - path.stat().st_mtime < CACHE_TTL:
        result = json.loads(path.read_text())
    else:
        result = _fetch(ticker)
        path.write_text(json.dumps(result))
    return _to_frame(result)
