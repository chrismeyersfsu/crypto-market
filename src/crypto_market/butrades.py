"""Binance.US aggregated trade history: every fill, not just minute closes.

aggTrades groups trades that landed at the same price and moment into one
row with a running id; ids are contiguous per symbol, so paging with
fromId never misses or repeats a trade. `buyer_maker` true means a seller
hit a resting bid (the tape prints on the bid); false means a buyer hit a
resting ask.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import httpx
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
BASE = "https://api.binance.us/api/v3/aggTrades"
HEADER = ["ts", "price", "qty", "buyer_maker"]
DEFAULT_SYMBOLS = ["BTCUSD", "ETHUSD", "BTCUSDT", "USDTUSD", "USDCUSD", "USDCUSDT"]
HOUR_MS = 3_600_000


def _get(params: dict) -> list[dict]:
    while True:
        try:
            r = httpx.get(BASE, params=params, timeout=30)
        except httpx.HTTPError:
            time.sleep(10)
            continue
        if r.status_code in (429, 418) or r.status_code >= 500:
            time.sleep(10)
            continue
        r.raise_for_status()
        return r.json()


def _find_start_id(symbol: str, since_ms: int, now_ms: int) -> int | None:
    """First agg id at or after since_ms, by walking 1-hour windows forward
    until one isn't empty."""
    cur = since_ms
    while cur < now_ms:
        end = min(cur + HOUR_MS, now_ms)
        page = _get({"symbol": symbol, "startTime": cur, "endTime": end})
        if page:
            return page[0]["a"]
        cur = end
    return None


def fetch(symbol: str, days: float = 30, data_dir: Path = DATA_DIR) -> Path:
    data_dir.mkdir(exist_ok=True)
    path = data_dir / f"trades_binanceus_{symbol.lower()}.csv"
    now_ms = int(time.time() * 1000)
    since_ms, rows_before = None, 0
    exists = path.exists()
    if exists:
        old = pd.read_csv(path)
        rows_before = len(old)
        if rows_before:
            since_ms = int(old["ts"].max())
    start_ms = since_ms if since_ms is not None else now_ms - int(days * 86_400_000)

    t0 = time.time()
    page_id = _find_start_id(symbol, start_ms, now_ms)
    new_rows, pages, last_t = 0, 0, start_ms
    with open(path, "a" if exists else "w", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(HEADER)
        while page_id is not None:
            page = _get({"symbol": symbol, "fromId": page_id, "limit": 1000})
            if not page:
                break
            pages += 1
            for t in page:
                if since_ms is not None and t["T"] <= since_ms:
                    continue
                w.writerow([t["T"], t["p"], t["q"], 1 if t["m"] else 0])
                new_rows += 1
            last_t = page[-1]["T"]
            if pages % 50 == 0:
                print(f"{symbol}: {pages} pages, {new_rows} rows, reached {pd.Timestamp(last_t, unit='ms')}")
            if len(page) < 1000 or last_t >= now_ms:
                break
            page_id = page[-1]["a"] + 1
            time.sleep(0.1)

    print(f"{symbol}: wrote {new_rows} rows ({rows_before + new_rows} total) in {time.time() - t0:.1f}s")
    return path


def load(symbol: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = data_dir / f"trades_binanceus_{symbol.lower()}.csv"
    df = pd.read_csv(path)
    df = df.drop_duplicates().sort_values("ts").reset_index(drop=True)
    return df.astype({"ts": "int64", "price": "float64", "qty": "float64", "buyer_maker": "bool"})


def main():
    p = argparse.ArgumentParser(description="Fetch Binance.US aggregated trade history")
    p.add_argument("symbols", nargs="*", default=DEFAULT_SYMBOLS)
    p.add_argument("--days", type=float, default=30)
    args = p.parse_args()
    for sym in args.symbols:
        fetch(sym.upper(), args.days)


if __name__ == "__main__":
    main()
