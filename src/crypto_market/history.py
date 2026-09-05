"""Candle history for Binance.US and Coinbase: 1m/1h/1d OHLCV, resumable.

Two paging schemes, one file layout. Binance.US klines page forward by
startTime (last candle's openTime + 1) and hand back up to 1000 candles at
once, each with the extra fields (trade count, taker-buy volume) this module
also stores. Coinbase's candle endpoint pages by an explicit start/end ISO
window and returns at most 300 candles per call, newest first, without a
trade count or taker-buy split -- those two columns are left empty for
Coinbase rows so the two exchanges share one CSV shape.

Each (exchange, symbol, interval) triple is its own file. Resuming a fetch
means reading the file's last ts and asking only for candles after it, so a
killed run just picks back up; nothing is ever re-fetched or re-written.
Gaps (a symbol not yet listed at the requested start, or genuinely missing
history) are left as gaps -- this module never fabricates a candle.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
HIST_DIR = DATA_DIR / "hist"
HEADER = ["ts", "open", "high", "low", "close", "volume", "trades", "taker_buy_volume"]
PACE_S = 0.1  # ~10 requests/s, comfortably under both exchanges' public limits

BINANCEUS_KLINES = "https://api.binance.us/api/v3/klines"
BINANCEUS_EXCHANGE_INFO = "https://api.binance.us/api/v3/exchangeInfo"
COINBASE_BASE = "https://api.exchange.coinbase.com"
COINBASE_HEADERS = {"User-Agent": "crypto-market/1.0 (history.py)"}
COINBASE_PAGE = 300  # candles per request, per Coinbase's own cap

INTERVAL_MS = {"1m": 60_000, "1h": 3_600_000, "1d": 86_400_000}
COINBASE_GRANULARITY = {"1m": 60, "1h": 3600, "1d": 86400}

BINANCEUS_1M_SYMBOLS = [
    "BTCUSD", "ETHUSD", "SOLUSD", "USDTUSD", "USDCUSD", "BTCUSDT", "ETHUSDT", "ETHBTC",
]
COINBASE_1M_SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "USDT-USD", "ETH-BTC"]


def _path(exchange: str, symbol: str, interval: str, data_dir: Path = HIST_DIR) -> Path:
    return data_dir / f"{exchange}_{symbol.lower()}_{interval}.csv"


def _get(url: str, params: dict, headers: dict | None = None) -> object:
    """GET with retry on transient failure/rate-limit, like butrades._get."""
    while True:
        try:
            r = httpx.get(url, params=params, headers=headers, timeout=30)
        except httpx.HTTPError:
            time.sleep(10)
            continue
        if r.status_code in (429, 418) or r.status_code >= 500:
            time.sleep(10)
            continue
        r.raise_for_status()
        return r.json()


def _last_ts(path: Path) -> tuple[bool, int | None, int]:
    """(file exists, last ts in it or None, row count)."""
    if not path.exists():
        return False, None, 0
    old = pd.read_csv(path)
    if len(old) == 0:
        return True, None, 0
    return True, int(old["ts"].max()), len(old)


def _summarize(path: Path, rows_before: int, elapsed: float) -> None:
    df = pd.read_csv(path)
    new_rows = len(df) - rows_before
    if len(df) == 0:
        print(f"{path.name}: empty, {elapsed:.1f}s")
        return
    first = pd.Timestamp(int(df["ts"].min()), unit="ms")
    last = pd.Timestamp(int(df["ts"].max()), unit="ms")
    print(f"{path.name}: +{new_rows} rows ({len(df)} total), {first.date()} to {last.date()}, {elapsed:.1f}s")


def _fetch_binanceus(symbol: str, interval: str, days: float, data_dir: Path) -> Path:
    path = _path("binanceus", symbol, interval, data_dir)
    exists, since_ms, rows_before = _last_ts(path)
    now_ms = int(time.time() * 1000)
    cur = since_ms + 1 if since_ms is not None else now_ms - int(days * 86_400_000)

    t0 = time.time()
    requests = 0
    with open(path, "a" if exists else "w", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(HEADER)
        while cur < now_ms:
            page = _get(BINANCEUS_KLINES, {
                "symbol": symbol, "interval": interval, "startTime": cur, "limit": 1000,
            })
            requests += 1
            if requests % 200 == 0:
                print(f"binanceus {symbol} {interval}: {requests} requests, reached {pd.Timestamp(cur, unit='ms')}")
            if not page:
                # No candles in this window at all -- either a gap or we're
                # still before the symbol's listing date. Skip forward a
                # full page's worth rather than looping on empty forever.
                cur += INTERVAL_MS[interval] * 1000
                time.sleep(PACE_S)
                continue
            for k in page:
                open_time, close_time = k[0], k[6]
                if close_time >= now_ms:
                    continue  # candle not closed yet
                w.writerow([open_time, k[1], k[2], k[3], k[4], k[5], k[8], k[9]])
            cur = page[-1][0] + 1
            time.sleep(PACE_S)

    elapsed = time.time() - t0
    _summarize(path, rows_before, elapsed)
    return path


def _fetch_coinbase(symbol: str, interval: str, days: float, data_dir: Path) -> Path:
    path = _path("coinbase", symbol, interval, data_dir)
    exists, since_ms, rows_before = _last_ts(path)
    granularity = COINBASE_GRANULARITY[interval]
    now_s = int(time.time())
    cur = (since_ms // 1000 + granularity) if since_ms is not None else now_s - int(days * 86_400)

    t0 = time.time()
    requests = 0
    with open(path, "a" if exists else "w", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(HEADER)
        while cur < now_s:
            end = min(cur + (COINBASE_PAGE - 1) * granularity, now_s)
            start_iso = datetime.fromtimestamp(cur, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            end_iso = datetime.fromtimestamp(end, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            page = _get(
                f"{COINBASE_BASE}/products/{symbol}/candles",
                {"granularity": granularity, "start": start_iso, "end": end_iso},
                headers=COINBASE_HEADERS,
            )
            requests += 1
            if requests % 200 == 0:
                print(f"coinbase {symbol} {interval}: {requests} requests, reached {pd.Timestamp(cur, unit='s')}")
            for t, low, high, o, c, v in sorted(page):
                ts_ms = t * 1000
                if t + granularity > now_s:
                    continue  # candle not closed yet
                w.writerow([ts_ms, o, high, low, c, v, "", ""])
            cur = end + granularity
            time.sleep(PACE_S)

    elapsed = time.time() - t0
    _summarize(path, rows_before, elapsed)
    return path


def fetch(exchange: str, symbol: str, interval: str, days: float, data_dir: Path = HIST_DIR) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    if exchange == "binanceus":
        return _fetch_binanceus(symbol, interval, days, data_dir)
    if exchange == "coinbase":
        return _fetch_coinbase(symbol, interval, days, data_dir)
    raise ValueError(f"unknown exchange {exchange!r}")


def load(exchange: str, symbol: str, interval: str, data_dir: Path = HIST_DIR) -> pd.DataFrame:
    path = _path(exchange, symbol, interval, data_dir)
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume", "trades", "taker_buy_volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ts"] = df["ts"].astype("int64")
    return df


def _binanceus_usd_symbols() -> list[str]:
    """Every symbol quoted in USD with status TRADING, per exchangeInfo."""
    info = _get(BINANCEUS_EXCHANGE_INFO, {})
    return sorted(
        s["symbol"] for s in info["symbols"]
        if s.get("quoteAsset") == "USD" and s.get("status") == "TRADING"
    )


def _coinbase_top_usd_products(n: int = 40) -> list[str]:
    """The n largest -USD products by 24h dollar volume (volume * last price).

    Simplest available approach: list all -USD products, hit /stats for
    each one (volume is in base units, so multiply by last price to rank by
    dollar volume rather than by coin count), sort, take the top n.
    """
    products = _get(f"{COINBASE_BASE}/products", {}, headers=COINBASE_HEADERS)
    ids = [p["id"] for p in products if p["id"].endswith("-USD") and not p.get("trading_disabled")]
    ranked = []
    for i, pid in enumerate(ids):
        try:
            stats = _get(f"{COINBASE_BASE}/products/{pid}/stats", {}, headers=COINBASE_HEADERS)
            dollar_vol = float(stats.get("volume", 0) or 0) * float(stats.get("last", 0) or 0)
        except (httpx.HTTPStatusError, ValueError, TypeError):
            dollar_vol = 0.0
        ranked.append((pid, dollar_vol))
        if (i + 1) % 200 == 0:
            print(f"coinbase stats: {i + 1}/{len(ids)} products checked")
        time.sleep(PACE_S)
    ranked.sort(key=lambda kv: kv[1], reverse=True)
    return [pid for pid, _ in ranked[:n]]


def _plan(only: str | None) -> list[tuple[str, str, str, float]]:
    plan: list[tuple[str, str, str, float]] = []
    if only in (None, "binanceus"):
        plan += [("binanceus", s, "1m", 365) for s in BINANCEUS_1M_SYMBOLS]
        usd_symbols = _binanceus_usd_symbols()
        print(f"binanceus: {len(usd_symbols)} USD symbols with status TRADING")
        plan += [("binanceus", s, "1h", 730) for s in usd_symbols]
        plan += [("binanceus", s, "1d", 2000) for s in usd_symbols]
    if only in (None, "coinbase"):
        plan += [("coinbase", s, "1m", 365) for s in COINBASE_1M_SYMBOLS]
        top = _coinbase_top_usd_products(40)
        print(f"coinbase: top {len(top)} USD products by 24h dollar volume")
        plan += [("coinbase", s, "1h", 730) for s in top]
        plan += [("coinbase", s, "1d", 2000) for s in top]
    return plan


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch Binance.US and Coinbase candle history")
    ap.add_argument("--only", choices=["binanceus", "coinbase"], default=None)
    ap.add_argument("--workers", type=int, default=6, help="targets fetched at once; each request costs ~1 s of exchange latency")
    args = ap.parse_args()

    plan = _plan(args.only)
    print(f"plan: {len(plan)} (exchange, symbol, interval) targets", flush=True)
    t0 = time.time()

    def one(target):
        exchange, symbol, interval, days = target
        try:
            fetch(exchange, symbol, interval, days)
        except Exception as e:  # noqa: BLE001 - keep going through the whole plan
            print(f"{exchange} {symbol} {interval}: FAILED: {e}", flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(one, plan))
    print(f"done: {len(plan)} targets in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
