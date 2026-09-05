"""Cross-exchange price gaps: is the same coin ever cheap on one venue and
dear on another for long enough to buy one and sell the other?

Two datasets answer two different questions.

Minute bars (Bitfinex, Coinbase, Bitstamp; public, 30 days) say whether a
gap *exists* and how it behaves minute to minute. The rule tested is the
only honest one at this resolution: notice the gap at the close of one
minute, trade at the close of the next, pay both venues' fees.

Live ticks (best bid/ask streamed from each venue's public WebSocket and
recorded here with the receive-time clock) say how long a gap *lasts* — the
number that decides whether a home connection could ever reach it.

Both separate a venue's steady premium (Bitfinex has traded above the others
for years because moving dollars off it is slow) from the flicker around it.
A steady premium is not tradeable more than once: after the first round you
hold the wrong asset on each venue and must pay to move it back. Only the
flicker is repeatable.
"""
from __future__ import annotations

import asyncio
import csv
import json
import logging
import time
from itertools import combinations
from pathlib import Path

import httpx
import pandas as pd

log = logging.getLogger(__name__)
CACHE_DIR = Path(__file__).resolve().parents[2] / "data"
TICKS_FILE = CACHE_DIR / "ticks_btcusd.csv"
HISTORY_DAYS = 30
KEEP_H = 48  # of live ticks on disk
MAX_GAP_MS = 60_000  # a hole this long in the tick stream means the collector was down; no episode spans it
STALE_S = 600

# ---------------------------------------------------------------- minute bars

def _bitfinex(start: pd.Timestamp) -> pd.DataFrame:
    rows, cur, now = [], start, pd.Timestamp.utcnow().tz_localize(None)
    while cur < now:
        r = httpx.get("https://api-pub.bitfinex.com/v2/candles/trade:1m:tBTCUSD/hist",
                      params={"start": int(cur.timestamp() * 1000), "limit": 10_000, "sort": 1}, timeout=30)
        r.raise_for_status()
        page = r.json()
        if not page:
            break
        rows += [(p[0], p[2]) for p in page]
        last = pd.Timestamp(page[-1][0], unit="ms")
        if last <= cur:
            break
        cur = last + pd.Timedelta(minutes=1)
        time.sleep(0.25)
    return pd.DataFrame(rows, columns=["ts", "close"])


def _coinbase(start: pd.Timestamp) -> pd.DataFrame:
    rows, cur, now = [], start, pd.Timestamp.utcnow().tz_localize(None)
    while cur < now:
        end = min(cur + pd.Timedelta(minutes=300), now)
        r = httpx.get("https://api.exchange.coinbase.com/products/BTC-USD/candles",
                      params={"granularity": 60, "start": cur.isoformat(), "end": end.isoformat()}, timeout=30)
        r.raise_for_status()
        rows += [(p[0] * 1000, p[4]) for p in r.json()]
        cur = end
        time.sleep(0.15)
    return pd.DataFrame(rows, columns=["ts", "close"])


def _bitstamp(start: pd.Timestamp) -> pd.DataFrame:
    rows, cur, now = [], start, pd.Timestamp.utcnow().tz_localize(None)
    while cur < now:
        r = httpx.get("https://www.bitstamp.net/api/v2/ohlc/btcusd/",
                      params={"step": 60, "limit": 1000, "start": int(cur.timestamp())}, timeout=30)
        r.raise_for_status()
        page = r.json()["data"]["ohlc"]
        if not page:
            break
        rows += [(int(p["timestamp"]) * 1000, float(p["close"])) for p in page]
        last = pd.Timestamp(int(page[-1]["timestamp"]), unit="s")
        if last <= cur:
            break
        cur = last + pd.Timedelta(minutes=1)
        time.sleep(0.2)
    return pd.DataFrame(rows, columns=["ts", "close"])


HISTORY = {"bitfinex": _bitfinex, "coinbase": _coinbase, "bitstamp": _bitstamp}


def minutes(refresh: bool = True) -> pd.DataFrame:
    """One column of 1-minute closes per exchange, aligned on the UTC minute."""
    CACHE_DIR.mkdir(exist_ok=True)
    floor = pd.Timestamp.utcnow().tz_localize(None).floor("min") - pd.Timedelta(days=HISTORY_DAYS)
    cols = {}
    for ex, fetch in HISTORY.items():
        f = CACHE_DIR / f"minute_{ex}.csv"
        cached = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["ts", "close"])
        cached = cached[cached["ts"] >= floor.timestamp() * 1000]
        since = pd.Timestamp(cached["ts"].max(), unit="ms") if len(cached) else floor
        stale = not len(cached) or time.time() - cached["ts"].max() / 1000 > STALE_S
        if refresh and stale:
            try:
                fresh = fetch(since - pd.Timedelta(minutes=1))
                cached = pd.concat([cached, fresh]).drop_duplicates("ts", keep="last").sort_values("ts")
                cached.to_csv(f, index=False)
            except Exception as e:  # keep serving what we have; one venue's outage shouldn't blank the tab
                log.warning("minute history %s: %s", ex, e)
        s = pd.Series(pd.to_numeric(cached["close"]).values,
                      index=pd.to_datetime(pd.to_numeric(cached["ts"]), unit="ms"), name=ex)
        cols[ex] = s[~s.index.duplicated()]
    return pd.DataFrame(cols).sort_index()


def gap(df: pd.DataFrame, a: str, b: str, fee_bp: float, max_hold: int = 60) -> dict:
    """Gap of a over b in basis points, split into a steady level (causal
    one-day rolling median) and the flicker around it.

    The trade: when the flicker exceeds the cost at the close of minute t,
    buy on the cheap venue and sell on the dear one at the close of t+1, then
    reverse both when the flicker crosses back through zero (or after
    `max_hold` minutes regardless). That is four fee-paying fills, and it
    leaves your balances where they started, so it can be repeated. Profit is
    the flicker captured at entry minus what remained at exit, minus fees."""
    both = df[[a, b]].dropna()
    g = (both[a] - both[b]) / ((both[a] + both[b]) / 2) * 1e4
    level = g.rolling("1D", min_periods=60).median().shift(1)
    dev = (g - level).dropna()
    cost = 4 * fee_bp
    d = dev.values
    trades, i, n = [], 0, len(d)
    while i < n - 1:
        if abs(d[i]) > cost:
            s = 1 if d[i] > 0 else -1
            e = i + 1
            x = e
            while x < min(e + max_hold, n - 1) and s * d[x] > 0:
                x += 1
            trades.append({"signal_bp": s * d[i], "entry_bp": s * d[e], "exit_bp": s * d[x],
                           "profit_bp": s * (d[e] - d[x]) - cost, "hold_min": x - e,
                           "converged": s * d[x] <= 0})
            i = x + 1
        else:
            i += 1
    t = pd.DataFrame(trades)
    ac = float(dev.autocorr(1)) if len(dev) > 10 else float("nan")
    step = max(len(g) // 2000, 1)
    days = (dev.index[-1] - dev.index[0]).total_seconds() / 86400 if len(dev) > 1 else float("nan")
    return {
        "pair": f"{a}/{b}", "n_minutes": int(len(dev)), "days": days,
        "level_bp": float(level.dropna().median()) if level.notna().any() else float("nan"),
        "abs_dev_median_bp": float(dev.abs().median()), "dev_p95_bp": float(dev.abs().quantile(0.95)),
        "lag1_autocorr": ac,
        "trades": int(len(t)),
        "still_open_at_entry": float((t["entry_bp"] > cost).mean()) if len(t) else float("nan"),
        "mean_profit_bp": float(t["profit_bp"].mean()) if len(t) else float("nan"),
        "total_profit_bp": float(t["profit_bp"].sum()) if len(t) else 0.0,
        "win_rate": float((t["profit_bp"] > 0).mean()) if len(t) else float("nan"),
        "median_hold_min": float(t["hold_min"].median()) if len(t) else float("nan"),
        "converged": float(t["converged"].mean()) if len(t) else float("nan"),
        "series": {"t": [ts.isoformat(timespec="minutes") for ts in g.index[::step]],
                   "gap": [round(v, 2) for v in g.values[::step]],
                   "level": [None if pd.isna(v) else round(v, 2) for v in level.values[::step]]},
    }


# ---------------------------------------------------------------- live ticks

class _Book:
    """Top-of-book kept from an order-book stream (price -> size per side).
    Venues only send deletes for levels inside the subscribed depth, so the
    book is trimmed to that depth after every update; otherwise a level that
    drifted out of the window lingers and later shows up as a crossed book."""

    def __init__(self, depth: int):
        self.depth = depth
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}

    def set(self, side: dict, price: float, size: float):
        if size > 0:
            side[price] = size
        else:
            side.pop(price, None)

    def best(self) -> tuple[float, float] | None:
        if len(self.bids) > self.depth:
            self.bids = dict(sorted(self.bids.items(), reverse=True)[: self.depth])
        if len(self.asks) > self.depth:
            self.asks = dict(sorted(self.asks.items())[: self.depth])
        if self.bids and self.asks and max(self.bids) < min(self.asks):
            return max(self.bids), min(self.asks)
        return None


class Collector:
    """Streams top-of-book from each venue and appends (recv_ms, venue, bid, ask)
    to a CSV. Timestamps are our receive clock, not the venue's: that's the
    view a trader on this connection would actually have. Venues whose ticker
    channel is throttled (Kraken, Bitfinex) are read from their order-book
    stream instead, so a quote is never seconds stale."""

    DEPTH = {"kraken": 10, "bitfinex": 25}
    FEEDS = {
        "coinbase": ("wss://ws-feed.exchange.coinbase.com",
                     {"type": "subscribe", "product_ids": ["BTC-USD"], "channels": ["ticker"]}),
        "kraken": ("wss://ws.kraken.com/v2",
                   {"method": "subscribe", "params": {"channel": "book", "symbol": ["BTC/USD"], "depth": 10}}),
        "bitfinex": ("wss://api-pub.bitfinex.com/ws/2",
                     {"event": "subscribe", "channel": "book", "symbol": "tBTCUSD", "prec": "P0", "freq": "F0", "len": "25"}),
        "bitstamp": ("wss://ws.bitstamp.net",
                     {"event": "bts:subscribe", "data": {"channel": "order_book_btcusd"}}),
    }

    def __init__(self):
        self.buf: list[tuple] = []
        self.tasks: list[asyncio.Task] = []
        self.started = time.time()
        self.counts = {ex: 0 for ex in self.FEEDS}

    @staticmethod
    def _parse(ex: str, m, book: _Book) -> tuple[float, float] | None:
        try:
            if ex == "coinbase" and m.get("type") == "ticker":
                return float(m["best_bid"]), float(m["best_ask"])
            if ex == "bitstamp" and m.get("event") == "data":
                d = m["data"]
                return float(d["bids"][0][0]), float(d["asks"][0][0])
            if ex == "kraken" and m.get("channel") == "book":
                d = m["data"][0]
                if m.get("type") == "snapshot":
                    book.bids.clear(); book.asks.clear()
                for lvl in d.get("bids", []):
                    book.set(book.bids, float(lvl["price"]), float(lvl["qty"]))
                for lvl in d.get("asks", []):
                    book.set(book.asks, float(lvl["price"]), float(lvl["qty"]))
                return book.best()
            if ex == "bitfinex" and isinstance(m, list) and isinstance(m[1], list):
                rows = m[1] if isinstance(m[1][0], list) else [m[1]]
                if isinstance(m[1][0], list):
                    book.bids.clear(); book.asks.clear()
                for price, count, amount in rows:
                    side = book.bids if amount > 0 else book.asks
                    book.set(side, float(price), abs(float(amount)) if count > 0 else 0.0)
                return book.best()
        except (KeyError, IndexError, TypeError, ValueError):
            return None
        return None

    async def _feed(self, ex: str):
        import websockets
        url, sub = self.FEEDS[ex]
        backoff = 1
        while True:
            try:
                book = _Book(self.DEPTH.get(ex, 0))
                async with websockets.connect(url, ping_interval=20, max_size=2**22) as ws:
                    await ws.send(json.dumps(sub))
                    backoff = 1
                    last = None
                    async for raw in ws:
                        q = self._parse(ex, json.loads(raw), book)
                        if q and q != last:  # book streams repeat unchanged tops; only record changes
                            last = q
                            self.buf.append((int(time.time() * 1000), ex, q[0], q[1]))
                            self.counts[ex] += 1
            except Exception as e:
                log.warning("tick feed %s: %s (retry in %ss)", ex, e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def _flush(self):
        CACHE_DIR.mkdir(exist_ok=True)
        new = not TICKS_FILE.exists()
        last_prune = time.time()
        while True:
            await asyncio.sleep(10)
            rows, self.buf = self.buf, []
            if rows:
                with TICKS_FILE.open("a", newline="") as f:
                    w = csv.writer(f)
                    if new:
                        w.writerow(["ts", "ex", "bid", "ask"]); new = False
                    w.writerows(rows)
            if time.time() - last_prune > 3600:  # keep the file a rolling KEEP_H window
                last_prune = time.time()
                await asyncio.to_thread(self._prune)

    @staticmethod
    def _prune():
        if TICKS_FILE.exists():
            df = ticks(KEEP_H)
            df.to_csv(TICKS_FILE, index=False)

    def start(self):
        self.tasks = [asyncio.create_task(self._feed(ex)) for ex in self.FEEDS] + [asyncio.create_task(self._flush())]

    def stop(self):
        for t in self.tasks:
            t.cancel()


def ticks(hours: float = 24) -> pd.DataFrame:
    if not TICKS_FILE.exists():
        return pd.DataFrame(columns=["ts", "ex", "bid", "ask"])
    df = pd.read_csv(TICKS_FILE)
    return df[df["ts"] >= (time.time() - hours * 3600) * 1000]


def episodes(df: pd.DataFrame, fee_bp: float, latency_ms: int) -> dict:
    """For every ordered venue pair (buy at B's ask, sell at A's bid): moments
    when that is profitable after both fees — one round trip, right now. An
    episode is an unbroken run of such moments; its duration is how long you
    had. `level_bp` is the pair's median spread: where it exceeds the fees, the
    pair is "executable" almost continuously, but only once — after that
    round you hold the wrong asset on each venue. `latency_ms` is how late you'd arrive; the capture
    figure is the edge still on the table then (zero if it had closed — this
    assumes you re-check before firing, which is generous)."""
    if df.empty:
        return {"pairs": [], "n_ticks": 0, "hours": 0.0, "venues": []}
    wide_bid = df.pivot_table(index="ts", columns="ex", values="bid", aggfunc="last").ffill()
    wide_ask = df.pivot_table(index="ts", columns="ex", values="ask", aggfunc="last").ffill()
    venues = [v for v in wide_bid.columns if wide_bid[v].notna().sum() > 100]
    ts = wide_bid.index.values.astype("int64")
    span_h = (ts[-1] - ts[0]) / 3.6e6 if len(ts) > 1 else 0.0
    out = []
    for a, b in [(x, y) for x in venues for y in venues if x != y]:
        bid_a, ask_b = wide_bid[a], wide_ask[b]
        ok = bid_a.notna() & ask_b.notna()
        mid = (bid_a + ask_b) / 2
        raw = ((bid_a - ask_b) / mid * 1e4)[ok]
        level = float(raw.median())
        edge = raw - 2 * fee_bp
        live = (edge > 0).values
        if not live.any():
            out.append({"sell_on": a, "buy_on": b, "level_bp": level, "episodes": 0, "share_time": 0.0,
                        "median_ms": None, "p90_ms": None, "max_ms": None, "share_open_after_latency": 0.0,
                        "capture_bp_per_day": 0.0, "durations_ms": []})
            continue
        t = raw.index.values.astype("int64")
        e = edge.values
        starts = [i for i in range(len(live)) if live[i] and (i == 0 or not live[i - 1])]
        durs, caps = [], []
        for i in starts:
            j = i
            while j + 1 < len(live) and live[j + 1] and t[j + 1] - t[j] <= MAX_GAP_MS:
                j += 1
            end = min(t[j + 1], t[j] + MAX_GAP_MS) if j + 1 < len(t) else t[j]
            durs.append(int(end - t[i]))
            arrive = t[i] + latency_ms
            k = i
            while k + 1 < len(t) and t[k + 1] <= arrive:
                k += 1
            caps.append(float(e[k]) if live[k] and k <= j else 0.0)
        d = pd.Series(durs)
        share_time = float(d.sum() / (t[-1] - t[0])) if t[-1] > t[0] else 0.0
        out.append({
            "sell_on": a, "buy_on": b, "level_bp": level,
            "episodes": len(durs), "share_time": share_time,
            "median_ms": float(d.median()), "p90_ms": float(d.quantile(0.9)), "max_ms": float(d.max()),
            "share_open_after_latency": float((d > latency_ms).mean()),
            "capture_bp_per_day": float(sum(caps) / span_h * 24) if span_h else 0.0,
            "durations_ms": [int(x) for x in d.sample(min(len(d), 2000), random_state=0)],
        })
    return {"pairs": out, "n_ticks": int(len(df)), "hours": span_h, "venues": venues,
            "per_venue": {v: int((df["ex"] == v).sum()) for v in venues}}
