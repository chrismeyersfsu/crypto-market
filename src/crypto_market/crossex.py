"""Cross-exchange price gaps: is the same coin ever cheap on one venue and
dear on another for long enough to buy one and sell the other?

Markets covered: BTC/USD, and the stablecoins against the dollar and each
other (USDT/USD, USDC/USD, USDC/USDT). Stablecoins are the interesting case
for a small account: the price barely moves, so all that is left is the
venues disagreeing with each other, and several venues charge far less on
these pairs than on BTC.

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
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
CACHE_DIR = Path(__file__).resolve().parents[2] / "data"
TICKS_FILE = CACHE_DIR / "ticks.csv"
HISTORY_DAYS = 30
KEEP_H = 48  # of live ticks on disk
MAX_GAP_MS = 60_000  # a hole this long in the tick stream means the collector was down; no episode spans it
STALE_S = 600

# Each venue's symbol for each market. A symbol starting with "1/" is quoted
# the other way round on that venue (Coinbase lists USDT priced in USDC) and is
# flipped on the way in so every column means the same thing. Kraken is live
# only: its public candle endpoint keeps twelve hours. Bitstamp lists USDC/USDT
# but nothing trades there. "base" is not an exchange but a Uniswap pool on
# the Base chain (see dex.py), polled once a block; live only.
MARKETS = {
    "btcusd": {"coinbase": "BTC-USD", "kraken": "BTC/USD", "bitstamp": "btcusd", "bitfinex": "tBTCUSD", "binanceus": "BTCUSD", "base": "btcusd"},
    "ethusd": {"coinbase": "ETH-USD", "kraken": "ETH/USD", "bitstamp": "ethusd", "bitfinex": "tETHUSD", "binanceus": "ETHUSD", "base": "ethusd"},
    "btcusdt": {"coinbase": "BTC-USDT", "kraken": "BTC/USDT", "bitstamp": "btcusdt", "bitfinex": "tBTCUST", "binanceus": "BTCUSDT"},
    "usdtusd": {"coinbase": "USDT-USD", "kraken": "USDT/USD", "bitstamp": "usdtusd", "bitfinex": "tUSTUSD", "binanceus": "USDTUSD"},
    "usdcusd": {"kraken": "USDC/USD", "bitstamp": "usdcusd", "bitfinex": "tUDCUSD", "binanceus": "USDCUSD"},
    "usdcusdt": {"coinbase": "1/USDT-USDC", "kraken": "USDC/USDT", "bitfinex": "tUDCUST", "binanceus": "USDCUSDT"},
}
MARKET_NAMES = {"btcusd": "BTC/USD", "ethusd": "ETH/USD", "btcusdt": "BTC/USDT", "usdtusd": "USDT/USD", "usdcusd": "USDC/USD", "usdcusdt": "USDC/USDT"}


def _sym(market: str, ex: str) -> tuple[str, bool]:
    """(venue symbol, quoted the other way round?)"""
    raw = MARKETS[market][ex]
    return (raw[2:], True) if raw.startswith("1/") else (raw, False)

# ---------------------------------------------------------------- minute bars

def _bitfinex(sym: str, start: pd.Timestamp) -> pd.DataFrame:
    rows, cur, now = [], start, pd.Timestamp.utcnow().tz_localize(None)
    while cur < now:
        r = httpx.get(f"https://api-pub.bitfinex.com/v2/candles/trade:1m:{sym}/hist",
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


def _coinbase(sym: str, start: pd.Timestamp) -> pd.DataFrame:
    rows, cur, now = [], start, pd.Timestamp.utcnow().tz_localize(None)
    while cur < now:
        end = min(cur + pd.Timedelta(minutes=300), now)
        r = httpx.get(f"https://api.exchange.coinbase.com/products/{sym}/candles",
                      params={"granularity": 60, "start": cur.isoformat(), "end": end.isoformat()}, timeout=30)
        r.raise_for_status()
        rows += [(p[0] * 1000, p[4]) for p in r.json()]
        cur = end
        time.sleep(0.15)
    return pd.DataFrame(rows, columns=["ts", "close"])


def _bitstamp(sym: str, start: pd.Timestamp) -> pd.DataFrame:
    rows, cur, now = [], start, pd.Timestamp.utcnow().tz_localize(None)
    while cur < now:
        r = httpx.get(f"https://www.bitstamp.net/api/v2/ohlc/{sym}/",
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


def _binanceus(sym: str, start: pd.Timestamp) -> pd.DataFrame:
    rows, cur, now = [], start, pd.Timestamp.utcnow().tz_localize(None)
    while cur < now:
        r = httpx.get("https://api.binance.us/api/v3/klines",
                      params={"symbol": sym, "interval": "1m", "startTime": int(cur.timestamp() * 1000), "limit": 1000}, timeout=30)
        r.raise_for_status()
        page = r.json()
        if not page:
            break
        rows += [(p[0], float(p[4])) for p in page]
        last = pd.Timestamp(page[-1][0], unit="ms")
        if last <= cur:
            break
        cur = last + pd.Timedelta(minutes=1)
        time.sleep(0.15)
    return pd.DataFrame(rows, columns=["ts", "close"])


HISTORY = {"bitfinex": _bitfinex, "coinbase": _coinbase, "bitstamp": _bitstamp, "binanceus": _binanceus}


def history_venues(market: str) -> list[str]:
    """Venues with 30 days of minute bars for this market, in a stable order."""
    return [ex for ex in HISTORY if ex in MARKETS[market]]


LIVE_ONLY = ("kraken", "base")


def minutes(market: str = "btcusd", refresh: bool = True) -> pd.DataFrame:
    """One column of 1-minute closes per exchange, aligned on the UTC minute."""
    CACHE_DIR.mkdir(exist_ok=True)
    floor = pd.Timestamp.utcnow().tz_localize(None).floor("min") - pd.Timedelta(days=HISTORY_DAYS)
    cols = {}
    for ex in history_venues(market):
        sym, inverted = _sym(market, ex)
        f = CACHE_DIR / f"minute_{market}_{ex}.csv"
        cached = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["ts", "close"])
        cached = cached[cached["ts"] >= floor.timestamp() * 1000]
        since = pd.Timestamp(cached["ts"].max(), unit="ms") if len(cached) else floor
        stale = not len(cached) or time.time() - cached["ts"].max() / 1000 > STALE_S
        if refresh and stale:
            try:
                fresh = HISTORY[ex](sym, since - pd.Timedelta(minutes=1))
                if inverted:
                    fresh["close"] = 1 / fresh["close"]
                cached = pd.concat([cached, fresh]).drop_duplicates("ts", keep="last").sort_values("ts")
                cached.to_csv(f, index=False)
            except Exception as e:  # keep serving what we have; one venue's outage shouldn't blank the tab
                log.warning("minute history %s %s: %s", market, ex, e)
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

    def __init__(self, depth: int | None):
        self.depth = depth  # None: the venue sends changes at every level, so nothing ever goes stale
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}

    def set(self, side: dict, price: float, size: float):
        if size > 0:
            side[price] = size
        else:
            side.pop(price, None)

    def best(self) -> tuple[float, float, float, float] | None:
        """(bid, ask, bid size, ask size) or None while the book is empty/crossed."""
        if self.depth and len(self.bids) > self.depth:
            self.bids = dict(sorted(self.bids.items(), reverse=True)[: self.depth])
        if self.depth and len(self.asks) > self.depth:
            self.asks = dict(sorted(self.asks.items())[: self.depth])
        if self.bids and self.asks and max(self.bids) < min(self.asks):
            b, a = max(self.bids), min(self.asks)
            return b, a, self.bids[b], self.asks[a]
        return None


class Collector:
    """Streams top-of-book from each venue and appends (recv_ms, market, venue,
    bid, ask, bid size, ask size) to a CSV. One connection per venue carries
    every market that venue lists. Timestamps are our receive clock, not the
    venue's: that's the view a trader on this connection would actually have.
    Only Binance.US offers a usable best-bid/ask stream; the others are read
    off their order-book streams because their ticker channels are throttled
    or (Coinbase) only speak when a trade happens, which on a thin market
    leaves the quote seconds stale."""

    DEPTH = {"kraken": 10, "bitfinex": 25}
    URLS = {
        "coinbase": "wss://ws-feed.exchange.coinbase.com",
        "kraken": "wss://ws.kraken.com/v2",
        "bitfinex": "wss://api-pub.bitfinex.com/ws/2",
        "bitstamp": "wss://ws.bitstamp.net",
        "binanceus": "wss://stream.binance.us:9443/stream?streams=",
    }

    def __init__(self):
        self.buf: list[tuple] = []
        self.tasks: list[asyncio.Task] = []
        self.started = time.time()
        self.counts = {m: {ex: 0 for ex in MARKETS[m]} for m in MARKETS}

    @staticmethod
    def _markets(ex: str) -> dict[str, str]:
        """venue symbol -> market, for everything this venue lists"""
        return {_sym(m, ex)[0]: m for m in MARKETS if ex in MARKETS[m]}

    @classmethod
    def _url_and_subs(cls, ex: str) -> tuple[str, list[dict]]:
        syms = list(cls._markets(ex))
        if ex == "coinbase":
            return cls.URLS[ex], [{"type": "subscribe", "product_ids": syms, "channels": ["level2_batch"]}]
        if ex == "kraken":
            return cls.URLS[ex], [{"method": "subscribe", "params": {"channel": "book", "symbol": syms, "depth": 10}}]
        if ex == "bitfinex":
            return cls.URLS[ex], [{"event": "subscribe", "channel": "book", "symbol": s, "prec": "P0", "freq": "F0", "len": "25"} for s in syms]
        if ex == "bitstamp":
            return cls.URLS[ex], [{"event": "bts:subscribe", "data": {"channel": f"order_book_{s}"}} for s in syms]
        if ex == "binanceus":
            return cls.URLS[ex] + "/".join(f"{s.lower()}@bookTicker" for s in syms), []
        raise KeyError(ex)

    @staticmethod
    def _route(ex: str, m, chan: dict[int, str]) -> str | None:
        """Which venue symbol a message is about; None for control traffic."""
        try:
            if ex == "coinbase" and m.get("type") in ("snapshot", "l2update"):
                return m["product_id"]
            if ex == "binanceus" and "data" in m:
                return m["data"]["s"]
            if ex == "bitstamp" and m.get("event") == "data":
                return m["channel"].removeprefix("order_book_")
            if ex == "kraken" and m.get("channel") == "book":
                return m["data"][0]["symbol"]
            if ex == "bitfinex":
                if isinstance(m, dict) and m.get("event") == "subscribed":
                    chan[m["chanId"]] = m["symbol"]
                    return None
                if isinstance(m, list) and isinstance(m[1], list):
                    return chan.get(m[0])
        except (KeyError, IndexError, TypeError):
            return None
        return None

    @staticmethod
    def _parse(ex: str, m, book: _Book) -> tuple[float, float, float, float] | None:
        try:
            if ex == "coinbase":
                if m["type"] == "snapshot":
                    book.bids.clear(); book.asks.clear()
                    for price, size in m["bids"]:
                        book.set(book.bids, float(price), float(size))
                    for price, size in m["asks"]:
                        book.set(book.asks, float(price), float(size))
                else:
                    for side, price, size in m["changes"]:
                        book.set(book.bids if side == "buy" else book.asks, float(price), float(size))
                return book.best()
            if ex == "binanceus":
                d = m["data"]
                return float(d["b"]), float(d["a"]), float(d["B"]), float(d["A"])
            if ex == "bitstamp":
                d = m["data"]
                return float(d["bids"][0][0]), float(d["asks"][0][0]), float(d["bids"][0][1]), float(d["asks"][0][1])
            if ex == "kraken":
                d = m["data"][0]
                if m.get("type") == "snapshot":
                    book.bids.clear(); book.asks.clear()
                for lvl in d.get("bids", []):
                    book.set(book.bids, float(lvl["price"]), float(lvl["qty"]))
                for lvl in d.get("asks", []):
                    book.set(book.asks, float(lvl["price"]), float(lvl["qty"]))
                return book.best()
            if ex == "bitfinex":
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

    @staticmethod
    def _flip(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """A venue quoting the market the other way round: its ask (sell X of
        the base at p) is our bid for X*p of the other coin at 1/p."""
        bid, ask, bq, aq = q
        return 1 / ask, 1 / bid, aq * ask, bq * bid

    async def _feed(self, ex: str):
        import websockets
        url, subs = self._url_and_subs(ex)
        markets = self._markets(ex)
        flipped = {sym for sym, m in markets.items() if _sym(m, ex)[1]}
        backoff = 1
        while True:
            try:
                books = {sym: _Book(self.DEPTH.get(ex, 0)) for sym in markets}
                chan: dict[int, str] = {}
                last: dict[str, tuple] = {}
                async with websockets.connect(url, ping_interval=20, max_size=2**26) as ws:
                    for sub in subs:
                        await ws.send(json.dumps(sub))
                    backoff = 1

                    def record(sym: str, q):
                        if q and sym in flipped:
                            q = self._flip(q)
                        if q and q[:2] != last.get(sym):  # record price changes; size-only changes would swamp the file
                            last[sym] = q[:2]
                            self.buf.append((int(time.time() * 1000), markets[sym], ex, *q))
                            self.counts[markets[sym]][ex] += 1

                    if ex == "binanceus":  # its stream sends nothing until a price changes, which on a stablecoin can be an hour
                        async with httpx.AsyncClient(timeout=10) as http:
                            for sym in markets:
                                r = await http.get("https://api.binance.us/api/v3/ticker/bookTicker", params={"symbol": sym})
                                record(sym, self._parse(ex, {"data": r.json()}, books[sym]))
                    async for raw in ws:
                        m = json.loads(raw)
                        sym = self._route(ex, m, chan)
                        if sym in books:
                            record(sym, self._parse(ex, m, books[sym]))
            except Exception as e:
                log.warning("tick feed %s: %s (retry in %ss)", ex, e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def _pool_feed(self):
        """The Uniswap pools on Base: one read per block (about two seconds)."""
        from . import dex
        markets = self._markets("base")
        last: dict[str, tuple] = {}
        while True:
            for sym, market in markets.items():
                try:
                    q = await asyncio.to_thread(dex.quote, sym)
                except Exception as e:
                    log.warning("pool %s: %s", sym, e)
                    continue
                if q[:2] != last.get(sym):
                    last[sym] = q[:2]
                    self.buf.append((int(time.time() * 1000), market, "base", *q[:4]))
                    self.counts[market]["base"] += 1
            await asyncio.sleep(2)

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
                        w.writerow(["ts", "market", "ex", "bid", "ask", "bid_qty", "ask_qty"]); new = False
                    w.writerows(rows)
            if time.time() - last_prune > 3600:  # keep the file a rolling KEEP_H window
                last_prune = time.time()
                await asyncio.to_thread(self._prune)

    @staticmethod
    def _prune():
        if TICKS_FILE.exists():
            df = ticks(KEEP_H, None)
            df.to_csv(TICKS_FILE, index=False)

    def start(self):
        venues = {ex for m in MARKETS.values() for ex in m} - {"base"}
        self.tasks = ([asyncio.create_task(self._feed(ex)) for ex in sorted(venues)]
                      + [asyncio.create_task(self._pool_feed()), asyncio.create_task(self._flush()),
                         asyncio.create_task(scan_loop())])

    def stop(self):
        for t in self.tasks:
            t.cancel()


def ticks(hours: float = 24, market: str | None = "btcusd") -> pd.DataFrame:
    """Recorded ticks in the window; one market, or all of them for `None`."""
    if not TICKS_FILE.exists():
        return pd.DataFrame(columns=["ts", "market", "ex", "bid", "ask", "bid_qty", "ask_qty"])
    df = pd.read_csv(TICKS_FILE)
    df = df[df["ts"] >= (time.time() - hours * 3600) * 1000]
    return df if market is None else df[df["market"] == market]


NO_RUNS = {"episodes": 0, "share_time": 0.0, "median_ms": None, "p90_ms": None, "max_ms": None,
           "share_open_after_latency": 0.0, "capture_bp_per_day": 0.0, "median_size_usd": None,
           "median_profit_usd": None, "durations_ms": []}


def _runs(t, e, sz, latency_ms: int, holes, span_h: float) -> dict:
    """Unbroken runs of e > 0 over receive times t (ms), with the dollars sz
    available at each moment. `holes` are the last receive times before each
    collector outage; no run spans one."""
    live = e > 0
    if not live.any():
        return dict(NO_RUNS)
    # index of the first outage at or after each tick; a run is broken between j and j+1 if one falls between them
    hole_at = np.searchsorted(holes, t, side="left")
    starts = [i for i in range(len(live)) if live[i] and (i == 0 or not live[i - 1])]
    durs, caps, sizes, profits = [], [], [], []
    for i in starts:
        j = i
        while j + 1 < len(live) and live[j + 1] and hole_at[j + 1] == hole_at[j]:
            j += 1
        broken = hole_at[j] < len(holes) and (j + 1 == len(t) or hole_at[j + 1] > hole_at[j])
        end = holes[hole_at[j]] if broken else (t[j + 1] if j + 1 < len(t) else t[j])
        durs.append(int(end - t[i]))
        arrive = t[i] + latency_ms
        k = i
        while k + 1 < len(t) and t[k + 1] <= arrive:
            k += 1
        caps.append(float(e[k]) if live[k] and k <= j else 0.0)
        sizes.append(float(sz[i]) if not pd.isna(sz[i]) else float("nan"))
        profits.append(float(e[i]) / 1e4 * sizes[-1])  # the whole edge, on all the size there was
    d = pd.Series(durs)
    return {
        "episodes": len(durs), "share_time": float(d.sum() / (t[-1] - t[0])) if t[-1] > t[0] else 0.0,
        "median_ms": float(d.median()), "p90_ms": float(d.quantile(0.9)), "max_ms": float(d.max()),
        "share_open_after_latency": float((d > latency_ms).mean()),
        "capture_bp_per_day": float(sum(caps) / span_h * 24) if span_h else 0.0,
        "median_size_usd": float(pd.Series(sizes).median()),
        "median_profit_usd": float(pd.Series(profits).median()),
        "durations_ms": [int(x) for x in d.sample(min(len(d), 2000), random_state=0)],
    }


def _holes(alive_ts):
    alive = np.sort(np.asarray(alive_ts, dtype="int64"))
    return alive[:-1][np.diff(alive) > MAX_GAP_MS]  # last receive time before each outage


def episodes(df: pd.DataFrame, fee_bp: float, latency_ms: int, alive_ts=None) -> dict:
    """For every ordered venue pair (buy at B's ask, sell at A's bid): moments
    when that is profitable after both fees — one round trip, right now. An
    episode is an unbroken run of such moments; its duration is how long you
    had. `level_bp` is the pair's median spread: where it exceeds the fees, the
    pair is "executable" almost continuously, but only once — after that
    round you hold the wrong asset on each venue. `latency_ms` is how late you'd arrive; the capture
    figure is the edge still on the table then (zero if it had closed — this
    assumes you re-check before firing, which is generous). `alive_ts` are
    receive times across every market: a hole in *those* means the collector
    was down and no episode may span it. A hole in one slow market's own
    ticks means nothing — a stablecoin quote can sit unchanged for minutes."""
    if df.empty:
        return {"pairs": [], "n_ticks": 0, "hours": 0.0, "venues": []}
    wide_bid = df.pivot_table(index="ts", columns="ex", values="bid", aggfunc="last").ffill()
    wide_ask = df.pivot_table(index="ts", columns="ex", values="ask", aggfunc="last").ffill()
    wide_bq = df.pivot_table(index="ts", columns="ex", values="bid_qty", aggfunc="last").ffill()
    wide_aq = df.pivot_table(index="ts", columns="ex", values="ask_qty", aggfunc="last").ffill()
    venues = [v for v in wide_bid.columns if wide_bid[v].notna().sum() >= 5]
    ts = wide_bid.index.values.astype("int64")
    holes = _holes(alive_ts if alive_ts is not None else ts)
    span_h = (ts[-1] - ts[0]) / 3.6e6 if len(ts) > 1 else 0.0
    out = []
    for a, b in [(x, y) for x in venues for y in venues if x != y]:
        bid_a, ask_b = wide_bid[a], wide_ask[b]
        ok = bid_a.notna() & ask_b.notna()
        mid = (bid_a + ask_b) / 2
        raw = ((bid_a - ask_b) / mid * 1e4)[ok]
        # dollars you could actually move at those two prices: the smaller of the two top-of-book sizes
        size_usd = (pd.concat([wide_bq[a], wide_aq[b]], axis=1).min(axis=1) * mid)[ok]
        level = float(raw.median())
        edge = raw - 2 * fee_bp
        out.append({"sell_on": a, "buy_on": b, "level_bp": level,
                    **_runs(raw.index.values.astype("int64"), edge.values, size_usd.values, latency_ms, holes, span_h)})
    return {"pairs": out, "n_ticks": int(len(df)), "hours": span_h, "venues": venues,
            "per_venue": {v: int((df["ex"] == v).sum()) for v in venues}}


# ---------------------------------------------------------------- triangle

TRIANGLE = ("btcusd", "btcusdt", "usdtusd")  # dollars -> BTC -> USDT -> dollars, all on one venue


def triangle_venues() -> list[str]:
    return [ex for ex in MARKETS["btcusd"] if all(ex in MARKETS[m] for m in TRIANGLE)]


def triangle_minutes(fee_bp: float, venue: str) -> dict:
    """Minute closes on one venue: is BTC/USD equal to BTC/USDT times USDT/USD?
    The mismatch, in basis points, is what a round trip through the three
    markets would make before spreads and fees. Positive means BTC is dear
    in dollars relative to the USDT route (so: dollars -> USDT -> BTC ->
    dollars); negative means the other way round. Three fills, so the cost
    is three fees. Closes cannot see the bid-ask spread, which on the thin
    BTC/USDT markets is several bp — the live figures are the ones to trust."""
    cols = {}
    for m in TRIANGLE:
        if venue in history_venues(m):
            cols[m] = minutes(m)[venue]
    if len(cols) < 3:
        return {"venue": venue, "n_minutes": 0}
    df = pd.DataFrame(cols).dropna()
    mm = (df["btcusd"] / (df["btcusdt"] * df["usdtusd"]) - 1) * 1e4
    cost = 3 * fee_bp
    beyond = mm.abs() > cost
    step = max(len(mm) // 2000, 1)
    return {
        "venue": venue, "n_minutes": int(len(mm)),
        "days": (mm.index[-1] - mm.index[0]).total_seconds() / 86400 if len(mm) > 1 else float("nan"),
        "level_bp": float(mm.median()), "abs_median_bp": float(mm.abs().median()), "p95_bp": float(mm.abs().quantile(0.95)),
        "minutes_beyond_cost": int(beyond.sum()), "share_beyond_cost": float(beyond.mean()),
        "sum_beyond_cost_bp": float((mm.abs() - cost)[beyond].sum()),
        "series": {"t": [ts.isoformat(timespec="minutes") for ts in mm.index[::step]],
                   "mismatch": [round(v, 2) for v in mm.values[::step]]},
    }


def triangle_live(df: pd.DataFrame, fee_bp: float, latency_ms: int, alive_ts=None) -> dict:
    """Live ticks, every venue that lists all three markets, both directions,
    priced at the bids and asks you would actually hit. `df` holds ticks of
    every market. Size is the dollars the thinnest of the three legs allows."""
    out, per_venue = [], {}
    holes = _holes(alive_ts) if alive_ts is not None else None
    for ex in triangle_venues():
        d = df[(df["ex"] == ex) & df["market"].isin(TRIANGLE)]
        if d.empty or d["market"].nunique() < 3:
            continue
        w = {f: d.pivot_table(index="ts", columns="market", values=f, aggfunc="last").ffill() for f in ("bid", "ask", "bid_qty", "ask_qty")}
        ok = w["bid"].notna().all(axis=1)
        bid, ask, bq, aq = (w[f][ok] for f in ("bid", "ask", "bid_qty", "ask_qty"))
        if len(bid) < 5:
            continue
        t = bid.index.values.astype("int64")
        span_h = (t[-1] - t[0]) / 3.6e6 if len(t) > 1 else 0.0
        h = holes if holes is not None else _holes(t)
        per_venue[ex] = int(len(d))
        # dollars -> BTC (pay BTC/USD ask) -> USDT (sell at BTC/USDT bid) -> dollars (sell at USDT/USD bid)
        via_btc = (bid["btcusdt"] * bid["usdtusd"] / ask["btcusd"] - 1) * 1e4 - 3 * fee_bp
        size_btc = pd.concat([aq["btcusd"] * ask["btcusd"], bq["btcusdt"] * bid["btcusdt"] * bid["usdtusd"], bq["usdtusd"] * bid["usdtusd"]], axis=1).min(axis=1)
        # dollars -> USDT (pay USDT/USD ask) -> BTC (pay BTC/USDT ask) -> dollars (sell at BTC/USD bid)
        via_usdt = (bid["btcusd"] / (ask["usdtusd"] * ask["btcusdt"]) - 1) * 1e4 - 3 * fee_bp
        size_usdt = pd.concat([aq["usdtusd"] * ask["usdtusd"], aq["btcusdt"] * ask["btcusdt"] * ask["usdtusd"], bq["btcusd"] * bid["btcusd"]], axis=1).min(axis=1)
        for route, edge, size in (("USD → BTC → USDT → USD", via_btc, size_btc), ("USD → USDT → BTC → USD", via_usdt, size_usdt)):
            out.append({"venue": ex, "route": route, "level_bp": float(edge.median() + 3 * fee_bp),
                        **_runs(t, edge.values, size.values, latency_ms, h, span_h)})
    return {"routes": out, "per_venue": per_venue}


# ---------------------------------------------------------------- Coinbase: every triangle

SCAN_FILE = CACHE_DIR / "triscan_coinbase.csv"
SCAN_S = 60
SCAN_COLS = ["ts", "coin", "via", "route", "gross_bp", "size_usd"]


def coinbase_triangles() -> list[tuple[str, str]]:
    """(coin, bridge) for every coin that Coinbase lists both in dollars and
    in a bridge currency that is itself listed in dollars: BTC, ETH, USDT."""
    ps = httpx.get("https://api.exchange.coinbase.com/products", timeout=20).json()
    pairs = {(p["base_currency"], p["quote_currency"]) for p in ps if p["status"] == "online" and not p.get("trading_disabled")}
    return sorted((x, q) for x, q in pairs if q in ("BTC", "ETH", "USDT") and (x, "USD") in pairs and (q, "USD") in pairs)


async def scan_coinbase(triangles: list[tuple[str, str]]) -> list[tuple]:
    """One pass over every triangle at Coinbase's current best bid/ask: the
    gross round-trip profit in basis points, both directions, and the
    dollars the thinnest leg allowed. Positive means the three prices are
    inconsistent by that much before fees."""
    products = sorted({f"{x}-USD" for x, _ in triangles} | {f"{x}-{q}" for x, q in triangles} | {f"{q}-USD" for _, q in triangles})
    sem = asyncio.Semaphore(2)  # public limit is 10 requests a second; stay well under it

    async def book(http, pid):
        async with sem:
            try:
                r = await http.get(f"https://api.exchange.coinbase.com/products/{pid}/book", params={"level": 1})
                j = r.json()
                await asyncio.sleep(0.25)
                return pid, (float(j["bids"][0][0]), float(j["asks"][0][0]), float(j["bids"][0][1]), float(j["asks"][0][1]))
            except Exception as e:
                log.warning("coinbase book %s: %s", pid, e)
                return pid, None
    async with httpx.AsyncClient(timeout=15) as http:
        q = dict(await asyncio.gather(*(book(http, pid) for pid in products)))
    ts, rows = int(time.time() * 1000), []
    for x, br in triangles:
        xu, xq, qu = q.get(f"{x}-USD"), q.get(f"{x}-{br}"), q.get(f"{br}-USD")
        if not (xu and xq and qu):
            continue
        # dollars -> coin (pay X/USD ask) -> bridge (sell at X/Q bid) -> dollars (sell bridge at Q/USD bid)
        rows.append((ts, x, br, f"USD → {x} → {br} → USD", (xq[0] * qu[0] / xu[1] - 1) * 1e4,
                     min(xu[3] * xu[1], xq[2] * xq[0] * qu[0], qu[2] * qu[0])))
        # dollars -> bridge (pay Q/USD ask) -> coin (pay X/Q ask) -> dollars (sell at X/USD bid)
        rows.append((ts, x, br, f"USD → {br} → {x} → USD", (xu[0] / (qu[1] * xq[1]) - 1) * 1e4,
                     min(qu[3] * qu[1], xq[3] * xq[1] * qu[1], xu[2] * xu[0])))
    return rows


async def scan_loop():
    """Runs with the collector: every triangle on Coinbase once a minute, appended to SCAN_FILE."""
    CACHE_DIR.mkdir(exist_ok=True)
    triangles, listed = [], 0.0
    while True:
        try:
            if time.time() - listed > 3600:
                triangles, listed = await asyncio.to_thread(coinbase_triangles), time.time()
            rows = await scan_coinbase(triangles)
            new = not SCAN_FILE.exists()
            with SCAN_FILE.open("a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(SCAN_COLS)
                w.writerows(rows)
        except Exception as e:
            log.warning("coinbase triangle scan: %s", e)
        await asyncio.sleep(SCAN_S)


def triangle_scan(fee_bp: float, hours: float = 24) -> dict:
    """Every Coinbase triangle over the last `hours` of scans: latest and
    median gross mismatch per direction, how often it beat three fees, and
    the dollars it was good for. Sorted by median gross, best first."""
    if not SCAN_FILE.exists():
        return {"rows": [], "scans": 0}
    df = pd.read_csv(SCAN_FILE)
    df = df[df["ts"] >= (time.time() - hours * 3600) * 1000]
    if len(df) > 400_000:  # keep the file to a couple of days
        df.to_csv(SCAN_FILE, index=False)
    cost = 3 * fee_bp
    out = []
    for (coin, via, route), g in df.groupby(["coin", "via", "route"]):
        g = g.sort_values("ts")
        out.append({"coin": coin, "via": via, "route": route, "scans": int(len(g)),
                    "latest_bp": float(g["gross_bp"].iloc[-1]), "median_bp": float(g["gross_bp"].median()),
                    "p90_bp": float(g["gross_bp"].quantile(0.9)), "best_bp": float(g["gross_bp"].max()),
                    "share_beyond_cost": float((g["gross_bp"] > cost).mean()),
                    "median_size_usd": float(g["size_usd"].median()),
                    "median_profit_usd": float(((g["gross_bp"] - cost).clip(lower=0) / 1e4 * g["size_usd"]).median())})
    out.sort(key=lambda r: -r["median_bp"])
    return {"rows": out, "scans": int(df["ts"].nunique()), "triangles": int(df.groupby(["coin", "via"]).ngroups),
            "since": float(df["ts"].min() / 1000) if len(df) else None}
