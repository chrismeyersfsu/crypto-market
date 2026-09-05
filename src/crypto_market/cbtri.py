"""Every triangle an exchange offers, watched from its own price feed.

A triangle is a coin X that the exchange lists in dollars and also in a
bridge currency Q (BTC, ETH, USDT or USDC) that is itself listed in
dollars. The three prices should agree: X/USD == X/Q x Q/USD. When they
don't, three trades in a loop end with more dollars than they started
with, before fees. There are two loops per triangle:

    direction 1:  USD -> X -> Q -> USD   pay the X/USD ask, sell X at the X/Q bid, sell Q at the Q/USD bid
    direction 2:  USD -> Q -> X -> USD   pay the Q/USD ask, pay the X/Q ask, sell X at the X/USD bid

One Watcher runs per venue. Each venue is a small adapter (a Venue
subclass) that knows how to list its triangles, what to connect to, how
to read a message into a best bid and ask, and which of its markets are
cheaper to trade than the rest; everything else -- re-pricing both loops
each time one of a triangle's three markets moves, the episode
bookkeeping and the files -- is shared. Two files come out per venue:

    <prefix>_samples.csv   every SAMPLE_S seconds: each loop's gross mismatch and the dollars its thinnest market allowed
    <prefix>_episodes.csv  one row per unbroken stretch of a loop being positive: when, how long, how big, and what
                           was still there LAT_MS after it opened (the question a real order has to answer)

Coinbase keeps its full order books from the `level2_batch` stream and
writes cbtri_*.csv; Binance.US reads its best-bid/ask stream and writes
butri_*.csv.
"""
from __future__ import annotations

import asyncio
import csv
import json
import logging
import time

import httpx
import pandas as pd

from .crossex import CACHE_DIR, KEEP_H, _Book

log = logging.getLogger(__name__)

SAMPLE_S = 20
LAT_MS = (100, 250, 500, 1000)  # what was left of an episode this long after it opened
SAMPLE_COLS = ["ts", "coin", "via", "dir", "gross_bp", "size_usd"]
EPISODE_COLS = (["start", "end", "coin", "via", "dir", "open_bp", "open_size_usd", "max_bp", "max_size_usd", "updates", "broken"]
                + [f"bp_at_{l}" for l in LAT_MS])

Quote = tuple[float, float, float, float]  # bid, ask, bid size, ask size


def route(coin: str, via: str, d: int) -> str:
    return f"USD → {coin} → {via} → USD" if d == 1 else f"USD → {via} → {coin} → USD"


# -- venues --------------------------------------------------------------------

class Venue:
    """What differs between exchanges. `fee_hint` is shown on the page
    under the fee boxes. `zero` and `reduced` are the markets that cost
    nothing / the reduced rate to fill immediately; everything else
    costs the ordinary rate. `default_fees` is (ordinary, reduced) in bp
    for an immediate fill at the bottom volume tier."""

    key: str
    name: str
    prefix: str
    bridges: tuple[str, ...]
    zero: frozenset[str] = frozenset()
    reduced: frozenset[str] = frozenset()
    reduced_label: str = "reduced-rate pair"
    default_fees: tuple[float, float]
    fee_hint: str
    url: str

    def __init__(self):
        self.samples_path = CACHE_DIR / f"{self.prefix}_samples.csv"
        self.episodes_path = CACHE_DIR / f"{self.prefix}_episodes.csv"

    def triangles(self) -> list[tuple[str, str]]:
        """(coin, bridge) for every coin listed both in dollars and in a bridge that is listed in dollars.
        Blocking; runs in a thread."""
        raise NotImplementedError

    def product(self, base: str, quote: str) -> str:
        """The venue's name for the base/quote market."""
        raise NotImplementedError

    def markets(self, coin: str, via: str) -> tuple[str, str, str]:
        """The three markets a triangle trades: X/USD, X/Q, Q/USD."""
        return self.product(coin, "USD"), self.product(coin, via), self.product(via, "USD")

    def cost_bp(self, coin: str, via: str, coin_fee: float, reduced_fee: float) -> float:
        """Three immediate fills; each market charges its own class of fee."""
        return sum(0.0 if p in self.zero else reduced_fee if p in self.reduced else coin_fee for p in self.markets(coin, via))

    def connect_url(self, products: list[str]) -> str:
        return self.url

    def subscribe(self, products: list[str]) -> list[dict]:
        """Messages to send right after connecting."""
        return []

    async def seed(self, products: list[str]) -> list[tuple[str, Quote]]:
        """Quotes to start from before the stream says anything (empty unless the stream is silent until a change)."""
        return []

    def parse(self, m: dict, books: dict[str, _Book]) -> tuple[str, Quote] | None:
        """(product, quote) from one message, or None for anything that isn't a price."""
        raise NotImplementedError


class Coinbase(Venue):
    """The level2_batch stream: a full order book per product, kept in a
    _Book, with the best bid and ask read off it after every change."""

    key, name, prefix = "coinbase", "Coinbase", "cbtri"
    bridges = ("BTC", "ETH", "USDT")
    reduced = frozenset({"USDT-USD", "USDC-USD", "USDT-USDC", "DAI-USD", "PYUSD-USD"})  # Coinbase's near-free "stable pairs"
    reduced_label = "stable pair"
    default_fees = (60.0, 60.0)
    url = "wss://ws-feed.exchange.coinbase.com"
    fee_hint = (
        "1 bp = 0.01%. A loop needs its orders filled immediately, which means paying Coinbase's rate for an immediate fill "
        "(its \"taker\" rate). By 30-day volume: under $10k traded, 60 bp; $10k+, 40; $50k+, 25; $100k+, 20; $1M+, 18; $15M+, 16; "
        "$75M+, 10; $250M+, 6; $400M+, 4. A resting order (one that waits on the book for someone else to fill it, the \"maker\" "
        "rate) is cheaper, 40 bp at the bottom tier and 0 at the top, but the mismatch is gone by the time it fills. Coinbase's "
        "\"stable pairs\" cost 0.1–0.45 bp to fill immediately, but USDT/USD — the one that appears here — was reportedly dropped "
        "from that list in 2025, so the stable-pair box starts at the ordinary rate; set it to 0.5 if your account still gets the "
        "stable rate on USDT/USD. A loop through USDT pays two ordinary fees and one stable fee; a loop through BTC or ETH pays "
        "three ordinary fees. Latency is snapped to the nearest of 100, 250, 500 and 1000 ms, which is what the recorder keeps. "
        "Source: https://www.coinbase.com/advanced-fees"
    )

    def triangles(self):
        ps = httpx.get("https://api.exchange.coinbase.com/products", timeout=20).json()
        pairs = {(p["base_currency"], p["quote_currency"]) for p in ps if p["status"] == "online" and not p.get("trading_disabled")}
        return sorted((x, q) for x, q in pairs if q in self.bridges and (x, "USD") in pairs and (q, "USD") in pairs)

    def product(self, base, quote):
        return f"{base}-{quote}"

    def subscribe(self, products):
        return [{"type": "subscribe", "product_ids": products, "channels": ["level2_batch"]}]

    def parse(self, m, books):
        pid = m.get("product_id")
        book = books.get(pid)
        if book is None:
            if m.get("type") == "error":
                log.warning("coinbase triangles: %s", m)
            return None
        if m["type"] == "snapshot":
            book.bids.clear(); book.asks.clear()
            for price, size in m["bids"]:
                book.set(book.bids, float(price), float(size))
            for price, size in m["asks"]:
                book.set(book.asks, float(price), float(size))
        elif m["type"] == "l2update":
            for side, price, size in m["changes"]:
                book.set(book.bids if side == "buy" else book.asks, float(price), float(size))
        else:
            return None
        q = book.best()
        return (pid, q) if q else None


class BinanceUS(Venue):
    """The combined bookTicker stream: one message per change of a
    market's best bid or ask, with the size at each -- no full book
    needed. It says nothing about a market until its top of book moves,
    so every market is seeded from the REST snapshot on connect. One
    connection may carry up to 1024 streams (Binance's documented cap);
    the ~115 needed here fit on one."""

    key, name, prefix = "binanceus", "Binance.US", "butri"
    bridges = ("BTC", "ETH", "USDT", "USDC")
    # Read from https://www.binance.us/fees on 2026-09-05: only BNB/USD is a "Tier 0" pair (0.01% immediate fill);
    # everything else, BTC and stablecoin pairs included, is "Tier I" (0.02%). No pair is free to fill immediately.
    reduced = frozenset({"BNBUSD"})
    reduced_label = "Tier 0 pair"
    default_fees = (2.0, 1.0)
    url = "wss://stream.binance.us:9443/stream?streams="
    fee_hint = (
        "1 bp = 0.01%. Binance.US's spot schedule, read from binance.us/fees on 2026-09-05: on every pair a resting order "
        "(one that waits on the book for someone else to fill it, the \"maker\" rate) costs 0 bp and an immediate fill (the "
        "\"taker\" rate) costs 2 bp, the same at every volume level up to $500M a month, where it drops to 1 bp. One \"Tier 0\" "
        "pair, BNB/USD, costs 1 bp to fill immediately. There is no longer a separate rate for BTC pairs or stablecoin pairs: "
        "BTC/USD, BTC/USDT, USDT/USD and USDC/USD all cost 2 bp. Paying the fee in BNB takes 5% off. A loop needs immediate "
        "fills, so it costs three ordinary fees (6 bp), or two ordinary fees and one Tier 0 fee through BNB/USD. Latency is "
        "snapped to the nearest of 100, 250, 500 and 1000 ms, which is what the recorder keeps. Source: https://www.binance.us/fees "
        "(schedule announced 2026-04-22, https://blog.binance.us/zero-fee-trading/)"
    )

    def __init__(self):
        super().__init__()
        self.symbols: dict[tuple[str, str], str] = {}  # (base, quote) -> symbol; not always base+quote (NANOUSD is XNO/USD)

    def triangles(self):
        info = httpx.get("https://api.binance.us/api/v3/exchangeInfo", timeout=20).json()
        self.symbols = {(s["baseAsset"], s["quoteAsset"]): s["symbol"] for s in info["symbols"] if s["status"] == "TRADING"}
        pairs = self.symbols
        return sorted((x, q) for x, q in pairs if q in self.bridges and (x, "USD") in pairs and (q, "USD") in pairs)

    def product(self, base, quote):
        return self.symbols.get((base, quote), f"{base}{quote}")

    def connect_url(self, products):
        return self.url + "/".join(f"{p.lower()}@bookTicker" for p in products)

    @staticmethod
    def _quote(d: dict) -> Quote | None:
        bid, ask = float(d.get("b") or d.get("bidPrice") or 0), float(d.get("a") or d.get("askPrice") or 0)
        if bid <= 0 or ask <= 0:  # an empty side; the REST snapshot lists halted symbols with zeros
            return None
        return bid, ask, float(d.get("B") or d.get("bidQty") or 0), float(d.get("A") or d.get("askQty") or 0)

    async def seed(self, products):
        async with httpx.AsyncClient(timeout=20) as http:
            r = await http.get("https://api.binance.us/api/v3/ticker/bookTicker")  # no params: every symbol, one call
        want = set(products)
        out = []
        for d in r.json():
            q = self._quote(d) if d.get("symbol") in want else None
            if q:
                out.append((d["symbol"], q))
        return out

    def parse(self, m, books):
        d = m.get("data")
        if not d or "s" not in d:
            if "error" in m:
                log.warning("binance.us triangles: %s", m)
            return None
        q = self._quote(d)
        return (d["s"], q) if q and d["s"] in books else None


VENUES: dict[str, Venue] = {v.key: v for v in (Coinbase(), BinanceUS())}


# -- the watcher -----------------------------------------------------------------

class Watcher:
    def __init__(self, venue: Venue):
        self.venue = venue
        self.tri: list[tuple[str, str]] = []
        self.books: dict[str, _Book] = {}  # one per product; Coinbase keeps the full book in it, Binance.US only its key
        self.quotes: dict[str, Quote] = {}  # the standing best bid and ask per product
        self.by_product: dict[str, list[int]] = {}
        self.state: dict[tuple, dict] = {}  # (coin, via, dir) -> now: gross, size, ts; open episode
        self.samples: list[tuple] = []
        self.episodes: list[tuple] = []
        self.updates = 0
        self.connected: float | None = None
        self.tasks: list[asyncio.Task] = []

    # -- pricing -------------------------------------------------------------

    def _reprice(self, i: int, now: int):
        x, q = self.tri[i]
        xu, xq, qu = (self.quotes.get(p) for p in self.venue.markets(x, q))
        if not (xu and xq and qu):
            return
        # bid, ask, bid size, ask size
        g1 = (xq[0] * qu[0] / xu[1] - 1) * 1e4
        s1 = min(xu[3] * xu[1], xq[2] * xq[0] * qu[0], qu[2] * qu[0])
        g2 = (xu[0] / (qu[1] * xq[1]) - 1) * 1e4
        s2 = min(qu[3] * qu[1], xq[3] * xq[1] * qu[1], xu[2] * xu[0])
        self._update((x, q, 1), g1, s1, now)
        self._update((x, q, 2), g2, s2, now)

    def _update(self, key: tuple, g: float, s: float, now: int):
        st = self.state.setdefault(key, {"gross": None, "size": None, "ts": 0, "open": None})
        prev = st["gross"]
        st["gross"], st["size"], st["ts"] = g, s, now
        ep = st["open"]
        if ep is not None:
            for lat in LAT_MS:  # the quote standing at each latency mark is the one that was there until this update
                if ep["at"][lat] is None and now >= ep["start"] + lat:
                    ep["at"][lat] = prev
        if g > 0:
            if ep is None:
                st["open"] = {"start": now, "open_bp": g, "open_size": s, "max_bp": g, "max_size": s, "n": 1,
                              "at": {lat: None for lat in LAT_MS}}
            else:
                ep["n"] += 1
                if g > ep["max_bp"]:
                    ep["max_bp"], ep["max_size"] = g, s
        elif ep is not None:
            self._close(key, now, broken=False)

    def _close(self, key: tuple, now: int, broken: bool):
        st = self.state[key]
        ep, st["open"] = st["open"], None
        x, q, d = key
        self.episodes.append((ep["start"], now, x, q, d, ep["open_bp"], ep["open_size"], ep["max_bp"], ep["max_size"],
                              ep["n"], int(broken), *(ep["at"][lat] if ep["at"][lat] is not None else 0.0 for lat in LAT_MS)))

    def _close_all(self, now: int):
        for key, st in self.state.items():
            if st["open"] is not None:
                self._close(key, now, broken=True)

    # -- feed ----------------------------------------------------------------

    def _take(self, product: str, quote: Quote):
        """A new best bid/ask for one market: re-price every triangle it is part of."""
        self.quotes[product] = quote
        now = int(time.time() * 1000)
        self.updates += 1
        for i in self.by_product[product]:
            self._reprice(i, now)

    async def _feed(self):
        import websockets
        v = self.venue
        backoff = 1
        while True:
            try:
                self.tri = await asyncio.to_thread(v.triangles)
                self.by_product = {}
                for i, (x, q) in enumerate(self.tri):
                    for p in v.markets(x, q):
                        self.by_product.setdefault(p, []).append(i)
                self.books = {p: _Book(None) for p in self.by_product}
                self.quotes = {}
                products = list(self.books)
                async with websockets.connect(v.connect_url(products), ping_interval=20, max_size=2**26) as ws:
                    for sub in v.subscribe(products):
                        await ws.send(json.dumps(sub))
                    backoff, self.connected, listed = 1, time.time(), time.time()
                    for product, quote in await v.seed(products):
                        self._take(product, quote)
                    async for raw in ws:
                        got = v.parse(json.loads(raw), self.books)
                        if got is None:
                            continue
                        self._take(*got)
                        if time.time() - listed > 6 * 3600:  # pick up newly listed products
                            break
            except Exception as e:
                log.warning("%s triangles feed: %s (retry in %ss)", v.name, e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
            self._close_all(int(time.time() * 1000))
            self.connected = None

    async def _sample(self):
        CACHE_DIR.mkdir(exist_ok=True)
        last_prune = time.time()
        while True:
            await asyncio.sleep(SAMPLE_S)
            now = int(time.time() * 1000)
            if self.connected:
                for (x, q, d), st in self.state.items():
                    if st["ts"] >= self.connected * 1000:  # priced on this connection; a quiet market's quote is still standing
                        self.samples.append((now, x, q, d, round(st["gross"], 3), round(st["size"], 2)))
            self._write(self.venue.samples_path, SAMPLE_COLS, self.samples); self.samples = []
            self._write(self.venue.episodes_path, EPISODE_COLS, self.episodes); self.episodes = []
            if time.time() - last_prune > 3600:
                last_prune = time.time()
                await asyncio.to_thread(self._prune)

    @staticmethod
    def _write(path, cols, rows):
        if not rows:
            return
        new = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(cols)
            w.writerows(rows)

    def _prune(self):
        cutoff = (time.time() - KEEP_H * 3600) * 1000
        for path, col in ((self.venue.samples_path, "ts"), (self.venue.episodes_path, "start")):
            if path.exists():
                df = pd.read_csv(path)
                df[df[col] >= cutoff].to_csv(path, index=False)

    def start(self):
        self.tasks = [asyncio.create_task(self._feed()), asyncio.create_task(self._sample())]

    def stop(self):
        for t in self.tasks:
            t.cancel()

    def now(self) -> dict[tuple, dict]:
        return {k: {"gross": st["gross"], "size": st["size"], "ts": st["ts"], "open_since": st["open"]["start"] if st["open"] else None}
                for k, st in self.state.items()}

    def feed(self) -> dict:
        return {"connected": self.connected, "updates": self.updates, "products": len(self.books)}


# -- analysis ------------------------------------------------------------------

def _load(path, cols, col, hours):
    if not path.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path)
    return df[df[col] >= (time.time() - hours * 3600) * 1000]


def report(venue: Venue, coin_fee: float, reduced_fee: float, latency_ms: int, hours: float, now: dict | None = None) -> dict:
    """Every loop over the last `hours`: where it stands now, its typical
    and best mismatch, how often and how long it was positive, what was
    left of it after your latency, and what beating your fees would have
    paid. Sorted by median gross, best first."""
    s = _load(venue.samples_path, SAMPLE_COLS, "ts", hours)
    e = _load(venue.episodes_path, EPISODE_COLS, "start", hours)
    lat = min(LAT_MS, key=lambda l: abs(l - latency_ms))
    alive_h = s["ts"].nunique() * SAMPLE_S / 3600 if len(s) else 0.0
    now = now or {}
    rows = []
    keys = {(r.coin, r.via, r.dir) for r in s[["coin", "via", "dir"]].drop_duplicates().itertuples()} | set(now)
    for x, q, d in sorted(keys):
        sg = s[(s["coin"] == x) & (s["via"] == q) & (s["dir"] == d)]
        g = sg["gross_bp"]
        ep = e[(e["coin"] == x) & (e["via"] == q) & (e["dir"] == d)]
        # the largest mismatch: from closed episodes, else (still open, or the feed just started) from the samples
        if len(ep) and (not len(g) or ep["max_bp"].max() >= g.max()):
            best_bp, best_size = float(ep["max_bp"].max()), float(ep.loc[ep["max_bp"].idxmax(), "max_size_usd"])
        elif len(g):
            best_bp, best_size = float(g.max()), float(sg.loc[g.idxmax(), "size_usd"])
        else:
            best_bp = best_size = None
        dur = ep["end"] - ep["start"]
        cost = venue.cost_bp(x, q, coin_fee, reduced_fee)
        after = ep[f"bp_at_{lat}"] if len(ep) else pd.Series(dtype=float)
        net = ((after - cost).clip(lower=0) / 1e4 * ep["max_size_usd"]) if len(ep) else pd.Series(dtype=float)
        cur = now.get((x, q, d), {})
        rows.append({
            "coin": x, "via": q, "dir": d, "route": route(x, q, d), "cost_bp": cost,
            "now_bp": cur.get("gross"), "now_size_usd": cur.get("size"), "open_since": cur.get("open_since"),
            "samples": int(len(g)), "median_bp": float(g.median()) if len(g) else None,
            "p90_bp": float(g.quantile(0.9)) if len(g) else None,
            "median_size_usd": float(sg["size_usd"].median()) if len(g) else None,
            "episodes": int(len(ep)), "share_time": float(dur.sum() / (alive_h * 3.6e6)) if alive_h else 0.0,
            "median_ms": float(dur.median()) if len(ep) else None, "p90_ms": float(dur.quantile(0.9)) if len(ep) else None,
            "max_ms": float(dur.max()) if len(ep) else None,
            "best_bp": best_bp, "best_size_usd": best_size,
            "share_open_after_latency": float((after > 0).mean()) if len(ep) else 0.0,
            "median_bp_after_latency": float(after[after > 0].median()) if (after > 0).any() else None,
            "beyond_cost": int((after > cost).sum()) if len(ep) else 0,
            "usd_per_day": float(net.sum() / alive_h * 24) if alive_h else 0.0,
        })
    rows.sort(key=lambda r: -(r["median_bp"] if r["median_bp"] is not None else -1e9))
    return {"venue": venue.key, "venue_name": venue.name, "rows": rows, "alive_h": alive_h, "hours": hours, "latency_ms_used": lat,
            "since": float(s["ts"].min() / 1000) if len(s) else None, "triangles": len({(r["coin"], r["via"]) for r in rows}),
            "fee_hint": venue.fee_hint, "fee_sets": {"zero": sorted(venue.zero), "reduced": sorted(venue.reduced)},
            "reduced_label": venue.reduced_label, "default_fees": {"coin_fee": venue.default_fees[0], "stable_fee": venue.default_fees[1]}}


def detail(venue: Venue, coin: str, via: str, d: int, hours: float, coin_fee: float, reduced_fee: float) -> dict:
    """One loop: its sampled mismatch over time and its largest episodes."""
    s = _load(venue.samples_path, SAMPLE_COLS, "ts", hours)
    e = _load(venue.episodes_path, EPISODE_COLS, "start", hours)
    s = s[(s["coin"] == coin) & (s["via"] == via) & (s["dir"] == d)].sort_values("ts")
    e = e[(e["coin"] == coin) & (e["via"] == via) & (e["dir"] == d)].sort_values("max_bp", ascending=False).head(50)
    return {"venue": venue.key, "route": route(coin, via, d), "legs": venue.markets(coin, via),
            "cost_bp": venue.cost_bp(coin, via, coin_fee, reduced_fee),
            "series": {"t": (s["ts"] // 1000).tolist(), "gross": s["gross_bp"].round(3).tolist(), "size": s["size_usd"].round(0).tolist()},
            "episodes": [{"start": int(r.start), "ms": int(r.end - r.start), "open_bp": float(r.open_bp), "max_bp": float(r.max_bp),
                          "max_size_usd": float(r.max_size_usd), "updates": int(r.updates), "broken": bool(r.broken),
                          **{f"bp_at_{l}": float(getattr(r, f"bp_at_{l}")) for l in LAT_MS}} for r in e.itertuples()]}
