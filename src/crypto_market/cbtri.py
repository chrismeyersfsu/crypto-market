"""Every triangle Coinbase offers, watched from its order-book stream.

A triangle is a coin X that Coinbase lists in dollars and also in a bridge
currency Q (BTC, ETH or USDT) that is itself listed in dollars. The three
prices should agree: X/USD == X/Q x Q/USD. When they don't, three trades in
a loop end with more dollars than they started with, before fees. There
are two loops per triangle:

    direction 1:  USD -> X -> Q -> USD   pay the X/USD ask, sell X at the X/Q bid, sell Q at the Q/USD bid
    direction 2:  USD -> Q -> X -> USD   pay the Q/USD ask, pay the X/Q ask, sell X at the X/USD bid

One WebSocket connection carries every product's full order book. Each
time one of a triangle's three books changes, both loops are re-priced at
the current bids and asks and the state of each loop is updated. Two
files come out of it:

    cbtri_samples.csv   every SAMPLE_S seconds: each loop's gross mismatch and the dollars its thinnest book allowed
    cbtri_episodes.csv  one row per unbroken stretch of a loop being positive: when, how long, how big, and what
                        was still there LAT_MS after it opened (the question a real order has to answer)
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

URL = "wss://ws-feed.exchange.coinbase.com"
BRIDGES = ("BTC", "ETH", "USDT")
STABLE = {"USDT-USD", "USDC-USD", "USDT-USDC", "DAI-USD", "PYUSD-USD"}  # Coinbase's near-free "stable pairs"
SAMPLE_S = 20
LAT_MS = (100, 250, 500, 1000)  # what was left of an episode this long after it opened
SAMPLES = CACHE_DIR / "cbtri_samples.csv"
EPISODES = CACHE_DIR / "cbtri_episodes.csv"
SAMPLE_COLS = ["ts", "coin", "via", "dir", "gross_bp", "size_usd"]
EPISODE_COLS = (["start", "end", "coin", "via", "dir", "open_bp", "open_size_usd", "max_bp", "max_size_usd", "updates", "broken"]
                + [f"bp_at_{l}" for l in LAT_MS])


def triangles() -> list[tuple[str, str]]:
    """(coin, bridge) for every coin listed both in dollars and in a bridge that is listed in dollars."""
    ps = httpx.get("https://api.exchange.coinbase.com/products", timeout=20).json()
    pairs = {(p["base_currency"], p["quote_currency"]) for p in ps if p["status"] == "online" and not p.get("trading_disabled")}
    return sorted((x, q) for x, q in pairs if q in BRIDGES and (x, "USD") in pairs and (q, "USD") in pairs)


def legs(coin: str, via: str) -> tuple[str, str, str]:
    return f"{coin}-USD", f"{coin}-{via}", f"{via}-USD"


def route(coin: str, via: str, d: int) -> str:
    return f"USD → {coin} → {via} → USD" if d == 1 else f"USD → {via} → {coin} → USD"


def cost_bp(coin: str, via: str, coin_fee: float, stable_fee: float) -> float:
    """Three fills; a leg on a stable pair pays the stable-pair fee."""
    return sum(stable_fee if p in STABLE else coin_fee for p in legs(coin, via))


class Watcher:
    def __init__(self):
        self.tri: list[tuple[str, str]] = []
        self.books: dict[str, _Book] = {}
        self.by_product: dict[str, list[int]] = {}
        self.state: dict[tuple, dict] = {}  # (coin, via, dir) -> now: gross, size, ts; open episode
        self.samples: list[tuple] = []
        self.episodes: list[tuple] = []
        self.updates = 0
        self.connected: float | None = None
        self.task: asyncio.Task | None = None

    # -- pricing -------------------------------------------------------------

    def _reprice(self, i: int, now: int):
        x, q = self.tri[i]
        xu, xq, qu = (self.books[p].best() for p in legs(x, q))
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

    async def _feed(self):
        import websockets
        backoff = 1
        while True:
            try:
                self.tri = await asyncio.to_thread(triangles)
                self.by_product = {}
                for i, (x, q) in enumerate(self.tri):
                    for p in legs(x, q):
                        self.by_product.setdefault(p, []).append(i)
                self.books = {p: _Book(None) for p in self.by_product}
                async with websockets.connect(URL, ping_interval=20, max_size=2**26) as ws:
                    await ws.send(json.dumps({"type": "subscribe", "product_ids": list(self.books), "channels": ["level2_batch"]}))
                    backoff, self.connected, listed = 1, time.time(), time.time()
                    async for raw in ws:
                        m = json.loads(raw)
                        pid = m.get("product_id")
                        book = self.books.get(pid)
                        if book is None:
                            if m.get("type") == "error":
                                log.warning("coinbase triangles: %s", m)
                            continue
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
                            continue
                        now = int(time.time() * 1000)
                        self.updates += 1
                        for i in self.by_product[pid]:
                            self._reprice(i, now)
                        if time.time() - listed > 6 * 3600:  # pick up newly listed products
                            break
            except Exception as e:
                log.warning("coinbase triangles feed: %s (retry in %ss)", e, backoff)
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
                    if now - st["ts"] < 120_000:
                        self.samples.append((now, x, q, d, round(st["gross"], 3), round(st["size"], 2)))
            self._write(SAMPLES, SAMPLE_COLS, self.samples); self.samples = []
            self._write(EPISODES, EPISODE_COLS, self.episodes); self.episodes = []
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

    @staticmethod
    def _prune():
        cutoff = (time.time() - KEEP_H * 3600) * 1000
        for path, col in ((SAMPLES, "ts"), (EPISODES, "start")):
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


# -- analysis ------------------------------------------------------------------

def _load(path, col, hours):
    if not path.exists():
        return pd.DataFrame(columns=SAMPLE_COLS if path is SAMPLES else EPISODE_COLS)
    df = pd.read_csv(path)
    return df[df[col] >= (time.time() - hours * 3600) * 1000]


def report(coin_fee: float, stable_fee: float, latency_ms: int, hours: float, now: dict | None = None) -> dict:
    """Every loop over the last `hours`: where it stands now, its typical
    and best mismatch, how often and how long it was positive, what was
    left of it after your latency, and what beating your fees would have
    paid. Sorted by median gross, best first."""
    s, e = _load(SAMPLES, "ts", hours), _load(EPISODES, "start", hours)
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
        cost = cost_bp(x, q, coin_fee, stable_fee)
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
    return {"rows": rows, "alive_h": alive_h, "hours": hours, "latency_ms_used": lat,
            "since": float(s["ts"].min() / 1000) if len(s) else None, "triangles": len({(r["coin"], r["via"]) for r in rows})}


def detail(coin: str, via: str, d: int, hours: float) -> dict:
    """One loop: its sampled mismatch over time and its largest episodes."""
    s, e = _load(SAMPLES, "ts", hours), _load(EPISODES, "start", hours)
    s = s[(s["coin"] == coin) & (s["via"] == via) & (s["dir"] == d)].sort_values("ts")
    e = e[(e["coin"] == coin) & (e["via"] == via) & (e["dir"] == d)].sort_values("max_bp", ascending=False).head(50)
    return {"route": route(coin, via, d), "legs": legs(coin, via),
            "series": {"t": (s["ts"] // 1000).tolist(), "gross": s["gross_bp"].round(3).tolist(), "size": s["size_usd"].round(0).tolist()},
            "episodes": [{"start": int(r.start), "ms": int(r.end - r.start), "open_bp": float(r.open_bp), "max_bp": float(r.max_bp),
                          "max_size_usd": float(r.max_size_usd), "updates": int(r.updates), "broken": bool(r.broken),
                          **{f"bp_at_{l}": float(getattr(r, f"bp_at_{l}")) for l in LAT_MS}} for r in e.itertuples()]}
