"""FastAPI: one page, one JSON endpoint."""
from __future__ import annotations

import math
from itertools import combinations
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from . import cbtri, crossex


@asynccontextmanager
async def lifespan(_):
    # The cross-exchange tab needs tick-level timestamps, which no free
    # history offers, so the app records its own for as long as it runs.
    collector = crossex.Collector()
    collector.start()
    app.state.collector = collector
    # every triangle on each venue, from the venue's own price feed
    watchers = {key: cbtri.Watcher(venue) for key, venue in cbtri.VENUES.items()}
    for w in watchers.values():
        w.start()
    app.state.watchers = watchers
    yield
    collector.stop()
    for w in watchers.values():
        w.stop()


app = FastAPI(title="crypto-market", lifespan=lifespan)
STATIC = Path(__file__).parent / "static"


def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/crossex")
def cross_exchange(
    market: str = Query("btcusd", pattern="^(btcusd|ethusd|btcusdt|usdtusd|usdcusd|usdcusdt)$"),
    a: str = Query("coinbase", pattern="^(coinbase|bitstamp|bitfinex|binanceus)$", description="minute bars: first venue"),
    b: str = Query("bitstamp", pattern="^(coinbase|bitstamp|bitfinex|binanceus)$", description="minute bars: second venue"),
    fee: float = Query(10, ge=0, le=100, description="basis points per fill, charged on every venue touched"),
    latency_ms: int = Query(150, ge=0, le=60_000, description="how late after a gap opens your order could arrive"),
    hours: float = Query(24, ge=0.1, le=48, description="live-tick window to analyse"),
    max_hold: int = Query(60, ge=1, le=1440, description="minutes to wait for a gap to close before giving up"),
):
    if a == b:
        raise HTTPException(400, "pick two different venues")
    venues = crossex.history_venues(market)
    if a not in venues or b not in venues:
        raise HTTPException(400, f"{crossex.MARKET_NAMES[market]} minute history covers "
                                 + ", ".join(venues) + " only")
    m = crossex.minutes(market)
    if m[[a, b]].dropna().empty:
        raise HTTPException(400, "no overlapping minute history for that pair yet")
    g = crossex.gap(m, a, b, fee, max_hold)
    series = g.pop("series")
    sweep = []
    for f in (0, 0.5, 1, 2, 5, 10, 25, 60):
        r = crossex.gap(m, a, b, f, max_hold)
        sweep.append({"fee_bp": f, "trades": r["trades"], "total_profit_bp": _clean(r["total_profit_bp"]),
                      "mean_profit_bp": _clean(r["mean_profit_bp"])})
    pairs = []
    for x, y in combinations(venues, 2):
        if m[[x, y]].dropna().empty:
            continue
        r = crossex.gap(m, x, y, fee, max_hold)
        r.pop("series")
        pairs.append({k: _clean(v) for k, v in r.items()})
    all_ticks = crossex.ticks(hours, None)
    live = crossex.episodes(all_ticks[all_ticks["market"] == market], fee, latency_ms, all_ticks["ts"].values)
    collector = getattr(app.state, "collector", None)
    live["collector"] = {"since": collector.started, "counts": collector.counts[market]} if collector else None
    live["pairs"] = [{k: (_clean(v) if not isinstance(v, list) else v) for k, v in p.items()} for p in live["pairs"]]
    return {"market": market, "history_venues": venues,
            "minutes": {**{k: _clean(v) for k, v in g.items()}, "series": series, "fee_sweep": sweep, "pairs": pairs},
            "live": live}


@app.get("/api/triangle")
def triangle(
    venue: str = Query("binanceus", pattern="^(coinbase|kraken|bitstamp|bitfinex|binanceus)$"),
    fee: float = Query(2, ge=0, le=100, description="basis points per fill, three fills per round trip"),
    latency_ms: int = Query(150, ge=0, le=60_000),
    hours: float = Query(24, ge=0.1, le=48),
):
    m = crossex.triangle_minutes(fee, venue)
    series = m.pop("series", None)
    all_ticks = crossex.ticks(hours, None)
    live = crossex.triangle_live(all_ticks, fee, latency_ms, all_ticks["ts"].values if len(all_ticks) else None)
    live["routes"] = [{k: (_clean(v) if not isinstance(v, list) else v) for k, v in r.items()} for r in live["routes"]]
    return {"minutes": {**{k: _clean(v) for k, v in m.items()}, "series": series}, "live": live,
            "history_venues": [v for v in crossex.triangle_venues() if all(v in crossex.history_venues(x) for x in crossex.TRIANGLE)]}


VENUE = "^(coinbase|binanceus)$"


@app.get("/api/triangles")
def triangles(
    venue: str = Query("coinbase", pattern=VENUE),
    coin_fee: float | None = Query(None, ge=0, le=200, description="basis points per immediate fill on an ordinary pair; default: the venue's bottom tier"),
    stable_fee: float | None = Query(None, ge=0, le=200, description="basis points per immediate fill on the venue's reduced-rate pairs (Coinbase's stable pairs, Binance.US's Tier 0 pairs)"),
    latency_ms: int = Query(150, ge=0, le=60_000, description="how late after a mismatch opens your first order could arrive"),
    hours: float = Query(24, ge=0.1, le=48),
):
    ven = cbtri.VENUES[venue]
    coin_fee = ven.default_fees[0] if coin_fee is None else coin_fee
    stable_fee = ven.default_fees[1] if stable_fee is None else stable_fee
    watcher = getattr(app.state, "watchers", {}).get(venue)
    r = cbtri.report(ven, coin_fee, stable_fee, latency_ms, hours, watcher.now() if watcher else None)
    r["rows"] = [{k: _clean(v) for k, v in row.items()} for row in r["rows"]]
    r["feed"] = watcher.feed() if watcher else None
    return r


@app.get("/api/triangles/detail")
def triangle_detail(venue: str = Query("coinbase", pattern=VENUE), coin: str = Query(..., max_length=12),
                    via: str = Query(..., pattern="^(BTC|ETH|USDT|USDC)$"), dir: int = Query(1, ge=1, le=2),
                    hours: float = Query(24, ge=0.1, le=48),
                    coin_fee: float | None = Query(None, ge=0, le=200), stable_fee: float | None = Query(None, ge=0, le=200)):
    ven = cbtri.VENUES[venue]
    coin_fee = ven.default_fees[0] if coin_fee is None else coin_fee
    stable_fee = ven.default_fees[1] if stable_fee is None else stable_fee
    return cbtri.detail(ven, coin.upper(), via, dir, hours, coin_fee, stable_fee)


@app.get("/api/markets")
def markets():
    """What the page can ask for: each market's display name and which venues carry minute history / live ticks."""
    return {m: {"name": crossex.MARKET_NAMES[m], "history": crossex.history_venues(m), "live": list(crossex.MARKETS[m])}
            for m in crossex.MARKETS}


def main():
    import uvicorn
    uvicorn.run("crypto_market.app:app", host="127.0.0.1", port=8870)
