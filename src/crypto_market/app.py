"""FastAPI: one page, one JSON endpoint."""
from __future__ import annotations

import math
from itertools import combinations
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from . import crossex


@asynccontextmanager
async def lifespan(_):
    # The cross-exchange tab needs tick-level timestamps, which no free
    # history offers, so the app records its own for as long as it runs.
    collector = crossex.Collector()
    collector.start()
    app.state.collector = collector
    yield
    collector.stop()


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
    market: str = Query("btcusd", pattern="^(btcusd|usdtusd|usdcusd|usdcusdt)$"),
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


@app.get("/api/markets")
def markets():
    """What the page can ask for: each market's display name and which venues carry minute history / live ticks."""
    return {m: {"name": crossex.MARKET_NAMES[m], "history": crossex.history_venues(m), "live": list(crossex.MARKETS[m])}
            for m in crossex.MARKETS}


def main():
    import uvicorn
    uvicorn.run("crypto_market.app:app", host="127.0.0.1", port=8870)
