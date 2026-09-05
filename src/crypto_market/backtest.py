"""Shared harness for the strategy search: one set of costs, one split, one score.

A strategy turns candles into a target weight series: the fraction of the
account held in the coin at each bar, 0 to 1 (spot: nothing can be sold
that isn't owned). The weight decided at bar t's close is held over bar
t+1, so a strategy can only use what it could have known. Every change
in weight pays fee + half the spread on the amount changed. The last
`OOS` share of the bars is held back: pick parameters on the first part,
report the held-back part separately, and never tune on it.

    from crypto_market import backtest as bt
    df = bt.candles("binanceus", "BTCUSD", "1h")
    w = (df.close > df.close.rolling(24).mean()).astype(float)   # a rule
    r = bt.score(df, w, exchange="binanceus", market="BTCUSD")
    bt.record("trend", [{**r, "strategy": "close above 24h average", "params": "n=24"}])
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .butrades import DATA_DIR

HIST_DIR = DATA_DIR / "hist"
RESULTS_DIR = DATA_DIR / "strategies"
OOS = 0.3  # share of the bars held back for the out-of-sample check

# fee per fill for an immediate fill at the bottom tier, bp
FEE_BP = {"binanceus": 2.0, "coinbase": 60.0}
# half the bid-ask spread, bp, from this server's tick recorder (median), with a
# guess for coins it doesn't record; a taker pays this on top of the fee
HALF_SPREAD_BP = {
    "binanceus": {"BTCUSD": 0.3, "BTCUSDT": 0.35, "ETHUSD": 0.5, "ETHUSDT": 0.6, "ETHBTC": 1.0,
                  "SOLUSD": 1.5, "USDTUSD": 0.15, "USDCUSD": 0.1, "USDCUSDT": 0.1, "*": 5.0},
    "coinbase": {"BTC-USD": 0.1, "BTC-USDT": 1.5, "ETH-USD": 0.25, "ETH-BTC": 1.0, "SOL-USD": 1.0,
                 "USDT-USD": 0.1, "USDC-USD": 0.1, "*": 3.0},
}
STEP = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "12h": "12h", "1d": "1D"}
BARS_PER_YEAR = {"1m": 525_600, "5m": 105_120, "15m": 35_040, "1h": 8_760, "4h": 2_190, "12h": 730, "1d": 365}


def candles(exchange, symbol, interval):
    """Candles as a DataFrame indexed by UTC time: open high low close volume trades taker_buy_volume.

    Bars the exchange has no candle for are present as blank rows, so a hole
    in the history (Binance.US has none from 2023-07-14 to 2025-02-19) is a
    stretch of no data, not one enormous bar; a rule's rolling window and the
    return across it come out NaN, and `score` skips those bars.
    """
    p = HIST_DIR / f"{exchange}_{symbol.lower()}_{interval}.csv"
    df = pd.read_csv(p)
    df.index = pd.to_datetime(df.ts, unit="ms", utc=True)
    df = df.drop(columns="ts")
    df = df[~df.index.duplicated()].sort_index()
    step = STEP.get(interval)
    if step is not None and len(df) > 1:
        df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq=step))
    return df


def holes(df, interval=None):
    """(start, end) of every stretch of blank bars in a candles frame."""
    missing = df.close.isna()
    if not missing.any():
        return []
    grp = (missing != missing.shift()).cumsum()
    return [(str(g.index[0])[:16], str(g.index[-1])[:16]) for _, g in missing[missing].groupby(grp[missing])]


def available(exchange=None, interval=None):
    """(exchange, symbol, interval) for every history file on disk."""
    out = []
    for p in sorted(HIST_DIR.glob("*.csv")):
        ex, rest = p.stem.split("_", 1)
        sym, iv = rest.rsplit("_", 1)
        if (exchange is None or ex == exchange) and (interval is None or iv == interval):
            out.append((ex, sym.upper(), iv))
    return out


def cost_bp(exchange, market, fee_bp=None, maker=False):
    """Round cost of changing the position by 100% of the account, one way, in bp."""
    hs = HALF_SPREAD_BP[exchange]
    half = hs.get(market, hs["*"])
    fee = FEE_BP[exchange] if fee_bp is None else fee_bp
    if maker:  # a resting fill: no fee at Binance.US; the fill is still at the bar's close, so no spread either way
        return 0.0 if exchange == "binanceus" and fee_bp is None else fee
    return fee + half


def split(index):
    """Boolean masks (in_sample, out_of_sample) over a time index."""
    n = len(index)
    cut = int(n * (1 - OOS))
    m = np.zeros(n, dtype=bool)
    m[:cut] = True
    return m, ~m


def _stats(ret, bars_per_year):
    ret = ret[~np.isnan(ret)]
    n = len(ret)
    if n == 0:
        return {"return_pct": 0.0, "ann_pct": 0.0, "sharpe": 0.0, "maxdd_pct": 0.0, "bars": 0}
    eq = np.cumprod(1 + ret)
    total = eq[-1] - 1
    years = n / bars_per_year
    ann = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else -1.0
    sd = ret.std(ddof=1) if n > 1 else 0.0
    sharpe = ret.mean() / sd * math.sqrt(bars_per_year) if sd > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    dd = ((eq - peak) / peak).min()
    return {"return_pct": round(total * 100, 3), "ann_pct": round(ann * 100, 2),
            "sharpe": round(sharpe, 2), "maxdd_pct": round(dd * 100, 2), "bars": int(n)}


def score(df, weight, exchange, market, interval=None, fee_bp=None, maker=False, price="close", late=0):
    """Score a target-weight series against candles; returns one flat result row.

    `weight` is aligned to `df.index`, in [0, 1], decided at each bar's close and
    held over the next bar. Fills happen at the next bar's `price` column
    (default close; pass "open" for a fill at the next open, if the data has it).
    Costs are charged on every change in weight. In-sample and out-of-sample
    are reported separately, with buy-and-hold on the same bars for comparison.
    `late=1` acts on each decision one bar later than the rule says: a result
    that survives that is a signal, one that vanishes was the fill timing.
    """
    interval = interval or _interval_of(df.index)
    bpy = BARS_PER_YEAR[interval]
    w = pd.Series(weight, index=df.index).astype(float).clip(0, 1).fillna(0.0)
    if late:
        w = w.shift(late).fillna(0.0)
    px = df[price]
    w = w.where(px.notna(), 0.0)  # nothing can be held through a hole in the data
    bar_ret = px.pct_change().shift(-1)  # return earned over the bar after the decision (NaN at a hole: skipped)
    held = w.shift(1).fillna(0.0)  # the position that was held during this bar
    turn = (w - held).abs()  # what changed at this bar's close
    c = cost_bp(exchange, market, fee_bp, maker) / 1e4
    strat = (w * bar_ret - turn * c).where(bar_ret.notna()).to_numpy()
    strat = strat[:-1]  # the last bar's forward return is unknown
    bh = bar_ret.to_numpy()[:-1]
    ins, oos = split(df.index[:-1])
    s_in, s_out = _stats(strat[ins], bpy), _stats(strat[oos], bpy)
    b_in, b_out = _stats(bh[ins], bpy), _stats(bh[oos], bpy)
    trades = int((turn > 1e-9).sum())
    gross_bp = (w * bar_ret).sum() * 1e4
    interval_holes = holes(df)
    return {
        "exchange": exchange, "market": market, "interval": interval,
        "fee_bp": FEE_BP[exchange] if fee_bp is None else fee_bp, "cost_bp": round(c * 1e4, 2),
        "maker": bool(maker), "bars": len(strat), "trades": trades,
        "turnover": round(float(turn.sum()), 1),
        "bp_per_trade": round(float(gross_bp / max(turn.sum() / 2, 1e-9)), 2),  # gross edge per round trip
        "share_in_market": round(float(held.mean()), 3),
        "is_return_pct": s_in["return_pct"], "is_ann_pct": s_in["ann_pct"], "is_sharpe": s_in["sharpe"], "is_maxdd_pct": s_in["maxdd_pct"],
        "oos_return_pct": s_out["return_pct"], "oos_ann_pct": s_out["ann_pct"], "oos_sharpe": s_out["sharpe"], "oos_maxdd_pct": s_out["maxdd_pct"],
        "bh_is_ann_pct": b_in["ann_pct"], "bh_oos_ann_pct": b_out["ann_pct"],
        "is_from": str(df.index[0])[:16], "oos_from": str(df.index[:-1][oos][0])[:16] if oos.any() else "", "to": str(df.index[-1])[:16],
        "late": int(late),
        "note": f"{len(interval_holes)} hole(s) in the data skipped: " + ", ".join(f"{a} to {b}" for a, b in interval_holes[:3]) if interval_holes else "",
    }


def score_returns(ret, index, interval, exchange, market, trades, bh=None, note=""):
    """Score a per-bar net return series you built yourself (multi-asset, pairs, resting fills...).

    Use this when `score` doesn't fit; you are then responsible for the costs
    and the no-look-ahead rule. `bh` is a benchmark return series, optional.
    """
    ret = np.asarray(ret, dtype=float)
    bpy = BARS_PER_YEAR[interval]
    ins, oos = split(index)
    s_in, s_out = _stats(ret[ins], bpy), _stats(ret[oos], bpy)
    b_in = _stats(np.asarray(bh)[ins], bpy) if bh is not None else {"ann_pct": None}
    b_out = _stats(np.asarray(bh)[oos], bpy) if bh is not None else {"ann_pct": None}
    return {
        "exchange": exchange, "market": market, "interval": interval, "bars": len(ret), "trades": int(trades),
        "is_return_pct": s_in["return_pct"], "is_ann_pct": s_in["ann_pct"], "is_sharpe": s_in["sharpe"], "is_maxdd_pct": s_in["maxdd_pct"],
        "oos_return_pct": s_out["return_pct"], "oos_ann_pct": s_out["ann_pct"], "oos_sharpe": s_out["sharpe"], "oos_maxdd_pct": s_out["maxdd_pct"],
        "bh_is_ann_pct": b_in["ann_pct"], "bh_oos_ann_pct": b_out["ann_pct"],
        "is_from": str(index[0])[:16], "oos_from": str(index[oos][0])[:16] if oos.any() else "", "to": str(index[-1])[:16],
        "note": note,
    }


def _interval_of(index):
    step = pd.Series(index).diff().median()
    return {60: "1m", 300: "5m", 900: "15m", 3600: "1h", 14400: "4h", 43200: "12h", 86400: "1d"}[int(step.total_seconds())]


COLUMNS = ["family", "strategy", "params", "exchange", "market", "interval", "fee_bp", "cost_bp", "maker",
           "bars", "trades", "turnover", "bp_per_trade", "share_in_market",
           "is_return_pct", "is_ann_pct", "is_sharpe", "is_maxdd_pct",
           "oos_return_pct", "oos_ann_pct", "oos_sharpe", "oos_maxdd_pct",
           "bh_is_ann_pct", "bh_oos_ann_pct", "is_from", "oos_from", "to", "late", "variants_tried", "note"]


def record(family, rows, variants_tried=None):
    """Write a family's results to data/strategies/<family>.csv (replacing it).

    Each row needs at least `strategy` and `params` on top of what `score`
    returns. `variants_tried` is how many parameter combinations were run in
    total, kept so a lucky best-of-many is read as such.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / f"{family}.csv"
    with p.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            r = {"family": family, "variants_tried": variants_tried if variants_tried is not None else len(rows), **r}
            r["params"] = r.get("params") if isinstance(r.get("params"), str) else json.dumps(r.get("params", {}))
            wr.writerow({k: r.get(k, "") for k in COLUMNS})
    return p


def results():
    """Every family's rows in one frame, best out-of-sample first."""
    parts = [pd.read_csv(p) for p in sorted(RESULTS_DIR.glob("*.csv"))]
    if not parts:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat(parts, ignore_index=True)
    return df.sort_values("oos_ann_pct", ascending=False).reset_index(drop=True)
