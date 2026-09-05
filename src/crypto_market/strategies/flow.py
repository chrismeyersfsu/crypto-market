"""Flow: does who was in a hurry to trade say anything about the next bar?

Every Binance.US candle records how much of its volume came from buyers who
took the offer (`taker_buy_volume`) and how many separate trades printed
(`trades`). The 30-day trade tape goes further and says, for each fill,
whether a buyer lifted a resting sell or a seller hit a resting buy. This
module asks four plain questions with that data:

1. buy share    -- when most of the recent volume was buyers in a hurry,
                   does the coin keep going up? (and the opposite reading:
                   maybe a crowd of eager buyers is a good moment to sell)
2. volume spike -- after one bar trades far more than usual, is the next
                   stretch worth holding? Split by whether that bar closed
                   up or down.
3. trade spike  -- the same question using the number of trades instead of
                   the size traded: many small trades vs a few big ones.
4. tape         -- from the raw 30-day tape: minute-by-minute buy vs sell
                   pressure, and what happens right after a big sell print.

Nothing here sells what it doesn't own: every rule produces a target share
of the account held in the coin, 0 or 1. Parameters are picked on the first
70% of the bars only; the rest is reported and never used for choosing.

    uv run python -m crypto_market.strategies.flow
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import backtest as bt
from ..butrades import load as load_trades

FAMILY = "flow"
MARKETS = ["BTCUSD", "ETHUSD", "SOLUSD"]
EXCHANGE = "binanceus"
BIG_TRADE_USD = 10_000


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Roll 1-minute candles up into longer bars (5m, 15m, ...)."""
    out = df.resample(rule).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
        trades=("trades", "sum"), taker_buy_volume=("taker_buy_volume", "sum"),
    )
    return out.dropna(subset=["close"])


def hysteresis(enter: pd.Series, exit_: pd.Series) -> pd.Series:
    """Hold once `enter` fires, keep holding until `exit_` fires.

    Both are read at a bar's close, so the share they set is what is held
    over the next bar. Between the two the previous state carries forward.
    """
    state = pd.Series(np.nan, index=enter.index)
    state[enter.to_numpy()] = 1.0
    state[exit_.to_numpy() & ~enter.to_numpy()] = 0.0
    return state.ffill().fillna(0.0)


def hold_after(signal: pd.Series, bars: int) -> pd.Series:
    """Hold for `bars` bars after each bar where `signal` is true."""
    return signal.astype(float).rolling(bars, min_periods=1).max().fillna(0.0)


def in_sample_best(rows: list[dict]) -> dict | None:
    """The variant with the best in-sample yearly return (never looks at OOS)."""
    return max(rows, key=lambda r: r["is_ann_pct"]) if rows else None


def mark_best(rows: list[dict], label: str) -> None:
    """Flag the in-sample winner of a sub-family so the CSV reads clearly."""
    best = in_sample_best(rows)
    if best is not None:
        best["note"] = (best.get("note", "") + " | " if best.get("note") else "") + \
            f"in-sample best of {label} ({len(rows)} variants)"


# --------------------------------------------------------------------------
# 1. buy share of recent volume
# --------------------------------------------------------------------------

def buy_share_rows() -> list[dict]:
    """r = buyer-in-a-hurry volume / all volume over the last N bars.

    Hold when r rises above 0.5 + k, sell when it falls back under 0.5; and
    the contrarian reading, hold when r drops under 0.5 - k and sell when it
    climbs back over 0.5. Grid: 3 bar sizes x 3 windows x 3 thresholds x 2
    readings = 54 variants per coin.
    """
    rows: list[dict] = []
    for market in MARKETS:
        one_min = bt.candles(EXCHANGE, market, "1m")
        frames = {"5m": resample(one_min, "5min"), "15m": resample(one_min, "15min"),
                  "1h": bt.candles(EXCHANGE, market, "1h")}
        for direction in ("with", "against"):
            sub: list[dict] = []
            for interval, df in frames.items():
                for n in (5, 15, 60):
                    taker = df.taker_buy_volume.rolling(n).sum()
                    total = df.volume.rolling(n).sum()
                    r = (taker / total.where(total > 0)).fillna(0.5)
                    for k in (0.05, 0.10, 0.20):
                        if direction == "with":
                            w = hysteresis(r > 0.5 + k, r < 0.5)
                            name = "hold while buyers are in a hurry"
                        else:
                            w = hysteresis(r < 0.5 - k, r > 0.5)
                            name = "hold while sellers are in a hurry"
                        res = bt.score(df, w, exchange=EXCHANGE, market=market, interval=interval)
                        res["strategy"] = name
                        res["params"] = f"buy_share bar={interval} n={n} k={k} read={direction}"
                        # the 5m/15m frames come from the 1-minute file, which
                        # starts later than the hourly one: different windows
                        res["note"] = "bars rolled up from 1m" if interval != "1h" else ""
                        sub.append(res)
            mark_best(sub, f"buy share / {direction} / {market}")
            rows += sub
    return rows


# --------------------------------------------------------------------------
# 2 and 3. one bar trades far more than usual
# --------------------------------------------------------------------------

def spike_rows(column: str, label: str) -> list[dict]:
    """Hold for H bars after a bar whose `column` beat m x its recent median.

    The median is over the 60 bars before the spike bar, so the test never
    uses the bar it is judging. Split by whether the spike bar closed up or
    down. Grid: 3 sizes x 4 holding times x 2 directions = 24 per coin, on
    hourly bars.
    """
    rows: list[dict] = []
    for market in MARKETS:
        df = bt.candles(EXCHANGE, market, "1h")
        base = df[column].rolling(60).median().shift(1)
        up = df.close > df.open
        sub: list[dict] = []
        for m in (3, 5, 10):
            big = df[column] > m * base.where(base > 0)
            for how, mask in (("up", big & up), ("down", big & ~up)):
                for h in (1, 3, 6, 24):
                    w = hold_after(mask.fillna(False), h)
                    res = bt.score(df, w, exchange=EXCHANGE, market=market, interval="1h")
                    res["strategy"] = f"hold {h}h after a {label} spike on an {how} bar"
                    res["params"] = f"{label}_spike m={m} hold={h} bar={how}"
                    sub.append(res)
        mark_best(sub, f"{label} spike / {market}")
        rows += sub
    return rows


# --------------------------------------------------------------------------
# 4. the raw trade tape
# --------------------------------------------------------------------------

def minute_tape(symbol: str) -> pd.DataFrame:
    """The 30-day tape rolled into 1-minute bars.

    close     last traded price (carried forward through quiet minutes)
    buy_qty   size bought by someone who lifted a resting sell
    sell_qty  size sold by someone who hit a resting buy
    big_sell  count of sells over $10k; big_buy the same for buys
    """
    t = load_trades(symbol)
    t["time"] = pd.to_datetime(t.ts, unit="ms", utc=True)
    t = t.set_index("time")
    usd = t.qty * t.price
    aggressive_buy = ~t.buyer_maker  # a buyer lifted the offer
    parts = pd.DataFrame({
        "close": t.price,
        "buy_qty": t.qty.where(aggressive_buy, 0.0),
        "sell_qty": t.qty.where(~aggressive_buy, 0.0),
        "big_buy": ((usd > BIG_TRADE_USD) & aggressive_buy).astype(float),
        "big_sell": ((usd > BIG_TRADE_USD) & ~aggressive_buy).astype(float),
    })
    out = parts.resample("1min").agg({"close": "last", "buy_qty": "sum", "sell_qty": "sum",
                                     "big_buy": "sum", "big_sell": "sum"})
    out["close"] = out.close.ffill()
    return out.dropna(subset=["close"])


def tape_rows() -> list[dict]:
    """Two readings of the tape, on 1-minute bars from 30 days of trades.

    imbalance: (bought - sold) / (bought + sold) over the last N minutes;
    hold the next minute when it is above k (and the contrarian reading,
    hold when it is below -k). Grid 3 windows x 3 thresholds x 2 readings.

    big sell: hold for H minutes after a sell print over $10k, on the idea
    that a forced seller pushes the price below where it settles; and the
    same after a big buy print. Grid 4 holding times x 2 sides.
    """
    rows: list[dict] = []
    for market in ["BTCUSD", "ETHUSD"]:
        df = minute_tape(market)
        note = "30 days of tape only; the held-back part is ~9 days"
        imb_num = (df.buy_qty - df.sell_qty)
        imb_den = (df.buy_qty + df.sell_qty)
        sub: list[dict] = []
        for n in (1, 5, 15):
            r = (imb_num.rolling(n).sum() / imb_den.rolling(n).sum().where(lambda s: s > 0)).fillna(0.0)
            for k in (0.2, 0.4, 0.6):
                for direction in ("with", "against"):
                    w = (r > k).astype(float) if direction == "with" else (r < -k).astype(float)
                    res = bt.score(df, w, exchange=EXCHANGE, market=market, interval="1m")
                    res["strategy"] = ("hold the minute after buyers dominated the tape" if direction == "with"
                                       else "hold the minute after sellers dominated the tape")
                    res["params"] = f"tape_imbalance n={n} k={k} read={direction}"
                    res["note"] = note
                    sub.append(res)
        mark_best(sub, f"tape imbalance / {market}")
        rows += sub

        sub = []
        for side in ("sell", "buy"):
            hits = (df.big_sell if side == "sell" else df.big_buy) > 0
            for h in (1, 3, 6, 24):
                w = hold_after(hits, h)
                res = bt.score(df, w, exchange=EXCHANGE, market=market, interval="1m")
                res["strategy"] = f"hold {h} min after a big {side} print (over ${BIG_TRADE_USD:,})"
                res["params"] = f"big_{side}_print hold={h}"
                res["note"] = note
                sub.append(res)
        mark_best(sub, f"big print / {market}")
        rows += sub
    return rows


# --------------------------------------------------------------------------

def run() -> list[dict]:
    rows: list[dict] = []
    rows += buy_share_rows()
    rows += spike_rows("volume", "volume")
    rows += spike_rows("trades", "trade count")
    rows += tape_rows()
    path = bt.record(FAMILY, rows, variants_tried=len(rows))
    print(f"{len(rows)} variants -> {path}")
    return rows


def main() -> None:
    rows = run()
    df = pd.DataFrame(rows)
    cols = ["strategy", "params", "market", "interval", "trades", "cost_bp", "bp_per_trade",
            "is_ann_pct", "is_sharpe", "oos_ann_pct", "oos_sharpe", "bh_oos_ann_pct"]
    best = df[df.note.fillna("").str.contains("in-sample best")] if "note" in df else df
    print(best[cols].to_string(index=False))


if __name__ == "__main__":
    main()
