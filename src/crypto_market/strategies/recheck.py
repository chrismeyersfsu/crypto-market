"""Re-run the rules that looked best in the other families, acted on one bar late.

Two families found the same thing independently: on Binance.US a minute or
hour with no trade carries the last price forward, so a rule that "buys the
dip" is often buying a stale print that the next trade corrects. Acting one
bar later than the rule says is the plain test -- a signal survives it, a
fill-timing artifact does not. `uv run python -m crypto_market.strategies.recheck`.
"""

from __future__ import annotations

import pandas as pd

from .. import backtest as bt


def _eth_4h():
    h = bt.candles("binanceus", "ETHUSD", "1h")
    return h.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


def rules():
    """(strategy, params, candles, weight, exchange, market, interval) for each rule worth re-checking."""
    d4 = _eth_4h()
    h = bt.candles("binanceus", "ETHUSD", "1h")
    r = h.close.pct_change()
    return [
        ("trend: price above its own average", "n=168", d4,
         (d4.close > d4.close.rolling(168).mean()).astype(float), "binanceus", "ETHUSD", "4h"),
        ("trend: short average above long average", "96/192", d4,
         (d4.close.rolling(96).mean() > d4.close.rolling(192).mean()).astype(float), "binanceus", "ETHUSD", "4h"),
        ("reversion: after a bar that moved down more than X%, hold H bars", "x=2 h=1", h,
         (r < -0.02).astype(float), "binanceus", "ETHUSD", "1h"),
        ("reversion control: after a bar that moved up more than X%, hold H bars", "x=2 h=1", h,
         (r > 0.02).astype(float), "binanceus", "ETHUSD", "1h"),
    ]


def run():
    rows = []
    for strategy, params, df, w, ex, mk, iv in rules():
        for late in (0, 1):
            res = bt.score(df, w, ex, mk, interval=iv, late=late)
            rows.append({**res, "strategy": strategy, "params": params,
                         "note": "acted one bar late" if late else "as the family ran it"})
    return bt.record("recheck", rows, variants_tried=len(rows))


def main():
    p = run()
    df = pd.read_csv(p)
    print(df[["strategy", "params", "late", "is_ann_pct", "oos_ann_pct", "oos_sharpe", "bh_oos_ann_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
