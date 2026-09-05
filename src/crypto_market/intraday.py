"""Sub-daily rules on hourly BTC bars — the timescale "day trading" implies.

Two things get tested: hour-of-day seasonality (is any UTC hour reliably
better/worse, the intraday analogue of the day-of-week check), and
momentum/reversal (does the sign of the last `lookback` hours predict the
next `lookback` hours). Both are marked to market every hour with a
per-flip slippage cost, because at hourly rebalancing frequency, cost is
usually the whole story.
"""
from __future__ import annotations

import math

import pandas as pd


def hour_of_day(close: pd.Series) -> pd.DataFrame:
    r = close.pct_change().dropna()
    g = r.groupby(r.index.hour)
    return pd.DataFrame({"mean": g.mean(), "median": g.median(), "n": g.size(),
                         "win_rate": g.apply(lambda x: (x > 0).mean())})


def _welch_t(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna(), b.dropna()
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se else float("nan")


def momentum(close: pd.Series, lookback: int, direction: str, slippage: float = 0.0005) -> dict:
    """direction: 'momentum' (long after a rise) or 'reversal' (long after a fall).
    Position decided once every `lookback` hours from data up to that point
    (no lookahead), held flat otherwise; a slippage cost is charged on every
    position change, not just entries, since this flips constantly."""
    ret = close.pct_change().fillna(0)
    up = close.pct_change(lookback) > 0
    sig = up if direction == "momentum" else ~up
    pos = sig.astype(float).shift(1).fillna(0)
    turn = pos.diff().abs().fillna(0)
    equity = (1 + ret * pos - slippage * turn).cumprod()

    fwd = close.pct_change(lookback).shift(-lookback)
    prior_up = up.shift(lookback).fillna(False).astype(bool)
    after_up = fwd[prior_up]
    after_down = fwd[~prior_up]
    a, b = (after_up, after_down) if direction == "momentum" else (after_down, after_up)

    return {"equity": equity, "flips": int(turn.sum()),
            "signal_t_stat": _welch_t(a, b),
            "signal_mean_with": float(a.mean()) if len(a) else float("nan"),
            "signal_mean_against": float(b.mean()) if len(b) else float("nan")}


def stats(equity: pd.Series) -> dict:
    if len(equity) < 2:
        return {}
    yrs = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = equity.iloc[-1] ** (1 / yrs) - 1
    d = equity.pct_change().dropna()
    vol = d.std() * math.sqrt(24 * 365) if len(d) > 1 else float("nan")
    dd = (equity / equity.cummax() - 1).min()
    return {"cagr": cagr, "max_drawdown": dd, "volatility": vol,
            "sharpe": cagr / vol if vol else float("nan"), "final": float(equity.iloc[-1])}
