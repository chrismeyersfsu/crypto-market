"""Three holding rules on one daily series, marked to market every bar.

  buy_hold  in from the first open to the last close
  weekday   in from the week's first bar (open) to its last bar (close);
            flat over the weekend
  weekend   in from the week's last bar (close) to the next week's first
            bar (open); flat during the week

Bars on Sat/Sun (crypto spot) are never fill days, so on every series
weekday = Mon open -> Fri close and weekend = Fri close -> Mon open (holidays
shift the boundary to the nearest trading day).

Costs: a per-side slippage on every fill (Fidelity charges $0 commission,
but you still cross the spread), and an expense ratio accrued per calendar
day held. ETF prices already embed their own expense ratio; the drag is
here so spot BTC/ETH can be compared as if held through a 0.25% ETF.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

STRATEGIES = ("buy_hold", "weekday", "weekend")


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["open", "close"]].copy()
    iso = df.index.isocalendar()
    df["week"] = iso["year"].astype(int) * 100 + iso["week"].astype(int)
    # Week boundaries are the first and last Mon-Fri bar of the ISO week;
    # Sat/Sun bars (crypto spot) are held through, never traded on. That
    # makes the weekend leg Fri close -> Mon open on every series.
    wk = df.index.dayofweek < 5
    key = df["week"].where(wk)
    df["first"] = wk & ~key.duplicated(keep="first")
    df["last"] = wk & ~key.duplicated(keep="last")
    return df


def _fills(df: pd.DataFrame, strategy: str) -> list[tuple[pd.Timestamp, str, str]]:
    """(date, 'open'|'close', 'buy'|'sell') in time order."""
    if strategy == "buy_hold":
        return [(df.index[0], "open", "buy"), (df.index[-1], "close", "sell")]
    out = []
    if strategy == "weekday":
        for d, r in df.iterrows():
            if r["first"]:
                out.append((d, "open", "buy"))
            if r["last"]:
                out.append((d, "close", "sell"))
    elif strategy == "weekend":
        for d, r in df.iterrows():
            if r["first"]:
                out.append((d, "open", "sell"))
            if r["last"]:
                out.append((d, "close", "buy"))
        # Leading sell (nothing held yet) and trailing buy (never closed) are
        # not trades.
        if out and out[0][2] == "sell":
            out.pop(0)
        if out and out[-1][2] == "buy":
            out.pop()
    else:
        raise ValueError(strategy)
    return out


def run(df: pd.DataFrame, strategy: str, expense_ratio: float = 0.0,
        slippage: float = 0.0, start_cash: float = 10_000.0) -> dict:
    df = _prepare(df)
    by_day: dict[pd.Timestamp, list] = {}
    for d, when, side in _fills(df, strategy):
        by_day.setdefault(d, []).append((when, side))

    daily_decay = (1.0 - expense_ratio) ** (1.0 / 365.0)
    cash, units = start_cash, 0.0
    entry_date = entry_cash = None
    prev_day = None
    equity, trades = [], []
    for d, r in df.iterrows():
        if units and prev_day is not None:
            units *= daily_decay ** (d - prev_day).days
        for when, side in by_day.get(d, []):
            if side == "buy":
                units, entry_cash, entry_date, cash = cash / (r[when] * (1 + slippage)), cash, d, 0.0
            else:
                cash = units * r[when] * (1 - slippage)
                trades.append({
                    "entry": entry_date.date().isoformat(),
                    "exit": d.date().isoformat(),
                    "days": (d - entry_date).days,
                    "ret": cash / entry_cash - 1,
                })
                units = 0.0
        equity.append(cash + units * r["close"])
        prev_day = d

    return {"equity": pd.Series(equity, index=df.index), "trades": pd.DataFrame(trades)}


def stats(equity: pd.Series, trades: pd.DataFrame) -> dict:
    days = (equity.index[-1] - equity.index[0]).days
    years = max(days / 365.25, 1e-9)
    total = equity.iloc[-1] / equity.iloc[0] - 1
    daily = equity.pct_change().dropna()
    # Bars per year differs between spot (365) and exchange-traded (~252).
    bars_per_year = len(equity) / years
    vol = daily.std() * math.sqrt(bars_per_year) if len(daily) > 1 else float("nan")
    cagr = (1 + total) ** (1 / years) - 1
    dd = (equity / equity.cummax() - 1).min()
    out = {
        "total_return": total, "cagr": cagr, "max_drawdown": dd,
        "volatility": vol, "sharpe": cagr / vol if vol else float("nan"),
        "final": equity.iloc[-1], "trades": int(len(trades)),
    }
    if len(trades):
        r = trades["ret"]
        out.update({
            "win_rate": float((r > 0).mean()),
            "avg_trade": float(r.mean()),
            "median_trade": float(r.median()),
            # One-sample t vs. zero mean: is the average leg return
            # distinguishable from noise?
            "t_stat": float(r.mean() / (r.std(ddof=1) / math.sqrt(len(r)))) if len(r) > 2 and r.std() else float("nan"),
        })
    return out


def welch_t(a: pd.Series, b: pd.Series) -> float:
    """Two-sample t (unequal variance) for mean(a) - mean(b)."""
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se else float("nan")


def day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and median close-to-close return by weekday (Mon=0)."""
    r = df["close"].pct_change().dropna()
    g = r.groupby(r.index.dayofweek)
    return pd.DataFrame({"mean": g.mean(), "median": g.median(), "n": g.size(),
                         "win_rate": g.apply(lambda x: (x > 0).mean())})
