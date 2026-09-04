"""Four holding rules on one daily series, marked to market every bar.

  buy_hold  in from the first open to the last close
  weekday   in from the week's first bar (open) to its last bar (close);
            flat over the weekend
  weekend   in from the week's last bar (close) to the next week's first
            bar (open); flat during the week
  custom    in from the close of weekday `entry_dow` to the close of weekday
            `exit_dow`, once a week; wraps over the weekend when exit <= entry
            (Thu -> Mon), and a holiday pushes a fill to the next bar
  breakout  in at the close that makes a new `breakout_n`-bar high, out at the
            close `hold_bars` bars later; legs never overlap

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

STRATEGIES = ("buy_hold", "weekday", "weekend", "custom")


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


def _custom_fills(df: pd.DataFrame, entry_dow: int, exit_dow: int) -> list:
    delta = (exit_dow - entry_dow) % 7 or 7
    dates = df.index
    out = []
    # Walk week by week from the Monday of the first bar.
    monday = dates[0] - pd.Timedelta(days=dates[0].dayofweek)
    last_exit = None
    while monday <= dates[-1]:
        target = monday + pd.Timedelta(days=entry_dow)
        i = dates.searchsorted(target)
        # Enter on the first bar on/after the target (holiday -> next bar),
        # but not one that already belongs to the following week or that
        # falls before the previous leg closed.
        if i < len(dates) and dates[i] < target + pd.Timedelta(days=7) \
                and (last_exit is None or dates[i] > last_exit):
            j = dates.searchsorted(dates[i] + pd.Timedelta(days=delta))
            if j < len(dates):
                out += [(dates[i], "close", "buy"), (dates[j], "close", "sell")]
                last_exit = dates[j]
        monday += pd.Timedelta(days=7)
    return out


def _breakout_fills(df: pd.DataFrame, n: int, hold: int) -> list:
    c = df["close"]
    new_high = (c >= c.rolling(n).max()) & c.rolling(n).max().notna()
    dates, out, i = df.index, [], 0
    while i < len(dates) - hold:
        if new_high.iloc[i]:
            out += [(dates[i], "close", "buy"), (dates[i + hold], "close", "sell")]
            i += hold + 1
        else:
            i += 1
    return out


def _fills(df: pd.DataFrame, strategy: str, entry_dow: int = 3, exit_dow: int = 0,
           breakout_n: int = 20, hold_bars: int = 5) -> list[tuple[pd.Timestamp, str, str]]:
    """(date, 'open'|'close', 'buy'|'sell') in time order."""
    if strategy == "custom":
        return _custom_fills(df, entry_dow, exit_dow)
    if strategy == "breakout":
        return _breakout_fills(df, breakout_n, hold_bars)
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


def _trend_gate(df: pd.DataFrame, fills: list, strategy: str, sma: int) -> list:
    """Only be long while the close is above its `sma`-day average.

    The signal is read at the prior close (for an open fill) or the fill
    bar's own close, so it never peeks. Buy & hold becomes the classic
    trend rule: in at the next open after a cross above, out at the next
    open after a cross below. The weekly rules just skip legs that would
    start below the average.
    """
    above = df["close"] > df["close"].rolling(sma).mean()
    if strategy == "buy_hold":
        out, held = [], False
        for prev, d in zip(df.index[:-1], df.index[1:]):
            if above[prev] and not held:
                out.append((d, "open", "buy")); held = True
            elif not above[prev] and held:
                out.append((d, "open", "sell")); held = False
        if held:
            out.append((df.index[-1], "close", "sell"))
        return out
    out = []
    for buy, sell in zip(fills[0::2], fills[1::2]):
        d, when, _ = buy
        i = df.index.get_loc(d)
        if when == "open":
            if i == 0:
                continue
            d = df.index[i - 1]
        if bool(above[d]):
            out += [buy, sell]
    return out


def run(df: pd.DataFrame, strategy: str, expense_ratio: float = 0.0,
        slippage: float = 0.0, start_cash: float = 10_000.0,
        entry_dow: int = 3, exit_dow: int = 0, sma: int = 0,
        breakout_n: int = 20, hold_bars: int = 5) -> dict:
    df = _prepare(df)
    fills = _fills(df, strategy, entry_dow, exit_dow, breakout_n, hold_bars)
    if sma:
        fills = _trend_gate(df, fills, strategy, sma)
    by_day: dict[pd.Timestamp, list] = {}
    for d, when, side in fills:
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
    if len(equity) < 2:
        return {"trades": 0}
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
