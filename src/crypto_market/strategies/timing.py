"""Timing: does *when* you hold the coin matter, apart from what the price is doing?

Four questions, all on 1h candles, all as "hold the coin (weight 1) or hold
cash (weight 0)" rules except the vol-target one, which holds a fraction:

1. Clock. Are some UTC hours, or some weekdays, reliably better than others?
   Pick the best K hours / D weekdays on the in-sample part only, then see
   whether they stay good out of sample. Plus two fixed rules that need no
   fitting at all: "hold only during US trading hours" and "hold only at
   weekends".
2. Calm or busy. Measure how jumpy the last N hours were (the standard
   deviation of hourly returns). Hold only when that is below its own 30-day
   median (calm), or only when above (busy) - two separate rules. Then a
   softer version: hold a fraction of the account so the jumpiness of what you
   hold stays near a target, weight = min(1, target / current jumpiness).
3. Quiet then a push. When the last N hours' high-to-low range is unusually
   narrow (bottom tenth of the last 30 days), wait for a close above the top
   of that range, then hold for H hours, selling early on a close back below
   the bottom of the range.
4. Thirds of the day. Hold only 00-08, only 08-16, or only 16-24 UTC. No
   fitting: just three fixed rules, to see if one third of the day carries
   the return.

Every variant that was run is written to data/strategies/timing.csv. The rows
whose `note` starts with "PICK" are the in-sample-best of their sub-family -
those are the only ones whose out-of-sample numbers mean anything, everything
else is there so the number of tries is visible.

    uv run python -m crypto_market.strategies.timing
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import backtest as bt

FAMILY = "timing"
HOUR = pd.Timedelta(hours=1)
BARS_30D = 720  # hours in 30 days
BARS_YEAR = 8760

# the markets to run everything on, plus Coinbase BTC as a cheap cross-check
MARKETS = [("binanceus", "BTCUSD"), ("binanceus", "ETHUSD"), ("binanceus", "SOLUSD")]
COMPARE = [("coinbase", "BTC-USD")]


# ---------------------------------------------------------------- helpers


def in_sample_mask(df):
    """True for the bars whose forward return is in the in-sample part."""
    ins, _ = bt.split(df.index[:-1])
    return np.append(ins, False)


def hold_when(index, picker):
    """Weight 1 at bar t when the *next* bar falls in the wanted slot.

    The weight decided at bar t's close is held over bar t+1, so a rule about
    "hold during hour 15" has to be switched on one bar earlier.
    """
    nxt = index + HOUR
    return pd.Series(np.asarray(picker(nxt), dtype=float), index=index)


def per_day(row):
    """A short plain-language line: how often it trades and what that costs a year."""
    days = row["bars"] / 24
    tpd = row["trades"] / days
    cost_year = row["turnover"] * row["cost_bp"] / days * 365 / 100
    return f"{tpd:.2f} trades/day, costs {cost_year:.1f}%/yr, in market {row['share_in_market'] * 100:.0f}% of the time"


def run_variant(rows, df, weight, exchange, market, strategy, params, note=""):
    r = bt.score(df, weight, exchange=exchange, market=market)
    r.update(strategy=strategy, params=params)
    r["note"] = (note + " | " if note else "") + per_day(r)
    rows.append(r)
    return r


def pick(rows, tag):
    """Mark the best in-sample row of a sub-family; ties broken by in-sample return."""
    group = [r for r in rows if r["params"].startswith(tag)]
    if not group:
        return None
    best = max(group, key=lambda r: (r["is_sharpe"], r["is_ann_pct"]))
    best["note"] = "PICK (best in-sample of " + tag + ") | " + best["note"]
    return best


def keep_all(rows, tag):
    """Mark every row of a sub-family: nothing was fitted, so all of them count."""
    for r in rows:
        if r["params"].startswith(tag):
            r["note"] = "PICK (fixed rule, nothing fitted) | " + r["note"]


# ---------------------------------------------------------------- 1. clock


def clock_rules(rows, df, exchange, market):
    """Best K hours of the day / best D weekdays, chosen on the in-sample part."""
    ret = df.close.pct_change()  # the return realised over the bar labelled t
    ins = in_sample_mask(df)
    by_hour = ret[ins].groupby(df.index[ins].hour).mean().sort_values(ascending=False)
    by_day = ret[ins].groupby(df.index[ins].dayofweek).mean().sort_values(ascending=False)

    for k in (4, 8, 12):
        best = set(by_hour.index[:k])
        w = hold_when(df.index, lambda ix, s=best: np.isin(ix.hour, list(s)))
        run_variant(rows, df, w, exchange, market,
                    f"hold only in the best {k} UTC hours",
                    f"hours:K={k}", f"hours picked in-sample: {sorted(best)}")

    for d in (2, 3, 4):
        best = set(by_day.index[:d])
        names = [["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i] for i in sorted(best)]
        w = hold_when(df.index, lambda ix, s=best: np.isin(ix.dayofweek, list(s)))
        run_variant(rows, df, w, exchange, market,
                    f"hold only on the best {d} weekdays",
                    f"weekdays:D={d}", f"days picked in-sample: {names}")

    # two rules with nothing fitted at all
    w = hold_when(df.index, lambda ix: (ix.dayofweek < 5) & (ix.hour >= 14) & (ix.hour <= 20))
    run_variant(rows, df, w, exchange, market,
                "hold only during US trading hours",
                "fixed:us_hours", "13:00-20:00 UTC Mon-Fri (bars ending 14:00 through 20:00)")

    w = hold_when(df.index, lambda ix: ix.dayofweek >= 5)
    run_variant(rows, df, w, exchange, market, "hold only at weekends",
                "fixed:weekend", "Saturday and Sunday UTC")


# ------------------------------------------------------- 2. calm or busy


def vol_rules(rows, df, exchange, market):
    ret = df.close.pct_change()
    for n in (24, 72, 168):
        vol = ret.rolling(n).std()  # uses bars up to and including t
        med = vol.rolling(BARS_30D, min_periods=BARS_30D // 3).median()

        w = (vol < med).astype(float)
        run_variant(rows, df, w, exchange, market,
                    f"hold only when the last {n}h were calm",
                    f"vol:calm,N={n}", "calm = jumpiness below its own 30-day median")

        w = (vol >= med).astype(float)
        run_variant(rows, df, w, exchange, market,
                    f"hold only when the last {n}h were busy",
                    f"vol:busy,N={n}", "busy = jumpiness at or above its own 30-day median")

        ann = vol * np.sqrt(BARS_YEAR)
        for target in (0.30, 0.50, 0.80):
            w = (target / ann).clip(upper=1.0).fillna(0.0)
            run_variant(rows, df, w, exchange, market,
                        f"hold a slice sized to keep jumpiness near {target:.0%}/yr",
                        f"voltarget:N={n},t={target:.2f}",
                        "weight = min(1, target / current jumpiness); it drifts every hour")

            wr = (np.round(w * 10) / 10)  # only move in 10% steps, to see the cost of the drift
            run_variant(rows, df, wr, exchange, market,
                        f"same, but only moved in 10% steps ({target:.0%}/yr target)",
                        f"voltargetstep:N={n},t={target:.2f}",
                        "the drifting weight rounded to the nearest 10% of the account")


# --------------------------------------------------- 3. quiet then a push


def squeeze_weights(df, n, h, q=0.10, win=BARS_30D):
    """Weight 1 for h bars after a close above the top of an unusually narrow range."""
    top = df.high.rolling(n).max()
    bot = df.low.rolling(n).min()
    width = (top - bot) / df.close
    thr = width.rolling(win, min_periods=win // 3).quantile(q)
    narrow = (width <= thr).to_numpy()
    topv, botv, close = top.to_numpy(), bot.to_numpy(), df.close.to_numpy()

    nb = len(df)
    w = np.zeros(nb)
    armed_top = armed_bot = np.nan
    armed_until = -1
    t = 0
    while t < nb:
        if narrow[t]:
            armed_top, armed_bot, armed_until = topv[t], botv[t], t + n
        if not np.isnan(armed_top) and t <= armed_until and close[t] > armed_top:
            stop = armed_bot
            armed_top = np.nan
            j = t
            end = min(nb, t + h)
            while j < end:
                w[j] = 1.0
                j += 1
                if j < nb and close[j] < stop:
                    break  # sold at the close of bar j
            t = j
            continue
        t += 1
    return pd.Series(w, index=df.index)


def squeeze_rules(rows, df, exchange, market):
    for n in (6, 12, 24):
        for h in (6, 12, 24):
            w = squeeze_weights(df, n, h)
            run_variant(rows, df, w, exchange, market,
                        f"buy the push out of a narrow {n}h range, hold {h}h",
                        f"squeeze:N={n},H={h}",
                        "narrow = range in the bottom tenth of the last 30 days; sell at H hours or on a close below the range bottom")


# --------------------------------------------------- 4. thirds of the day


def thirds_rules(rows, df, exchange, market):
    for lo, hi in ((0, 8), (8, 16), (16, 24)):
        w = hold_when(df.index, lambda ix, a=lo, b=hi: (ix.hour >= a) & (ix.hour < b))
        run_variant(rows, df, w, exchange, market,
                    f"hold only between {lo:02d}:00 and {hi:02d}:00 UTC",
                    f"third:{lo:02d}-{hi:02d}", "fixed rule, nothing fitted")


# ---------------------------------------------------------------- driver


def run():
    rows = []
    for exchange, market in MARKETS:
        df = bt.candles(exchange, market, "1h")
        start = len(rows)
        clock_rules(rows, df, exchange, market)
        vol_rules(rows, df, exchange, market)
        squeeze_rules(rows, df, exchange, market)
        thirds_rules(rows, df, exchange, market)
        mine = rows[start:]
        for tag in ("hours:", "weekdays:", "vol:calm", "vol:busy", "voltarget:",
                    "voltargetstep:", "squeeze:"):
            pick(mine, tag)
        for tag in ("third:", "fixed:"):
            keep_all(mine, tag)

    # Coinbase BTC for comparison: only the rules that need no fitting, since a
    # 60 bp fee per fill makes anything that trades often pointless there.
    for exchange, market in COMPARE:
        df = bt.candles(exchange, market, "1h")
        start = len(rows)
        clock_rules(rows, df, exchange, market)
        thirds_rules(rows, df, exchange, market)
        mine = rows[start:]
        for tag in ("hours:", "weekdays:"):
            pick(mine, tag)
        for tag in ("third:", "fixed:"):
            keep_all(mine, tag)

    for r in rows:
        if not r["note"].startswith("PICK"):
            r["note"] = "also ran | " + r["note"]

    path = bt.record(FAMILY, rows, variants_tried=len(rows))
    return path, rows


def main():
    path, rows = run()
    picks = [r for r in rows if r["note"].startswith("PICK")]
    cols = ["exchange", "market", "params", "trades", "cost_bp", "share_in_market",
            "is_ann_pct", "is_sharpe", "bh_is_ann_pct",
            "oos_ann_pct", "oos_sharpe", "oos_maxdd_pct", "bh_oos_ann_pct"]
    out = pd.DataFrame(picks)[cols].sort_values("oos_ann_pct", ascending=False)
    print(out.to_string(index=False))
    print(f"\n{len(rows)} variants -> {path}")


if __name__ == "__main__":
    main()
