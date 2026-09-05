"""Buy-the-dip rules: does a price that fell below its own recent average come back?

Five ideas, each scored on its own small grid. Everything here is a
"buy low, sell when it has come back" rule; nothing sells what it does
not own.

1. Far below the recent average (z-score of the close against a rolling
   mean and standard deviation): buy when the close is k standard
   deviations below the average, sell when it has climbed back.
2. The same shape drawn as bands (Bollinger): buy at the lower band,
   sell at the middle band or at the upper band.
3. A very short RSI (2 or 3 bars): buy when it is very low, sell when it
   recovers or after a fixed number of bars.
4. One big down bar: after a bar that fell more than X%, hold the coin
   for H bars. The mirror rule (after a big UP bar) is run as a control:
   if both make money the rule is not about dips, it is about the coin
   going up in that period.
5. Stablecoin off its peg: buy USDT or USDC a few basis points below a
   dollar, sell when it is back at a dollar.

Parameters are chosen on the first 70% of the bars only. Every variant
that was run is written to the results file, so a best-of-many is
visible as such.

    uv run python -m crypto_market.strategies.reversion
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_market import backtest as bt

FAMILY = "reversion"

COINS = ["BTCUSD", "ETHUSD", "SOLUSD"]
CB_OF = {"BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD", "SOLUSD": "SOL-USD"}

# how many parameter combinations were scored in total, kept honest
_tried = 0


# ---------------------------------------------------------------- data helpers

def _downsample(df, rule):
    """Roll 1-minute candles up into bigger ones (5m, 15m)."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    for extra in ("trades", "taker_buy_volume"):
        if extra in df.columns:
            agg[extra] = "sum"
    out = df.resample(rule).agg(agg)
    return out.dropna(subset=["close"])


_CACHE = {}


def bars(exchange, market, interval):
    """Candles at `interval`, built from 1-minute data for 5m and 15m."""
    key = (exchange, market, interval)
    if key not in _CACHE:
        if interval in ("5m", "15m"):
            rule = {"5m": "5min", "15m": "15min"}[interval]
            _CACHE[key] = _downsample(bt.candles(exchange, market, "1m"), rule)
        else:
            _CACHE[key] = bt.candles(exchange, market, interval)
    return _CACHE[key]


# ------------------------------------------------------------- signal helpers

def hold_between(enter, leave):
    """1 from the bar `enter` is true until the bar `leave` is true, else 0.

    Both are decided at a bar's close from data up to that bar, so the
    weight this returns is only ever based on the past. A bar that says
    both "buy" and "sell" is read as "buy".
    """
    s = pd.Series(np.nan, index=enter.index)
    s[leave.fillna(False).to_numpy()] = 0.0
    s[enter.fillna(False).to_numpy()] = 1.0
    return s.ffill().fillna(0.0)


def hold_with_timeout(enter, leave, max_hold):
    """Same, but also sell after `max_hold` bars whatever the sell rule says."""
    e = enter.fillna(False).to_numpy()
    x = leave.fillna(False).to_numpy()
    n = len(e)
    w = np.zeros(n)
    held = 0
    age = 0
    for i in range(n):
        if held:
            age += 1
            if x[i] or (max_hold and age >= max_hold):
                held = 0
        if not held and e[i]:
            held = 1
            age = 0
        w[i] = held
    return pd.Series(w, index=enter.index)


def rsi(close, period):
    """RSI with Wilder's smoothing, from past bars only."""
    d = close.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    au = up.ewm(alpha=1 / period, adjust=False).mean()
    ad = dn.ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + au / ad.replace(0, np.nan))


def _run(rows, df, w, exchange, market, interval, strategy, params, note="", maker=False):
    """Score one variant and append the row."""
    global _tried
    _tried += 1
    r = bt.score(df, w, exchange=exchange, market=market, interval=interval, maker=maker)
    r.update(strategy=strategy, params=params, note=note)
    rows.append(r)
    return r


def pick_best(rows, min_trades=20):
    """The in-sample best of a batch: best in-sample Sharpe among rules that actually traded.

    Only in-sample numbers are read here. The out-of-sample columns exist
    on the same row but are not used to choose anything.
    """
    ok = [r for r in rows if r["trades"] >= min_trades and r["share_in_market"] > 0.01]
    if not ok:
        ok = [r for r in rows if r["trades"] >= 1] or rows
    return max(ok, key=lambda r: (r["is_sharpe"], r["is_ann_pct"]))


# ---------------------------------------------------------------- the ideas

def idea_zscore(rows):
    """1. Buy when the close is far below its recent average, sell when it comes back."""
    out = []
    for market in COINS:
        for interval in ("5m", "15m", "1h"):
            df = bars("binanceus", market, interval)
            c = df.close
            for n in (20, 60, 240):
                mean = c.rolling(n).mean()
                sd = c.rolling(n).std()
                z = (c - mean) / sd
                for k in (1.5, 2.0, 3.0):
                    for exit_name, exit_z in (("back to average", 0.0), ("halfway back", -k / 2)):
                        w = hold_between(z < -k, z > exit_z)
                        out.append(_run(
                            rows, df, w, "binanceus", market, interval,
                            "buy when far below the recent average, sell when it comes back",
                            f"n={n} k={k} sell_at_z>{exit_z:g} ({exit_name})"))
    return out


def idea_bands(rows):
    """2. The same shape as bands: buy at the lower band, sell at the middle or upper band."""
    out = []
    for market in COINS:
        for interval in ("5m", "15m", "1h"):
            df = bars("binanceus", market, interval)
            c = df.close
            for n in (20, 60, 240):
                mid = c.rolling(n).mean()
                sd = c.rolling(n).std()
                for k in (1.5, 2.0, 3.0):
                    lower = mid - k * sd
                    upper = mid + k * sd
                    for target, level in (("middle band", mid), ("upper band", upper)):
                        w = hold_between(c <= lower, c >= level)
                        out.append(_run(
                            rows, df, w, "binanceus", market, interval,
                            "buy at the lower band, sell at the " + target,
                            f"n={n} k={k} sell_at={target}"))
    return out


def idea_rsi(rows):
    """3. A very short RSI: buy when it is very low, sell on recovery or after H bars."""
    out = []
    for market in COINS:
        for interval in ("1h", "1d"):
            df = bars("binanceus", market, interval)
            for period in (2, 3):
                r = rsi(df.close, period)
                for entry in (5, 10):
                    for exit_lvl in (60, 70):
                        for max_hold in (6, 24):
                            w = hold_with_timeout(r < entry, r > exit_lvl, max_hold)
                            out.append(_run(
                                rows, df, w, "binanceus", market, interval,
                                f"buy when the {period}-bar RSI is very low, sell when it recovers or after H bars",
                                f"period={period} buy_below={entry} sell_above={exit_lvl} max_hold={max_hold}"))
    return out


def idea_big_down_bar(rows):
    """4. After one big down bar, hold for H bars. The big-up-bar mirror is the control."""
    out = []
    for market in COINS:
        for interval, drops in (("1h", (1.0, 2.0, 3.0)), ("1d", (3.0, 5.0, 8.0))):
            df = bars("binanceus", market, interval)
            ret = df.close.pct_change() * 100
            for x in drops:
                for direction, sig in (("down", ret < -x), ("up (control)", ret > x)):
                    for h in (1, 3, 6, 24):
                        w = sig.rolling(h).max().fillna(0.0).astype(float)
                        out.append(_run(
                            rows, df, w, "binanceus", market, interval,
                            f"after a bar that moved {direction} more than X%, hold the coin for H bars",
                            f"x={x}% h={h} bar={direction}",
                            note="control: buying after a big UP bar" if direction.startswith("up") else ""))
    return out


def idea_peg(rows):
    """5. Buy a stablecoin a few basis points under a dollar, sell when it is back."""
    out = []
    for market in ("USDTUSD", "USDCUSD"):
        for interval in ("1m", "5m"):
            df = bars("binanceus", market, interval)
            c = df.close
            ins, oos = bt.split(df.index)
            for d_bp in (2, 5, 10, 20):
                below = c < 1 - d_bp / 1e4
                n_in, n_out = int(below[ins].sum()), int(below[oos].sum())
                share = round(100 * float(below.mean()), 2)
                for exit_name, lvl in (("back to $1", 1.0), ("halfway back", 1 - d_bp / 2e4)):
                    w = hold_between(below, c >= lvl)
                    for maker in (False, True):
                        note = (f"{n_in + n_out} of {len(c)} bars ({share}%) closed below $1 - {d_bp}bp: "
                                f"{n_in} in the first 70% of the year, only {n_out} in the held-back last 30%. ")
                        note += ("resting fill: the price does trade through this level, so an order left "
                                 "sitting there would often be hit — but only when someone chooses to sell "
                                 "into it, and a queue of other buyers sits at the same round level, so read "
                                 "this as a best case. Note the fill price here is still the next bar's close; "
                                 "only the cost changed."
                                 if maker else
                                 "immediate fill: 2 bp fee plus half the spread, taken at the next bar's close.")
                        out.append(_run(
                            rows, df, w, "binanceus", market, interval,
                            f"buy the stablecoin below $1 - {d_bp}bp, sell {exit_name}",
                            f"d={d_bp}bp sell_at={exit_name} maker={maker}",
                            note=note, maker=maker))
    return out


def coinbase_check(rows, winners):
    """Re-run the in-sample winners where a fill costs 60 bp instead of 2."""
    out = []
    for r in winners:
        market = CB_OF.get(r["market"])
        if market is None:
            continue
        interval = r["interval"]
        try:
            df = bars("coinbase", market, interval)
        except FileNotFoundError:
            continue
        w = _rebuild(df, r)
        if w is None:
            continue
        out.append(_run(
            rows, df, w, "coinbase", market, interval, r["strategy"], r["params"],
            note="same rule on Coinbase, where one fill costs 60 bp"))
    return out


def _rebuild(df, r):
    """Rebuild a winner's weight series on another exchange's candles."""
    p = dict(kv.split("=", 1) for kv in r["params"].split(" ") if "=" in kv)
    s = r["strategy"]
    c = df.close
    if s.startswith("buy when far below"):
        n, k = int(p["n"]), float(p["k"])
        z = (c - c.rolling(n).mean()) / c.rolling(n).std()
        exit_z = float(r["params"].split("sell_at_z>")[1].split(" ")[0])
        return hold_between(z < -k, z > exit_z)
    if s.startswith("buy at the lower band"):
        n, k = int(p["n"]), float(p["k"])
        mid, sd = c.rolling(n).mean(), c.rolling(n).std()
        level = mid if "middle" in r["params"] else mid + k * sd
        return hold_between(c <= mid - k * sd, c >= level)
    if s.startswith("buy when the"):
        rr = rsi(c, int(p["period"]))
        return hold_with_timeout(rr < float(p["buy_below"]), rr > float(p["sell_above"]), int(p["max_hold"]))
    if s.startswith("after a bar that moved"):
        x = float(p["x"].rstrip("%"))
        ret = c.pct_change() * 100
        sig = ret > x if "up" in r["params"] else ret < -x
        return sig.rolling(int(p["h"])).max().fillna(0.0).astype(float)
    return None


# ------------------------------------------------------------------- runner

def run():
    rows = []
    best = {}
    z = idea_zscore(rows)
    best["far below the average"] = pick_best(z)
    b = idea_bands(rows)
    best["bands"] = pick_best(b)
    r = idea_rsi(rows)
    best["short RSI"] = pick_best(r)
    d = idea_big_down_bar(rows)
    best["one big down bar"] = pick_best([x for x in d if "control" not in x["params"]])
    best["one big up bar (control)"] = pick_best([x for x in d if "control" in x["params"]])
    p = idea_peg(rows)
    best["stablecoin off peg (immediate fill)"] = pick_best([x for x in p if not x["maker"]])
    best["stablecoin off peg (resting fill)"] = pick_best([x for x in p if x["maker"]])

    for name, row in best.items():
        row["note"] = (f"in-sample best of: {name}. " + (row.get("note") or "")).strip()

    # the two ideas with the best in-sample Sharpe on a real coin, priced at Coinbase fees
    coin_best = [best["far below the average"], best["bands"], best["short RSI"], best["one big down bar"]]
    coin_best.sort(key=lambda x: x["is_sharpe"], reverse=True)
    coinbase_check(rows, coin_best[:2])

    path = bt.record(FAMILY, rows, variants_tried=_tried)
    return path, rows, best


def main():
    path, rows, best = run()
    cols = ["strategy", "market", "interval", "trades", "cost_bp", "is_ann_pct", "is_sharpe",
            "oos_ann_pct", "oos_sharpe", "oos_maxdd_pct", "bh_oos_ann_pct"]
    print(f"{len(rows)} rows, {_tried} variants -> {path}\n")
    for name, r in best.items():
        print(f"[{name}] {r['params']}")
        print("   " + "  ".join(f"{c}={r[c]}" for c in cols))
    cb = [r for r in rows if r["exchange"] == "coinbase"]
    if cb:
        print("\nSame rules at Coinbase fees (60 bp a fill):")
        for r in cb:
            print(f"   {r['market']} {r['interval']} {r['params']}")
            print("   " + "  ".join(f"{c}={r[c]}" for c in cols))


if __name__ == "__main__":
    main()
