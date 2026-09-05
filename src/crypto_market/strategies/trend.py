"""Trend rules: does "keep holding what has been going up" pay after costs?

Four ideas, each with a small grid, all decided at a bar's close and held
over the next bar:

1. Averages      hold the coin while its price is above an N-bar average,
                 or while a short average is above a long one.
2. Breakout      buy when the price makes a new N-bar high, sell when it
                 makes a new M-bar low.
3. Past return   hold while the return over the last N bars is positive,
                 re-deciding every bar or every K bars.
4. Best movers   across every Binance.US USD coin, hold equal shares of the
                 K coins with the best return over the last N days.

Parameters are chosen on the first 70% of the bars only. Every variant that
was run is written to data/strategies/trend.csv; the in-sample winner of each
idea is marked in its `note`.

    uv run python -m crypto_market.strategies.trend
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import backtest as bt

COINS = ["BTCUSD", "ETHUSD", "SOLUSD"]
# Coinbase names for the same three coins, used for the "what does a 60 bp fee do" run
CB_COINS = {"BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD", "SOLUSD": "SOL-USD"}
# dollar-pegged coins: they have no trend to ride, and they would win a
# "least volatile" contest for the wrong reason
PEGGED = {"USDTUSD", "USDCUSD", "USDCUSDT"}


# ---------------------------------------------------------------- candles

def bars(market, interval, exchange="binanceus"):
    """Candles for one coin at 1h, 4h (built from 1h) or 1d."""
    if interval == "4h":
        h = bt.candles(exchange, market, "1h")
        return h.resample("4h").agg({"open": "first", "high": "max", "low": "min",
                                     "close": "last", "volume": "sum"}).dropna()
    return bt.candles(exchange, market, interval)


def in_sample_only(row):
    """The score we are allowed to choose on: risk-adjusted return on the first 70%."""
    return row["is_sharpe"]


def pick_best(rows, label):
    """Mark and return the in-sample winner of one idea."""
    usable = [r for r in rows if r["trades"] > 0]
    if not usable:
        return None
    best = max(usable, key=in_sample_only)
    best["note"] = f"in-sample best of the '{label}' idea ({len(rows)} variants)"
    return best


# ------------------------------------------------------- 1. averages

def averages(out):
    """Price above an N-bar average, and a short average above a long one."""
    single, crossed = [], []
    for interval in ("1h", "4h", "1d"):
        ns = (24, 48, 96, 168, 336) if interval != "1d" else (10, 20, 50, 100, 200)
        fasts = (12, 24, 48, 96) if interval != "1d" else (5, 10, 20, 50)
        for market in COINS:
            df = bars(market, interval)
            close = df.close
            for n in ns:
                w = (close > close.rolling(n).mean()).astype(float)
                r = bt.score(df, w, exchange="binanceus", market=market)
                r.update(strategy="price above its own average", params=f"n={n}")
                single.append(r)
            for fast in fasts:
                for mult in (2, 4, 8):
                    slow = fast * mult
                    if slow >= len(close) // 4:
                        continue
                    w = (close.rolling(fast).mean() > close.rolling(slow).mean()).astype(float)
                    r = bt.score(df, w, exchange="binanceus", market=market)
                    r.update(strategy="short average above long average", params=f"fast={fast},slow={slow}")
                    crossed.append(r)
    out += single + crossed
    return pick_best(single, "price above its own average"), pick_best(crossed, "short average above long average")


# ------------------------------------------------------- 2. breakout

def _breakout_weight(close, n, m):
    """Buy on a new n-bar high, sell on a new m-bar low, hold in between."""
    high_n = close.rolling(n).max()
    low_m = close.rolling(m).min()
    buy = close >= high_n
    sell = close <= low_m
    state = pd.Series(np.nan, index=close.index)
    state[sell] = 0.0
    state[buy] = 1.0  # a bar that is both a high and a low counts as a buy
    return state.ffill().fillna(0.0)


def breakout(out):
    rows = []
    for interval in ("1h", "1d"):
        ns = (24, 48, 96, 168) if interval == "1h" else (10, 20, 50, 100)
        for market in COINS:
            df = bars(market, interval)
            for n in ns:
                for m in (max(2, n // 2), n):
                    w = _breakout_weight(df.close, n, m)
                    r = bt.score(df, w, exchange="binanceus", market=market)
                    r.update(strategy="buy new highs, sell new lows", params=f"n={n},m={m}")
                    rows.append(r)
    out += rows
    return pick_best(rows, "buy new highs, sell new lows")


# ------------------------------------------------------- 3. past return

def past_return(out):
    """Hold while the return over the last N bars is positive."""
    rows = []
    for interval in ("1h", "1d"):
        ns = (24, 72, 168, 336) if interval == "1h" else (7, 14, 30, 60, 90)
        ks = (1, 24) if interval == "1h" else (1, 7)
        for market in COINS:
            df = bars(market, interval)
            close = df.close
            for n in ns:
                up = (close / close.shift(n) - 1 > 0).astype(float)
                for k in ks:
                    if k == 1:
                        w = up
                    else:  # only look every k bars, hold the answer in between
                        keep = np.arange(len(up)) % k == 0
                        w = up.where(keep).ffill().fillna(0.0)
                    r = bt.score(df, w, exchange="binanceus", market=market)
                    r.update(strategy="hold while the last N bars were up", params=f"n={n},every={k}")
                    rows.append(r)
    out += rows
    return pick_best(rows, "hold while the last N bars were up")


# ------------------------------------------------------- 4. best movers

def _universe(interval):
    """Closes for every Binance.US USD coin, one column each, plus each coin's one-way cost."""
    syms = [s for _, s, _ in bt.available("binanceus", interval)
            if s.endswith("USD") and s not in PEGGED]
    close = pd.DataFrame({s: bt.candles("binanceus", s, interval).close for s in syms}).sort_index()
    cost = pd.Series({s: bt.cost_bp("binanceus", s) / 1e4 for s in syms})
    return close, cost


def _targets_top_k(close, lookback, top_k, every):
    """Equal shares of the top_k coins by past return, re-picked every `every` bars.

    Only the re-pick bars carry a target; in between, whatever is held is left
    alone (its share drifts with price, as it would in a real account).
    """
    past = close / close.shift(lookback) - 1
    ok = past.notna() & close.notna()
    tgt = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    for t in range(0, len(close), every):
        winners = past.iloc[t].where(ok.iloc[t]).dropna().nlargest(top_k)
        new = pd.Series(0.0, index=close.columns)
        if len(winners):
            new[winners.index] = 1.0 / len(winners)
        tgt.iloc[t] = new.to_numpy()
    return tgt


def _targets_equal_weight(close, every):
    """The yardstick: an equal share of every coin that trades, re-set every `every` bars."""
    alive = close.notna()
    tgt = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    ew = alive.div(alive.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    take = np.zeros(len(close), dtype=bool)
    take[::every] = True
    tgt[take] = ew[take]
    return tgt


def _run_account(tgt, close, cost):
    """Walk an account bar by bar: rebalance on the target bars, drift in between.

    Between rebalances nothing is bought or sold, so each holding's share of the
    account moves with its price; the cost is only paid on what actually changes
    hands. Returns (net return per bar, total turnover, number of buys and sells).
    """
    px = close.to_numpy(dtype=float)
    fwd = close.pct_change().shift(-1).to_numpy(dtype=float)
    tg = tgt.to_numpy(dtype=float)
    c = cost.reindex(close.columns).to_numpy(dtype=float)
    n, m = px.shape
    held = np.zeros(m)
    net = np.empty(n - 1)
    turnover = 0.0
    trades = 0
    listed = ~np.isnan(px)
    last_bar = np.where(listed.any(axis=0), n - 1 - np.argmax(listed[::-1], axis=0), -1)  # each coin's last priced bar
    for t in range(n - 1):
        charge = 0.0
        # a coin with no price for the rest of the series has stopped trading and has to go; a blank
        # bar with prices after it is a hole in the data, held through at no return
        gone = (t > last_bar) & (held > 0)
        if gone.any():
            charge += float((held[gone] * c[gone]).sum())
            turnover += float(held[gone].sum())
            trades += int(gone.sum())
            held = np.where(gone, 0.0, held)
        row = tg[t]
        if not np.isnan(row).all():
            want = np.where(np.isnan(row), held, row)  # a NaN coin on a rebalance bar is left as it is
            if want.sum() > 1.0 + 1e-9:  # what is bought has to come out of cash: scale the buys down
                touched = ~np.isnan(row)
                room = 1.0 - want[~touched].sum()
                want[touched] *= max(room, 0.0) / want[touched].sum()
            change = np.abs(want - held)
            charge += float((change * c).sum())
            turnover += float(change.sum())
            trades += int((change > 1e-9).sum())
            held = want.copy()
        step = np.nan_to_num(fwd[t])
        gain = float((held * step).sum())
        net[t] = gain - charge
        grown = held * (1 + step)
        total = 1.0 + gain
        held = grown / total if total > 1e-9 else np.zeros(m)
    return net, turnover, trades


def _score_account(tgt, close, cost, interval, label, params, bh=None, note=""):
    """Score one universe-wide account."""
    net, turnover, trades = _run_account(tgt, close, cost)
    r = bt.score_returns(net, close.index[:-1], interval, "binanceus", "USD universe",
                         trades, bh=bh, note=note)
    r.update(strategy=label, params=params, turnover=round(turnover, 1),
             cost_bp=round(float(cost.mean() * 1e4), 2),
             fee_bp=bt.FEE_BP["binanceus"], maker=False)
    return r


def best_movers(out):
    rows, marks = [], []
    for interval, lookbacks, rebalances in (("1d", (7, 14, 30), (1, 7)),
                                            ("1h", (168, 336, 720), (24, 168))):
        close, cost = _universe(interval)
        slow = max(rebalances)

        # yardsticks: an equal share of every coin that trades, and BTC alone
        ew_tgt = _targets_equal_weight(close, slow)
        bench = _score_account(ew_tgt, close, cost, interval, "hold every coin equally",
                               f"every={slow}b",
                               note="yardstick, not a trend rule: equal shares of the whole universe")
        btc_tgt = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
        first = int(close["BTCUSD"].notna().argmax())
        btc_tgt.iloc[first] = 0.0
        btc_tgt.iloc[first, close.columns.get_loc("BTCUSD")] = 1.0
        bench_btc = _score_account(btc_tgt, close, cost, interval, "hold BTC only", "buy once",
                                   note="yardstick, not a trend rule")
        rows += [bench, bench_btc]
        ew_ret, _, _ = _run_account(ew_tgt, close, cost)  # the yardstick returns, bar by bar

        group = []
        for lookback in lookbacks:
            for top_k in (3, 5, 10):
                for every in rebalances:
                    tgt = _targets_top_k(close, lookback, top_k, every)
                    days = lookback if interval == "1d" else lookback // 24
                    r = _score_account(tgt, close, cost, interval, "hold the best movers",
                                       f"top={top_k},lookback={days}d,every={every}b", bh=ew_ret)
                    group.append(r)
        rows += group
        best = pick_best(group, f"hold the best movers ({interval})")
        if best:
            marks.append(best)
    out += rows
    return marks


# ------------------------------------------------------- what a 60 bp fee does

def with_coinbase_fees(out, winners):
    """Re-run the best single-coin rules where a fill costs 60 bp instead of 2."""
    rows = []
    for w_row in winners:
        market, interval, params = w_row["market"], w_row["interval"], w_row["params"]
        if market not in CB_COINS:
            continue
        cb = CB_COINS[market]
        # same bars as the Binance.US run, so the only thing that changes is the fee
        window = bars(market, interval).index
        df = bars(cb, interval, exchange="coinbase")
        df = df[df.index.isin(window)]
        p = dict(kv.split("=") for kv in params.split(","))
        if w_row["strategy"] == "price above its own average":
            weight = (df.close > df.close.rolling(int(p["n"])).mean()).astype(float)
        elif w_row["strategy"] == "short average above long average":
            weight = (df.close.rolling(int(p["fast"])).mean() > df.close.rolling(int(p["slow"])).mean()).astype(float)
        elif w_row["strategy"] == "buy new highs, sell new lows":
            weight = _breakout_weight(df.close, int(p["n"]), int(p["m"]))
        else:
            up = (df.close / df.close.shift(int(p["n"])) - 1 > 0).astype(float)
            every = int(p["every"])
            weight = up if every == 1 else up.where(np.arange(len(up)) % every == 0).ffill().fillna(0.0)
        r = bt.score(df, weight, exchange="coinbase", market=cb)
        r.update(strategy=w_row["strategy"], params=params,
                 note="same rule on Coinbase, where a fill costs 60 bp instead of 2")
        rows.append(r)
    out += rows
    return rows


# ------------------------------------------------------------------ run

def run():
    rows = []
    best_single, best_cross = averages(rows)
    best_break = breakout(rows)
    best_past = past_return(rows)
    movers = best_movers(rows)
    winners = [r for r in (best_single, best_cross, best_break, best_past) if r] + movers
    # every single-coin idea's in-sample winner gets the expensive-fee run
    with_coinbase_fees(rows, [r for r in winners if r.get("market") in COINS])
    # honest count: every rule that was run, not counting the two yardsticks
    tried = sum(1 for r in rows if r["params"] not in ("benchmark", "buy once")
                and not str(r.get("note", "")).startswith("yardstick"))
    path = bt.record("trend", rows, variants_tried=tried)
    return path, winners


def main():
    path, winners = run()
    cols = ["strategy", "params", "market", "interval", "trades", "cost_bp",
            "is_ann_pct", "is_sharpe", "oos_ann_pct", "oos_sharpe", "oos_maxdd_pct", "bh_oos_ann_pct"]
    print(f"wrote {path}")
    for r in winners:
        print(" | ".join(f"{c}={r.get(c)}" for c in cols))


if __name__ == "__main__":
    main()
