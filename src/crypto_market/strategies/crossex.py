"""Does one exchange's price tell you where the other one is about to go?

Coinbase is much the bigger USD market; Binance.US trades the same coins
thinly. Three things are tried here:

1. Lead-lag. If Coinbase just moved up, does Binance.US move up in the
   next minute? Hold the coin on Binance.US when it does.
2. Gap reversion. When Binance.US is cheaper than Coinbase by more than a
   few hundredths of a percent, buy on Binance.US and sell when the two
   prices come back together.
3. The same lead-lag rule priced off the real Binance.US trade tape
   instead of minute closes, to find out whether the price the backtest
   assumes was actually available.

The headline warning, which the numbers below make very hard to miss:
Binance.US BTCUSD does not trade in about 56% of minutes. A minute with
no trades has no new close, so its "close" is just the last one carried
forward. A stale close mechanically lags Coinbase and then "catches up"
the moment somebody trades, which looks exactly like a prediction. The
minute-close backtests in ideas 1 and 2 therefore print a huge, fake edge;
the liquidity filter and the trade-tape run in idea 3 are the honest
tests, and they say there is nothing here. At prices that were really
printed, following a Coinbase move captures about 0.5 to 1.7 bp per round
trip, against 4.6 bp of cost to do it. Every honest variant loses money.

One trap worth writing down: a candle's timestamp is the minute it opens,
so the close of the bar labelled T is only known at T+60s. Pricing a
trade off the tape at T instead of T+60s buys a minute before the signal
exists, and that alone turned this from a loser into a fake 60% a year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import backtest as bt
from ..butrades import load as load_tape

PAIRS = [("BTC-USD", "BTCUSD"), ("ETH-USD", "ETHUSD"), ("SOL-USD", "SOLUSD"), ("USDT-USD", "USDTUSD")]


# ----------------------------------------------------------------- data


def joined(cb_symbol, bu_symbol, interval="1m"):
    """Binance.US candles plus the Coinbase close, on the minutes both have."""
    cb = bt.candles("coinbase", cb_symbol, interval)
    bu = bt.candles("binanceus", bu_symbol, interval)
    idx = cb.index.intersection(bu.index)
    out = bu.loc[idx].copy()
    out["cb_close"] = cb.loc[idx, "close"]
    return out


def resample(df, rule):
    """Roll a minute frame up to a slower bar (fewer stale closes)."""
    how = {"open": "first", "high": "max", "low": "min", "close": "last",
           "volume": "sum", "trades": "sum", "cb_close": "last"}
    how = {k: v for k, v in how.items() if k in df.columns}
    return df.resample(rule).agg(how).dropna(subset=["close", "cb_close"])


def alignment_report():
    """How well the two exchanges line up, and how much one leads the other."""
    rows = []
    for cb_symbol, bu_symbol in PAIRS:
        cb = bt.candles("coinbase", cb_symbol, "1m")
        bu = bt.candles("binanceus", bu_symbol, "1m")
        j = joined(cb_symbol, bu_symbol)
        diff_bp = (j.close / j.cb_close - 1) * 1e4
        cb_ret, bu_ret = j.cb_close.pct_change(), j.close.pct_change()
        ins, _ = bt.split(j.index)
        rows.append({
            "market": f"{cb_symbol} vs {bu_symbol}",
            "coinbase_minutes": len(cb), "binanceus_minutes": len(bu), "minutes_on_both": len(j),
            "median_diff_bp": round(float(diff_bp.median()), 2),
            "iqr_diff_bp": f"{diff_bp.quantile(.25):+.2f}..{diff_bp.quantile(.75):+.2f}",
            "binanceus_minutes_with_no_trade_pct": round(float((j.volume == 0).mean()) * 100, 1),
            "corr_coinbase_now_binanceus_next": round(float(cb_ret[ins].corr(bu_ret.shift(-1)[ins])), 4),
            "corr_binanceus_now_coinbase_next": round(float(bu_ret[ins].corr(cb_ret.shift(-1)[ins])), 4),
            "corr_same_minute": round(float(cb_ret[ins].corr(bu_ret[ins])), 4),
        })
    return pd.DataFrame(rows)


def staleness_report(cb_symbol="BTC-USD", bu_symbol="BTCUSD"):
    """The lead-lag correlation split by how busy the Binance.US minute was."""
    j = joined(cb_symbol, bu_symbol)
    cb_ret, bu_ret = j.cb_close.pct_change(), j.close.pct_change()
    ins, _ = bt.split(j.index)
    rows = []
    for lo, hi, label in [(0, 0, "no trades"), (1, 2, "1-2"), (3, 10, "3-10"),
                          (11, 50, "11-50"), (51, 10**9, "51+")]:
        m = ins & (j.trades >= lo).to_numpy() & (j.trades <= hi).to_numpy()
        rows.append({"trades_in_the_minute": label, "minutes": int(m.sum()),
                     "corr_coinbase_now_binanceus_next": round(float(cb_ret[m].corr(bu_ret.shift(-1)[m])), 4)})
    return pd.DataFrame(rows)


# ------------------------------------------------------- idea 1: lead-lag


def follow_weight(df, k, thresh_bp, hold, min_trades=0):
    """Hold the coin for `hold` bars after Coinbase rose more than `thresh_bp` over `k` bars."""
    move_bp = (df.cb_close / df.cb_close.shift(k) - 1) * 1e4
    sig = move_bp > thresh_bp
    if min_trades:  # only act when the Binance.US bar really traded
        sig &= df.trades.fillna(0) >= min_trades
    return sig.astype(float).rolling(hold, min_periods=1).max().fillna(0.0)


def lead_lag(rows, variants):
    """Coinbase moves first; follow it on Binance.US."""
    grid = [(k, t, h) for k in (1, 3, 5) for t in (5, 10, 20) for h in (1, 3, 5)]

    btc = joined("BTC-USD", "BTCUSD")
    ins, _ = bt.split(btc.index[:-1])
    best, best_is = None, -1e9
    for k, t, h in grid:
        w = follow_weight(btc, k, t, h)
        r = bt.score(btc, w, exchange="binanceus", market="BTCUSD")
        variants.append(r)
        if r["is_sharpe"] > best_is:
            best_is, best = r["is_sharpe"], (k, t, h)
    k, t, h = best

    def tag(r, strategy, params, note):
        return {**r, "strategy": strategy, "params": params, "note": note}

    warn = ("PROBABLY NOT REAL: fills are assumed at the next minute's close, but 56% of "
            "Binance.US BTCUSD minutes have no trade at all, so that close is a carried-forward "
            "stale price nobody could have traded at. See the liquidity-filter and trade-tape rows.")

    # the in-sample best, kept as the headline of the sub-family
    w = follow_weight(btc, k, t, h)
    rows.append(tag(bt.score(btc, w, exchange="binanceus", market="BTCUSD"),
                    "follow Coinbase up, hold the coin on Binance.US",
                    f"k={k} thresh={t}bp hold={h}", "best in-sample of 27 minute-close variants. " + warn))

    # the same parameters on the other two coins
    for cb_symbol, bu_symbol in [("ETH-USD", "ETHUSD"), ("SOL-USD", "SOLUSD")]:
        d = joined(cb_symbol, bu_symbol)
        r = bt.score(d, follow_weight(d, k, t, h), exchange="binanceus", market=bu_symbol)
        variants.append(r)
        rows.append(tag(r, "follow Coinbase up, hold the coin on Binance.US",
                        f"k={k} thresh={t}bp hold={h}", "BTC's best parameters, untouched. " + warn))

    # control: the other direction. Binance.US moves first, follow it on Coinbase.
    rev = btc.copy()
    rev["bu_close"], rev["close"], rev["cb_close"] = rev.close, rev.cb_close, rev.close
    for fee, why in [(None, "real Coinbase taker fee, 60bp"), (2.0, "Binance.US fee forced on Coinbase, to isolate the signal from the fee")]:
        r = bt.score(rev, follow_weight(rev, k, t, h), exchange="coinbase", market="BTC-USD", fee_bp=fee)
        variants.append(r)
        rows.append(tag(r, "CONTROL: follow Binance.US up, hold the coin on Coinbase",
                        f"k={k} thresh={t}bp hold={h} fee={why}",
                        "The reverse direction. Coinbase's minute return barely responds to "
                        "Binance.US's last minute (correlation about -0.01), which is what you "
                        "expect if the whole effect is Binance.US's stale close catching up."))

    # honesty check 1: only act on minutes that actually traded on Binance.US
    for m in (1, 3, 10):
        r = bt.score(btc, follow_weight(btc, k, t, h, min_trades=m), exchange="binanceus", market="BTCUSD")
        variants.append(r)
        rows.append(tag(r, "follow Coinbase up, only on minutes Binance.US really traded",
                        f"k={k} thresh={t}bp hold={h} min_trades={m}",
                        "Same rule, but it only acts when the Binance.US bar had at least "
                        f"{m} trade(s), so the close is a price somebody paid."))

    # honesty check 2: 15-minute bars, where only 5% of bars are empty
    b15 = resample(btc, "15min")
    best15, best15_is = None, -1e9
    for k2 in (1, 2):
        for t2 in (5, 10, 20):
            for h2 in (1, 2):
                r = bt.score(b15, follow_weight(b15, k2, t2, h2), exchange="binanceus", market="BTCUSD", interval="15m")
                variants.append(r)
                if r["is_sharpe"] > best15_is:
                    best15_is, best15 = r["is_sharpe"], (k2, t2, h2, r)
    k2, t2, h2, r = best15
    rows.append(tag(r, "follow Coinbase up, on 15-minute bars", f"k={k2} thresh={t2}bp hold={h2}",
                    "best in-sample of 12. Only 5% of 15-minute bars are empty, so the closes "
                    "are mostly real prices."))
    return best


# --------------------------------------------------- idea 2: gap reversion


def _gap_weight(gap_bp, enter, hold, tradeable=None):
    """Buy when Binance.US is `enter` bp below Coinbase; sell at half that gap or after `hold` bars."""
    g = np.asarray(gap_bp, dtype=float)
    ok = np.ones(len(g), dtype=bool) if tradeable is None else np.asarray(tradeable, dtype=bool)
    w = np.zeros(len(g))
    holding, bars = False, 0
    for i in range(len(g)):
        if holding:
            bars += 1
            if not (g[i] > enter / 2) or bars >= hold:
                holding = False
        if not holding and g[i] > enter and ok[i]:
            holding, bars = True, 0
        w[i] = 1.0 if holding else 0.0
    return w


def gap_reversion(rows, variants):
    """Buy the cheaper exchange and wait for the two prices to meet."""
    jobs = [("BTC-USD", "BTCUSD", (3, 5, 10, 20)), ("USDT-USD", "USDTUSD", (1, 2, 5, 10))]
    for cb_symbol, bu_symbol, enters in jobs:
        d = joined(cb_symbol, bu_symbol)
        gap_bp = (d.cb_close / d.close - 1) * 1e4  # positive: Binance.US is the cheaper one
        best, best_is = None, -1e9
        for enter in enters:
            for hold in (3, 5, 10, 30):
                w = pd.Series(_gap_weight(gap_bp, enter, hold), index=d.index)
                r = bt.score(d, w, exchange="binanceus", market=bu_symbol)
                variants.append(r)
                if r["is_sharpe"] > best_is:
                    best_is, best = r["is_sharpe"], (enter, hold, r)
        enter, hold, r = best
        artifact = ("NOT REAL. " if r["is_ann_pct"] > 1e3 else "")
        rows.append({**r, "strategy": f"buy Binance.US {bu_symbol} when it is below Coinbase, sell when they meet",
                     "params": f"enter={enter}bp exit_at={enter/2}bp hold<={hold} bars",
                     "note": (artifact + "best in-sample of 16. Taker costs. The gap is measured off "
                              "minute closes, so it inherits the stale-close problem: most of the "
                              "'gap' is just Binance.US not having traded yet, and the 'reversion' "
                              "is the next real trade printing. An impossible-looking return here is "
                              "the artifact, not an edge. See the next row for the same rule "
                              "restricted to minutes that actually traded.")})

        # the same rule, but it may only buy on a minute Binance.US really traded
        w = pd.Series(_gap_weight(gap_bp, enter, hold, tradeable=(d.trades.fillna(0) > 0).to_numpy()), index=d.index)
        r2 = bt.score(d, w, exchange="binanceus", market=bu_symbol)
        variants.append(r2)
        rows.append({**r2, "strategy": f"buy Binance.US {bu_symbol} below Coinbase, only on minutes it really traded",
                     "params": f"enter={enter}bp exit_at={enter/2}bp hold<={hold} bars min_trades=1",
                     "note": ("Same parameters as the row above, but a buy is only allowed on a bar "
                              "whose close is a price somebody actually paid. This is the honest "
                              "version of the gap trade.")})


# ------------------------------------------- idea 3: the real trade tape


def tape_prices(symbol="BTCUSD"):
    """Every Binance.US fill, as a price series indexed by time."""
    tp = load_tape(symbol)
    px = pd.Series(tp.price.to_numpy(dtype=float),
                   index=pd.to_datetime(tp.ts, unit="ms", utc=True)).sort_index()
    return px[~px.index.duplicated(keep="last")]


def _tape_window(px, cb_symbol="BTC-USD", bu_symbol="BTCUSD"):
    """The joined minutes the trade tape actually covers."""
    j = joined(cb_symbol, bu_symbol)
    return j.loc[(j.index >= px.index[0]) & (j.index <= px.index[-1] - pd.Timedelta(minutes=10))]


def _tape_trade(px, j, sig, hold_minutes, delay_s):
    """Hold the coin at prices that were really printed; proper per-minute returns.

    A bar labelled T closes at T+60s, so T+60s+delay is the first moment the
    signal could be acted on. `mark[i]` is the first Binance.US price printed
    at or after that moment: the price a market order would have got. Holding
    from one decision point to the next earns mark[i+1]/mark[i] - 1, so each
    element of the returned series really is one minute of return. A new buy
    is not started while one is still running, so nothing is double counted.
    """
    ticks, vals = px.index.to_numpy(), px.to_numpy()
    cost = bt.cost_bp("binanceus", "BTCUSD") / 1e4
    t = (j.index + pd.Timedelta(seconds=60 + delay_s)).to_numpy()
    i = np.searchsorted(ticks, t, side="left")
    ok = i < len(ticks)
    i = np.clip(i, 0, len(ticks) - 1)
    ok &= (ticks[i] - t) <= np.timedelta64(5, "m")  # nothing printed for 5 minutes: unusable
    mark = np.where(ok, vals[i], np.nan)

    sig = np.asarray(sig, dtype=bool) & ok
    n = len(j)
    w = np.zeros(n)
    until = -1
    for x in range(n):
        if sig[x] and x >= until:
            until = x + hold_minutes
        w[x] = 1.0 if x < until else 0.0

    step = np.full(n, np.nan)
    step[:-1] = mark[1:] / mark[:-1] - 1
    step = np.where(np.isfinite(step), step, 0.0)
    turn = np.abs(np.diff(np.concatenate([[0.0], w])))
    ret = w * step - turn * cost
    gross = float((w * step).sum() / max(turn.sum() / 2, 1e-9) * 1e4)
    return ret, w, int((turn > 1e-9).sum()), float(turn.sum()), gross


def gap_tape_check(rows, variants):
    """Idea 2 priced off the tape: can you actually buy the cheap print?"""
    px = tape_prices("BTCUSD")
    j = _tape_window(px)
    gap_bp = (j.cb_close / j.close - 1) * 1e4
    bh = j.close.pct_change().shift(-1).fillna(0.0).to_numpy()
    cost = bt.cost_bp("binanceus", "BTCUSD")
    for enter in (5, 10, 20):
        for hold in (1, 5, 30):
            sig = (gap_bp > enter).fillna(False).to_numpy()
            ret, w, trades, turn, gross = _tape_trade(px, j, sig, hold, delay_s=5)
            r = bt.score_returns(ret, j.index, "1m", "binanceus", "BTCUSD", trades=trades, bh=bh)
            variants.append(r)
            rows.append({**r, "cost_bp": cost, "maker": False, "turnover": round(turn, 1),
                         "bp_per_trade": round(gross, 2), "share_in_market": round(float(w.mean()), 4),
                         "strategy": "buy Binance.US below Coinbase, at real Binance.US trade prices",
                         "params": f"enter={enter}bp hold={hold}min delay=5s",
                         "note": ("The honest version of idea 2, on the 30 days the tape covers. "
                                  "The minute-close backtest gets to buy at the stale low print that "
                                  "raised the signal; here it has to buy at the next price actually "
                                  f"printed. bp_per_trade is the gross move captured against {cost:.1f}bp "
                                  "of cost each way.")})


def tape_check(rows, variants, k=1, holds_minutes=1):
    """Price the lead-lag rule at prices that were actually printed.

    A candle's timestamp is the minute it *opens*, so the close of the bar
    labelled T is only known at T+60s. That is the earliest an order could
    be sent. For each such minute where Coinbase rose more than the
    threshold, buy at the first Binance.US trade at or after T+60s (plus a
    delay for getting the order there) and sell at the first trade a minute
    after that. If nothing trades within five minutes, the trade is skipped.
    """
    px = tape_prices("BTCUSD")
    j = _tape_window(px)
    move_bp = (j.cb_close / j.cb_close.shift(k) - 1) * 1e4
    cost = bt.cost_bp("binanceus", "BTCUSD")
    bh = j.close.pct_change().shift(-1).fillna(0.0).to_numpy()

    for thresh in (5, 10, 20):
        for delay_s in (0, 5):
            sig = (move_bp > thresh).fillna(False).to_numpy()
            ret, w, trades, turn, gross = _tape_trade(px, j, sig, holds_minutes, delay_s)
            r = bt.score_returns(ret, j.index, "1m", "binanceus", "BTCUSD", trades=trades, bh=bh)
            variants.append(r)
            rows.append({**r, "cost_bp": cost, "maker": False,
                         "turnover": round(turn, 1), "bp_per_trade": round(gross, 2),
                         "share_in_market": round(float(w.mean()), 4),
                         "strategy": "follow Coinbase up, bought and sold at real Binance.US trade prices",
                         "params": f"k={k} thresh={thresh}bp hold=1min delay={delay_s}s",
                         "note": ("Only the last 30 days: that is all the trade tape covers, so this is "
                                  "a different and much shorter window than the other rows, and its "
                                  "in/out split is 30 days cut 70/30. Buys at the first fill printed "
                                  f"at or after the minute close plus {delay_s}s, sells at the first "
                                  "fill a minute later. bp_per_trade here is the gross move captured, "
                                  f"to compare against the {cost:.1f}bp each way it costs.")})


# ------------------------------------------------------------------ run


def run():
    variants, rows = [], []

    align = alignment_report()
    stale = staleness_report()
    print("How well the two exchanges line up (in-sample correlations):")
    print(align.to_string(index=False))
    print("\nBinance.US BTCUSD: the lead-lag correlation, split by how busy the minute was:")
    print(stale.to_string(index=False))
    print()

    lead_lag(rows, variants)
    gap_reversion(rows, variants)
    tape_check(rows, variants)
    gap_tape_check(rows, variants)

    path = bt.record("crossex", rows, variants_tried=len(variants))
    print(pd.DataFrame(rows)[["strategy", "params", "market", "interval", "trades", "cost_bp",
                              "bp_per_trade", "is_ann_pct", "is_sharpe", "oos_ann_pct",
                              "oos_sharpe", "oos_maxdd_pct", "bh_oos_ann_pct"]].to_string(index=False))
    print(f"\n{len(variants)} variants run; {len(rows)} kept -> {path}")
    return path


def main():
    run()


if __name__ == "__main__":
    main()
