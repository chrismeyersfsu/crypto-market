"""The spot version of the approach in Pavel Kycek's Algorithmic Crypto Playbook (2025).

The book itself isn't on hand; this follows what its author describes in
public (his Better System Trader interview and the Robuxio course outline):
simple rules on slow bars (4-hour, 12-hour, daily), each run over a basket
of coins with a cap per coin rather than on one coin; long-term momentum
("price above its long average") and moving-average crossovers as the trend
rules, RSI dips as the reversion rule, and breakouts; then the rule types
combined so their bad stretches fall at different times; and a lot of
weight on survivorship bias and parameter stability. He also trades
futures and bets on falls, which Binance.US spot cannot do, so this is the
"hold or hold cash" half of it.

Universe: Binance.US USD coins with a median day of $100k or more; at 4h and
12h only the 8 that trade in most hours, since a bar without a trade carries
a stale price. Binance.US's own daily history has a hole from 2023-07-14 to
2025-02-19 (the API has nothing there), so the daily basket takes its prices
from Coinbase, which is complete since 2021, for the 15 of those coins
Coinbase lists, and charges Binance.US's costs. Coins that were delisted are
not on disk, so the universe is survivors only -- the author says that
flatters results 5-10x, and nothing here corrects it.

    uv run python -m crypto_market.strategies.playbook
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .. import backtest as bt
from .trend import _run_account

PEGGED = {"USDTUSD", "USDCUSD", "USDCUSDT"}
MIN_USD_DAY = 100_000  # median dollars traded a day to be in the daily basket
MIN_HOURS_TRADED = 0.75  # share of hours with a trade to be in the 4h/12h baskets
CAPS = (0.1, 0.2)  # most of the account one coin may hold


def universe(interval):
    """(closes, one-way cost per coin, note) for the coins liquid enough for this bar size."""
    syms = [s for _, s, _ in bt.available("binanceus", "1d") if s.endswith("USD") and s not in PEGGED]
    on_coinbase = {s for _, s, _ in bt.available("coinbase", "1d")}
    keep, closes = [], {}
    for s in syms:
        d = bt.candles("binanceus", s, "1d")
        if (d.volume * d.close).median() < MIN_USD_DAY:
            continue
        if interval == "1d":
            cb = s[:-3] + "-USD"
            if cb not in on_coinbase:
                continue
            keep.append(s)
            closes[s] = bt.candles("coinbase", cb, "1d").close
        else:
            h = bt.candles("binanceus", s, "1h")
            if (h.trades > 0).mean() < MIN_HOURS_TRADED:
                continue
            keep.append(s)
            closes[s] = h.close.resample(interval).last()
    close = pd.DataFrame(closes).sort_index()
    cost = pd.Series({s: bt.cost_bp("binanceus", s) / 1e4 for s in keep})
    src = "Coinbase prices, Binance.US costs" if interval == "1d" else "Binance.US"
    return close, cost, f"{len(keep)} coins ({src}): {' '.join(keep)}"


def _bars_per_day(interval):
    return {"1d": 1, "12h": 2, "4h": 6}[interval]


def targets_from_signal(sig, close, cap):
    """Equal shares of the coins whose signal is on, each at most `cap`; cash for the rest.

    Only a coin whose signal just changed gets a target (bought at the share
    the count of held coins implies, or sold); the others are left to drift,
    so the account walk pays cost on real changes only.
    """
    sig = sig & close.notna()
    n = sig.sum(axis=1)
    share = (1.0 / n.clip(lower=1.0 / cap)).where(n > 0, 0.0)
    want = sig.astype(float).mul(share, axis=0)
    changed = sig != sig.shift(1).fillna(False)
    return want.where(changed, np.nan)


def rsi(close, n):
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


# ---- the rules, each returning a boolean "hold this coin" frame ----

def sig_long_momentum(close, days, interval):
    n = days * _bars_per_day(interval)
    return close > close.rolling(n).mean()


def sig_crossover(close, fast_days, slow_days, interval):
    k = _bars_per_day(interval)
    return close.rolling(fast_days * k).mean() > close.rolling(slow_days * k).mean()


def sig_rsi_dip(close, n, low, high, filter_days, interval):
    """Buy a coin when its RSI drops below `low`, sell when it climbs back above `high`.

    With `filter_days` set, only coins above their long average qualify --
    buying the dip in something that is trending, the usual pairing.
    """
    r = rsi(close, n)
    enter = r < low
    exit_ = r > high
    if filter_days:
        enter &= close > close.rolling(filter_days * _bars_per_day(interval)).mean()
    state = pd.DataFrame(False, index=close.index, columns=close.columns)
    held = np.zeros(close.shape[1], dtype=bool)
    en, ex = enter.to_numpy(), exit_.to_numpy()
    out = np.zeros(close.shape, dtype=bool)
    for t in range(len(close)):
        held = (held & ~ex[t]) | en[t]
        out[t] = held
    state[:] = out
    return state


def sig_breakout(close, n_days, interval):
    """Hold after a new N-day high, until a new N/2-day low."""
    k = _bars_per_day(interval)
    n, m = n_days * k, max(n_days * k // 2, 2)
    hi = close.rolling(n).max().shift(1)
    lo = close.rolling(m).min().shift(1)
    enter, exit_ = (close > hi).to_numpy(), (close < lo).to_numpy()
    out = np.zeros(close.shape, dtype=bool)
    held = np.zeros(close.shape[1], dtype=bool)
    for t in range(len(close)):
        held = (held & ~exit_[t]) | enter[t]
        out[t] = held
    return pd.DataFrame(out, index=close.index, columns=close.columns)


# ---- scoring ----

def score_basket(sig, close, cost, interval, cap, label, params, bh, note):
    tgt = targets_from_signal(sig, close, cap)
    net, turnover, trades = _run_account(tgt, close, cost)
    r = bt.score_returns(net, close.index[:-1], interval, "binanceus", f"basket of {close.shape[1]}",
                         trades, bh=bh, note=note)
    r.update(strategy=label, params=params, turnover=round(turnover, 1),
             cost_bp=round(float(cost.mean() * 1e4), 2), fee_bp=bt.FEE_BP["binanceus"], maker=False,
             share_in_market=round(float(tgt.ffill().sum(axis=1).mean()), 3))
    return r, net


def _by_year(net, index):
    """'by year: 2021 +37% 2022 -17% ...' for a note."""
    y = pd.Series(np.nan_to_num(net), index=index).groupby(index.year).apply(lambda v: (np.prod(1 + v) - 1) * 100)
    return "by year: " + " ".join(f"{yr} {pct:+.0f}%" for yr, pct in y.items())


def _in_sample_sharpe(net, index, interval):
    ins, _ = bt.split(index)
    r = net[ins]
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(bt.BARS_PER_YEAR[interval])) if sd > 0 else 0.0


def run():
    rows = []
    variants = 0
    for interval in ("1d", "12h", "4h"):
        close, cost, uni = universe(interval)
        btc = close["BTCUSD"].pct_change().shift(-1).to_numpy()[:-1]
        idx = close.index[:-1]
        note_uni = f"{uni}; survivors only"

        # yardsticks
        ew = targets_from_signal(close.notna(), close, 1.0)
        net, turnover, trades = _run_account(ew, close, cost)
        r = bt.score_returns(net, idx, interval, "binanceus", f"basket of {close.shape[1]}", trades, bh=btc,
                             note="yardstick: an equal share of every coin, reset when one appears or disappears; " + note_uni)
        r.update(strategy="hold every coin equally", params="", turnover=round(turnover, 1),
                 cost_bp=round(float(cost.mean() * 1e4), 2), fee_bp=bt.FEE_BP["binanceus"], maker=False)
        rows.append(r)

        # each rule type: a grid, in-sample best kept for the combination
        families = {}  # name -> list of (in-sample sharpe, net, label, params)
        for cap in CAPS:
            grid = []
            for days in (50, 100, 200):
                grid.append(("long-term momentum: hold coins above their N-day average", f"n={days}d cap={cap}",
                             sig_long_momentum(close, days, interval)))
            for fast, slow in ((10, 50), (20, 100), (20, 200), (50, 200)):
                grid.append(("trend: short average above long average", f"fast={fast}d slow={slow}d cap={cap}",
                             sig_crossover(close, fast, slow, interval)))
            for n, (low, high), filt in itertools.product((2, 4, 14), ((20, 60), (30, 50)), (0, 200)):
                grid.append(("reversion: buy an RSI dip, sell when it recovers" + (" (only coins above their 200-day average)" if filt else ""),
                             f"rsi={n} buy<{low} sell>{high} filter={filt}d cap={cap}",
                             sig_rsi_dip(close, n, low, high, filt, interval)))
            for days in (20, 50, 100):
                grid.append(("breakout: buy a new N-day high, sell a new N/2-day low", f"n={days}d cap={cap}",
                             sig_breakout(close, days, interval)))
            for label, params, sig in grid:
                variants += 1
                r, net = score_basket(sig, close, cost, interval, cap, label, params, btc, note_uni)
                rows.append(r)
                families.setdefault(label.split(":")[0], []).append(
                    (_in_sample_sharpe(net, idx, interval), net, label, params, sig, cap))

        # combinations: equal capital to each rule type
        picks = {k: max(v, key=lambda x: x[0]) for k, v in families.items()}
        nets = [p[1] for p in picks.values()]
        combo = np.mean(nets, axis=0)
        variants += 1
        r = bt.score_returns(combo, idx, interval, "binanceus", f"basket of {close.shape[1]}",
                             sum(1 for _ in nets), bh=btc,
                             note="equal capital to the in-sample-best of each rule type: " +
                                  "; ".join(f"{p[2]} [{p[3]}]" for p in picks.values()) + "; " + note_uni)
        r.update(strategy="combined: best of each rule type", params=f"{len(nets)} rule types", maker=False,
                 cost_bp=round(float(cost.mean() * 1e4), 2), fee_bp=bt.FEE_BP["binanceus"],
                 note=r["note"] + "; " + _by_year(combo, idx))
        rows.append(r)

        # the same combination acted on one bar late: a rule that only works on the closing print is a data artifact
        late_nets = []
        for _, _, label, params, sig, cap in picks.values():
            _, net = score_basket(sig.shift(1).fillna(False), close, cost, interval, cap, label, params, btc, note_uni)
            late_nets.append(net)
        late_combo = np.mean(late_nets, axis=0)
        variants += 1
        r = bt.score_returns(late_combo, idx, interval, "binanceus", f"basket of {close.shape[1]}", len(late_nets), bh=btc,
                             note="the combination above, every change made one bar after the signal; " + _by_year(late_combo, idx))
        r.update(strategy="combined: best of each rule type", params=f"{len(nets)} rule types", maker=False, late=1,
                 cost_bp=round(float(cost.mean() * 1e4), 2), fee_bp=bt.FEE_BP["binanceus"])
        rows.append(r)

        # the robust version: every variant of every rule type, averaged, nothing picked
        allnets = [x[1] for v in families.values() for x in v]
        variants += 1
        r = bt.score_returns(np.mean(allnets, axis=0), idx, interval, "binanceus", f"basket of {close.shape[1]}",
                             len(allnets), bh=btc,
                             note=f"equal capital to all {len(allnets)} variants of all rule types, nothing chosen in-sample; " + note_uni)
        r.update(strategy="combined: every variant equally, nothing picked", params=f"{len(allnets)} variants", maker=False,
                 cost_bp=round(float(cost.mean() * 1e4), 2), fee_bp=bt.FEE_BP["binanceus"])
        rows.append(r)

        # parameter stability: how the in-sample sharpe spreads across each rule type's grid
        for k, v in families.items():
            s = np.array([x[0] for x in v])
            print(f"{interval} {k:22s} in-sample sharpe over {len(v)} variants: median {np.median(s):.2f} "
                  f"min {s.min():.2f} max {s.max():.2f}", flush=True)
    return bt.record("playbook", rows, variants_tried=variants)


def main():
    p = run()
    df = pd.read_csv(p)
    cols = ["strategy", "params", "interval", "trades", "turnover", "is_ann_pct", "is_sharpe", "oos_ann_pct", "oos_sharpe",
            "oos_maxdd_pct", "bh_oos_ann_pct"]
    pd.set_option("display.width", 250)
    for iv in ("1d", "12h", "4h"):
        d = df[df.interval == iv]
        print(f"\n== {iv}: yardsticks and combinations")
        print(d[d.strategy.str.startswith(("hold", "combined"))][cols].to_string(index=False))
        print(f"== {iv}: best in-sample of each rule type")
        d = d[~d.strategy.str.startswith(("hold", "combined"))]
        print(d.loc[d.groupby(d.strategy.str.split(":").str[0]).is_sharpe.idxmax()][cols].to_string(index=False))


if __name__ == "__main__":
    main()
