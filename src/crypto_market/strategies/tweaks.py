"""Tweaks to the daily playbook basket, each tried on its own against the same baseline.

The baseline is the combination the paper run trades (strategies/playbook.py,
1d): equal money in "above the 50-day average", the 10/50-day crossover, the
20-day breakout and the RSI-14 dip, 15 Coinbase-priced coins, a tenth of the
account per coin at most. Every row here is that combination with one thing
changed, so the rows answer "does this change help?", not "what is best?"
-- picking the best row by its held-back number would be choosing on the
held-back data, which is the thing the split exists to prevent. The
changes fall in three groups: things a person would do for a reason other
than the backtest (drop the coins that are expensive to trade, charge
today's measured bid-ask gaps, act once a week), things that trade less
(wait for a signal to hold for two days, only act when a majority of the
rules agree), and things that manage risk (a trailing stop, sizing by
volatility, only holding when BTC itself is in an uptrend).

    uv run python -m crypto_market.strategies.tweaks
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import backtest as bt
from . import playbook as pb
from .trend import _run_account

INTERVAL = "1d"
CAP = 0.1
# the in-sample-best of each rule type; the same four the paper run trades
RULES = {
    "above 50-day average": lambda close: pb.sig_long_momentum(close, 50, INTERVAL),
    "10/50-day crossover": lambda close: pb.sig_crossover(close, 10, 50, INTERVAL),
    "RSI-14 dip, buy <20 sell >60": lambda close: pb.sig_rsi_dip(close, 14, 20, 60, 0, INTERVAL),
    "20-day breakout": lambda close: pb.sig_breakout(close, 20, INTERVAL),
}
# half the bid-ask gap on Binance.US, measured from the live book on 2026-09-06 (see paper.book); the
# backtest's standing assumption is 5 bp for every coin the tick recorder never watched
MEASURED_HALF_SPREAD_BP = {
    "BTCUSD": 0.3, "ETHUSD": 0.4, "SOLUSD": 0.5, "XRPUSD": 1.1, "DOGEUSD": 1.7, "BNBUSD": 2.0, "ADAUSD": 3.2,
    "LINKUSD": 3.3, "HBARUSD": 3.7, "AVAXUSD": 4.6, "XLMUSD": 5.4, "ZECUSD": 14.9, "LTCUSD": 16.6,
    "ALGOUSD": 46.2, "ATOMUSD": 204.3,
}
THIN = ("ATOMUSD", "ALGOUSD", "LTCUSD", "ZECUSD")  # a measured gap of 15 bp or more


# ---- signal transforms: each takes a boolean "hold" frame and returns another ----

def confirm(sig, bars):
    """Only start holding after the signal has been on `bars` bars in a row; stop as soon as it is off."""
    on = sig.rolling(bars).min().fillna(0).astype(bool)
    s, o = sig.to_numpy(), on.to_numpy()
    out = np.zeros(sig.shape, dtype=bool)
    held = np.zeros(sig.shape[1], dtype=bool)
    for t in range(len(sig)):
        held = o[t] | (held & s[t])
        out[t] = held
    return pd.DataFrame(out, index=sig.index, columns=sig.columns)


def btc_filter(sig, close, days):
    """Hold nothing while BTC is below its N-day average: the whole market's weather."""
    btc = close["BTCUSD"]
    up = (btc > btc.rolling(days).mean()).to_numpy()
    return sig & up[:, None]


def trailing_stop(sig, close, drop):
    """Sell a coin that falls `drop` from its highest close since it was bought; stay out until the rule resets."""
    px, s = close.to_numpy(dtype=float), sig.to_numpy()
    out = np.zeros(sig.shape, dtype=bool)
    held = np.zeros(sig.shape[1], dtype=bool)
    stopped = np.zeros(sig.shape[1], dtype=bool)
    high = np.full(sig.shape[1], np.nan)
    for t in range(len(sig)):
        stopped &= s[t]  # the stop clears once the rule itself says sell
        entering = s[t] & ~held & ~stopped
        high = np.where(entering, px[t], np.fmax(high, px[t]))
        hit = held & (px[t] < high * (1 - drop))
        stopped |= hit
        held = s[t] & ~stopped
        out[t] = held
    return pd.DataFrame(out, index=sig.index, columns=sig.columns)


def weekly(sig):
    """Look at the signal once a week (Monday's close) and hold what it said until the next look."""
    look = sig.index.dayofweek == 0
    return sig[look].reindex(sig.index).ffill().fillna(False).astype(bool)


def targets_by_volatility(sig, close, cap, target_vol):
    """Like playbook.targets_from_signal, but a coin's share is cut when its recent volatility is high.

    The share is the usual equal split (capped), times target_vol / the coin's
    30-day volatility (a year's worth), never more than 1: a coin moving 100%
    a year gets half the share of one moving 50% when the target is 50%.
    """
    tgt = pb.targets_from_signal(sig, close, cap)
    vol = close.pct_change().rolling(30).std() * np.sqrt(365)
    scale = (target_vol / vol).clip(upper=1.0).fillna(1.0)
    return tgt * scale


# ---- scoring ----

def _combined(sigs, close, cost, cap, targets=pb.targets_from_signal):
    """Equal money in each rule, each run as its own account; (net per bar, turnover, trades)."""
    nets, turnover, trades = [], 0.0, 0
    for sig in sigs:
        net, t, k = _run_account(targets(sig, close, cap), close, cost)
        nets.append(net)
        turnover += t
        trades += k
    return np.mean(nets, axis=0), turnover / len(sigs), trades


def _row(net, idx, close, cost, btc, trades, turnover, strategy, params, note):
    r = bt.score_returns(net, idx, INTERVAL, "binanceus", f"basket of {close.shape[1]}", trades, bh=btc, note=note)
    r.update(strategy=strategy, params=params, turnover=round(turnover, 1), maker=False,
             cost_bp=round(float(cost.mean() * 1e4), 2), fee_bp=bt.FEE_BP["binanceus"],
             note=note + "; " + pb._by_year(net, idx))
    return r


def run():
    close, cost, uni = pb.universe(INTERVAL)
    btc = close["BTCUSD"].pct_change().shift(-1).to_numpy()[:-1]
    idx = close.index[:-1]
    base_note = "the paper run's four rules, 15 coins, cap 0.1; " + uni + "; survivors only"
    rows = []
    variants = 0

    def add(strategy, params, note, close_=close, cost_=cost, cap=CAP, transform=None, targets=pb.targets_from_signal):
        nonlocal variants
        variants += 1
        sigs = [fn(close_) for fn in RULES.values()]
        if transform is not None:
            sigs = [transform(s) for s in sigs]
        net, turnover, trades = _combined(sigs, close_, cost_, cap, targets)
        rows.append(_row(net, idx, close_, cost_, btc, trades, turnover, strategy, params, note))
        print(f"{strategy:60s} {params:28s} IS {rows[-1]['is_ann_pct']:6.1f}%  OOS {rows[-1]['oos_ann_pct']:6.1f}% "
              f"sharpe {rows[-1]['oos_sharpe']:5.2f}  maxdd {rows[-1]['oos_maxdd_pct']:6.1f}%  trades {trades}", flush=True)

    add("baseline: the four rules as the paper run trades them", "", base_note)

    # -- the coin that made the held-back result --
    add("without ZEC", "dropped: ZECUSD",
        "ZEC's late-2025 rise is the whole held-back result: summed over the held-back bars the combination's gross "
        "return is 56 points of the account, of which ZEC gave 61 and the other 14 coins together -5; this is the "
        "same four rules without it; " + base_note,
        close_=close.drop(columns=["ZECUSD"]), cost_=cost.drop(index=["ZECUSD"]))

    # -- reasons other than the backtest --
    thin = [s for s in THIN if s in close.columns]
    add("without the coins that are expensive to trade", f"dropped: {', '.join(thin)}",
        "the four coins whose measured half gap is 15 bp or more (ATOM 204, ALGO 46, LTC 17, ZEC 15) left out; " + base_note,
        close_=close.drop(columns=thin), cost_=cost.drop(index=thin))
    measured = pd.Series({s: (bt.FEE_BP["binanceus"] + MEASURED_HALF_SPREAD_BP.get(s, 5.0)) / 1e4 for s in close.columns})
    add("charged the bid-ask gaps measured on 2026-09-06", "fee 2 bp + measured half gap",
        "every change of position pays 2 bp plus half the gap seen in Binance.US's live book on 2026-09-06, "
        "instead of the standing 5 bp guess for coins the recorder never watched; " + base_note, cost_=measured)
    add("without the expensive coins, at the measured gaps", f"dropped: {', '.join(thin)}",
        "both of the above; " + base_note, close_=close.drop(columns=thin), cost_=measured.drop(index=thin))
    add("signals looked at once a week", "Monday close",
        "each rule read at Monday's close only and held until the next Monday, so there is one order day a week; " + base_note,
        transform=weekly)

    # -- trading less --
    for bars in (2, 3):
        add("wait for the signal to hold N days before buying", f"n={bars}",
            f"a coin is bought only after its rule has said hold for {bars} closes in a row, and sold the first close it says sell; " + base_note,
            transform=lambda s, b=bars: confirm(s, b))
    for k in (2, 3):
        variants += 1
        sigs = [fn(close) for fn in RULES.values()]
        vote = sum(s.astype(int) for s in sigs) >= k
        for cap in (CAP, 0.2):
            net, turnover, trades = _run_account(pb.targets_from_signal(vote, close, cap), close, cost)
            r = _row(net, idx, close, cost, btc, trades, turnover, "hold a coin only when N of the four rules agree",
                     f"n={k} cap={cap}", f"one account instead of four: a coin is held while at least {k} of the four rules say hold; " + base_note)
            rows.append(r)
            print(f"{'majority vote':60s} {f'n={k} cap={cap}':28s} IS {r['is_ann_pct']:6.1f}%  OOS {r['oos_ann_pct']:6.1f}% "
                  f"sharpe {r['oos_sharpe']:5.2f}  maxdd {r['oos_maxdd_pct']:6.1f}%  trades {trades}", flush=True)

    # -- managing risk --
    for days in (100, 200):
        add("hold nothing while BTC is below its N-day average", f"n={days}d",
            f"every rule is switched off while BTC's close is under its {days}-day average, the whole market's weather; " + base_note,
            transform=lambda s, d=days: btc_filter(s, close, d))
    for drop in (0.15, 0.25):
        add("sell a coin that falls N% from its high since bought", f"n={drop:.0%}",
            f"a trailing stop: a held coin {drop:.0%} below its highest close since it was bought is sold and not re-bought until its rule resets; " + base_note,
            transform=lambda s, d=drop: trailing_stop(s, close, d))
    for tv in (0.5, 0.8):
        add("size each coin by its volatility", f"target {tv:.0%}/yr",
            f"a coin's share is scaled down by {tv:.0%} over its 30-day volatility (a year's worth), never up; " + base_note,
            targets=lambda sig, close_, cap, t=tv: targets_by_volatility(sig, close_, cap, t))
    for cap in (1 / 15, 0.15, 0.2):
        add("a different cap per coin", f"cap={cap:.3f}",
            f"at most {cap:.1%} of the account in one coin; " + base_note, cap=cap)

    return bt.record("tweaks", rows, variants_tried=variants)


def main():
    p = run()
    df = pd.read_csv(p)
    pd.set_option("display.width", 250)
    cols = ["strategy", "params", "trades", "turnover", "is_ann_pct", "is_sharpe", "oos_ann_pct", "oos_sharpe", "oos_maxdd_pct"]
    print()
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
