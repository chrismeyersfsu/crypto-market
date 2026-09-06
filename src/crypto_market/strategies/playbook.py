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


def _whole_series_stats(net, interval):
    """(annualized %, sharpe, max drawdown %) over an entire return series, split ignored."""
    r = net[~np.isnan(net)]
    if len(r) == 0:
        return 0.0, 0.0, 0.0
    eq = np.cumprod(1 + r)
    years = len(r) / bt.BARS_PER_YEAR[interval]
    ann = eq[-1] ** (1 / years) - 1 if years > 0 else 0.0
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    sharpe = r.mean() / sd * np.sqrt(bt.BARS_PER_YEAR[interval]) if sd > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    dd = ((eq - peak) / peak).min()
    return ann * 100, sharpe, dd * 100


def _contributions(tgt, close, cost):
    """Per-bar, per-coin gross contribution (held share x that bar's return) to a basket's return.

    Mirrors `_run_account`'s walk (same rebalance-then-drift bar-by-bar loop) but
    keeps the per-coin breakdown instead of summing it into one net return, and
    skips the cost charge -- this is used only to see which coins drove the
    gross result, not to re-score anything.
    """
    px = close.to_numpy(dtype=float)
    fwd = close.pct_change().shift(-1).to_numpy(dtype=float)
    tg = tgt.to_numpy(dtype=float)
    n, m = px.shape
    held = np.zeros(m)
    contrib = np.zeros((n - 1, m))
    listed = ~np.isnan(px)
    last_bar = np.where(listed.any(axis=0), n - 1 - np.argmax(listed[::-1], axis=0), -1)
    for t in range(n - 1):
        gone = (t > last_bar) & (held > 0)
        if gone.any():
            held = np.where(gone, 0.0, held)
        row = tg[t]
        if not np.isnan(row).all():
            want = np.where(np.isnan(row), held, row)
            if want.sum() > 1.0 + 1e-9:
                touched = ~np.isnan(row)
                room = 1.0 - want[~touched].sum()
                want[touched] *= max(room, 0.0) / want[touched].sum()
            held = want.copy()
        step = np.nan_to_num(fwd[t])
        contrib[t] = held * step
        gain = float((held * step).sum())
        grown = held * (1 + step)
        total = 1.0 + gain
        held = grown / total if total > 1e-9 else np.zeros(m)
    return pd.DataFrame(contrib, index=close.index[:-1], columns=close.columns)


def _pick_families(close, cost, interval, bh, note_uni, caps=CAPS):
    """Rebuild the four-rule-type grid (same as the one in `run`) on a given universe.

    Used to re-pick the in-sample best of each rule type when the coin universe
    changes (a coin dropped), without touching `run`'s own grid or its rows.
    Returns (families, variants_run); families maps rule-type name to a list of
    (in-sample sharpe, net, label, params, sig, cap), same shape as in `run`.
    """
    idx = close.index[:-1]
    families = {}
    variants = 0
    for cap in caps:
        grid = []
        for days in (50, 100, 200):
            grid.append(("long-term momentum: hold coins above their N-day average", f"n={days}d cap={cap}",
                         sig_long_momentum(close, days, interval)))
        for fast, slow in ((10, 50), (20, 100), (20, 200), (50, 200)):
            grid.append(("trend: short average above long average", f"fast={fast}d slow={slow}d cap={cap}",
                         sig_crossover(close, fast, slow, interval)))
        for n, (low, high), filt in itertools.product((2, 4, 14), ((20, 60), (30, 50)), (0, 200)):
            grid.append(("reversion: buy an RSI dip, sell when it recovers" + (" (only coins above their 200-day average)" if filt else ""),
                         f"rsi={n} buy<{low} sell>{high} filter={filt}d",
                         sig_rsi_dip(close, n, low, high, filt, interval)))
        for days in (20, 50, 100):
            grid.append(("breakout: buy a new N-day high, sell a new N/2-day low", f"n={days}d cap={cap}",
                         sig_breakout(close, days, interval)))
        for label, params, sig in grid:
            variants += 1
            r, net = score_basket(sig, close, cost, interval, cap, label, params, bh, note_uni)
            families.setdefault(label.split(":")[0], []).append(
                (_in_sample_sharpe(net, idx, interval), net, label, params, sig, cap))
    return families, variants


def run():
    rows = []
    extra_rows = []  # the new robustness-check rows (daily only), kept aside for main()'s summary print
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

        if interval == "1d":
            # --- robustness checks the daily combined-best-of-each-rule-type basket still needed:
            #     drop the biggest contributors, walk-forward re-picking, a finer settings grid.

            # 1. drop the coins that contributed most, re-pick, re-run. Ranked two ways: by in-sample
            #    contribution (no held-back data used) and by held-back contribution, which is the one that
            #    answers "is the held-back result one coin's rally?" -- dropping a coin because it did well
            #    held back can only make the check harder to pass, so it is the honest direction to peek in
            ins_mask, oos_mask = bt.split(idx)
            contrib_is = pd.Series(0.0, index=close.columns)
            contrib_oos = pd.Series(0.0, index=close.columns)
            for _, _, _, _, sig, cap in picks.values():
                tgt = targets_from_signal(sig, close, cap)
                c = _contributions(tgt, close, cost)
                contrib_is = contrib_is.add(c[ins_mask].sum() / len(picks), fill_value=0.0)
                contrib_oos = contrib_oos.add(c[oos_mask].sum() / len(picks), fill_value=0.0)
            for where, contrib, ks in (("in-sample", contrib_is, (3, 5)), ("held-back", contrib_oos, (1, 3))):
                ranked = contrib.sort_values(ascending=False)
                for k in ks:
                    drop = list(ranked.index[:k])
                    close2, cost2 = close.drop(columns=drop), cost.drop(index=drop)
                    note_uni2 = (f"{close2.shape[1]} coins, after dropping the {k} biggest {where} "
                                 f"contributors ({', '.join(drop)}) from: {uni}")
                    families2, v2 = _pick_families(close2, cost2, interval, btc, note_uni2)
                    variants += v2
                    picks2 = {kk: max(v, key=lambda x: x[0]) for kk, v in families2.items()}
                    nets2 = [p[1] for p in picks2.values()]
                    combo2 = np.mean(nets2, axis=0)
                    variants += 1
                    r = bt.score_returns(combo2, idx, interval, "binanceus", f"basket of {close2.shape[1]}",
                                         len(nets2), bh=btc,
                                         note=(f"dropped the {k} coin{'s' if k > 1 else ''} that contributed most {where} to the "
                                               f"combo above ({', '.join(drop)}: "
                                               + ", ".join(f"{contrib[d] * 100:+.0f}" for d in drop)
                                               + " points of account return); picks: " +
                                               "; ".join(f"{p[2]} [{p[3]}]" for p in picks2.values()) +
                                               "; " + note_uni2))
                    r.update(strategy=f"combined: best of each rule type, without the {k} coin{'s' if k > 1 else ''} that contributed most {where}",
                             params=f"dropped: {', '.join(drop)}", maker=False,
                             cost_bp=round(float(cost2.mean() * 1e4), 2), fee_bp=bt.FEE_BP["binanceus"],
                             note=r["note"] + "; " + _by_year(combo2, idx))
                    rows.append(r)
                    extra_rows.append(r)

            # 2. walk-forward: re-pick each rule type's best (by sharpe on bars before that year), year by year
            years = (2023, 2024, 2025, 2026)
            chained = pd.Series(np.nan, index=idx)
            year_notes = []
            years_used = 0
            for y in years:
                cutoff = pd.Timestamp(f"{y}-01-01", tz=idx.tz)
                before = np.asarray(idx < cutoff)
                if not before.any():
                    continue
                years_used += 1

                def _sharpe_before(net, before=before):
                    r = net[before]
                    sd = r.std(ddof=1)
                    return float(r.mean() / sd * np.sqrt(bt.BARS_PER_YEAR[interval])) if sd > 0 else 0.0

                best = {fam: max(lst, key=lambda x: _sharpe_before(x[1])) for fam, lst in families.items()}
                combo_y = np.mean([p[1] for p in best.values()], axis=0)
                in_year = np.asarray(idx.year == y)
                chained[in_year] = combo_y[in_year]
                year_notes.append(f"{y}: " + ", ".join(f"{p[2]} [{p[3]}]" for p in best.values()))

            mask = np.asarray(idx.year >= 2023)
            chain_idx, chain_net, chain_bh = idx[mask], chained.to_numpy()[mask], btc[mask]
            ann, sh, dd = _whole_series_stats(chain_net, interval)
            variants += 1
            r = bt.score_returns(chain_net, chain_idx, interval, "binanceus", f"basket of {close.shape[1]}",
                                 4 * years_used, bh=chain_bh,
                                 note=("each test year's picks are the in-sample-best of each rule type using "
                                       "only bars before that year; " + "; ".join(year_notes) +
                                       f"; whole chained series {str(chain_idx[0])[:10]} to {str(chain_idx[-1])[:10]}: "
                                       f"ann {ann:+.1f}% sharpe {sh:.2f} maxdd {dd:.1f}%; " +
                                       _by_year(chain_net, chain_idx) + "; " + note_uni))
            r.update(strategy="combined: walk-forward, picks re-chosen each January from the years before",
                     params=f"test years: {', '.join(str(y) for y in years)}", maker=False,
                     cost_bp=round(float(cost.mean() * 1e4), 2), fee_bp=bt.FEE_BP["binanceus"])
            rows.append(r)
            extra_rows.append(r)

            # 3. finer neighbours around the winning long-term-momentum and crossover settings, cap 0.1 only
            for days in (30, 40, 50, 60, 75):
                variants += 1
                r, _ = score_basket(sig_long_momentum(close, days, interval), close, cost, interval, 0.1,
                                    "long-term momentum: hold coins above their N-day average", f"n={days}d cap=0.1",
                                    btc, note_uni)
                rows.append(r)
                extra_rows.append(r)
            for fast, slow in ((5, 50), (10, 40), (10, 50), (10, 60), (15, 50)):
                variants += 1
                r, _ = score_basket(sig_crossover(close, fast, slow, interval), close, cost, interval, 0.1,
                                    "trend: short average above long average", f"fast={fast}d slow={slow}d cap=0.1",
                                    btc, note_uni)
                rows.append(r)
                extra_rows.append(r)

    return bt.record("playbook", rows, variants_tried=variants), extra_rows


def main():
    p, extra = run()
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

    if extra:
        print("\n== new robustness-check rows (daily basket)")
        ecols = ["strategy", "params", "is_ann_pct", "is_sharpe", "oos_ann_pct", "oos_sharpe", "oos_maxdd_pct"]
        print(pd.DataFrame(extra)[ecols].to_string(index=False))


if __name__ == "__main__":
    main()
