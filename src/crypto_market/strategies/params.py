"""The paper run's four rule types with their numbers changed, one rule at a time.

Same 15 coins, same cap (a tenth of the account per coin), same daily bars
as strategies/playbook.py. For each rule type a grid of settings is run two
ways: the rule on its own, and the rule swapped into the paper run's
combination in place of the setting it uses now (the other three rules
unchanged). Each row is scored with and without ZEC, since ZEC's 2025 rise
is most of the held-back return (the "without ZEC" numbers are in the
note); a setting that only looks good with ZEC in the basket is a setting
that held ZEC longer, not a better rule.

    uv run python -m crypto_market.strategies.params
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .. import backtest as bt
from . import playbook as pb
from .trend import _run_account

INTERVAL = "1d"
CAP = 0.1
BASE = {"average": 50, "crossover": (10, 50), "breakout": 20, "rsi": (14, 20, 60)}
GRID = {
    "average": [10, 20, 30, 40, 50, 75, 100, 150, 200],
    "crossover": [(5, 20), (5, 50), (10, 30), (10, 50), (20, 50), (10, 100), (20, 100), (50, 200)],
    "breakout": [10, 20, 30, 50, 100],
    "rsi": [(n, lo, hi) for n in (7, 14, 21) for lo, hi in ((20, 60), (30, 50), (30, 70), (25, 55))],
}
LABEL = {
    "average": "hold coins above their N-day average",
    "crossover": "hold coins whose fast average is above their slow average",
    "breakout": "buy a new N-day high, sell a new N/2-day low",
    "rsi": "buy an RSI dip, sell when it recovers",
}


def signal(kind, p, close):
    if kind == "average":
        return pb.sig_long_momentum(close, p, INTERVAL)
    if kind == "crossover":
        return pb.sig_crossover(close, p[0], p[1], INTERVAL)
    if kind == "breakout":
        return pb.sig_breakout(close, p, INTERVAL)
    n, lo, hi = p
    return pb.sig_rsi_dip(close, n, lo, hi, 0, INTERVAL)


def params_text(kind, p):
    if kind == "average":
        return f"n={p}d"
    if kind == "crossover":
        return f"fast={p[0]}d slow={p[1]}d"
    if kind == "breakout":
        return f"n={p}d exit={max(p // 2, 2)}d"
    return f"rsi={p[0]} buy<{p[1]} sell>{p[2]}"


def _net(sigs, close, cost):
    nets, turnover, trades = [], 0.0, 0
    for s in sigs:
        n, t, k = _run_account(pb.targets_from_signal(s, close, CAP), close, cost)
        nets.append(n)
        turnover += t
        trades += k
    return np.mean(nets, axis=0), turnover / len(sigs), trades


def run():
    close, cost, uni = pb.universe(INTERVAL)
    btc = close["BTCUSD"].pct_change().shift(-1).to_numpy()[:-1]
    idx = close.index[:-1]
    close_x, cost_x = close.drop(columns=["ZECUSD"]), cost.drop(index=["ZECUSD"])
    base_note = f"15 coins, cap {CAP}; {uni}; survivors only"
    rows, variants = [], 0

    def score(sigs, sigs_x, strategy, params, note):
        nonlocal variants
        variants += 1
        net, turnover, trades = _net(sigs, close, cost)
        net_x, _, _ = _net(sigs_x, close_x, cost_x)
        rx = bt.score_returns(net_x, idx, INTERVAL, "binanceus", "x", 0)
        r = bt.score_returns(net, idx, INTERVAL, "binanceus", f"basket of {close.shape[1]}", trades, bh=btc,
                             note=f"without ZEC: {rx['is_ann_pct']:+.1f}% in-sample, {rx['oos_ann_pct']:+.1f}% held back "
                                  f"(sharpe {rx['oos_sharpe']:.2f}); {note}; " + pb._by_year(net, idx))
        r.update(strategy=strategy, params=params, turnover=round(turnover, 1), maker=False,
                 cost_bp=round(float(cost.mean() * 1e4), 2), fee_bp=bt.FEE_BP["binanceus"])
        rows.append(r)
        print(f"{strategy[:58]:58s} {params:26s} IS {r['is_ann_pct']:6.1f}%  OOS {r['oos_ann_pct']:6.1f}%  "
              f"no ZEC {rx['oos_ann_pct']:6.1f}%  sharpe {r['oos_sharpe']:5.2f}", flush=True)
        return r

    base = {k: signal(k, p, close) for k, p in BASE.items()}
    base_x = {k: signal(k, p, close_x) for k, p in BASE.items()}
    score(list(base.values()), list(base_x.values()), "the paper run's combination", "as run",
          "the four rules at the settings the paper run uses; " + base_note)

    for kind, grid in GRID.items():
        for p in grid:
            s, s_x = signal(kind, p, close), signal(kind, p, close_x)
            tag = " (as run)" if p == BASE[kind] else ""
            score([s], [s_x], f"{LABEL[kind]}, alone", params_text(kind, p) + tag,
                  f"this rule on its own with all the money; " + base_note)
            combo = dict(base, **{kind: s})
            combo_x = dict(base_x, **{kind: s_x})
            score(list(combo.values()), list(combo_x.values()), f"{LABEL[kind]}, in the combination",
                  params_text(kind, p) + tag,
                  f"the paper run's combination with this rule's setting changed and the other three as run; " + base_note)

    return bt.record("params", rows, variants_tried=variants)


def main():
    p = run()
    df = pd.read_csv(p)
    pd.set_option("display.width", 250)
    df["oos_no_zec"] = df.note.str.extract(r"without ZEC: [-+\d.]+% in-sample, ([-+\d.]+)% held back").astype(float)
    print()
    print(df[["strategy", "params", "trades", "is_ann_pct", "is_sharpe", "oos_ann_pct", "oos_sharpe", "oos_maxdd_pct", "oos_no_zec"]].to_string(index=False))


if __name__ == "__main__":
    main()
