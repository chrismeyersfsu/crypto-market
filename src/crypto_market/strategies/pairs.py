"""Switch between two coins: hold coin A, hold coin B, or split the account.

Spot only. The account is always fully invested in one of the two coins or
split between them; nothing is ever borrowed or sold that isn't owned.

The rules all come from the price of one coin measured in the other:

  ratio = price of A in dollars / price of B in dollars

* "cheap side" (reversion): score the ratio against its own recent average
  (a z-score over the last N bars). When A looks cheap against B, hold A;
  when A looks dear, hold B; in between, split the account 50/50.
* "winner" (trend): hold whichever of the two rose more over the last N bars.

Costs. Moving x of the account out of one coin and into the other means two
fills: sell x of A on its dollar market and buy x of B on its dollar market.
So every change of weight pays x * (fee + half spread on A's USD market)
plus x * (fee + half spread on B's USD market). For ETH vs BTC there is a
direct market (ETHBTC), where the same switch is one fill instead of two, so
the ETH/BTC winners are also scored with ETHBTC's cost charged once - those
rows say "direct ETHBTC fill" in the note.

Benchmarks. `bh` in every row is holding BTC over the same bars. Holding the
pair 50/50 and never rebalancing is recorded as its own row per pair.

Run it:  uv run python -m crypto_market.strategies.pairs
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import backtest as bt

EX = "binanceus"
FAMILY = "pairs"

# grids, small and round
GRID_1H_N = (24, 72, 168, 336)
GRID_1D_N = (7, 14, 30)
GRID_K = (1.0, 1.5, 2.0)
STABLE_GAPS_BP = (2, 5, 10, 20, 50)


# ---------------------------------------------------------------- data helpers

def closes(symbols, interval):
    """Closing prices for several markets on one shared time index (bars all of them have)."""
    out = {}
    for s in symbols:
        out[s] = bt.candles(EX, s, interval).close
    df = pd.DataFrame(out).dropna()
    return df


def usd_cost(market):
    """Cost of moving 100% of the account through one dollar market, one way, as a fraction."""
    return bt.cost_bp(EX, market) / 1e4


# ---------------------------------------------------------------- the two rules

def weight_cheap_side(ratio, n, k):
    """Weight on coin A: 1 when A looks cheap against B, 0 when dear, 0.5 in between.

    The z-score uses only bars up to and including the one being decided.
    """
    lr = np.log(ratio)
    mean = lr.rolling(n).mean()
    sd = lr.rolling(n).std()
    z = (lr - mean) / sd.replace(0.0, np.nan)
    w = pd.Series(0.5, index=ratio.index)
    w[z < -k] = 1.0
    w[z > k] = 0.0
    w[z.isna()] = 0.5  # before the window fills, just sit split
    return w


def weight_winner(ratio, n):
    """Weight on coin A: 1 if A rose more than B over the last n bars, else 0."""
    w = (ratio > ratio.shift(n)).astype(float)
    w[ratio.shift(n).isna()] = 0.5
    return w


# ---------------------------------------------------------------- scoring

def switch_returns(w_a, px_a, px_b, cost_a, cost_b):
    """Net per-bar returns of holding w_a of coin A and the rest of coin B.

    `w_a` is decided at a bar's close and held over the next bar. Costs are
    charged at the bar where the weight changes, on both coins' dollar
    markets. Returns (net return array, number of bars that traded); the last
    bar is dropped because its forward return isn't known yet.
    """
    ret_a = px_a.pct_change().shift(-1)
    ret_b = px_b.pct_change().shift(-1)
    w_a = w_a.astype(float).clip(0, 1)
    w_b = 1.0 - w_a
    turn_a = (w_a - w_a.shift(1).fillna(0.0)).abs()  # the first bar buys in from cash
    turn_b = (w_b - w_b.shift(1).fillna(0.0)).abs()
    gross = w_a * ret_a + w_b * ret_b
    net = gross - turn_a * cost_a - turn_b * cost_b
    trades = int(((turn_a + turn_b) > 1e-9).sum())
    return net.to_numpy()[:-1], trades, edge_per_switch(gross, turn_a)


def switch_returns_direct(w_a, px_a, px_b, cost_direct, cost_in):
    """Same, but the switch is one fill on the A/B market instead of two dollar fills.

    Only the initial buy-in pays a dollar market (`cost_in`); after that, moving
    weight between the coins pays `cost_direct` once per unit moved.
    """
    ret_a = px_a.pct_change().shift(-1)
    ret_b = px_b.pct_change().shift(-1)
    w_a = w_a.astype(float).clip(0, 1)
    turn = (w_a - w_a.shift(1)).abs()
    turn.iloc[0] = 0.0
    gross = w_a * ret_a + (1.0 - w_a) * ret_b
    net = gross - turn * cost_direct
    net.iloc[0] -= cost_in  # buying in the first time
    trades = int((turn > 1e-9).sum()) + 1
    return net.to_numpy()[:-1], trades, edge_per_switch(gross, turn)


def edge_per_switch(gross, turn):
    """Gross edge earned per 100% switch, in bp: what the rule wins before costs."""
    moved = float(turn.sum())
    if moved <= 0:
        return 0.0
    return float(gross.fillna(0).sum() * 1e4 / moved)


def hold_both_returns(px_a, px_b, cost_a, cost_b):
    """Buy half of each coin once and never touch it again."""
    ret_a = px_a.pct_change().shift(-1)
    ret_b = px_b.pct_change().shift(-1)
    eq = 0.5 * (1 + ret_a.fillna(0)).cumprod() + 0.5 * (1 + ret_b.fillna(0)).cumprod()
    net = eq.pct_change()
    net.iloc[0] = eq.iloc[0] - 1.0
    net.iloc[0] -= 0.5 * cost_a + 0.5 * cost_b
    return net.to_numpy()[:-1], 1


def row(name, params, ret, index, interval, market, trades, bh, note=""):
    r = bt.score_returns(ret, index, interval, EX, market, trades, bh=bh, note=note)
    r["strategy"] = name
    r["params"] = params
    return r


def is_sharpe(r):
    return r["is_sharpe"]


# ---------------------------------------------------------------- the ideas

def run_pair(sym_a, sym_b, interval, btc_bh, rows, tag=""):
    """Both rules over their grids on one pair; returns the in-sample best of each rule."""
    px = closes([sym_a, sym_b, "BTCUSD"], interval)
    idx = px.index[:-1]
    bh = btc_bh(px)
    ratio = px[sym_a] / px[sym_b]
    ca, cb = usd_cost(sym_a), usd_cost(sym_b)
    market = f"{sym_a}/{sym_b}"
    label = f"{sym_a[:-3]}/{sym_b[:-3]}"

    ns = GRID_1H_N if interval == "1h" else GRID_1D_N
    cheap, winner, weights = [], [], {}
    for n in ns:
        for k in GRID_K:
            w = weight_cheap_side(ratio, n, k)
            ret, tr, edge = switch_returns(w, px[sym_a], px[sym_b], ca, cb)
            weights[f"cheap n={n} k={k}"] = w
            cheap.append(row(f"{label} hold the cheap side", f"n={n} k={k}", ret, idx,
                             interval, market, tr, bh,
                             note=f"z-score of {label} over {n} bars; earns {edge:.1f} bp a switch "
                                  f"against {1e4*(ca+cb):.2f} bp of cost (two dollar fills){tag}"))
    for n in ns:
        w = weight_winner(ratio, n)
        ret, tr, edge = switch_returns(w, px[sym_a], px[sym_b], ca, cb)
        weights[f"winner n={n}"] = w
        winner.append(row(f"{label} hold the winner", f"n={n}", ret, idx, interval, market, tr, bh,
                          note=f"whichever rose more over {n} bars; earns {edge:.1f} bp a switch "
                               f"against {1e4*(ca+cb):.2f} bp of cost (two dollar fills){tag}"))

    ret, tr = hold_both_returns(px[sym_a], px[sym_b], ca, cb)
    both = row(f"{label} hold 50/50", "-", ret, idx, interval, market, tr, bh,
               note="buy half of each once, never rebalance (the do-nothing benchmark)")

    best_c = max(cheap, key=is_sharpe)
    best_w = max(winner, key=is_sharpe)
    for r in (best_c, best_w):
        r["note"] = "IS-BEST of its grid; " + r["note"]
    rows.extend(cheap + winner + [both])

    # the patience check: same rule, but act one bar later. If the whole edge is
    # in reacting to the newest print, waiting one bar should kill it - and a
    # print in a market that trades only some hours is often not fillable anyway.
    late = []
    for best, key in ((best_c, "cheap "), (best_w, "winner ")):
        w = weights[key + best["params"]].shift(1).fillna(0.5)
        ret, tr, edge = switch_returns(w, px[sym_a], px[sym_b], ca, cb)
        late.append(row(best["strategy"] + " (one bar late)", best["params"], ret, idx, interval,
                        market, tr, bh,
                        note=f"patience check on the in-sample best: same rule acted on one bar "
                             f"later; earns {edge:.1f} bp a switch against {1e4*(ca+cb):.2f} bp of cost"))
    rows.extend(late)
    return {"px": px, "idx": idx, "bh": bh, "ratio": ratio, "best_cheap": best_c,
            "best_winner": best_w, "late": late}


def run_ethbtc_direct(res, rows):
    """The ETH/BTC in-sample winners again, switching on the ETHBTC market (one fill, not two)."""
    px, idx, bh = res["px"], res["idx"], res["bh"]
    ratio = res["ratio"]
    direct = bt.cost_bp(EX, "ETHBTC") / 1e4
    cost_in = usd_cost("BTCUSD")
    out = []
    for best, mk in ((res["best_cheap"], weight_cheap_side), (res["best_winner"], weight_winner)):
        p = dict(kv.split("=") for kv in best["params"].split())
        w = (mk(ratio, int(p["n"]), float(p["k"])) if "k" in p else mk(ratio, int(p["n"])))
        ret, tr, edge = switch_returns_direct(w, px["ETHUSD"], px["BTCUSD"], direct, cost_in)
        r = row(best["strategy"] + " (switch on ETHBTC)", best["params"], ret, idx,
                best["interval"], "ETHUSD/BTCUSD", tr, bh,
                note=f"same rule, direct ETHBTC fill: {1e4*direct:.2f} bp a switch instead of "
                     f"{1e4*(usd_cost('ETHUSD') + usd_cost('BTCUSD')):.2f} bp; earns {edge:.1f} bp a switch")
        out.append(r)
    rows.extend(out)
    return out


def half_life(spread, mask):
    """Reversion speed of a spread, from a plain AR(1) fit on the in-sample part.

    Fits  s[t+1] - s[t] = a + b * s[t]. A negative b means the spread is pulled
    back toward its average; the half-life is how many bars it takes to close
    half the gap. Returns (half_life_bars, b). No half-life if b >= 0.
    """
    s = spread[mask].to_numpy()
    x, y = s[:-1], np.diff(s)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 100:
        return np.inf, 0.0
    xm, ym = x.mean(), y.mean()
    var = ((x - xm) ** 2).sum()
    if var <= 0:
        return np.inf, 0.0
    b = ((x - xm) * (y - ym)).sum() / var
    if b >= 0 or 1 + b <= 0:
        return np.inf, b
    return -np.log(2) / np.log(1 + b), b


def market_stats():
    """Per 1h market: dollars traded in-sample, and the share of in-sample hours with any trade.

    The second number matters more than the first. A market that goes hours
    without a trade only has a stale last price, and a rule that "buys the dip"
    on a stale price is buying a print nobody could have filled.
    """
    out = {}
    for _, s, _ in bt.available(EX, "1h"):
        d = bt.candles(EX, s, "1h")
        if len(d) < 12_000:  # skip markets that listed too recently to have a full history
            continue
        ins, _ = bt.split(d.index)
        out[s] = (float((d.close * d.volume)[ins].sum()), float((d.trades[ins] > 0).mean()))
    return out


STABLES = {"USDTUSD", "USDCUSD", "SUSD"}


def hunt(universe, tag, rows, n, k, note_extra=""):
    """Pick the 10 fastest-reverting pairs in a universe in-sample, trade them on the whole sample.

    The picking is itself the fit, so the held-back part is the honest test of
    "does a pair that reverted keep reverting".
    """
    px = closes(list(universe) + ["BTCUSD"], "1h")
    idx = px.index[:-1]
    bh = px["BTCUSD"].pct_change().shift(-1).to_numpy()[:-1]
    ins_mask, _ = bt.split(px.index)
    logs = np.log(px)

    scored = []
    for i, a in enumerate(universe):
        for b_sym in universe[i + 1:]:
            hl, _ = half_life(logs[a] - logs[b_sym], ins_mask)
            if np.isfinite(hl) and hl >= 2:  # a 1-bar half-life is the fit fitting noise
                scored.append((hl, a, b_sym))
    scored.sort()
    picked = scored[:10]

    kept, late_kept = [], []
    for hl, a, b_sym in picked:
        ratio = px[a] / px[b_sym]
        w = weight_cheap_side(ratio, n, k)
        ca, cb = usd_cost(a), usd_cost(b_sym)
        ret, tr, edge = switch_returns(w, px[a], px[b_sym], ca, cb)
        hold, _ = hold_both_returns(px[a], px[b_sym], ca, cb)
        hold_oos = bt.score_returns(hold, idx, "1h", EX, f"{a}/{b_sym}", 1)["oos_ann_pct"]
        r = row(f"{a[:-3]}/{b_sym[:-3]} hold the cheap side ({tag})", f"n={n} k={k}", ret, idx,
                "1h", f"{a}/{b_sym}", tr, bh,
                note=f"picked in-sample for fast reversion (half-life {hl:.0f}h), rule fixed from the "
                     f"ETH/BTC in-sample best; earns {edge:.1f} bp a switch against "
                     f"{1e4*(ca+cb):.2f} bp of charged cost; holding this pair 50/50 makes "
                     f"{hold_oos:.0f}%/yr out of sample{note_extra}")
        rows.append(r)
        if r["oos_ann_pct"] > 0 and r["oos_ann_pct"] > hold_oos:
            kept.append((a, b_sym))

        ret_l, tr_l, edge_l = switch_returns(w.shift(1).fillna(0.5), px[a], px[b_sym], ca, cb)
        rl = row(f"{a[:-3]}/{b_sym[:-3]} hold the cheap side ({tag}, one bar late)", f"n={n} k={k}",
                 ret_l, idx, "1h", f"{a}/{b_sym}", tr_l, bh,
                 note=f"patience check: same pair and rule acted on one bar later; earns "
                      f"{edge_l:.1f} bp a switch against {1e4*(ca+cb):.2f} bp of cost")
        rows.append(rl)
        if rl["oos_ann_pct"] > 0 and rl["oos_ann_pct"] > hold_oos:
            late_kept.append((a, b_sym))
    return picked, kept, late_kept


def run_hunted_pairs(rows, n, k):
    """Idea 3, run twice: over the 40 biggest dollar markets, and over the ones that really trade."""
    stats = market_stats()
    by_dollars = sorted(stats, key=lambda s: -stats[s][0])
    top40 = [s for s in by_dollars if s not in STABLES][:40]
    liquid = [s for s in by_dollars if s not in STABLES and stats[s][1] >= 0.75]

    thin = [s for s in top40 if stats[s][1] < 0.75]
    picked_a, kept_a, late_a = hunt(top40, "hunted pair, thin markets", rows, n, k,
                            note_extra="; WARNING both markets trade in well under half of all hours, "
                                       "so most of this is bouncing between stale last prices")
    picked_b, kept_b, late_b = hunt(liquid, "hunted pair, markets that trade", rows, n, k)
    return {"top40": top40, "thin": thin, "liquid": liquid, "stats": stats,
            "picked_top40": picked_a, "kept_top40": kept_a, "late_top40": late_a,
            "picked_liquid": picked_b, "kept_liquid": kept_b, "late_liquid": late_b}


def run_ethbtc_market(rows):
    """The ETH/BTC rules again, but the ratio comes from the ETHBTC market's own prints.

    ETHUSD/BTCUSD divides two markets whose last prints happen at different
    moments, and a "cheap side" rule can end up trading that mismatch rather
    than anything real. The ETHBTC market prices the same ratio in one book, so
    running the same rules on it says whether the edge was ever there.
    """
    ratio = bt.candles(EX, "ETHBTC", "1m").close.resample("1h").last().dropna()
    btc = bt.candles(EX, "BTCUSD", "1h").close
    px = pd.DataFrame({"ratio": ratio, "BTCUSD": btc}).dropna()
    px["ETH"] = px.ratio * px.BTCUSD  # ETH in dollars, priced through the ETHBTC book
    idx = px.index[:-1]
    bh = px["BTCUSD"].pct_change().shift(-1).to_numpy()[:-1]
    direct = bt.cost_bp(EX, "ETHBTC") / 1e4
    cost_in = usd_cost("BTCUSD")

    out = []
    for n in GRID_1H_N:
        for k in GRID_K:
            w = weight_cheap_side(px.ratio, n, k)
            ret, tr, edge = switch_returns_direct(w, px["ETH"], px["BTCUSD"], direct, cost_in)
            out.append(row("ETH/BTC hold the cheap side (ETHBTC book)", f"n={n} k={k}", ret, idx, "1h",
                           "ETHBTC", tr, bh,
                           note=f"ratio from the ETHBTC book itself, not ETHUSD/BTCUSD; earns "
                                f"{edge:.1f} bp a switch against {1e4*direct:.2f} bp of cost; "
                                f"one year of data, so a shorter window than the other rows"))
    best = max(out, key=is_sharpe)
    best["note"] = "IS-BEST of its grid; " + best["note"]
    rows.extend(out)
    return out


def run_stablecoins(rows):
    """Idea 4: hold whichever dollar coin is cheaper, and wait for it to come back to a dollar."""
    px = closes(["USDCUSD", "USDTUSD", "BTCUSD"], "1h")
    idx = px.index[:-1]
    bh = px["BTCUSD"].pct_change().shift(-1).to_numpy()[:-1]
    cc, ct = usd_cost("USDCUSD"), usd_cost("USDTUSD")
    gap_bp = (px["USDCUSD"] / px["USDTUSD"] - 1.0) * 1e4  # negative: USDC is the cheaper one

    out = []
    for d in STABLE_GAPS_BP:
        s = pd.Series(np.nan, index=px.index)
        s[gap_bp < -d] = 1.0  # USDC cheaper by more than d: hold USDC
        s[gap_bp > d] = 0.0
        w = s.ffill().fillna(0.5)
        ret, tr, edge = switch_returns(w, px["USDCUSD"], px["USDTUSD"], cc, ct)
        out.append(row("hold the cheaper dollar coin", f"gap={d}bp", ret, idx, "1h",
                       "USDCUSD/USDTUSD", tr, bh,
                       note=f"swap when one is more than {d} bp cheaper; earns {edge:.1f} bp a swap "
                            f"against {1e4*(cc+ct):.2f} bp of cost (two dollar fills)"))
    best = max(out, key=is_sharpe)
    best["note"] = "IS-BEST of its grid; " + best["note"]

    d = int(best["params"].split("=")[1].rstrip("bp"))
    s = pd.Series(np.nan, index=px.index)
    s[gap_bp < -d] = 1.0
    s[gap_bp > d] = 0.0
    w = s.ffill().fillna(0.5)
    direct = bt.cost_bp(EX, "USDCUSDT") / 1e4
    ret, tr, edge = switch_returns_direct(w, px["USDCUSD"], px["USDTUSD"], direct, ct)
    out.append(row("hold the cheaper dollar coin (swap on USDCUSDT)", best["params"], ret, idx, "1h",
                   "USDCUSD/USDTUSD", tr, bh,
                   note=f"same rule, direct USDCUSDT fill: {1e4*direct:.2f} bp a swap instead of "
                        f"{1e4*(cc+ct):.2f} bp; earns {edge:.1f} bp a swap"))
    ret, tr, edge = switch_returns(w.shift(1).fillna(0.5), px["USDCUSD"], px["USDTUSD"], cc, ct)
    out.append(row("hold the cheaper dollar coin (one bar late)", best["params"], ret, idx, "1h",
                   "USDCUSD/USDTUSD", tr, bh,
                   note=f"patience check on the in-sample best: same rule acted on one bar later; "
                        f"earns {edge:.1f} bp a swap against {1e4*(cc+ct):.2f} bp of cost"))
    rows.extend(out)
    return out


# ---------------------------------------------------------------- driver

def run():
    rows = []
    variants = 0

    def btc_bh(px):
        return px["BTCUSD"].pct_change().shift(-1).to_numpy()[:-1]

    pairs = [("ETHUSD", "BTCUSD"), ("SOLUSD", "ETHUSD"), ("SOLUSD", "BTCUSD")]
    results = {}
    for a, b in pairs:
        for interval in ("1h", "1d"):
            res = run_pair(a, b, interval, btc_bh, rows)
            results[(a, b, interval)] = res
            ns = GRID_1H_N if interval == "1h" else GRID_1D_N
            variants += len(ns) * len(GRID_K) + len(ns) + 2  # + the two patience checks

    # ETH/BTC again with the direct market's cost
    for interval in ("1h", "1d"):
        run_ethbtc_direct(results[("ETHUSD", "BTCUSD", interval)], rows)
        variants += 2

    # the rule that idea 3 borrows: the in-sample best cheap-side rule on ETH/BTC at 1h
    p = dict(kv.split("=") for kv in results[("ETHUSD", "BTCUSD", "1h")]["best_cheap"]["params"].split())
    n, k = int(p["n"]), float(p["k"])
    hunted = run_hunted_pairs(rows, n, k)
    variants += 2 * (len(hunted["picked_top40"]) + len(hunted["picked_liquid"]))

    run_ethbtc_market(rows)
    variants += len(GRID_1H_N) * len(GRID_K)

    variants += len(STABLE_GAPS_BP) + 2
    run_stablecoins(rows)

    # holding BTC alone, for the record
    px = closes(["BTCUSD"], "1h")
    r = bt.score_returns(px["BTCUSD"].pct_change().shift(-1).to_numpy()[:-1], px.index[:-1], "1h",
                         EX, "BTCUSD", 1, bh=px["BTCUSD"].pct_change().shift(-1).to_numpy()[:-1],
                         note="buy BTC once and hold it (the benchmark in bh_* on every row)")
    r["strategy"], r["params"] = "hold BTC", "-"
    rows.append(r)

    path = bt.record(FAMILY, rows, variants_tried=variants)
    print(f"{len(rows)} rows, {variants} variants tried -> {path}")
    for tag in ("top40", "liquid"):
        picked, kept = hunted[f"picked_{tag}"], hunted[f"kept_{tag}"]
        print(f"[{tag}] {len(kept)} of {len(picked)} hunted pairs still beat holding the pair "
              f"50/50 out of sample: " + ", ".join(f"{a[:-3]}/{b[:-3]}" for a, b in kept))
        print(f"[{tag}] {len(hunted['late_' + tag])} of {len(picked)} still beat it when acted on "
              f"one bar later")
        print(f"[{tag}] picked:", ", ".join(f"{a[:-3]}/{b[:-3]} ({hl:.0f}h)" for hl, a, b in picked))
    st = hunted["stats"]
    print(f"of the 40 biggest dollar markets, {len(hunted['thin'])} trade in under 75% of hours; "
          f"only {len(hunted['liquid'])} clear that bar: " +
          ", ".join(f"{s}({st[s][1]:.0%})" for s in hunted["liquid"]))
    df = pd.DataFrame(rows)
    cols = ["strategy", "params", "market", "interval", "trades", "is_ann_pct", "is_sharpe",
            "oos_ann_pct", "oos_sharpe", "oos_maxdd_pct", "bh_oos_ann_pct"]
    pd.set_option("display.width", 200)
    print(df.sort_values("is_sharpe", ascending=False)[cols].head(25).to_string(index=False))
    return path


def main():
    run()


if __name__ == "__main__":
    main()
