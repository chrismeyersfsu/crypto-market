"""Resting-order strategies on Binance.US, replayed against real trade history.

A resting order is filled by somebody else's trade, so the only honest test
is to replay the trades that actually happened and ask, for each one, whether
it would have filled an order we had sitting there. Two fill rules bracket
the truth. "through": a trade printed at a price beyond ours, so our whole
price level was eaten and we were filled whatever our place in the queue --
the pessimistic rule and the headline. "touch": a trade at exactly our price
fills us up to its size, as if we were first in the queue -- the optimistic
rule. Real life sits between them.

The best bid and ask at each moment are estimated from the trades themselves:
a seller hitting a bid tells us where the best bid was, a buyer hitting an
ask where the best ask was. `check_quotes` measures how far that estimate
sits from the quotes the tick recorder saw over the same hours.

Fees are Binance.US's base tier: 0 bp for a resting fill, 2 bp for an
immediate one. Orders are $100 each, below the ~$165 usually sitting at the
best price, so an immediate fill is assumed to get the best price in full.
"""

from __future__ import annotations

import csv
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from .butrades import DATA_DIR, load

TAKER_BP = 2.0
ORDER_USD = 100.0
CAP_USD = 500.0  # how far inventory may drift from where it started, either way
LAT_MS = 200  # an order takes this long to arrive; trades before that can't fill it

TICK = {"BTCUSD": 0.01, "ETHUSD": 0.01, "BTCUSDT": 0.01,
        "USDTUSD": 0.00001, "USDCUSD": 0.00001, "USDCUSDT": 0.00001}
BASE = {"BTCUSD": "BTC", "ETHUSD": "ETH", "BTCUSDT": "BTC",
        "USDTUSD": "USDT", "USDCUSD": "USDC", "USDCUSDT": "USDC"}
QUOTE = {"BTCUSD": "USD", "ETHUSD": "USD", "BTCUSDT": "USDT",
         "USDTUSD": "USD", "USDCUSD": "USD", "USDCUSDT": "USDT"}
REF = {"USD": None, "BTC": "BTCUSD", "ETH": "ETHUSD", "USDT": "USDTUSD", "USDC": "USDCUSD"}
SYMBOLS = list(TICK)

RESULTS_FILE = DATA_DIR / "resting_results.csv"
CURVES_FILE = DATA_DIR / "resting_curves.csv"


class Book:
    """Our estimate of one market's best bid and ask, from its trades."""

    __slots__ = ("bid", "ask", "tick", "last")

    def __init__(self, tick):
        self.bid = self.ask = None
        self.tick = tick
        self.last = None

    def trade(self, p, buyer_maker):
        self.last = p
        if buyer_maker:  # a seller hit the bid: the bid was here
            self.bid = p
            if self.ask is None or self.ask <= p:
                self.ask = p + self.tick
        else:
            self.ask = p
            if self.bid is None or self.bid >= p:
                self.bid = p - self.tick

    @property
    def mid(self):
        return (self.bid + self.ask) / 2 if self.bid is not None else None


class Order:
    __slots__ = ("price", "qty", "placed", "done")

    def __init__(self, price, qty, placed):
        self.price, self.qty, self.placed, self.done = price, qty, placed, 0.0


class Sim:
    """One strategy's orders, balances and fills under one fill rule."""

    def __init__(self, strategy, rule, books):
        self.s = strategy
        self.touch = rule == "touch"
        self.books = books
        self.orders = {}  # (symbol, side) -> Order
        self.bal = defaultdict(float)  # asset -> change from the start
        self.resting = self.taken = self.crossed = 0
        self.volume = self.fees = 0.0
        self.curve = []  # (ts, equity)
        self.peak = self.dd = 0.0
        self.max_inv = 0.0
        self.busy_ms = 0
        self.last_ts = None
        self.state = {}  # scratch space for the strategy

    # -- what the strategy may do --------------------------------------
    def rest(self, sym, side, price, qty, now):
        """Keep an order at `price` for `qty`, or cancel with price None."""
        key = (sym, side)
        b = self.books[sym]
        if price is None or qty <= 0 or b.bid is None:
            self.orders.pop(key, None)
            return
        tick = b.tick
        if side == "buy":
            price = min(np.floor(price / tick + 1e-9) * tick, b.ask - tick)
        else:
            price = max(np.ceil(price / tick - 1e-9) * tick, b.bid + tick)
        price = round(price, 8)
        o = self.orders.get(key)
        if o is None or abs(o.price - price) > tick / 2:
            self.orders[key] = Order(price, qty, now)  # replacing loses the queue spot
        else:
            o.qty = qty

    def take(self, sym, side, qty, now):
        """Fill `qty` immediately at the estimated best price, paying the fee."""
        b = self.books[sym]
        price = b.ask if side == "buy" else b.bid
        self._fill(sym, side, price, qty, TAKER_BP, now)
        self.taken += 1

    def inventory_usd(self, asset):
        return self.bal[asset] * self.ref_price(asset)

    def ref_price(self, asset):
        sym = REF[asset]
        if sym is None:
            return 1.0
        m = self.books[sym].mid
        return m if m is not None else 1.0

    # -- engine side ---------------------------------------------------
    def _fill(self, sym, side, price, qty, fee_bp, now):
        base, quote = BASE[sym], QUOTE[sym]
        sign = 1 if side == "buy" else -1
        self.bal[base] += sign * qty
        self.bal[quote] -= sign * qty * price
        fee = qty * price * fee_bp / 1e4
        self.bal[quote] -= fee
        self.fees += fee * self.ref_price(quote)
        self.volume += qty * price * self.ref_price(quote)

    def on_trade(self, sym, ts, p, q, buyer_maker):
        for side in ("buy", "sell"):
            o = self.orders.get((sym, side))
            if o is None or ts < o.placed + LAT_MS:
                continue
            left = o.qty - o.done
            same = abs(p - o.price) < self.books[sym].tick / 2
            if side == "buy":
                through, at = p < o.price and not same, same and buyer_maker
                crossed = through and not buyer_maker  # a buyer lifted an ask below our bid: it was never resting
            else:
                through, at = p > o.price and not same, same and not buyer_maker
                crossed = through and buyer_maker
            if through:
                got = left
            elif at and self.touch:
                got = min(left, q)
            else:
                continue
            o.done += got
            if o.done >= o.qty - 1e-12:
                del self.orders[(sym, side)]
            # our estimate of the other side was stale and the order would have
            # filled at once as an immediate order: charge it as one
            fee = TAKER_BP if crossed else 0.0
            self.resting += not crossed
            self.crossed += crossed
            self._fill(sym, side, o.price, got, fee, ts)
            self.s.on_fill(self, sym, side, o.price, got, ts)

    def mark(self, ts):
        eq = sum(self.bal[a] * self.ref_price(a) for a in self.bal)
        inv = sum(abs(self.bal[a]) * self.ref_price(a) for a in self.bal if a != "USD")
        self.max_inv = max(self.max_inv, inv)
        if self.last_ts is not None and inv > 1.0:
            self.busy_ms += ts - self.last_ts
        self.last_ts = ts
        self.peak = max(self.peak, eq)
        self.dd = max(self.dd, self.peak - eq)
        self.curve.append((ts, eq))
        return eq


# -- strategies ---------------------------------------------------------

class Strategy:
    name = ""
    symbols = ()
    capital = 2000.0  # $1,000 in dollars and $1,000 in the coin, so both sides can be quoted
    about = ""

    def quote(self, sim, now):
        pass

    def on_fill(self, sim, sym, side, price, qty, now):
        pass


class MakeBoth(Strategy):
    """Rest a buy and a sell around the market, hold the coin until the other side fills.

    `offset_bp` is the distance from the mid price; `beyond_ticks` instead
    places relative to the best price (negative improves it by that many
    ticks, 0 joins it, positive sits behind it).
    """

    def __init__(self, sym, offset_bp=None, beyond_ticks=None):
        self.sym = sym
        self.symbols = (sym,)
        self.offset_bp = offset_bp
        self.beyond_ticks = beyond_ticks
        base = BASE[sym]
        if offset_bp is not None:
            self.name = f"{sym} both sides {offset_bp:g} bp from mid"
            self.about = (f"Keep a buy order {offset_bp:g} bp below the mid price and a sell order "
                          f"{offset_bp:g} bp above it, re-placed as the price moves. A buy fill leaves "
                          f"{base} to be sold by the sell order later, and the reverse.")
        else:
            w = {-1: "one tick better than the best", 0: "at the best", 1: "one tick behind the best"}
            w = w.get(beyond_ticks, f"{beyond_ticks} ticks behind the best")
            self.name = f"{sym} both sides, {w}"
            self.about = (f"Keep a buy order {w} bid and a sell order {w} ask, following them as "
                          f"they move. A buy fill leaves {base} to be sold by the sell order later, "
                          f"and the reverse.")

    def prices(self, b):
        if self.offset_bp is not None:
            m = b.mid
            return m * (1 - self.offset_bp / 1e4), m * (1 + self.offset_bp / 1e4)
        k = self.beyond_ticks * b.tick
        return b.bid - k, b.ask + k

    def quote(self, sim, now):
        b = sim.books[self.sym]
        if b.mid is None:
            return
        bid, ask = self.prices(b)
        inv = sim.inventory_usd(BASE[self.sym])
        qty = ORDER_USD / b.mid
        sim.rest(self.sym, "buy", bid if inv < CAP_USD else None, qty, now)
        sim.rest(self.sym, "sell", ask if inv > -CAP_USD else None, qty, now)


class MakeAndClose(MakeBoth):
    """Rest both sides; the moment one fills, undo it with an immediate fill."""

    def __init__(self, sym, offset_bp):
        super().__init__(sym, offset_bp=offset_bp)
        self.name = f"{sym} {offset_bp:g} bp from mid, close at once"
        self.about = (f"Same orders as the {offset_bp:g} bp strategy, but the moment a resting order "
                      f"fills, undo it immediately at the best price, paying the 2 bp fee. Keeps "
                      f"{offset_bp:g} bp minus half the spread minus 2 bp if the price hasn't moved.")

    def on_fill(self, sim, sym, side, price, qty, now):
        sim.take(sym, "sell" if side == "buy" else "buy", qty, now)


class MakeWithStop(MakeBoth):
    """Rest both sides; if a fill isn't undone by the other side within `stop_s`, close it."""

    def __init__(self, sym, offset_bp, stop_s):
        super().__init__(sym, offset_bp=offset_bp)
        self.stop_s = stop_s
        self.name = f"{sym} {offset_bp:g} bp from mid, {stop_s:g} s stop"
        self.about = (f"Same orders as the {offset_bp:g} bp strategy, but if the other side hasn't "
                      f"filled within {stop_s:g} seconds of a fill, close what's left immediately "
                      f"at the 2 bp fee instead of holding it.")

    def quote(self, sim, now):
        base = BASE[self.sym]
        inv = sim.bal[base]
        since = sim.state.get("since")  # per sim, not per strategy: a strategy runs under both rules
        if abs(inv) * sim.ref_price(base) < 1.0:
            sim.state["since"] = None
        elif since is None:
            sim.state["since"] = now
        elif now - since > self.stop_s * 1000:
            sim.take(self.sym, "sell" if inv > 0 else "buy", abs(inv), now)
            sim.state["since"] = None
        super().quote(sim, now)


class TriangleRest(Strategy):
    """Rest on BTC/USDT at the price BTC/USD and USDT/USD imply, finish the loop immediately."""

    symbols = ("BTCUSDT", "BTCUSD", "USDTUSD")
    capital = 3000.0  # $1,000 each of dollars, USDT and BTC

    def __init__(self, offset_bp):
        self.offset_bp = offset_bp
        self.name = f"BTC/USDT rest {offset_bp:g} bp off the implied price"
        self.about = (f"BTC/USD divided by USDT/USD says what BTC/USDT should cost. Keep a buy order "
                      f"{offset_bp:g} bp under that and a sell order {offset_bp:g} bp over it. When one "
                      f"fills, immediately do the other two sides of the triangle at the best price "
                      f"(2 bp each) so the balances return to where they started.")

    def quote(self, sim, now):
        bu, bt = sim.books["BTCUSD"], sim.books["USDTUSD"]
        if bu.mid is None or bt.mid is None or sim.books["BTCUSDT"].mid is None:
            return
        implied = bu.mid / bt.mid
        qty = ORDER_USD / bu.mid
        sim.rest("BTCUSDT", "buy", implied * (1 - self.offset_bp / 1e4), qty, now)
        sim.rest("BTCUSDT", "sell", implied * (1 + self.offset_bp / 1e4), qty, now)

    def on_fill(self, sim, sym, side, price, qty, now):
        if sym != "BTCUSDT":
            return
        if side == "buy":  # paid USDT for BTC: sell the BTC for dollars, buy the USDT back
            sim.take("BTCUSD", "sell", qty, now)
            sim.take("USDTUSD", "buy", qty * price, now)
        else:
            sim.take("BTCUSD", "buy", qty, now)
            sim.take("USDTUSD", "sell", qty * price, now)


def strategies():
    out = []
    for sym in ("USDTUSD", "USDCUSD", "USDCUSDT"):
        for k in (-1, 0, 1):
            out.append(MakeBoth(sym, beyond_ticks=k))
    for s in (2, 5, 10, 25):
        out.append(MakeBoth("BTCUSD", offset_bp=s))
    out.append(MakeBoth("ETHUSD", offset_bp=5))
    out.append(MakeAndClose("BTCUSD", 5))
    out.append(MakeAndClose("BTCUSD", 10))
    out.append(MakeWithStop("BTCUSD", 5, 60))
    out.append(MakeWithStop("BTCUSD", 5, 600))
    out.append(TriangleRest(5))
    out.append(TriangleRest(10))
    return out


# -- replay -------------------------------------------------------------

def trades(days=None):
    """Every trade on every symbol, in time order: arrays ts, sym index, price, qty, buyer_maker."""
    parts = []
    for i, sym in enumerate(SYMBOLS):
        try:
            df = load(sym)
        except FileNotFoundError:
            print(f"no trades on disk for {sym}; strategies needing it will sit idle")
            continue
        df["sym"] = i
        parts.append(df)
    df = pd.concat(parts).sort_values("ts", kind="stable")
    if days:
        df = df[df.ts >= df.ts.max() - days * 86400_000]
    return df.reset_index(drop=True)


def run(strats, rules=("through", "touch"), days=None, mark_every_s=300, log=print):
    df = trades(days)
    ts, si, px, qty, bm = (df.ts.to_numpy(), df.sym.to_numpy(), df.price.to_numpy(),
                           df.qty.to_numpy(), df.buyer_maker.to_numpy())
    span_days = (ts[-1] - ts[0]) / 86400_000
    log(f"{len(df):,} trades over {span_days:.1f} days")
    books = {s: Book(TICK[s]) for s in SYMBOLS}
    sims = [Sim(s, r, books) for s in strats for r in rules]
    by_sym = defaultdict(list)
    for sim in sims:
        for s in sim.s.symbols:
            by_sym[SYMBOLS.index(s)].append(sim)
    next_mark = ts[0]
    t0 = time.time()
    for n in range(len(ts)):
        sym = SYMBOLS[si[n]]
        now = ts[n]
        books[sym].trade(px[n], bm[n])
        for sim in by_sym[si[n]]:
            sim.on_trade(sym, now, px[n], qty[n], bm[n])
            sim.s.quote(sim, now)
        if now >= next_mark:
            for sim in sims:
                sim.mark(now)
            next_mark = now + mark_every_s * 1000
        if n % 200_000 == 0 and n:
            log(f"  {n:,} trades, {time.time() - t0:.0f} s")
    rows = []
    for sim in sims:
        eq = sim.mark(ts[-1])
        # what it would be worth if everything were closed now with immediate fills
        closing_fee = sum(abs(sim.bal[a]) * sim.ref_price(a) for a in sim.bal if a != "USD") * TAKER_BP / 1e4
        s = sim.s
        rows.append({
            "strategy": s.name, "market": "+".join(s.symbols), "rule": "through" if not sim.touch else "touch",
            "about": s.about, "days": round(span_days, 2), "capital_usd": s.capital,
            "resting_fills": sim.resting, "immediate_fills": sim.taken, "crossed_fills": sim.crossed,
            "volume_usd": round(sim.volume, 2), "fees_usd": round(sim.fees, 4),
            "pnl_usd": round(eq, 4), "pnl_closed_usd": round(eq - closing_fee, 4),
            "usd_per_day": round(eq / span_days, 4) if span_days else 0.0,
            "pct_per_year": round(eq / span_days * 365 / s.capital * 100, 2) if span_days else 0.0,
            "max_drawdown_usd": round(sim.dd, 4), "max_inventory_usd": round(sim.max_inv, 2),
            "share_time_holding": round(sim.busy_ms / max(ts[-1] - ts[0], 1), 3),
            "end_balances": " ".join(f"{a} {v:+.6g}" for a, v in sorted(sim.bal.items()) if abs(v) > 1e-9),
        })
    res = pd.DataFrame(rows)
    curves = pd.DataFrame([(sim.s.name, "touch" if sim.touch else "through", t, round(e, 4))
                           for sim in sims for t, e in sim.curve],
                          columns=["strategy", "rule", "ts", "pnl_usd"])
    return res, curves


def check_quotes(days=2):
    """How far the trade-estimated bid/ask sits from the recorded one, per market, in bp."""
    from .crossex import TICKS_FILE
    tk = pd.read_csv(TICKS_FILE)
    tk = tk[tk.ex == "binanceus"]
    out = {}
    for sym in SYMBOLS:
        mk = sym.lower()
        t = tk[tk.market == mk].sort_values("ts")
        if t.empty:
            continue
        try:
            tr = load(sym)
        except FileNotFoundError:
            continue
        tr = tr[(tr.ts >= t.ts.min()) & (tr.ts <= t.ts.max())]
        b = Book(TICK[sym])
        est = []
        for p, m in zip(tr.price.to_numpy(), tr.buyer_maker.to_numpy()):
            b.trade(p, m)
            est.append((b.bid, b.ask))
        est = pd.DataFrame(est, columns=["ebid", "eask"])
        est["ts"] = tr.ts.to_numpy()
        # the recorded quote standing at each trade
        rec = pd.merge_asof(est, t[["ts", "bid", "ask"]].drop_duplicates("ts", keep="last"), on="ts")
        ok = rec.bid.notna()
        emid = (rec.ebid + rec.eask) / 2
        rmid = (rec.bid + rec.ask) / 2
        err = ((emid - rmid) / rmid * 1e4)[ok]
        est = rec
        out[sym] = {"trades": int(ok.sum()), "median_abs_bp": round(float(err.abs().median()), 3),
                    "p90_abs_bp": round(float(err.abs().quantile(0.9)), 3),
                    "bid_exact": round(float((est.ebid[ok] == rec.bid[ok]).mean()), 3),
                    "ask_exact": round(float((est.eask[ok] == rec.ask[ok]).mean()), 3)}
    return out


def results():
    if not RESULTS_FILE.exists():
        return None
    return pd.read_csv(RESULTS_FILE)


def curves(strategy):
    if not CURVES_FILE.exists():
        return None
    c = pd.read_csv(CURVES_FILE)
    return c[c.strategy == strategy]


def main():
    days = None
    args = sys.argv[1:]
    if args and args[0] == "--days":
        days = float(args[1])
        args = args[2:]
    if args and args[0] == "check":
        for sym, r in check_quotes().items():
            print(sym, r)
        return
    res, cv = run(strategies(), days=days)
    res.to_csv(RESULTS_FILE, index=False, quoting=csv.QUOTE_MINIMAL)
    cv.to_csv(CURVES_FILE, index=False)
    cols = ["strategy", "rule", "resting_fills", "immediate_fills", "crossed_fills", "fees_usd", "pnl_usd",
            "usd_per_day", "pct_per_year", "max_drawdown_usd", "max_inventory_usd"]
    with pd.option_context("display.width", 200, "display.max_rows", 100, "display.max_colwidth", 48):
        print(res[cols].to_string(index=False))


if __name__ == "__main__":
    main()
