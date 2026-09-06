"""Run the daily playbook basket on paper: real prices, real timing, pretend money.

Once a day, just after the daily bar closes (00:00 UTC), this fetches the
day's Coinbase closes, runs the same signal code the backtest ran
(strategies/playbook.py), and turns any changed signal into an order. Each
order is filled at Binance.US's live best ask (buys) or best bid (sells) at
that moment, plus the 2 bp fee -- the backtest assumed a fill at the bar's
close, so the difference between the two is one of the things this run
measures. Four rules, $2,500 each, run as separate sub-accounts that never
rebalance between each other (the backtest averaged their returns daily,
which is a small difference and is noted on the page).

State lives in data/paper/: state.json (the accounts), orders.csv (every
order), daily.csv (the value of everything at each step, with "hold the
basket" and "hold BTC" yardsticks bought on day one). Nothing here trades
for real. Order sizes ignore Binance.US's minimum lot sizes.

    uv run python -m crypto_market.paper step     # once a day, from the timer
    uv run python -m crypto_market.paper status   # what the page shows
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

from . import backtest as bt, history
from .butrades import DATA_DIR
from .strategies import playbook as pb

PAPER_DIR = DATA_DIR / "paper"
STATE = PAPER_DIR / "state.json"
ORDERS = PAPER_DIR / "orders.csv"
DAILY = PAPER_DIR / "daily.csv"
CAPITAL = 10_000.0
DAYS = 180
FEE = bt.FEE_BP["binanceus"] / 1e4
CAP = 0.1
BOOK_URL = "https://api.binance.us/api/v3/ticker/bookTicker"

# the in-sample-best of each rule type from the backtest (data/strategies/playbook.csv, 1d)
RULES = {
    "above 50-day average": lambda close: pb.sig_long_momentum(close, 50, "1d"),
    "10/50-day crossover": lambda close: pb.sig_crossover(close, 10, 50, "1d"),
    "RSI-14 dip, buy <20 sell >60": lambda close: pb.sig_rsi_dip(close, 14, 20, 60, 0, "1d"),
    "20-day breakout": lambda close: pb.sig_breakout(close, 20, "1d"),
}
ORDER_COLS = ["ts", "date", "rule", "symbol", "side", "qty", "price", "usd", "fee_usd", "share_before", "share_after", "why"]


def _now():
    return datetime.now(timezone.utc)


def _load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else None


def _save_state(st):
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1))


def closes(coins):
    """Daily Coinbase closes for the basket's coins, columns named the Binance.US way."""
    return pd.DataFrame({s: bt.candles("coinbase", s[:-3] + "-USD", "1d").close for s in coins}).sort_index()


def refresh(coins):
    for s in coins:
        history.fetch("coinbase", s[:-3] + "-USD", "1d", 30)


def book(coins):
    """Binance.US best bid and ask right now, {symbol: (bid, ask)}."""
    rows = httpx.get(BOOK_URL, timeout=20).raise_for_status().json()
    have = {r["symbol"]: (float(r["bidPrice"]), float(r["askPrice"])) for r in rows}
    return {s: have[s] for s in coins if s in have and have[s][0] > 0 and have[s][1] > 0}


def start(coins=None):
    """Open the paper account on the backtest's universe with today's prices."""
    if STATE.exists():
        raise SystemExit(f"{STATE} exists; remove data/paper to start over")
    if coins is None:
        _, _, uni = pb.universe("1d")
        coins = uni.split(": ", 1)[1].split()
    refresh(coins)
    bk = book(coins)
    missing = [s for s in coins if s not in bk]
    if missing:
        raise SystemExit(f"no Binance.US book for {missing}")
    mid = {s: (b + a) / 2 for s, (b, a) in bk.items()}
    st = {
        "started": _now().isoformat(timespec="seconds"), "days": DAYS, "capital": CAPITAL, "coins": coins,
        "accounts": {rule: {"cash": CAPITAL / len(RULES), "holdings": {}, "signal": {}} for rule in RULES},
        "yardsticks": {  # bought once, on day one, at the same prices
            "basket": {s: CAPITAL / len(coins) / mid[s] for s in coins},
            "btc": {"BTCUSD": CAPITAL / mid["BTCUSD"]},
        },
        "last_bar": None, "steps": 0,
    }
    _save_state(st)
    _record_day(st, mid, _now(), "started")
    print(f"paper account opened: ${CAPITAL:,.0f} over {len(RULES)} rules on {len(coins)} coins")


def _value(acct, mid):
    return acct["cash"] + sum(q * mid[s] for s, q in acct["holdings"].items() if s in mid)


def _record_day(st, mid, when, what):
    new = not DAILY.exists()
    vals = {rule: _value(a, mid) for rule, a in st["accounts"].items()}
    total = sum(vals.values())
    basket = sum(q * mid[s] for s, q in st["yardsticks"]["basket"].items() if s in mid)
    btc = st["yardsticks"]["btc"]["BTCUSD"] * mid["BTCUSD"]
    in_market = sum(sum(q * mid[s] for s, q in a["holdings"].items() if s in mid) for a in st["accounts"].values()) / total
    with DAILY.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "date", "what", "total", *[f"rule_{i}" for i in range(len(RULES))], "basket", "btc", "in_market"])
        w.writerow([when.isoformat(timespec="seconds"), when.date().isoformat(), what, round(total, 2),
                    *[round(v, 2) for v in vals.values()], round(basket, 2), round(btc, 2), round(in_market, 4)])


def _order(f, when, rule, s, side, qty, price, before, after, why):
    usd = qty * price
    fee = usd * FEE
    f.writerow([when.isoformat(timespec="seconds"), when.date().isoformat(), rule, s, side, f"{qty:.8f}", price,
                round(usd, 2), round(fee, 4), round(before, 4), round(after, 4), why])
    return usd, fee


def step():
    """The daily step: new bar in, orders out, value recorded. Safe to run more than once a day."""
    st = _load_state()
    if st is None:
        raise SystemExit("no paper account; run `start` first")
    coins = st["coins"]
    refresh(coins)
    close = closes(coins)
    bar = str(close.index[-1].date())
    now = _now()
    if bar == st["last_bar"]:
        print(f"no new bar since {bar}; nothing to do")
        return
    bk = book(coins)
    mid = {s: (b + a) / 2 for s, (b, a) in bk.items()}
    if "BTCUSD" not in mid:
        raise SystemExit("no Binance.US book for BTCUSD; not stepping")

    new_orders = not ORDERS.exists()
    with ORDERS.open("a", newline="") as fh:
        f = csv.writer(fh)
        if new_orders:
            f.writerow(ORDER_COLS)
        for rule, sig_fn in RULES.items():
            acct = st["accounts"][rule]
            sig = sig_fn(close)
            tgt = pb.targets_from_signal(sig, close, CAP).iloc[-1]  # NaN = unchanged, else the share to hold
            equity = _value(acct, mid)
            held = {s: q * mid[s] / equity for s, q in acct["holdings"].items() if s in mid}
            want = {s: float(t) for s, t in tgt.items() if pd.notna(t) and s in mid}
            if st["steps"] == 0:  # day one: the account is empty, so buy everything whose signal is already on
                on = [s for s, v in sig.iloc[-1].items() if v and s in mid]
                want = {s: 1.0 / max(len(on), 1.0 / CAP) for s in on}
            # sells first, so the cash is there for the buys
            for s, t in sorted(want.items(), key=lambda kv: kv[1] - held.get(kv[0], 0.0)):
                cur = held.get(s, 0.0)
                delta_usd = (t - cur) * equity
                if abs(delta_usd) < 1.0:
                    continue
                bid, ask = bk[s]
                if delta_usd < 0:
                    qty = min(acct["holdings"].get(s, 0.0), -delta_usd / mid[s])
                    usd, fee = _order(f, now, rule, s, "sell", qty, bid, cur, t, "signal off" if t == 0 else "signal changed")
                    acct["cash"] += usd - fee
                    acct["holdings"][s] = acct["holdings"].get(s, 0.0) - qty
                    if acct["holdings"][s] <= 1e-12:
                        del acct["holdings"][s]
                else:
                    usd = min(delta_usd, acct["cash"] / (1 + FEE))  # what is bought comes out of cash, as in the backtest
                    if usd < 1.0:
                        continue
                    qty = usd / ask
                    usd, fee = _order(f, now, rule, s, "buy", qty, ask, cur, t,
                                      "signal on" + (" (short of cash, scaled down)" if usd < delta_usd - 1.0 else ""))
                    acct["cash"] -= usd + fee
                    acct["holdings"][s] = acct["holdings"].get(s, 0.0) + qty
            acct["signal"] = {s: bool(v) for s, v in sig.iloc[-1].items()}
    st["last_bar"] = bar
    st["steps"] += 1
    _save_state(st)
    _record_day(st, mid, now, f"bar {bar}")
    print(f"stepped on bar {bar}: total ${sum(_value(a, mid) for a in st['accounts'].values()):,.2f}")


def status():
    """What the page shows: the run so far, holdings, orders, and the backtest to compare against."""
    st = _load_state()
    if st is None:
        return {"running": False}
    daily = pd.read_csv(DAILY) if DAILY.exists() else pd.DataFrame()
    orders = pd.read_csv(ORDERS) if ORDERS.exists() else pd.DataFrame(columns=ORDER_COLS)
    started = datetime.fromisoformat(st["started"])
    day = (_now() - started).days
    coins = st["coins"]
    try:
        mid = {s: (b + a) / 2 for s, (b, a) in book(coins).items()}
    except Exception:  # noqa: BLE001 -- the page still shows the last recorded value
        mid = None
    live = None
    if mid and "BTCUSD" in mid:
        vals = {rule: _value(a, mid) for rule, a in st["accounts"].items()}
        live = {"total": round(sum(vals.values()), 2), "rules": {k: round(v, 2) for k, v in vals.items()},
                "basket": round(sum(q * mid[s] for s, q in st["yardsticks"]["basket"].items() if s in mid), 2),
                "btc": round(st["yardsticks"]["btc"]["BTCUSD"] * mid["BTCUSD"], 2)}
    holdings = []
    for rule, a in st["accounts"].items():
        for s, q in a["holdings"].items():
            holdings.append({"rule": rule, "symbol": s, "qty": q, "usd": round(q * mid[s], 2) if mid and s in mid else None})
    # what the backtest would lead you to expect over this many days
    bt_row = None
    try:
        r = bt.results()
        r = r[(r.family == "playbook") & (r.interval == "1d") & (r.strategy == "combined: best of each rule type") & (r.late != 1)]
        bt_row = r.iloc[0].to_dict() if len(r) else None
    except Exception:  # noqa: BLE001
        pass
    return {
        "running": True, "started": st["started"], "day": day, "days": st["days"], "capital": st["capital"],
        "coins": coins, "rules": list(RULES), "last_bar": st["last_bar"], "steps": st["steps"],
        "live": live, "holdings": holdings, "cash": {k: round(a["cash"], 2) for k, a in st["accounts"].items()},
        "signals": {k: sorted(s for s, on in a["signal"].items() if on) for k, a in st["accounts"].items()},
        "daily": daily.to_dict(orient="records"), "orders": orders.to_dict(orient="records"),
        "backtest": {k: bt_row[k] for k in ("oos_ann_pct", "oos_sharpe", "oos_maxdd_pct", "bh_oos_ann_pct", "oos_from", "to")} if bt_row else None,
    }


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "start":
        start()
    elif cmd == "step":
        step()
    elif cmd == "status":
        s = status()
        s.pop("daily", None); s.pop("orders", None)
        print(json.dumps(s, indent=1, default=str))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
