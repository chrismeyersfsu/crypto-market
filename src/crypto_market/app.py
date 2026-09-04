"""FastAPI: one page, one JSON endpoint."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from . import backtest, data, funding, onchain

app = FastAPI(title="crypto-market")
STATIC = Path(__file__).parent / "static"

# What a Fidelity 401(k) BrokerageLink account can actually trade.
# FBTC/FETH carry the 0.25% expense ratio the strategy assumes.
TICKERS = [
    ("FBTC", "Fidelity Wise Origin Bitcoin Fund (0.25%)"),
    ("FETH", "Fidelity Ethereum Fund (0.25%)"),
    ("IBIT", "iShares Bitcoin Trust (0.25%)"),
    ("ETHA", "iShares Ethereum Trust (0.25%)"),
    ("BITO", "ProShares Bitcoin Strategy ETF (0.95%)"),
    ("GBTC", "Grayscale Bitcoin Trust (1.5%)"),
    ("BTC-USD", "Bitcoin spot (trades weekends)"),
    ("ETH-USD", "Ether spot (trades weekends)"),
    ("MSTR", "Strategy Inc"),
    ("COIN", "Coinbase"),
]


def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/tickers")
def tickers():
    return [{"symbol": s, "name": n} for s, n in TICKERS]


@app.get("/api/backtest")
def run_backtest(
    ticker: str = Query("FBTC"),
    start: str | None = None,
    end: str | None = None,
    expense_ratio: float = Query(0.25, ge=0, le=5, description="percent per year"),
    slippage: float = Query(0.0, ge=0, le=5, description="percent per side"),
    entry_dow: int = Query(3, ge=0, le=4, description="custom: buy at this weekday's close (Mon=0)"),
    exit_dow: int = Query(0, ge=0, le=4, description="custom: sell at this weekday's close"),
    split: str | None = Query(None, description="in-sample before this date, out-of-sample from it"),
    sma: int = Query(0, ge=0, le=400, description="trend filter: only long above this N-day average (0 = off)"),
    satellite: float = Query(25, ge=0, le=100, description="percent of the portfolio running the custom rule; the rest is held"),
    rule: str = Query("days", pattern="^(days|breakout)$", description="which custom rule: weekday->weekday or N-bar-high breakout"),
    breakout_n: int = Query(20, ge=2, le=250),
    hold_bars: int = Query(5, ge=1, le=60),
):
    try:
        df = data.load(ticker)
    except Exception as e:  # bad symbol, Yahoo hiccup
        raise HTTPException(400, f"{ticker}: {e}")
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    if len(df) < 10:
        raise HTTPException(400, f"{ticker}: only {len(df)} bars in range")

    er, sl = expense_ratio / 100, slippage / 100
    kw = dict(entry_dow=entry_dow, exit_dow=exit_dow, sma=sma, breakout_n=breakout_n, hold_bars=hold_bars)
    results = {s: backtest.run(df, s, er, sl, **kw) for s in ("buy_hold", "weekday", "weekend")}
    results["custom"] = backtest.run(df, "breakout" if rule == "breakout" else "custom", er, sl, **kw)
    # Core + satellite: two sleeves run side by side, never rebalanced.
    w = satellite / 100
    results["blend"] = {
        "equity": (1 - w) * results["buy_hold"]["equity"] + w * results["custom"]["equity"],
        "trades": results["custom"]["trades"],
    }
    dates = [d.date().isoformat() for d in df.index]
    series = {s: [round(v, 2) for v in r["equity"]] for s, r in results.items()}
    stats = {s: {k: _clean(v) for k, v in backtest.stats(r["equity"], r["trades"]).items()}
             for s, r in results.items()}
    wd, we = results["weekday"]["trades"], results["weekend"]["trades"]

    # Same equity paths, cut at the split: each half re-based to its own start
    # so the before/after stats are comparable.
    sample = None
    if split:
        cut = pd.Timestamp(split)
        sample = {}
        for s, r in results.items():
            eq, tr = r["equity"], r["trades"]
            halves = {"in": (eq[eq.index < cut], tr[tr["exit"] < split] if len(tr) else tr),
                      "out": (eq[eq.index >= cut], tr[tr["exit"] >= split] if len(tr) else tr)}
            sample[s] = {k: {kk: _clean(v) for kk, v in backtest.stats(e, t).items()}
                         for k, (e, t) in halves.items()}
    dow = backtest.day_of_week(df)
    return {
        "meta": df.attrs["meta"] | {"first": dates[0], "last": dates[-1], "bars": len(df)},
        "dates": dates,
        "close": [round(v, 4) for v in df["close"]],
        "equity": series,
        "stats": stats,
        "weekday_vs_weekend_t": _clean(backtest.welch_t(wd["ret"], we["ret"]) if len(wd) and len(we) else float("nan")),
        "day_of_week": [
            {"dow": int(i), "mean": _clean(float(r["mean"])), "median": _clean(float(r["median"])),
             "n": int(r["n"]), "win_rate": _clean(float(r["win_rate"]))}
            for i, r in dow.iterrows()
        ],
        "sample": sample,
        "trades": {s: results[s]["trades"].to_dict("records") for s in ("weekday", "weekend", "custom")},
    }


@app.get("/api/onchain")
def onchain_signal(
    z_thresh: float = Query(2.0, ge=0.5, le=5),
    hold_days: int = Query(7, ge=1, le=30),
    start: str | None = None,
    end: str | None = None,
    split: str | None = Query(None, description="in-sample before this date, out-of-sample from it"),
):
    d = onchain.load()
    btc = data.load("BTC-USD")["close"]
    btc.index = btc.index.normalize()
    if start:
        d = d[d.index >= pd.Timestamp(start)]
    if end:
        d = d[d.index <= pd.Timestamp(end)]
    if len(d) < 30:
        raise HTTPException(400, "not enough on-chain history in range")

    r = onchain.backtest(d, btc, z_thresh, hold_days)
    bh_eq = (btc.reindex(r["equity"].index).ffill() / btc.reindex(r["equity"].index).ffill().iloc[0])
    st = {k: _clean(v) for k, v in backtest.stats(r["equity"], r["trades"]).items()}

    sample = None
    if split:
        cut = pd.Timestamp(split)
        halves = {}
        for h, sub in (("in", d[d.index < cut]), ("out", d[d.index >= cut])):
            rr = onchain.backtest(sub, btc, z_thresh, hold_days) if len(sub) > 30 else None
            halves[h] = {k: _clean(v) for k, v in backtest.stats(rr["equity"], rr["trades"]).items()} if rr else {}
            halves[h]["fwd_return_spike_mean"] = _clean(rr["fwd_return_spike_mean"]) if rr else None
            halves[h]["fwd_return_base_mean"] = _clean(rr["fwd_return_base_mean"]) if rr else None
            halves[h]["signal_t_stat"] = _clean(rr["t_stat"]) if rr else None
            halves[h]["n_spikes"] = rr["n_spikes"] if rr else 0
        sample = halves

    return {
        "dates": [dt.date().isoformat() for dt in r["equity"].index],
        "equity_signal": [round(v, 2) for v in 10_000 * r["equity"]],
        "equity_hold": [round(v, 2) for v in 10_000 * bh_eq],
        "z": [round(v, 2) for v in d["z"].reindex(r["equity"].index)],
        "z_thresh": z_thresh,
        "stats": st,
        "n_spikes": r["n_spikes"],
        "fwd_return_spike_mean": _clean(r["fwd_return_spike_mean"]),
        "fwd_return_base_mean": _clean(r["fwd_return_base_mean"]),
        "signal_t_stat": _clean(r["t_stat"]),
        "sample": sample,
        "trades": r["trades"].to_dict("records"),
    }


@app.get("/api/funding")
def funding_series(
    start: str | None = None,
    end: str | None = None,
    capital_multiple: float = Query(2.0, ge=1, le=5, description="capital tied up per $1 of exposure (spot + short collateral)"),
):
    d = funding.daily()
    if start:
        d = d[d.index >= pd.Timestamp(start)]
    if end:
        d = d[d.index <= pd.Timestamp(end)]
    if len(d) < 2:
        raise HTTPException(400, "not enough funding history in range")

    carry = (1 + d["rate"]).cumprod()
    carry_adj = (1 + d["rate"] / capital_multiple).cumprod()  # same $ earned, more $ tied up
    spot = d["close"] / d["close"].iloc[0]

    def stats(eq: pd.Series) -> dict:
        yrs = (eq.index[-1] - eq.index[0]).days / 365.25
        cagr = eq.iloc[-1] ** (1 / yrs) - 1
        dly = eq.pct_change().dropna()
        vol = dly.std() * (365 ** 0.5) if len(dly) > 1 else float("nan")
        dd = (eq / eq.cummax() - 1).min()
        return {"cagr": _clean(cagr), "max_drawdown": _clean(dd), "volatility": _clean(vol),
                "sharpe": _clean(cagr / vol if vol else float("nan")), "final": _clean(10_000 * eq.iloc[-1])}

    by_year = []
    for y, g in d.groupby(d.index.year):
        ann = g["rate"] * 365
        by_year.append({"year": int(y), "mean_annualized": _clean(float(ann.mean())),
                         "min_annualized": _clean(float(ann.min())), "max_annualized": _clean(float(ann.max())),
                         "days": len(g), "days_negative": int((g["rate"] < 0).sum())})

    return {
        "dates": [dt.date().isoformat() for dt in d.index],
        "daily_rate": [round(v, 6) for v in d["rate"]],
        "annualized_rate": [round(v * 365, 4) for v in d["rate"]],
        "equity": {
            "carry": [round(10_000 * v, 2) for v in carry],
            "carry_adjusted": [round(10_000 * v, 2) for v in carry_adj],
            "spot": [round(10_000 * v, 2) for v in spot],
        },
        "stats": {"carry": stats(carry), "carry_adjusted": stats(carry_adj), "spot": stats(spot)},
        "by_year": by_year,
        "share_days_negative": _clean(float((d["rate"] < 0).mean())),
    }


def main():
    import uvicorn
    uvicorn.run("crypto_market.app:app", host="127.0.0.1", port=8870)
