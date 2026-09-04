# crypto-market

Backtests "buy during the week, sell for the weekend" against buy-and-hold on
the crypto ETFs a Fidelity 401(k) BrokerageLink account can trade (FBTC, FETH,
IBIT, …) and on BTC/ETH spot.

    uv sync
    uv run crypto-market        # http://127.0.0.1:8870

Two tabs: the weekday/weekend backtester above, and a funding-rate-arbitrage
tab (own spot BTC, short an equal-size perpetual future, collect the funding
rate — price exposure cancels). Funding history comes from Deribit's public
API (hourly, back to Apr 2019 — the only free source with this much of it)
and is cached to `data/funding_btc_perpetual.csv`, appended to on refresh.

Daily bars come from Yahoo Finance (no key) and are cached under `data/` for
six hours. Three rules are marked to market every bar:

- **buy_hold** — first open to last close
- **weekday** — Mon open → Fri close (first/last trading day of the week), flat over the weekend
- **weekend** — Fri close → Mon open, flat during the week
- **custom** — either close of one weekday to close of another, once a week
  (default Thu → Mon; wraps the weekend when exit ≤ entry), or a breakout:
  in at the close that sets a new N-day high, out H trading days later

A trend filter (long only above an N-day average) can gate any rule, and a
"core + satellite" line shows most of the money held with a slice in the
custom rule, never rebalanced.

An optional split date reports every rule's stats before and after it, so a
rule picked on old data can be tracked forward.

Costs: $0 commission, optional per-side slippage, and an expense ratio accrued
per calendar day held. ETF prices already embed their own expense ratio; the
input exists so spot BTC/ETH can be compared as if held through a 0.25% fund.
