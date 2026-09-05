# crypto-market

Is the same coin ever cheaper on one exchange than another, for long enough
to catch? And, within one exchange, do its own prices ever disagree with
each other?

    uv sync
    uv run crypto-market        # http://127.0.0.1:8870

## Across exchanges

Markets: BTC/USD, ETH/USD, BTC/USDT, and the stablecoins against the dollar
and each other (USDT/USD, USDC/USD, USDC/USDT). The stablecoins are the
interesting case for a small account: their price barely moves, so what's
left is purely the venues disagreeing with each other, and several venues
charge far less on these pairs than on BTC.

Venues: Coinbase, Kraken, Bitstamp, Bitfinex, Binance.US, and for BTC and
ETH one Uniswap v3 pool on the Base chain (cbBTC/USDC and WETH/USDC), read
every two seconds over public JSON-RPC. The pool is a venue like any other
in the live table, with its 5 bp pool fee shown as its spread and its size
being how many coins you can trade before moving its price by 1 bp.

Two datasets feed the question. Thirty days of 1-minute closes from
Coinbase, Bitstamp, Bitfinex and Binance.US (all public APIs; Kraken's
public candles only keep twelve hours, and the pool has no candles) feed a
"see the gap this minute, trade it the next, unwind when it closes" rule
that pays a fee on all four fills (buy and sell on both venues, then unwind
both). A fee sweep shows what fee per fill it would take to break even.

For the question minute bars can't answer -- how many milliseconds a gap
actually lasts -- the server runs a live tick recorder while it's up: one
WebSocket connection per venue carrying every market it lists. Every
venue is read off its order-book stream (Coinbase's ticker only fires on a
trade, so it goes stale on thin markets; the others throttle their ticker
channels). Each tick records the best bid and ask and the size on offer at
each, so an episode reports not just how wide the gap was but how many
dollars it was good for. Ticks land in `data/ticks.csv`, a rolling 48-hour
window, and are turned into executable episodes against a latency you
supply. Collector downtime is detected from the union of all markets'
ticks, so a stablecoin quote sitting unchanged for minutes is not mistaken
for an outage.

## Within one exchange (triangles)

On one venue, BTC/USD should equal BTC/USDT × USDT/USD. When it doesn't,
three trades in a loop (dollars → BTC → USDT → dollars, or the reverse)
would end with more dollars than they started with. The triangle section
checks this two ways: 30 days of minute closes, and live from the recorded
bids and asks, which is the honest one -- the minute-close version reports
"gaps" on thin BTC/USDT markets that are just stale prints.

A separate scan lists every triangle Coinbase offers (every coin that
trades against USD and also against BTC, ETH or USDT -- about fifty),
reads the best bid and ask of all three legs once a minute, and keeps the
results in `data/triscan_coinbase.csv`. The table shows each loop's gross
mismatch now, its median and best over the window, how often it exceeded
the fee you supply, and how many dollars the smallest of the three books
would let through.

## Findings so far

Across exchanges. BTC: the Coinbase/Bitstamp gap is real but small, about
1 bp, and only pays at zero fees -- every retail fee tier is 20-60x larger.
ETH looks the same (0.9 bp wobble). Binance.US charges nothing on BTC but
holds about $165 at its best price. Stablecoins: the venues sit at steady
offsets from each other (Bitfinex about 10 bp above everyone, Binance.US
about 1 bp above Coinbase and Bitstamp) and wobble around those by
0.3-0.5 bp a minute, less than BTC does; a steady offset can be collected
once, not repeatedly. The Base pool sits 4-5 bp away from the exchanges in
both directions -- exactly its own fee -- with about $2-3k of depth within
1 bp.

Triangles. Using real bids and asks, the BTC/USDT/USD loop loses money
before fees on every venue in both directions (-0.1 to -5 bp). Across all
fifty Coinbase triangles the best routes hover at zero to +1 bp with a few
hundred dollars of size; most are negative, some by 50-100 bp on illiquid
coins. Nothing there clears even Coinbase's cheapest fee.
