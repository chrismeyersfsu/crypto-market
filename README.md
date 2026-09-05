# crypto-market

Do Coinbase's own three prices for a coin ever disagree with each other,
for long enough to catch? And is the same coin ever cheaper on one
exchange than another?

    uv sync
    uv run crypto-market        # http://127.0.0.1:8870

## Coinbase triangles (the main page)

Every coin Coinbase lists both in dollars and in a bridge currency (BTC,
ETH or USDT) that is itself listed in dollars is a triangle -- about fifty
of them, a hundred loops counting both directions. The three prices should
agree: X/USD == X/Q x Q/USD. When they don't, dollars -> X -> Q -> dollars
(or the reverse) ends with more dollars than it started with, before fees.

`cbtri.py` holds one WebSocket connection to Coinbase's `level2_batch`
feed carrying all ~90 order books (about 300 updates a second) and
re-prices both loops of a triangle every time one of its three books
changes. It keeps two files, both rolling 48 hours: `cbtri_samples.csv`,
every loop's gross mismatch every 20 seconds, and `cbtri_episodes.csv`,
one row per unbroken stretch of a loop being positive -- when it opened,
how long it lasted, its peak and the dollars the thinnest book allowed at
the peak, and the quote that was standing 100, 250, 500 and 1000 ms after
it opened, which is what an order sent on seeing it would actually get.
The page ranks every loop, charts one, and lists its episodes; fees are
per fill with a separate rate for Coinbase's stable pairs, so a loop
through USDT costs two ordinary fees and one stable fee.

## Across exchanges (under "Everything else")

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

## The BTC/USDT/USD triangle on the other venues

The same check for the one triangle every venue has: 30 days of minute
closes, and live from the recorded bids and asks, which is the honest one
-- the minute-close version reports "gaps" on thin BTC/USDT markets that
are just stale prints.

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
before fees on every venue in both directions (-0.1 to -5 bp). On
Coinbase's full set, the loops that are ever positive are the ones through
BTC, ETH and USDT on the big coins, by a fraction of a basis point to
about 1.5 bp, on a few hundred dollars, for seconds at a time; the rest
are negative, some by 50-100 bp on coins nobody trades. Coinbase's
taker rate is 60 bp per fill at the bottom tier and 4 bp at the top
($400M a month), so a loop costs 12-180 bp; nothing seen so far comes
close to even the top tier.
