# crypto-market

Is the same coin ever cheaper on one exchange than another, for long enough
to catch?

    uv sync
    uv run crypto-market        # http://127.0.0.1:8870

Markets: BTC/USD, and the stablecoins against the dollar and each other
(USDT/USD, USDC/USD, USDC/USDT). The stablecoins are the interesting case
for a small account: their price barely moves, so what's left is purely
the venues disagreeing with each other, and several venues charge far less
on these pairs than on BTC.

Two datasets feed the question. Thirty days of 1-minute closes from
Coinbase, Bitstamp, Bitfinex and Binance.US (all public APIs; Kraken's
public candles only keep twelve hours) feed a "see the gap this minute,
trade it the next, unwind when it closes" rule that pays a fee on all four
fills (buy and sell on both venues, then unwind both). A fee sweep shows
what fee per fill it would take to break even.

For the question minute bars can't answer -- how many milliseconds a gap
actually lasts -- the server runs a live tick recorder while it's up: one
WebSocket connection per venue carrying every market it lists. Coinbase
and Binance.US supply a ticker; Kraken, Bitfinex and Bitstamp are read
off their order-book streams because their ticker channels are throttled.
Each tick records the best bid and ask and the size on offer at each, so
an episode reports not just how wide the gap was but how many dollars it
was good for. Ticks land in `data/ticks.csv`, a rolling 48-hour window,
and are turned into executable episodes against a latency you supply.
Collector downtime is detected from the union of all markets' ticks, so a
stablecoin quote sitting unchanged for minutes is not mistaken for an
outage.

Findings so far. BTC: the Coinbase/Bitstamp gap is real but small, about
1 bp, and only pays at zero fees -- every retail fee tier is 20-60x larger.
Binance.US charges nothing on BTC but holds about $165 at its best price.
Stablecoins: the venues sit at steady offsets from each other (Bitfinex
about 10 bp above everyone, Binance.US about 1 bp above Coinbase and
Bitstamp) and wobble around those by 0.3-0.5 bp a minute, less than BTC
does; a steady offset can be collected once, not repeatedly.
