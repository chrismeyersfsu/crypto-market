# crypto-market

Is BTC ever cheaper on one exchange than another, for long enough to catch?

    uv sync
    uv run crypto-market        # http://127.0.0.1:8870

Two datasets feed the question. Thirty days of 1-minute closes from
Coinbase, Bitstamp, Bitfinex and Binance.US (all public APIs) feed a "see the gap this
minute, trade it the next, unwind when it closes" rule that pays a fee on
all four fills (buy and sell on both venues, then unwind both). A fee sweep
shows what fee per fill it would take to break even.

For the question minute bars can't answer -- how many milliseconds a gap
actually lasts -- the server runs a live tick recorder while it's up: the
Coinbase and Binance.US ticker channels plus order-book WebSocket feeds
from Kraken, Bitfinex and Bitstamp (their ticker channels are throttled,
so top-of-book is read off the order book instead). Each tick records the
best bid and ask and the size on offer at each, so an episode reports not
just how wide the gap was but how many dollars it was good for. Ticks land
in `data/ticks_btcusd.csv`, a rolling 48-hour window, and are turned into
executable episodes against a latency you supply.

Finding so far: the Coinbase/Bitstamp gap is real but small, about 1 bp,
and only pays at zero fees -- every retail fee tier is 20-60x larger than
that.
