# crypto-market

Do an exchange's own three prices for a coin ever disagree with each other,
for long enough to catch? And is the same coin ever cheaper on one
exchange than another?

    uv sync
    uv run crypto-market        # http://127.0.0.1:8870

The landing page is one card per analysis, each with its question and its
one-line finding; the analyses are separate pages, `/triangles`,
`/crossex`, `/resting`, `/search` and `/paper`, sharing `static/site.css`
and `static/site.js`.

## Triangles on one exchange (`/triangles`)

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

The same watcher runs against Binance.US, picked with the Venue box on
the page. Its triangles come from `exchangeInfo` (about sixty: every
coin listed in dollars and in BTC, USDT or USDC that is itself listed in
dollars; nothing there is quoted in ETH), and its prices from one
combined `bookTicker` stream carrying all ~115 markets, which sends the
best bid and ask and the size at each whenever a market's top of book
moves -- no full order book to keep. That stream is silent about a
market until it changes, so every market is seeded from the REST
`ticker/bookTicker` snapshot on connect. Files are `butri_samples.csv`
and `butri_episodes.csv`, same columns as the Coinbase ones. Fees are
per fill as before; Binance.US's schedule (read from binance.us/fees on
2026-09-05) is 0 bp for a resting order and 2 bp for an immediate fill
on every pair, BTC and stablecoin pairs included, with one "Tier 0" pair
(BNB/USD) at 1 bp, so a loop costs 6 bp -- a far lower bar than
Coinbase's, and the reason it is worth watching.

Kraken is the third venue. Its bridges are BTC, ETH, USDT, USDC and
the euro: Kraken lists EUR/USD as a market of its own and most of its
coins in euros as well as dollars, so a coin priced in both is a
triangle through EUR/USD, and that is where most of Kraken's ~660
triangles (1,200 markets) come from. Listings come from the REST
`AssetPairs` call, with Kraken's own names (XBT, XDG) turned into the
everyday ones (BTC, DOGE), which is also what its v2 stream wants.
Prices come from one WebSocket v2 connection on the `ticker` channel,
told to push on every change of the best bid or ask rather than only on
trades; it sends a snapshot of every market on subscribe, so nothing is
seeded, but the subscribe has to go out in batches of a couple of
hundred symbols or Kraken closes the connection. Files are
`krtri_samples.csv` and `krtri_episodes.csv`. Fees, read from
kraken.com/features/fee-schedule on 2026-09-05: 80 bp for an immediate
fill on an ordinary pair at the bottom tier (40 to rest an order),
falling to 10 bp at $10M a month, and 20 bp on the stablecoin and
currency pairs -- EUR/USD, USDT/USD, USDC/USD and the like -- so a loop
through BTC or ETH costs 240 bp and one through EUR, USDT or USDC 180
bp; which pairs sit on which schedule is read off `AssetPairs`, which
carries a fee table per pair. Kraken's API still reports the older 40 bp
bottom tier for ordinary pairs; the website's number is the default and
the box takes either.

## Across exchanges (`/crossex`)

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

## The BTC/USDT/USD triangle on the other venues (bottom of `/crossex`)

The same check for the one triangle every venue has: 30 days of minute
closes, and live from the recorded bids and asks, which is the honest one
-- the minute-close version reports "gaps" on thin BTC/USDT markets that
are just stale prints.

## Resting orders on Binance.US (`/resting`)

Binance.US charges nothing for an order that rests on the book and is
filled by somebody else's trade, so the loops above could in principle be
run at zero fee -- if the orders fill. A resting order only fills when
someone trades at your price, and the only honest test of that is to
replay the trades that actually happened. `butrades.py` pulls a month of
every aggregated trade on the six markets (`trades_binanceus_*.csv`,
resumable) and `resting.py` replays them through twenty strategies under
two fill rules that bracket the truth: "through", a trade printed beyond
your price so your level was eaten whatever your place in the queue, and
"touch", a trade at your price fills you as if you were first in line.
The best bid and ask at each moment are estimated from the trades
themselves (a seller hitting a bid says where the bid was); when that
estimate turns out stale and an order would have crossed the book, the
fill is charged the 2 bp immediate rate. The strategies: quote both sides
of a stablecoin pair one tick better than, at, or one tick behind the best
price; quote both sides of BTC/USD or ETH/USD 2-25 bp from the mid, held
until the other side fills, or closed at once, or closed after a 60 s or
600 s stop; and rest on BTC/USDT at the price BTC/USD and USDT/USD imply,
finishing the triangle with two immediate fills. Orders are $100, the
position may drift $500 either way, orders take 200 ms to arrive.
`uv run python -m crypto_market.resting` writes `resting_results.csv` and
the profit curves; `/resting` shows them.

## Strategy search (`/search`)

A wider net than triangles. `history.py` pulls candles (`data/hist/`,
resumable, `--workers`): Binance.US 1-minute for a year on eight markets,
hourly since 2025-02 and daily since 2021 for every USD market it lists;
Coinbase 1-minute on five, hourly and daily on the forty biggest. One
harness, `backtest.py`, scores every idea the same way: a strategy is the
share of the account to hold in the coin, decided at each bar's close and
held over the next bar, so it can only use what it could have known; spot
only, nothing sold that isn't held; every change of position pays the fee
plus half the bid-ask spread measured by this server's tick recorder; the
last 30% of the bars is held back and parameters are chosen on the first
70% only; buy-and-hold over the same bars is reported beside every row.
`score(..., late=1)` acts on each decision one bar later than the rule
says, the test that separates a signal from a fill-timing artifact.

Six families, each a module under `strategies/` writing
`data/strategies/<family>.csv`: trend (averages, crossovers, breakouts,
past-return rules on one coin; holding the coins that rose most across
the whole Binance.US list), reversion (dips by z-score, bands and RSI, one
big down bar, stablecoins off their peg), crossex (Coinbase's last minute
predicting Binance.US's next; Binance.US below Coinbase; the same on the
trade tape), flow (buyer-aggressor share, volume and trade-count spikes,
imbalance and big prints from the tape), pairs (switching between ETH,
BTC and SOL by which is cheap or which rose; pairs hunted by how fast
their spread reverts; USDT vs USDC), timing (hour of day, weekday, US
hours, weekends, volatility regimes and targeting, range squeezes), plus
recheck, the best rules acted on one bar late. About 1,600 recorded
variants; `uv run python -m crypto_market.strategies.<family>` re-runs
one. `/search` shows them all, with each family's reading.

Binance.US's own history has a hole: its API returns no daily, hourly or
minute candles between 2023-07-14 and 2025-02-19. `candles()` leaves
blank rows where data is missing instead of joining the two ends into one
bar, and `score()` skips them, so a held position rides through a hole
without a return and without a trade; each row's note says how many holes
it skipped. Coinbase's daily history is complete since 2021 and is used
where a long daily series matters.

`strategies/playbook.py` is the spot version of the approach in Pavel
Kycek's *The Algorithmic Crypto Playbook* (2025), from his public
descriptions of it (the book itself is not on hand): a basket of coins
with a cap per coin, at daily, 12-hour and 4-hour bars; long-term
momentum (price above its N-day average), moving-average crossovers, RSI
dips and breakouts, 44 variants; the in-sample-best of each rule type
combined with equal money, and all 44 combined with nothing picked; every
coin equally and BTC as yardsticks; the combination acted on one bar late;
year-by-year results in the note. The daily basket is the 15 Binance.US
USD coins with a median $100k a day that Coinbase lists, at Coinbase's
prices and Binance.US's costs; the 12-hour and 4-hour baskets are the 7
that trade in most hours. He also trades futures and bets on falls, which
spot cannot, so this is half of what he runs; and the coins on disk are
the ones that survived to 2026, which he says flatters this kind of test
several times over. The robustness rows (drop the biggest contributors
in-sample and held-back, re-pick each January, the finer settings grid)
are in the same family; `strategies/tweaks.py` is the paper run's four
rules with one thing changed at a time (drop a coin, measured bid-ask
gaps, act weekly, wait for a signal to hold, a majority vote, a BTC
filter, a trailing stop, sizing by volatility, other caps), family
`tweaks`; `strategies/params.py` is the same four rules with their
numbers changed one rule at a time (34 settings, each alone and each
swapped into the combination, scored with and without ZEC), family
`params`.

## Paper run (`/paper`)

The playbook's daily basket, run forward for six months from 2026-09-06
with pretend money: `paper.py`. A systemd user timer
(`crypto-market-paper.timer`, 00:10 UTC daily) runs `python -m
crypto_market.paper step`, which fetches the day's Coinbase closes, runs
the same four rules as the backtest (above the 50-day average, a 10/50
crossover, a 20-day breakout, an RSI-14 dip) on the same 15 coins, and
fills each changed signal at Binance.US's live best bid or ask from
`/api/v3/ticker/bookTicker`, plus the 2 bp fee. $10,000, $2,500 a rule,
each rule its own account (the backtest re-balanced between rules daily,
a small difference); no coin over 10% of a rule's money; lot sizes
ignored. `data/paper/` holds state.json, orders.csv and daily.csv (the
account, the basket held from day one, and BTC held from day one, valued
at each step). `paper.py start` opens the account; `status` prints what
the page shows. Six months is one sample, and the backtest gives no
expectation to test against: its 31% a year held back is ZEC's rise, and
without ZEC the same rules made -7% (see the findings). The run tests
whether live fills track the model and whether the rules make anything on
the other 14 coins; a ZEC-sized move in some coin would settle nothing.

## Findings so far

Across exchanges. BTC: the Coinbase/Bitstamp gap is real but small, about
1 bp, and only pays at zero fees -- every retail fee tier is 20-60x larger.
ETH looks the same (0.9 bp wobble). Binance.US charges 2 bp on everything but
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

Resting orders. Over 30 days of Binance.US trades (August 6 to September
5, 2026), every BTC and ETH strategy lost money, $30 to $1,400 a month on
$2,000: a resting order on a moving coin fills only when the price is
moving through it, so each fill is on the wrong side of the move by about
4 bp, and closing at once or on a timer just adds the 2 bp fee. The
stablecoin quotes are the one place the zero fee shows: USDT/USD quoted
one tick better than the best price made about $6 a month on $2,000
(about 4% a year) under the pessimistic rule and $14 under the optimistic
one, with a $2.50 drawdown; USDC/USD and USDC/USDT trade too little to
matter (a few thousand trades a month) and came out within a dollar of
zero. The USDT figure would also have been about 6% of that market's
volume, which the market would notice.

Strategy search. Over the year to September 2026 (hourly bars since
February 2025, daily since 2021), the single most useful result is a
warning: Binance.US's minute and hour bars are stale. BTC/USD has no
trade in 56% of minutes, ETH/USD 72%, so a bar's close is often the last
price carried forward, and any rule that buys "the dip" or "the cheaper
exchange" or "the cheap coin of a pair" is buying a print the next trade
corrects. That one effect produced every spectacular number: Coinbase's
last minute "predicting" Binance.US's next (correlation 0.22, which is
0.30 on minutes with no trades and -0.05 on busy ones), ETH-vs-BTC
switching at +60% a year (-38% acted on one bar late; +5% on the ETHBTC
market's own prices), pairs among thin coins at millions of percent a
year, and the "buy after a -2% hour" rule (+14% a year held back, -6% one
bar late). On the actual trade tape the cross-exchange edge is 0.5-1.8 bp
a round trip against 4.6 bp of cost, and aggressive buying predicts the
next minute by 0.7-2.9 bp, real but half the cost of acting on it.
Stablecoins stopped leaving their peg in April 2026, so the peg trade has
no held-back sample. Every clock, weekday and volatility rule that beat
holding did so by being out of the market during the falling first part
of the year and lagged holding during the rally; the fitted ones flip
sign between the two parts. Holding the coins that rose most over the
past week or month lost 80-99% out of sample in every variant. What is
left is the slow trend rules: on ETH at 4-hour bars, "price above its
168-bar average" and a 96/192 crossover beat holding in the held-back
five months (59% and 55% a year vs 38%, acted one bar late, 21-93
trades); on Binance.US daily bars, with the 2023-2025 hole skipped, the
same shape of rule is a wash (-1% a year median vs -12% holding). That is
a well-known effect, ETH is the best of three coins tried, and one rally
is one sample; it is the only thing here worth a second look, and it is
not a business.

The playbook. The one strategy that came out of a book rather than a
guess looked better than anything above, and then failed the check that
matters. On 15 coins at daily bars since 2021, equal money in the
in-sample-best of each rule type (above the 50-day average, a 10/50
crossover, a 20-day breakout, an RSI-14 dip, each coin capped at 10%)
made 31% a year in the held-back 2025-01 to 2026-09 (Sharpe 0.96, worst
dip -33%) while holding the same coins made -1% and BTC -8%; acted a day
late it made 27%; by year +37, -17, +76, +57, +50, +1, against a basket
that lost 74% in 2022. It passed the author's checks: neighbours, where
every average from 30 to 75 days makes 37-54% held back and the five
crossovers around 10/50 make 24-31%, while 100 and 200 days and the slow
crossovers lose money and all 44 variants averaged make 0%; dropping the
3 or 5 coins that contributed most in-sample (34% and 33% held back); and
re-choosing the picks each January from the years before, which chooses
the same four every year and makes 46% a year over 2023-2026 (+69, +57,
+50, +2). But the held-back return is one coin: ZEC rose from $58 to over
$1,000 in the held-back part and the rules held it the whole way; of the
56 points of account return the combination made held back, ZEC gave 61
and the other 14 coins together -5. Without ZEC the same four rules,
re-picked, made -7% a year held back (2025 -8%, 2026 -5%), and without
the three biggest held-back contributors -10%. The drop-coins check had
passed because it dropped the 2021-2024 winners, which said nothing about
2025. `strategies/tweaks.py` tries one change at a time and finds the
same thing from the other side: every change that raises the held-back
number (a 15% or 20% cap, holding a coin only when a majority of the
rules agree) does it by holding more ZEC, and every change that spreads
risk (a smaller cap, sizing by volatility, a trailing stop, holding
nothing while BTC is below its long average) lowers it; charging the
bid-ask gaps measured on 2026-09-06 costs 4 points a year, and dropping
the four coins with a gap of 15 bp or more gives -1% because ZEC is one
of them. Changing the rules' numbers (`strategies/params.py`: 34
settings across the four rule types, each alone and each swapped into
the combination) does not change the answer: with ZEC every combination
makes 11-36% held back, without it every one loses 4-15% a year, and the
only settings above zero without ZEC are the RSI dip rule on its own at
14 or 21 days (+5-7%), which is rarely in the market. What is left for
the rules is 2023 and 2024 (+76, +57), which
the walk-forward test chose rules for without seeing; what is against
them is 2025 without ZEC and 2026, the 12-hour and 4-hour versions (1-3%
a year held back against 28% for holding), and the survivor-only universe.
It is running forward on paper from 2026-09-06 (`/paper`), which is the
only remaining way to tell.