"""One module per strategy family; each has run() writing data/strategies/<family>.csv via backtest.record.

READINGS is each family's honest one-paragraph reading of its own results,
written after the run (2026-09-05) and shown on the page next to the table.
"""

READINGS = {
    "trend": "Slow trend rules on one coin (price above its 168-bar average, a 96/192 crossover, at 4 hours) kept "
             "most of their result when acted on a bar late and beat holding ETH in the held-back part (55-59% vs 38%), "
             "but ETH was the best of three coins and the held-back window is one 5-month rally; on daily bars over "
             "five years the same rules mostly just sat out the falling stretches (+5% a year median vs -23% holding). "
             "Holding the 3-10 coins that rose most over the past week or month lost 80-99% out of sample in every variant.",
    "reversion": "Buying dips (z-score, bands, RSI) beat holding the coin in under 10% of variants and made 7-8% a year "
                 "while the coin made 96%. The one rule that looked right, buy after a -2% hour on ETH, flips from +14% "
                 "to -6% held back when acted on one bar late: it was buying a stale print. The stablecoin peg trade "
                 "made 47% a year in the first 70% of the year and 0.4% in the last 30% because USDC stopped leaving $1 "
                 "after April 2026 (69 bars below 0.9995 in five months).",
    "crossex": "Coinbase's last minute predicts Binance.US's next minute with correlation 0.22, and it is fake: Binance.US "
               "BTC/USD has no trade in 56% of minutes, so its close is a carried-forward price that catches up when someone "
               "trades. The correlation is 0.30 on minutes with no trades and -0.05 on busy ones; on the actual trades the "
               "edge is 0.5-1.8 bp a round trip against 4.6 bp of cost.",
    "flow": "Aggressive buying does predict the next minute a little (0.7-2.9 bp a round trip, every contrarian variant "
            "negative), which is about half the cost of trading it; all 36 variants lose. The volume-spike rules that beat "
            "holding did so by being in the market 20-30% of the time during a rally, and their sibling settings are all over the place.",
    "pairs": "Every switch-between-coins result is the bid-ask bounce on stale prints: ETH/BTC 'hold the cheap side' goes from "
             "+60% a year to -38% when acted on a bar late, and the same rule on the ETHBTC market's own prices makes 5%. "
             "The hunted pairs (ZIL/USD trades in 25% of hours) show millions of percent a year, which is the tell, not a result.",
    "timing": "Every clock or volatility rule that beat holding did so by being out of the market during the falling first "
              "part; in-sample and held-back Sharpe flip sign on the same rules (US hours: -1.2 then +2.0). Picking the "
              "best 4 hours of the day costs 50% a year in fees at Binance.US and everything at Coinbase.",
    "recheck": "The rules that looked best, acted on one bar late. The slow trend rules keep their result; the dip rule loses it.",
}
