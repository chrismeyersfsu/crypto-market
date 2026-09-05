"""One module per strategy family; each has run() writing data/strategies/<family>.csv via backtest.record.

READINGS is each family's honest one-paragraph reading of its own results,
written after the run (2026-09-05) and shown on the page next to the table.
"""

READINGS = {
    "trend": "Slow trend rules on one coin (price above its 168-bar average, a 96/192 crossover, at 4 hours) kept "
             "most of their result when acted on a bar late and beat holding ETH in the held-back part (55-59% vs 38%), "
             "but ETH was the best of three coins and the held-back window is one 5-month rally. On Binance.US daily "
             "bars the same rules are a wash: -1% a year median vs -12% holding in the held-back part, which is "
             "2025-02 to 2026-09 because Binance.US's daily history has a hole from 2023-07 to 2025-02 that is skipped. "
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
    "playbook": "The spot half of Pavel Kycek's Algorithmic Crypto Playbook, from his public descriptions of it: simple rules "
                "on a basket of coins with a cap per coin, at daily, 12-hour and 4-hour bars, the rule types combined. On "
                "15 Coinbase-priced coins at daily bars since 2021, equal money in the in-sample-best of each rule type "
                "(above the 50-day average, a 10/50 crossover, a 20-day breakout, an RSI-14 dip) made 30% a year in the "
                "held-back part (Sharpe 0.9, worst dip -33%) while holding the same coins made -1% and BTC -8%; acted a "
                "day late it made 26%; by year +37, -17, +76, +57, +50, +1, against a basket that lost 74% in 2022. "
                "Three things count against it. The neighbours fail: the 100- and 200-day averages and the three other "
                "crossovers lose money, and all 44 variants averaged make -1%, so the result rests on picks, not on a "
                "family that works. The 12-hour and 4-hour versions make 1-3% a year held back against 28% for holding. "
                "And the universe is the coins that survived to 2026, which the author himself says flatters this kind "
                "of test several times over. No short selling and no futures here, so half of what he runs is missing.",
}
