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
                "(above the 50-day average, a 10/50 crossover, a 20-day breakout, an RSI-14 dip) made 31% a year in the "
                "held-back part (Sharpe 0.96, worst dip -33%) while holding the same coins made -1% and BTC -8%; acted a "
                "day late it made 27%; by year +37, -17, +76, +57, +50, +1. It passed the author's neighbours check (every "
                "average from 30 to 75 days makes 37-54% held back) and his walk-forward check (re-choosing the picks each "
                "January chooses the same four every year, 46% a year over 2023-2026). It fails the one that matters: the "
                "held-back return is one coin. ZEC rose from $58 to over $1,000 in the held-back part and the rules held it "
                "the whole way; of the 56 points of account return the combination made held back, ZEC gave 61 and the "
                "other 14 coins together -5. Without ZEC the same four rules, re-picked, made -7% a year held back (2025 -8%, "
                "2026 -5%); without the three biggest held-back contributors, -10%. The check that dropped the biggest "
                "in-sample contributors (SOL, DOGE, ADA) passed because those were the 2021-2024 winners, which said nothing "
                "about 2025. The tweaks family confirms it: every change that raises the held-back number (a bigger cap, a "
                "majority vote) does so by holding more ZEC, and every change that spreads risk (a smaller cap, sizing by "
                "volatility, a trailing stop) lowers it. The 2023 and 2024 years (+76, +57), which the walk-forward test "
                "chose rules for without seeing them, are the case for the rules; 2025 without ZEC and 2026 are the case "
                "against. It is running forward on paper from 2026-09-06, which is the only remaining way to tell.",
    "tweaks": "The paper run's four rules with one thing changed at a time, on the same 15 coins. Dropping ZEC turns the "
              "held-back 31% a year into -7%, and the changes sort by how much ZEC they hold: a 15% or 20% cap makes 43% "
              "and 48%, holding a coin only when 2 or 3 of the 4 rules agree makes 42-84%, and all of those are -8% to -14% "
              "without ZEC; a cap of one fifteenth makes 21%, sizing by volatility 22-31%, a 15% or 25% trailing stop 0-9%, "
              "holding nothing while BTC is under its 100- or 200-day average 6% and -10%. Charging the bid-ask gaps "
              "measured on Binance.US on 2026-09-06 instead of the 5 bp guess costs 4 points a year (28%); leaving out the "
              "four coins with a gap of 15 bp or more (ATOM 204, ALGO 46, LTC 17, ZEC 15) gives -1%, because ZEC is one of "
              "them. Reading the signals once a week makes 17%; waiting two or three days before acting on one 27% and 23%. "
              "Nothing here improves the strategy; the rows show what the held-back number is made of.",
}
