Kalboy — Crypto Statistical Arbitrage Pairs Trading Engine

A research and paper-trading system for statistical arbitrage on cryptocurrency pairs. Discovers cointegrated crypto pairs, trades their spread using a Kalman-filter-estimated dynamic hedge ratio, and validates the strategy with rigorous out-of-sample backtesting before any live paper trading.

What this is (and isn't)

This is a signal-generation and validation system, not an automated trading bot moving real capital. It identifies statistically valid mean-reverting relationships between crypto assets, sizes hypothetical positions using empirical edge estimates, and produces live paper-trading signals. Turning it into something that trades real money is a deliberate, separate future step.

Pipeline
Data — ~900 days of daily OHLCV for ~30 liquid crypto assets, fetched via ccxt (KuCoin).
Pair discovery (pair_pipeline.py) — PCA + clustering to narrow the search space, bidirectional Engle-Granger cointegration testing, and a Benjamini-Hochberg FDR correction applied across every pair tested, followed by Hurst exponent and half-life filtering on the survivors. Of 300 pairs tested within the cluster, 4 survived correction at alpha 0.05.
Signal generation (kalman_strategy.py) — a single shared KalmanState class (used identically by the backtester and the live bot, eliminating implementation drift) estimates each pair's dynamic hedge ratio. The trading z-score is computed from a rolling realized spread, not the filter's own internal prediction variance — an earlier version used the raw Kalman variance directly, which silently collapsed toward zero over long runs and produced spurious extreme signals; this was found, diagnosed, and fixed.
Backtesting (backtest.py) — strict in-sample/out-of-sample split (720 IS / 180 OOS days) with zero data leakage into pair selection. Reports both gross and net returns so trading-cost drag is always visible, not just the net number. Runs all surviving pairs concurrently in a single portfolio to correctly capture overlapping/netted exposure.
Position sizing (compute_kelly.py) — quarter-Kelly sizing computed from in-sample trade statistics only, floored at zero for negative-edge pairs and capped at full Kelly before the quarter scaling is applied, with an added haircut for pairs with thin trade counts. 1 of the 4 FDR-surviving pairs cleared this bar; the other three carried negative in-sample edge and size to zero.
Risk controls — a portfolio-level max asset-concentration cap (30%), a portfolio max-drawdown circuit breaker (15%), and a per-pair cointegration-breakdown exit that halts a pair if its spread volatility spikes past 3x its in-sample historical average.
Live paper trading (live_bot.py) — an asyncio-based multi-pair engine polling live prices via free CCXT REST calls (no paid WebSocket dependency, no exchange account required), sharing one PortfolioManager across concurrently running pairs.
Honest baseline result

After all corrections above, pair discovery yields 4 statistically surviving pairs, only one of which carries positive in-sample edge and therefore trades. Its out-of-sample result over the 180-day window is +0.75% net of costs across 8 round trips, with the portfolio drawdown breaker never triggered.

A single tradeable pair is not a portfolio, and a sub-1% return on one pair is noise rather than demonstrated edge. The honest conclusion is that a 27-asset universe collapsing into one cluster does not contain enough genuine cointegration to survive a correction across 300 simultaneous tests. A generic, untuned cointegration + Kalman pairs strategy does not hold a durable edge in crypto without additional signal or refinement. The route forward is a larger or better-segmented universe, not a looser statistical threshold. This is treated as a legitimate, reportable finding, not a failure.

Correctness fixes

Five defects were found and corrected in the statistical and backtesting path. Each one flattered the reported results, so all prior numbers in this README have been regenerated.

FDR correction was only ever shown pre-filtered pairs. Pair discovery discarded every pair whose raw p-value exceeded 0.05 before passing the list to Benjamini-Hochberg, so the correction only saw candidates that a hand-chosen cutoff had already selected. On the current dataset the effect was total: 300 pairs were tested, 44 passed the raw cutoff, and Benjamini-Hochberg then rejected the null for all 44. The raw threshold performed the entire selection and the correction confirmed everything handed to it, which is not multiple-testing control in any meaningful sense. Every tested p-value is now recorded and passed into the correction, and the returned rejection array alone determines which pairs survive. Surviving pairs fell from 40 to 4. The now-inert coint_p command-line flag was removed rather than left in the help text implying control it no longer had.

Kelly fractions could exceed full leverage. Raw Kelly was floored at zero but never capped before the quarter-Kelly scaling, so a pair with a raw Kelly of 18.5 produced a final fraction of 4.6, or 460% of portfolio value allocated to a single pair. Raw Kelly is now capped at 1.0 before scaling, bounding any pair at 25%.

The portfolio backtest was not out-of-sample. The out-of-sample start date was hardcoded to 2025-07-25 while the dataset's actual split falls on 2026-03-29. Out-of-sample trading therefore began eight months early and ran through the in-sample period, meaning every previously reported portfolio return included data the pairs were selected on. The split is now derived from the test_days argument the same way the main orchestrator derives it, so the two cannot drift apart.

The burn-in diagnostic was never applied. The diagnostic printed a recommended per-pair Kalman burn-in and instructed the operator to write it into the pair file by hand. No run had ever done so, so every backtest silently used the 7-bar default and the diagnostic had no effect on any result. Recommended values are now written into the pair file before sizing and backtesting.

Data fetching contained an unreachable duplicated loop header. A duplicated while/try block in the pagination path was removed.

Known limitations / explicit non-goals
Cointegration is verified once at pair-selection time, not continuously re-tested during live trading. A pair could in principle stop being cointegrated while still being traded; a rolling re-verification step is a natural extension, not yet implemented.
No real capital or real exchange account is connected. Paper trading is simulated locally.
Position sizing is based on in-sample statistics estimated from a finite, sometimes small, number of trades per pair — Kelly fractions should be interpreted with appropriate caution given sample size.
The burn-in convergence detector fires on the first run of consecutive small changes in the estimated beta, which can register a temporary flat stretch as convergence. On the current pair set it recommends 22 bars for FIL/SHIB while that pair's beta continues drifting until roughly bar 65. The recommendation is used as generated rather than overridden by hand; tightening the detector is the correct fix and is not yet implemented.
Only one pair currently clears both the corrected significance test and the positive-edge sizing bar, so portfolio-level concentration and netting behaviour is presently untested against a realistic number of concurrent positions.
Status

Core pipeline (data, discovery, signal, backtest, sizing, risk controls) is complete, and all reported figures have been regenerated on frozen data after the correctness fixes above, so the data itself is not a variable in the before and after comparison. Live paper trading has been run only as a short smoke test; a sustained live paper-trading run is the next step, to be run against the pushed repo.

Setup
powershell
py -3.11 -m venv venv311
venv311\Scripts\Activate.ps1
pip install -r requirements.txt

Requires Python 3.11 (backtrader is not compatible with Python 3.10+ stdlib changes carried through to 3.14).

Usage
powershell
python main.py --mode fetch                      # pull historical data
python main.py --mode pipeline --test_days 180   # discover pairs (FDR-corrected across all tests)
python burnin_diagnostic.py                      # recommend per-pair Kalman burn-in, then write it into selected_pairs.csv
python compute_kelly.py                          # in-sample Kelly sizing on the surviving pairs
python backtest.py --portfolio --test_days 180   # portfolio backtest with sizing + risk controls
python live_bot.py                               # start live paper trading

Run the four discovery and validation stages in that order. Each one consumes the previous stage's output, so changing pair selection invalidates the sizing and backtest results downstream of it.
