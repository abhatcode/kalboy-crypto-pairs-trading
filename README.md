Kalboy — Crypto Statistical Arbitrage Pairs Trading Engine

A research and paper-trading system for statistical arbitrage on cryptocurrency pairs. Discovers cointegrated crypto pairs, trades their spread using a Kalman-filter-estimated dynamic hedge ratio, and validates the strategy with rigorous out-of-sample backtesting before any live paper trading.

What this is (and isn't)

This is a signal-generation and validation system, not an automated trading bot moving real capital. It identifies statistically valid mean-reverting relationships between crypto assets, sizes hypothetical positions using empirical edge estimates, and produces live paper-trading signals. Turning it into something that trades real money is a deliberate, separate future step.

Pipeline
Data — ~900 days of daily OHLCV for ~30 liquid crypto assets, fetched via ccxt (KuCoin).
Pair discovery (pair_pipeline.py) — PCA + clustering to narrow the search space, bidirectional Engle-Granger cointegration testing, Hurst exponent and half-life filtering, and a Benjamini-Hochberg FDR correction applied globally across all candidate pairs to control for false discoveries from testing many pairs at once. 37 pairs cleared a raw threshold; 11 survived FDR correction.
Signal generation (kalman_strategy.py) — a single shared KalmanState class (used identically by the backtester and the live bot, eliminating implementation drift) estimates each pair's dynamic hedge ratio. The trading z-score is computed from a rolling realized spread, not the filter's own internal prediction variance — an earlier version used the raw Kalman variance directly, which silently collapsed toward zero over long runs and produced spurious extreme signals; this was found, diagnosed, and fixed.
Backtesting (backtest.py) — strict in-sample/out-of-sample split (720 IS / 180 OOS days) with zero data leakage into pair selection. Reports both gross and net returns so trading-cost drag is always visible, not just the net number. Runs all surviving pairs concurrently in a single portfolio to correctly capture overlapping/netted exposure.
Position sizing (compute_kelly.py) — quarter-Kelly sizing computed from in-sample trade statistics only, floored at zero for negative-edge pairs, with an added haircut for pairs with thin trade counts. 5 of 11 FDR-surviving pairs cleared this bar.
Risk controls — a portfolio-level max asset-concentration cap (30%), a portfolio max-drawdown circuit breaker (15%), and a per-pair cointegration-breakdown exit that halts a pair if its spread volatility spikes past 3x its in-sample historical average.
Live paper trading (live_bot.py) — an asyncio-based multi-pair engine polling live prices via free CCXT REST calls (no paid WebSocket dependency, no exchange account required), sharing one PortfolioManager across concurrently running pairs.
Honest baseline result

After all corrections above, the raw strategy's out-of-sample performance across the 11 FDR-surviving pairs was mostly flat-to-negative net of trading costs — consistent with the expectation that a generic, untuned cointegration + Kalman pairs strategy does not hold a durable edge in crypto without additional signal or refinement. This is treated as a legitimate, reportable finding, not a failure.

Known limitations / explicit non-goals
Cointegration is verified once at pair-selection time, not continuously re-tested during live trading. A pair could in principle stop being cointegrated while still being traded; a rolling re-verification step is a natural extension, not yet implemented.
No real capital or real exchange account is connected. Paper trading is simulated locally.
Position sizing is based on in-sample statistics estimated from a finite, sometimes small, number of trades per pair — Kelly fractions should be interpreted with appropriate caution given sample size.
Status

Core pipeline (data → discovery → signal → backtest → sizing → risk controls) is complete and validated. Live paper trading has been run only as a short smoke test; a sustained live paper-trading run is the next step, to be run against the pushed repo.

Setup
powershell
py -3.11 -m venv venv311
venv311\Scripts\Activate.ps1
pip install -r requirements.txt

Requires Python 3.11 (backtrader is not compatible with Python 3.10+ stdlib changes carried through to 3.14).

Usage
powershell
python main.py --mode fetch      # pull historical data
python main.py --mode pipeline   # discover cointegrated pairs (FDR-corrected)
python main.py --mode backtest   # run portfolio backtest with sizing + risk controls
python live_bot.py                # start live paper trading
