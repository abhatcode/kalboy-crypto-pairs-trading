# Kalboy

Kalman-filter pairs trading on crypto perpetual futures. The code selects pairs of
KuCoin Futures contracts whose prices are cointegrated, tracks each pair's hedge ratio
with a Kalman filter, and trades the spread when it moves more than two standard
deviations from its mean. Strategies are tested with a walk-forward backtest net of
fees, slippage and funding. The chosen configuration was run once on a holdout period
and is now paper trading.

No configuration had a statistically significant edge after costs, and the final
configuration lost 1.16% on the holdout. REPORT.md has the method and results.

## Code

kalboy/ is the package. data.py downloads market data, selection.py picks pairs,
kalman.py and calibration.py fit them, engine.py and execution.py simulate trading,
walkforward.py runs the backtest and paper.py runs it on live data. stats.py,
ledger.py, report.py and tables.py produce the statistics and tables. Strategy settings
are JSON files in configs/, and tests/ has 44 tests that run on synthetic data.

## Running it

Tested with Python 3.14.2.

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    python -m pytest tests -n 8
    python -m kalboy fetch --timeframe 1h --start 2024-08-05 --out data/kucoinf_1h
    python -m kalboy walkforward --config configs/h1_tail.json
    python -m kalboy tables

Paper trading, which resumes from its saved state after a restart:

    powershell -ExecutionPolicy Bypass -File scripts\start_paper.ps1 -Config configs\h1_tail.json -Label paper_h1_tail

Set KALBOY_WEBHOOK_URL to a Discord webhook URL to receive trade alerts.
