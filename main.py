import os
import argparse
import sys
from data.fetch_data import fetch_historical_data
from pair_pipeline import run_pipeline
from backtest import run_all_backtests

def main():
    parser = argparse.ArgumentParser(description="Kalboy Crypto Pairs Trading Orchestrator")
    parser.add_argument('--mode', type=str, default='all', choices=['fetch', 'pipeline', 'backtest', 'all'],
                        help="Execution mode (default: all - fetches data, runs pipeline, and runs backtests)")
    
    parser.add_argument('--exchange', type=str, default='binance', help='CCXT exchange ID')
    parser.add_argument('--symbols', type=str, nargs='+', help='List of symbols (default: top 10 USDT pairs)')
    parser.add_argument('--timeframe', type=str, default='1d', help='Timeframe (e.g., 1d, 1h)')
    parser.add_argument('--days', type=int, default=730, help='Days of historical data to pull')
    parser.add_argument('--datadir', type=str, default='data', help='Data folder')
    parser.add_argument('--test_days', type=int, default=180, help='Number of days for Out-of-Sample testing')
    
    parser.add_argument('--n_components', type=int, default=3, help='PCA components')
    parser.add_argument('--eps', type=float, default=1.0, help='DBSCAN epsilon parameter')
    parser.add_argument('--min_samples', type=int, default=2, help='DBSCAN min_samples parameter')
    parser.add_argument('--hurst', type=float, default=0.5, help='Max Hurst exponent')
    parser.add_argument('--max_hl', type=int, default=100, help='Max half-life in days')
    parser.add_argument('--resdir', type=str, default='results', help='Results folder')

    args = parser.parse_args()

    os.makedirs(args.datadir, exist_ok=True)
    os.makedirs(args.resdir, exist_ok=True)

    print("Kalboy Crypto Pairs Trading System")

    if args.mode in ['fetch', 'all']:
        print("Stage 1: Fetching historical OHLCV data...")
        fetch_historical_data(
            exchange_id=args.exchange,
            symbols=args.symbols,
            timeframe=args.timeframe,
            days=args.days,
            output_dir=args.datadir
        )

    if args.mode in ['pipeline', 'all']:
        print("Stage 2: Running Pairs Selection Pipeline...")
        combined_close_path = os.path.join(args.datadir, "combined_close.csv")
        if not os.path.exists(combined_close_path):
            print(f"Error: Combined closing prices not found at {combined_close_path}. Please fetch data first.")
            sys.exit(1)
            
        import pandas as pd
        df_combined = pd.read_csv(combined_close_path, index_col=0, parse_dates=True)
        split_idx = max(1, len(df_combined) - args.test_days)
        split_date = df_combined.index[split_idx]
        print(f"Splitting data at {split_date.date()} (OOS Window: {args.test_days} days)")
            
        run_pipeline(
            data_path=combined_close_path,
            n_components=args.n_components,
            eps=args.eps,
            min_samples=args.min_samples,
            hurst_h=args.hurst,
            max_hl=args.max_hl,
            outdir=args.resdir,
            train_end=split_date
        )

    if args.mode in ['backtest', 'all']:
        print("Stage 3: Running Backtests on Selected Pairs...")
        selected_pairs_path = os.path.join(args.resdir, "selected_pairs.csv")
        if not os.path.exists(selected_pairs_path):
            print(f"No selected pairs file found at {selected_pairs_path}. Cannot backtest.")
            sys.exit(1)
            
        if 'split_date' not in locals():
            combined_close_path = os.path.join(args.datadir, "combined_close.csv")
            import pandas as pd
            df_combined = pd.read_csv(combined_close_path, index_col=0, parse_dates=True)
            split_idx = max(1, len(df_combined) - args.test_days)
            split_date = df_combined.index[split_idx]
            print(f"Testing on data from {split_date.date()} onwards (OOS Window: {args.test_days} days)...")
            
        run_all_backtests(
            selected_pairs_csv=selected_pairs_path,
            data_dir=args.datadir,
            results_dir=args.resdir,
            split_date=split_date
        )
        
    print("Kalboy Run Complete")

if __name__ == "__main__":
    main()
