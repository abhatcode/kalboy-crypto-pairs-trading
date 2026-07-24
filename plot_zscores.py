import os
import pandas as pd
import backtrader as bt
import matplotlib.pyplot as plt
from datetime import datetime
from kalman_strategy import KalmanPairs
from backtest import run_test

def main():
    import sys
    pair_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    data_dir = 'data'
    pairs_df = pd.read_csv('results/selected_pairs.csv')
    print(f"Running full history Z-score extraction for {pairs_df.iloc[pair_idx]['asset_a']} and {pairs_df.iloc[pair_idx]['asset_b']}...")
    pair_row = pairs_df.iloc[pair_idx]

    res = run_test(
        asset_y=pair_row['asset_a'],
        asset_x=pair_row['asset_b'],
        initial_beta=pair_row['beta'],
        initial_alpha=pair_row['alpha'],
        burn_in=pair_row['burn_in'],
        data_dir=data_dir,
        results_dir='results',
        start_date=None,
        end_date=None,
        trade_start_date=None,
        run_label='ZSCORE_EXTRACT_'
    )

    strat = res['strat']
    zscores = strat.zscore_history
    
    zscores = [z for z in zscores if z != 0.0]
    
    plt.figure(figsize=(10, 6))
    plt.hist(zscores, bins=50, color='skyblue', edgecolor='black')
    plt.title('Z-Score Distribution (LTC/USDT vs AVAX/USDT) over Full History')
    plt.xlabel('Z-Score')
    plt.ylabel('Frequency')
    plt.axvline(x=2.0, color='r', linestyle='--', label='Entry Threshold')
    plt.axvline(x=-2.0, color='r', linestyle='--')
    plt.axvline(x=0.5, color='g', linestyle='--', label='Exit Threshold')
    plt.axvline(x=-0.5, color='g', linestyle='--')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = 'results/zscore_histogram.png'
    plt.savefig(out_path)
    print(f"Saved Z-Score histogram to {out_path}")
    print(f"Min Z-Score: {min(zscores):.2f}, Max Z-Score: {max(zscores):.2f}")

if __name__ == '__main__':
    main()
