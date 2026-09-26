import os
import pandas as pd
import numpy as np
from datetime import datetime
from backtest import run_test

def compute_kelly_and_vol_for_pairs(data_dir='data', res_dir='results', test_days=180):
    pairs_file = os.path.join(res_dir, 'selected_pairs.csv')
    if not os.path.exists(pairs_file):
        print(f"File {pairs_file} not found.")
        return
    
    pairs_df = pd.read_csv(pairs_file)
    combined_close_path = os.path.join(data_dir, "combined_close.csv")
    df_combined = pd.read_csv(combined_close_path, index_col=0, parse_dates=True)
    
    split_idx = max(1, len(df_combined) - test_days)
    split_date = df_combined.index[split_idx]
    is_end = split_date - pd.Timedelta(days=1)
    
    print(f"Computing Kelly fractions on IS data (up to {is_end.date()})...")
    
    results = []
    
    for idx, row in pairs_df.iterrows():
        print(f"\nProcessing IS backtest for {row['asset_a']} vs {row['asset_b']}...")
        burn_in_val = int(row.get('burn_in', 7))
        
        res_is = run_test(
            asset_y=row['asset_a'],
            asset_x=row['asset_b'],
            initial_beta=row['beta'],
            initial_alpha=row['alpha'],
            burn_in=burn_in_val,
            data_dir=data_dir,
            results_dir=res_dir,
            start_date=None,
            end_date=is_end,
            run_label='TEMP_IS_'
        )
        
        if res_is is None:
            continue
            
        strat = res_is['strat']
        
        trade_returns = strat.trade_returns
        num_trades = len(trade_returns)
        
        if num_trades > 1:
            mean_ret = np.mean(trade_returns)
            var_ret = np.var(trade_returns, ddof=1)
            
            if var_ret > 0:
                raw_kelly = mean_ret / var_ret
            else:
                raw_kelly = 0.0
        else:
            raw_kelly = 0.0
            
        kelly = min(max(0.0, raw_kelly), 1.0)
        kelly *= 0.25
        
        haircut = min(1.0, num_trades / 30.0)
        kelly *= haircut
        
        spread_stds = strat.all_spread_stds
        is_spread_vol_avg = np.mean(spread_stds) if len(spread_stds) > 0 else 0.0
        
        row_dict = row.to_dict()
        row_dict['kelly_fraction'] = kelly
        row_dict['is_spread_vol_avg'] = is_spread_vol_avg
        row_dict['is_trade_count'] = num_trades
        results.append(row_dict)
        
        print(f"Trades: {num_trades}, Raw Kelly: {raw_kelly:.4f}, Final Kelly: {kelly:.4f}, IS Vol: {is_spread_vol_avg:.4f}")
        
    out_df = pd.DataFrame(results)
    out_file = os.path.join(res_dir, 'selected_pairs_with_kelly.csv')
    out_df.to_csv(out_file, index=False)
    print(f"\nSaved {len(out_df)} pairs with Kelly to {out_file}")

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    compute_kelly_and_vol_for_pairs()
