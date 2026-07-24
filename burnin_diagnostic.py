import os
import pandas as pd
import numpy as np
from kalman_strategy import KalmanState
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def check_convergence(log_df, col_name, threshold=0.01, consecutive_bars=5):
    val_range = log_df[col_name].max() - log_df[col_name].min()
    if val_range == 0:
        return 0
        
    diffs = log_df[col_name].diff().abs()
    is_small_change = diffs <= (val_range * threshold)
    
    rolling_sum = is_small_change.rolling(window=consecutive_bars).sum()
    converged = rolling_sum[rolling_sum == consecutive_bars]
    
    if len(converged) > 0:
        return int(converged.index[0] - consecutive_bars + 1)
        
    return len(log_df)

def run_diagnostic():
    results_dir = 'results'
    data_dir = 'data'
    pairs_file = os.path.join(results_dir, 'selected_pairs.csv')
    combined_path = os.path.join(data_dir, 'combined_close.csv')
    
    if not os.path.exists(pairs_file) or not os.path.exists(combined_path):
        print("Required files not found. Please run the data and pipeline stages first.")
        return

    pairs_df = pd.read_csv(pairs_file)
    df_combined = pd.read_csv(combined_path, index_col=0, parse_dates=True)
    is_start_df = df_combined.iloc[:100]
    
    print(f"Running burn-in diagnostic on the first 100 bars of the dataset for ALL pairs.")
    
    burn_in_values = []
    fig, axes = plt.subplots(len(pairs_df), 1, figsize=(10, 3 * len(pairs_df)))
    if len(pairs_df) == 1:
        axes = [axes]
    
    for idx, row in pairs_df.iterrows():
        asset_y = row['asset_a']
        asset_x = row['asset_b']
        beta = row['beta']
        alpha = row['alpha']
        
        kf = KalmanState(initial_beta=beta, initial_alpha=alpha)
        
        logs = []
        
        for step in range(len(is_start_df)):
            y_val = is_start_df[asset_y].iloc[step]
            x_val = is_start_df[asset_x].iloc[step]
            
            if np.isnan(y_val) or np.isnan(x_val):
                continue
                
            kf.update(y_val, x_val)
            logs.append({'bar': step + 1, 'beta': kf.beta, 'alpha': kf.alpha})
            
        log_df = pd.DataFrame(logs).set_index('bar')
        
        if not log_df.empty:
            beta_conv = check_convergence(log_df, 'beta', threshold=0.01, consecutive_bars=5)
            final_burn_in = max(15, min(100, beta_conv + 5))
            burn_in_values.append(final_burn_in)
            print(f"[{asset_y} vs {asset_x}] Converged at bar {beta_conv} -> Recommended burn_in: {final_burn_in}")
            
            axes[idx].plot(log_df.index, log_df['beta'], label=f'Beta ({asset_y}/{asset_x})')
            axes[idx].axvline(x=final_burn_in, color='r', linestyle='--', label=f'Recommended Burn-in ({final_burn_in})')
            axes[idx].set_title(f"Burn-in Convergence: {asset_y} vs {asset_x}")
            axes[idx].legend()
        else:
            burn_in_values.append(30)
            
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'burnin_convergence.png')
    plt.savefig(plot_path)
    print(f"\nSaved convergence plots to {plot_path}. Please review them manually.")
    print("If they look correct, manually update 'burn_in' column in selected_pairs.csv.")

if __name__ == "__main__":
    run_diagnostic()
