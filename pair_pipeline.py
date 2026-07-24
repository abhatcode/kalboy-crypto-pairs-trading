import os
import argparse
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts

def calculate_hurst(ts_data):
    ts_data = np.asarray(ts_data)
    if len(ts_data) < 100:
        return 0.5
    
    lags = range(2, 100)
    tau = [np.sqrt(np.std(np.subtract(ts_data[lag:], ts_data[:-lag]))) for lag in lags]
    
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    
    return poly[0] * 2.0

def calculate_half_life(spread):
    spread = np.asarray(spread)
    spread_lag = spread[:-1]
    spread_diff = np.diff(spread)
    
    spread_lag_const = sm.add_constant(spread_lag)
    model = sm.OLS(spread_diff, spread_lag_const)
    results = model.fit()
    
    if len(results.params) < 2:
        return np.nan
        
    beta = results.params[1]
    
    if beta >= 0:
        return np.nan
        
    half_life = -np.log(2) / beta
    return half_life

def find_cointegrated_pairs(df, clusters, labels, coint_threshold=0.05, hurst_threshold=0.5, max_half_life=100):
    asset_names = df.columns.tolist()
    
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == -1:
            continue
            
        cluster_assets = [asset_names[i] for i, l in enumerate(labels) if l == label]
        n_cluster_assets = len(cluster_assets)
        
        if n_cluster_assets < 2:
            continue
            
        print(f"Testing cointegration in cluster {label} with {n_cluster_assets} assets: {cluster_assets}")
        
        for i in range(n_cluster_assets):
            for j in range(i + 1, n_cluster_assets):
                asset_a = cluster_assets[i]
                asset_b = cluster_assets[j]
                
                series_a = df[asset_a]
                series_b = df[asset_b]
                
                coint_t_ab, p_value_ab, _ = ts.coint(series_a, series_b)
                coint_t_ba, p_value_ba, _ = ts.coint(series_b, series_a)
                
                p_pair = max(p_value_ab, p_value_ba)
                
                candidate_tests.append({
                    'asset_a': asset_a,
                    'asset_b': asset_b,
                    'p_pair': p_pair,
                    'p_value_ab': p_value_ab,
                    'p_value_ba': p_value_ba,
                    'cluster': label,
                    'series_a': series_a,
                    'series_b': series_b
                })
                
    pairs = []
    if not candidate_tests:
        return pd.DataFrame(pairs)
        
    p_values = [t['p_pair'] for t in candidate_tests]
    reject, pvals_corrected, _, _ = sm.stats.multipletests(p_values, alpha=coint_threshold, method='fdr_bh')
    
    print(f"FDR Correction applied to {len(p_values)} total candidate pairs (alpha={coint_threshold}).")
    print(f"Pairs surviving FDR correction: {sum(reject)}")
    
    for idx, test in enumerate(candidate_tests):
        if reject[idx]:
            asset_a = test['asset_a']
            asset_b = test['asset_b']
            series_a = test['series_a']
            series_b = test['series_b']
            
            x = sm.add_constant(series_b)
            model = sm.OLS(series_a, x)
            results = model.fit()
            
            alpha = results.params.iloc[0] if hasattr(results.params, 'iloc') else results.params[0]
            beta = results.params.iloc[1] if hasattr(results.params, 'iloc') else results.params[1]
            
            spread = series_a - (beta * series_b + alpha)
            
            hurst_val = calculate_hurst(spread)
            half_life_val = calculate_half_life(spread)
            
            if hurst_val < hurst_threshold and not np.isnan(half_life_val) and 1 <= half_life_val <= max_half_life:
                pairs.append({
                    'asset_a': asset_a,
                    'asset_b': asset_b,
                    'p_value_ab': test['p_value_ab'],
                    'p_value_ba': test['p_value_ba'],
                    'p_pair': test['p_pair'],
                    'p_fdr': pvals_corrected[idx],
                    'beta': beta,
                    'alpha': alpha,
                    'hurst': hurst_val,
                    'half_life': half_life_val,
                    'cluster': test['cluster']
                })
                print(f"  -> Found valid pair: {asset_a} & {asset_b} | p-fdr: {pvals_corrected[idx]:.4f} | Hurst: {hurst_val:.3f} | Half-life: {half_life_val:.1f} days")
                        
    return pd.DataFrame(pairs)

def run_pipeline(data_path="data/combined_close.csv", n_components=3, eps=1.0, min_samples=2, coint_p=0.05, hurst_h=0.5, max_hl=100, outdir="results", train_end=None):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Combined close price data not found at {data_path}. Run fetch_data.py first.")
        
    df = pd.read_csv(data_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    
    if train_end is not None:
        df = df.loc[:train_end]
        print(f"Data sliced to {train_end} for training phase (Rows: {len(df)})")
    
    returns = df.pct_change().dropna()
    
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns)
    
    pca = PCA(n_components=n_components)
    pca.fit(scaled_returns)
    
    asset_pca_loadings = pca.components_.T
    
    loadings_scaler = StandardScaler()
    scaled_loadings = loadings_scaler.fit_transform(asset_pca_loadings)
    
    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(scaled_loadings)
    labels = db.labels_
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"PCA shape: {asset_pca_loadings.shape} | DBSCAN identified {n_clusters} clusters (excluding noise)")
    
    for cluster_id in set(labels):
        assets_in_c = [df.columns[i] for i, l in enumerate(labels) if l == cluster_id]
        print(f"  Cluster {cluster_id}: {assets_in_c}")
        
    selected_pairs_df = find_cointegrated_pairs(
        df=df,
        clusters=labels,
        labels=labels,
        coint_threshold=coint_p,
        hurst_threshold=hurst_h,
        max_half_life=max_hl
    )
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    if not selected_pairs_df.empty:
        out_path = os.path.join(outdir, "selected_pairs.csv")
        selected_pairs_df.to_csv(out_path, index=False)
        print(f"Saved {len(selected_pairs_df)} selected pairs to {out_path}")
    else:
        print("No pairs passed the cointegration, Hurst, and half-life filters.")
        
    return selected_pairs_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pairs selection pipeline using PCA, Clustering, and Cointegration.')
    parser.add_argument('--data', type=str, default='data/combined_close.csv', help='Path to combined close CSV')
    parser.add_argument('--n_components', type=int, default=3, help='Number of PCA components')
    parser.add_argument('--eps', type=float, default=1.0, help='DBSCAN epsilon parameter')
    parser.add_argument('--min_samples', type=int, default=2, help='DBSCAN min_samples parameter')
    parser.add_argument('--coint_p', type=float, default=0.05, help='Max p-value for cointegration')
    parser.add_argument('--hurst', type=float, default=0.5, help='Max Hurst exponent')
    parser.add_argument('--max_hl', type=int, default=100, help='Max half-life in days')
    parser.add_argument('--outdir', type=str, default='results', help='Output directory for results')
    
    args = parser.parse_args()
    
    run_pipeline(
        data_path=args.data,
        n_components=args.n_components,
        eps=args.eps,
        min_samples=args.min_samples,
        coint_p=args.coint_p,
        hurst_h=args.hurst,
        max_hl=args.max_hl,
        outdir=args.outdir
    )
