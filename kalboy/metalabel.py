"""Meta-labeling experiment: predict at entry which trades will make money."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from kalboy.stats import trade_stats

FEATURES = ['f_abs_z', 'f_z_velocity', 'f_vol_ratio', 'f_innovation_sd', 'f_beta_drift', 'half_life',
            'zscore_window', 'hurst', 'p_fdr', 'formation_trades', 'formation_mean_return', 'kelly',
            'hour_utc', 'btc_vol_1d', 'direction']
MIN_TRAIN = 150
THRESHOLD = 0.5


def models():
    return {
        'logistic': make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05,
                                                        subsample=0.8, random_state=0),
    }


def run(trades, windows):
    trades = trades.copy()
    trades['entry_time'] = pd.to_datetime(trades['entry_time'])
    trades['exit_time'] = pd.to_datetime(trades['exit_time'])
    X = trades[FEATURES].astype(float).fillna(0.0).values
    y = (trades['return'] > 0).astype(int).values
    starts = pd.to_datetime(windows.set_index('window')['start'])

    out = {}
    for name, proto in models().items():
        prob = np.full(len(trades), np.nan)
        for w, start in starts.items():
            test = (trades['window'] == w).values
            train = (trades['exit_time'] < start).values
            if not test.any() or train.sum() < MIN_TRAIN or len(set(y[train])) < 2:
                continue
            model = models()[name]
            model.fit(X[train], y[train])
            prob[test] = model.predict_proba(X[test])[:, 1]
        scored = ~np.isnan(prob)
        taken = scored & (prob > THRESHOLD)
        r = trades['return'].values
        out[name] = {
            'scored_trades': int(scored.sum()),
            'auc': float(roc_auc_score(y[scored], prob[scored])) if scored.sum() > 10 and len(set(y[scored])) == 2 else None,
            'all_scored': trade_stats(r[scored]),
            'taken': trade_stats(r[taken]),
            'skipped': trade_stats(r[scored & ~taken]),
        }
        trades[f'p_{name}'] = prob
    return out, trades
