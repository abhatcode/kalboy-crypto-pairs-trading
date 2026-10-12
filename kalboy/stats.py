"""Trade and equity statistics, including the probabilistic and deflated Sharpe."""
import math

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def trade_stats(returns, n_boot=10_000, seed=0):
    r = np.asarray(returns, dtype=float)
    n = len(r)
    out = {'n_trades': n}
    if n == 0:
        return out
    out.update(win_rate=float((r > 0).mean()), mean_return=float(r.mean()), median_return=float(np.median(r)),
               total_pnl_share=float(r.sum()))
    if n < 2:
        return out
    sd = float(r.std(ddof=1))
    t = float(r.mean() / sd * math.sqrt(n)) if sd > 0 else float('nan')
    boot = np.random.default_rng(seed).choice(r, size=(n_boot, n), replace=True).mean(axis=1)
    worst = np.sort(r)[: max(1, n // 20)]
    out.update(std_return=sd, t_stat=t,
               p_value_one_sided=float(stats.t.sf(t, n - 1)) if sd > 0 else float('nan'),
               mean_ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
               worst_5pct_sum=float(worst.sum()), rest_sum=float(r.sum() - worst.sum()))
    return out


def probabilistic_sharpe(returns, benchmark_sr=0.0):
    """Bailey and Lopez de Prado (2012)."""
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n < 3 or r.std(ddof=1) == 0:
        return float('nan')
    sr = r.mean() / r.std(ddof=1)
    skew, kurt = stats.skew(r), stats.kurtosis(r, fisher=False)
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2))
    return float(stats.norm.cdf((sr - benchmark_sr) * math.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials, sharpe_variance):
    """Expected best Sharpe of n_trials strategies with no skill (Bailey and Lopez de Prado 2014)."""
    if n_trials <= 1 or sharpe_variance <= 0:
        return 0.0
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * math.e))
    return math.sqrt(sharpe_variance) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def deflated_sharpe(returns, n_trials, trial_sharpes):
    """Probabilistic Sharpe against the expected best of n_trials."""
    var = float(np.var(trial_sharpes, ddof=1)) if len(trial_sharpes) > 1 else 0.0
    return probabilistic_sharpe(returns, expected_max_sharpe(n_trials, var))


def daily_returns(equity, initial=None):
    daily = equity.resample('1D').last().dropna()
    if initial is not None:
        start = pd.Series([initial], index=[daily.index[0] - pd.Timedelta(days=1)])
        daily = pd.concat([start, daily])
    return daily, daily.pct_change().dropna()


def equity_stats(equity, initial=None):
    daily, rets = daily_returns(equity, initial)
    out = {'start_value': float(daily.iloc[0]), 'end_value': float(daily.iloc[-1]),
           'total_return': float(daily.iloc[-1] / daily.iloc[0] - 1), 'days': int(len(daily))}
    if len(rets) > 2 and rets.std() > 0:
        out.update(ann_vol=float(rets.std() * math.sqrt(365)),
                   sharpe=float(rets.mean() / rets.std() * math.sqrt(365)),
                   daily_sharpe=float(rets.mean() / rets.std()),
                   probabilistic_sharpe=probabilistic_sharpe(rets.values))
    peak = daily.cummax()
    out['max_drawdown'] = float(((peak - daily) / peak).max())
    return out
