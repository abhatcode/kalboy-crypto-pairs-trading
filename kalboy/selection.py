"""Pair selection from formation-window data only."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.tsatools import add_trend, lagmat
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def build_universe(closes, turnover, min_coverage, universe_size):
    """Most liquid contracts that traded through the window."""
    coverage = closes.notna().mean()
    eligible = coverage[coverage >= min_coverage].index
    liquidity = turnover[eligible].median().sort_values(ascending=False)
    return list(liquidity.index[:universe_size]) if universe_size else list(liquidity.index)


def cluster_assets(returns, n_market_factors=1, max_cluster_size=8):
    """Cluster assets on return correlations after removing the market factor."""
    z = StandardScaler().fit_transform(returns.values)
    n = z.shape[1]
    if n < 2:
        return {}
    factors = PCA(n_components=min(n_market_factors, n), random_state=0).fit_transform(z)
    coef, *_ = np.linalg.lstsq(factors, z, rcond=None)
    resid = z - factors @ coef
    corr = np.nan_to_num(np.corrcoef(resid, rowvar=False), nan=0.0)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    tree = linkage(squareform(dist, checks=False), method='average')
    for k in range(1, n + 1):
        labels = fcluster(tree, t=k, criterion='maxclust')
        if np.bincount(labels).max() <= max_cluster_size:
            break
    groups = {}
    for asset, lab in zip(returns.columns, labels):
        groups.setdefault(int(lab), []).append(asset)
    return {k: v for k, v in groups.items() if len(v) >= 2}


SQRTEPS = np.sqrt(np.finfo(np.double).eps)


def _aic_lag(x, maxlag):
    """The lag adfuller picks by AIC, from a single QR decomposition."""
    xdiff = np.diff(x)
    xdall = lagmat(xdiff[:, None], maxlag, trim='both', original='in')
    nobs = xdall.shape[0]
    xdall[:, 0] = x[-nobs - 1:-1]
    xdshort = xdiff[-nobs:]
    q, r = np.linalg.qr(xdall)
    d = np.abs(np.diag(r))
    if d.min() <= 1e-10 * d.max():
        return None
    c = q.T @ xdshort
    e = xdshort - q @ c
    suffix = np.cumsum((c * c)[::-1])[::-1]
    ssr = float(e @ e) + np.append(suffix[1:], 0.0)
    nobs2 = nobs / 2.0
    llf = -nobs2 * np.log(2 * np.pi) - nobs2 * np.log(ssr / nobs) - nobs2
    aic = -2.0 * llf + 2.0 * np.arange(1, maxlag + 2)
    return int(np.argmin(aic))


def coint_pvalue(y0, y1):
    """Same p-value as statsmodels ts.coint(y0, y1, autolag='aic'), about 14 times faster."""
    y0 = np.asarray(y0, dtype=float)
    y1 = np.asarray(y1, dtype=float)
    res_co = sm.OLS(y0, add_trend(y1[:, None], trend='c', prepend=False)).fit()
    if not res_co.rsquared < 1 - 100 * SQRTEPS:
        return ts.coint(y0, y1, autolag='aic')[1]
    resid = res_co.resid
    nobs = resid.shape[0]
    maxlag = min(nobs // 2 - 1, int(np.ceil(12.0 * np.power(nobs / 100.0, 1 / 4.0))))
    lag = _aic_lag(resid, maxlag)
    if lag is None:
        return ts.coint(y0, y1, autolag='aic')[1]
    stat = ts.adfuller(resid, maxlag=lag, autolag=None, regression='n')[0]
    return ts.mackinnonp(stat, regression='c', N=2)


def cointegration_tests(prices, clusters):
    """Engle-Granger both ways for each pair in a cluster. The smaller p-value is doubled."""
    rows = []
    for cid, assets in clusters.items():
        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                a, b = assets[i], assets[j]
                p_ab = coint_pvalue(prices[a].values, prices[b].values)
                p_ba = coint_pvalue(prices[b].values, prices[a].values)
                y, x = (a, b) if p_ab <= p_ba else (b, a)
                rows.append({'asset_y': y, 'asset_x': x, 'cluster': cid, 'p_ab': p_ab, 'p_ba': p_ba,
                             'p_pair': min(1.0, 2.0 * min(p_ab, p_ba))})
    return pd.DataFrame(rows)


def apply_fdr(tests, alpha):
    if tests.empty:
        return tests.assign(p_fdr=[], fdr_pass=[])
    reject, p_adj, _, _ = sm.stats.multipletests(tests['p_pair'].values, alpha=alpha, method='fdr_bh')
    return tests.assign(p_fdr=p_adj, fdr_pass=reject)


def hurst_exponent(series, max_lag=100):
    """Hurst exponent from the scaling of lagged differences."""
    s = np.asarray(series, dtype=float)
    if len(s) < 2 * max_lag:
        return 0.5
    lags = np.arange(2, max_lag)
    sd = np.array([np.std(s[lag:] - s[:-lag]) for lag in lags])
    ok = sd > 0
    return float(np.polyfit(np.log(lags[ok]), np.log(sd[ok]), 1)[0]) if ok.sum() > 2 else 0.5


def half_life(spread):
    """Bars for a spread deviation to halve."""
    s = np.asarray(spread, dtype=float)
    lam = sm.OLS(np.diff(s), sm.add_constant(s[:-1])).fit().params[1]
    return float(-np.log(2) / lam) if lam < 0 else float('nan')


def characterize(prices, y, x):
    fit = sm.OLS(prices[y].values, sm.add_constant(prices[x].values)).fit()
    alpha, beta = float(fit.params[0]), float(fit.params[1])
    spread = prices[y].values - (beta * prices[x].values + alpha)
    return {'beta': beta, 'alpha': alpha, 'hurst': hurst_exponent(spread), 'half_life': half_life(spread)}


def select_pairs(closes, turnover, sel, bars_per_day, trading_bars):
    """Run the selection funnel on one formation window."""
    universe = build_universe(closes, turnover, sel.min_coverage, sel.universe_size)
    prices = closes[universe].ffill().bfill()
    clusters = cluster_assets(prices.pct_change().dropna(), sel.n_market_factors, sel.max_cluster_size)
    tests = apply_fdr(cointegration_tests(prices, clusters), sel.fdr_alpha)

    max_hl = sel.max_half_life_frac * trading_bars
    survivors = []
    for row in tests[tests['fdr_pass']].to_dict('records') if len(tests) else []:
        stats = characterize(prices, row['asset_y'], row['asset_x'])
        row.update(stats)
        hl = stats['half_life']
        row['filter_pass'] = bool(stats['hurst'] < sel.max_hurst and not np.isnan(hl)
                                  and sel.min_half_life_bars <= hl <= max_hl)
        survivors.append(row)
    pairs = pd.DataFrame(survivors)
    funnel = {
        'n_eligible': int((closes.notna().mean() >= sel.min_coverage).sum()),
        'n_universe': len(universe),
        'cluster_sizes': sorted((len(v) for v in clusters.values()), reverse=True),
        'n_tests': int(len(tests)),
        'n_fdr_pass': int(tests['fdr_pass'].sum()) if len(tests) else 0,
        'n_filter_pass': int(pairs['filter_pass'].sum()) if len(pairs) else 0,
        'max_half_life_bars': max_hl,
    }
    return (pairs[pairs['filter_pass']].reset_index(drop=True) if len(pairs) else pairs), funnel
