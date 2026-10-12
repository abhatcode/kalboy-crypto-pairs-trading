import numpy as np

from kalboy.config import Selection
from kalboy.selection import build_universe, cluster_assets, cointegration_tests, apply_fdr, select_pairs


def formation(panel, n=480):
    return panel['close'].iloc[:n], panel['turnover'].iloc[:n]


def test_finds_the_planted_pairs_and_no_pure_random_walk_pair(panel):
    closes, turnover = formation(panel)
    sel = Selection(universe_size=0, max_cluster_size=8)
    pairs, funnel = select_pairs(closes, turnover, sel, bars_per_day=24, trading_bars=120)
    found = {frozenset((r['asset_y'], r['asset_x'])) for r in pairs.to_dict('records')}
    assert frozenset(('A1', 'A2')) in found and frozenset(('B1', 'B2')) in found
    assert not any(p <= {'R0', 'R1', 'R2', 'R3'} for p in found)
    assert funnel['n_tests'] >= funnel['n_fdr_pass'] >= funnel['n_filter_pass'] == len(pairs)


def test_fdr_sees_every_test_not_a_prefiltered_subset(panel):
    closes, _ = formation(panel)
    clusters = {0: list(closes.columns)}
    tests = apply_fdr(cointegration_tests(closes, clusters), 0.05)
    n = len(closes.columns)
    assert len(tests) == n * (n - 1) // 2
    assert (tests['p_pair'] >= tests[['p_ab', 'p_ba']].min(axis=1)).all()


def test_planted_pair_is_strongly_significant(panel):
    closes, _ = formation(panel)
    t = cointegration_tests(closes[['A1', 'A2']], {0: ['A1', 'A2']})
    assert t['p_pair'].iloc[0] < 0.001


def test_cluster_size_cap(panel):
    closes, _ = formation(panel)
    for cap in (2, 3, 8):
        groups = cluster_assets(closes.pct_change().dropna(), 1, cap)
        assert all(2 <= len(g) <= cap for g in groups.values())


def test_universe_is_point_in_time(panel):
    closes, turnover = formation(panel)
    closes = closes.copy()
    closes.loc[closes.index[:100], 'R0'] = float('nan')
    assert 'R0' not in build_universe(closes, turnover, 0.98, 0)
    top2 = build_universe(closes, turnover, 0.98, 2)
    assert len(top2) == 2


def test_fast_coint_matches_statsmodels():
    import statsmodels.tsa.stattools as ts
    from kalboy.selection import coint_pvalue
    rng = np.random.default_rng(3)
    for n in (300, 1500, 6000):
        for kind in ('coint', 'independent', 'slow'):
            x = np.cumsum(rng.normal(size=n)) + 100
            if kind == 'coint':
                y = 0.8 * x + rng.normal(size=n)
            elif kind == 'independent':
                y = np.cumsum(rng.normal(size=n)) + 100
            else:
                e = np.zeros(n)
                for i in range(1, n):
                    e[i] = 0.995 * e[i - 1] + rng.normal()
                y = 1.2 * x + e
            assert coint_pvalue(y, x) == ts.coint(y, x, autolag='aic')[1]
            assert coint_pvalue(x, y) == ts.coint(x, y, autolag='aic')[1]
