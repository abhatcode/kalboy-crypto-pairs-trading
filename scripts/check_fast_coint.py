"""Check that selection.coint_pvalue matches statsmodels on real data."""
import argparse
import itertools
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import statsmodels.tsa.stattools as ts

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kalboy.data import load_panel, bars_per_day
from kalboy.selection import coint_pvalue


def main():
    p = argparse.ArgumentParser()
    p.add_argument('data_dir')
    p.add_argument('--windows', type=int, default=6)
    p.add_argument('--coins', type=int, default=12)
    p.add_argument('--days', type=float, default=60.0)
    args = p.parse_args()
    warnings.simplefilter('ignore')

    closes = load_panel(args.data_dir)['close']
    n = int(round(args.days * bars_per_day(closes.index)))
    rng = np.random.default_rng(1)
    same = differ = 0
    t_statsmodels = t_fast = 0.0
    for _ in range(args.windows):
        end = int(rng.integers(n, len(closes)))
        window = closes.iloc[end - n:end]
        coins = [c for c in window.columns if window[c].notna().mean() > 0.98][:args.coins]
        window = window[coins].ffill().bfill()
        for a, b in itertools.combinations(coins, 2):
            t = time.perf_counter()
            expected = ts.coint(window[a], window[b], autolag='aic')[1]
            t_statsmodels += time.perf_counter() - t
            t = time.perf_counter()
            got = coint_pvalue(window[a].values, window[b].values)
            t_fast += time.perf_counter() - t
            if got == expected:
                same += 1
            else:
                differ += 1
                print(f"{a}/{b}: {expected} vs {got}")
    print(f"{same} of {same + differ} identical, {t_statsmodels / t_fast:.1f}x faster")
    sys.exit(1 if differ else 0)


if __name__ == '__main__':
    main()
