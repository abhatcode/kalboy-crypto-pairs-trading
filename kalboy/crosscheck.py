"""Compare KuCoin perp closes with Coinbase spot closes."""
import os
import time

import numpy as np
import pandas as pd
import requests

from kalboy.data import load_panel

COINBASE_URL = 'https://api.exchange.coinbase.com'
CANDLE_PAGE = 300


class CoinbaseExchange:
    def __init__(self, base_url=COINBASE_URL, timeout=30, retries=5, pause=0.15):
        self.base_url, self.timeout, self.retries, self.pause = base_url, timeout, retries, pause
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'kalboy-crosscheck'

    def get(self, path, params=None):
        for attempt in range(self.retries):
            try:
                r = self.session.get(self.base_url + path, params=params, timeout=self.timeout)
                if r.status_code == 429:
                    time.sleep(1 + 2 * attempt)
                    continue
                r.raise_for_status()
                time.sleep(self.pause)
                return r.json()
            except requests.RequestException:
                if attempt == self.retries - 1:
                    raise
                time.sleep(1 + 2 * attempt)
        raise RuntimeError(f'{path}: rate limited {self.retries} times')

    def usd_products(self):
        return {p['base_currency'] for p in self.get('/products')
                if p['quote_currency'] == 'USD' and p['status'] == 'online'}

    def closes(self, product, start, end, granularity=3600):
        out = {}
        step = pd.Timedelta(seconds=granularity)
        t = pd.Timestamp(start)
        end = pd.Timestamp(end)
        while t < end:
            t2 = min(end, t + step * CANDLE_PAGE)
            rows = self.get(f'/products/{product}/candles', {
                'granularity': granularity, 'start': t.isoformat() + 'Z',
                'end': (t2 - pd.Timedelta(seconds=1)).isoformat() + 'Z'})
            for ts, _low, _high, _open, close, _vol in rows:
                out[pd.Timestamp(ts, unit='s')] = float(close)
            t = t2
        s = pd.Series(out, dtype=float).sort_index()
        return s[(s.index >= pd.Timestamp(start)) & (s.index < end)]


def compare(kucoin, coinbase_usd, usdt_usd):
    """Price gap and return correlation for one coin."""
    df = pd.concat({'k': kucoin, 'c': coinbase_usd / usdt_usd}, axis=1).dropna()
    if len(df) < 48:
        return None
    gap_bps = (df['k'] / df['c'] - 1) * 1e4
    rk, rc = np.log(df['k']).diff(), np.log(df['c']).diff()
    ok = rk.notna() & rc.notna()
    return {'bars': len(df), 'first': df.index[0], 'last': df.index[-1],
            'median_gap_bps': float(gap_bps.median()), 'median_abs_gap_bps': float(gap_bps.abs().median()),
            'p99_abs_gap_bps': float(gap_bps.abs().quantile(0.99)),
            'return_corr': float(np.corrcoef(rk[ok], rc[ok])[0, 1]),
            'hours_gap_over_1pct': int((gap_bps.abs() > 100).sum())}


def run(data_dir, start=None, end=None, client=None, verbose=True):
    """Compare every coin also listed on Coinbase and write crosscheck_coinbase.csv."""
    client = client or CoinbaseExchange()
    panel = load_panel(data_dir)
    closes = panel['close']
    if pd.Series(closes.index).diff().median() != pd.Timedelta(hours=1):
        raise ValueError('cross-check expects an hourly dataset (e.g. data/kucoinf_1h)')
    start = pd.Timestamp(start) if start else closes.index[0]
    end = pd.Timestamp(end) if end else closes.index[-1] + pd.Timedelta(hours=1)
    listed = client.usd_products()
    coins = [c for c in closes.columns if c in listed]
    if verbose:
        print(f"{len(coins)} coins listed on Coinbase", flush=True)
    usdt = client.closes('USDT-USD', start, end)
    rows = []
    for c in coins:
        stats = compare(closes[c].loc[start:end], client.closes(f'{c}-USD', start, end), usdt)
        if stats:
            rows.append({'symbol': c, **stats})
    report = pd.DataFrame(rows)
    report.to_csv(os.path.join(data_dir, 'crosscheck_coinbase.csv'), index=False)
    return report
