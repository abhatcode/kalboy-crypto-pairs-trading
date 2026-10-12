"""Download and load KuCoin Futures candles and funding rates."""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

BASE_URL = 'https://api-futures.kucoin.com'
TIMEFRAME_MINUTES = {'5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440}
KLINE_PAGE = 200
FUNDING_PAGE_MS = 30 * 86400 * 1000

# not independent assets
EXCLUDED_BASES = {
    'USDC', 'USDT', 'TUSD', 'DAI', 'FDUSD', 'USDP', 'BUSD', 'PYUSD', 'USDD', 'USDE', 'USD1',
    'EUR', 'XAUT', 'PAXG', 'WBTC', 'WETH', 'STETH', 'WSTETH', 'CBETH', 'RETH', 'WBETH',
}


class KucoinFutures:

    def __init__(self, base_url=BASE_URL, timeout=30, retries=5):
        self.base_url = base_url
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()

    def get(self, path, params=None):
        for attempt in range(self.retries):
            try:
                r = self.session.get(self.base_url + path, params=params, timeout=self.timeout)
                if r.status_code == 429:
                    time.sleep(2 + 3 * attempt)
                    continue
                r.raise_for_status()
                body = r.json()
                if body.get('code') != '200000':
                    raise RuntimeError(f"{path}: {body}")
                return body['data']
            except (requests.RequestException, RuntimeError, ValueError):
                if attempt == self.retries - 1:
                    raise
                time.sleep(1 + 2 * attempt)

    def server_time_ms(self):
        return int(self.get('/api/v1/timestamp'))

    def active_contracts(self):
        return self.get('/api/v1/contracts/active')

    def klines(self, symbol, timeframe, start_ms, end_ms):
        gran = TIMEFRAME_MINUTES[timeframe]
        step = gran * 60_000
        rows, cursor = {}, start_ms
        while cursor < end_ms:
            to = min(end_ms, cursor + KLINE_PAGE * step)
            page = self.get('/api/v1/kline/query', {'symbol': symbol, 'granularity': gran,
                                                    'from': cursor, 'to': to})
            for bar in page or []:
                if start_ms <= bar[0] < end_ms:
                    rows[int(bar[0])] = bar
            cursor = to
        return [rows[k] for k in sorted(rows)]

    def funding_history(self, symbol, start_ms, end_ms):
        out, cursor = {}, start_ms
        while cursor < end_ms:
            to = min(end_ms, cursor + FUNDING_PAGE_MS)
            page = self.get('/api/v1/contract/funding-rates', {'symbol': symbol, 'from': cursor, 'to': to})
            for row in page or []:
                if start_ms <= row['timepoint'] < end_ms:
                    out[int(row['timepoint'])] = float(row['fundingRate'])
            cursor = to
        return sorted(out.items())


def candidate_contracts(client, top_n):
    """Today's top USDT perpetuals by 24h turnover, so delisted coins are missing."""
    rows = []
    for c in client.active_contracts():
        if c.get('type') != 'FFWCSX' or c.get('isInverse') or c.get('quoteCurrency') != 'USDT':
            continue
        if c.get('assetClass', 'CRYPTO') != 'CRYPTO':
            continue
        if c.get('status') != 'Open' or c.get('baseCurrency', '').upper() in EXCLUDED_BASES:
            continue
        rows.append(c)
    rows.sort(key=lambda c: float(c.get('turnoverOf24h') or 0), reverse=True)
    return rows[:top_n]


def base_name(contract):
    base = contract['baseCurrency'].upper()
    return 'BTC' if base == 'XBT' else base


def _ms(ts):
    return int(pd.Timestamp(ts, tz='UTC').timestamp() * 1000) if not isinstance(ts, (int, np.integer)) else int(ts)


def fetch_symbol(client, contract, out_dir, timeframe, start_ms, end_ms):
    """Download or extend one contract's candles and funding."""
    base = base_name(contract)
    path = os.path.join(out_dir, f'{base}.csv')
    fpath = os.path.join(out_dir, 'funding', f'{base}.csv')

    old = pd.read_csv(path, index_col=0, parse_dates=True) if os.path.exists(path) else None
    fetch_from = start_ms
    if old is not None and len(old):
        fetch_from = int(old.index[-1].tz_localize('UTC').timestamp() * 1000) + TIMEFRAME_MINUTES[timeframe] * 60_000
    bars = client.klines(contract['symbol'], timeframe, fetch_from, end_ms)
    new = pd.DataFrame([b[:5] + [b[6]] for b in bars], columns=['time', 'open', 'high', 'low', 'close', 'turnover'])
    new['time'] = pd.to_datetime(new['time'], unit='ms')
    new = new.set_index('time')
    df = new if old is None else pd.concat([old, new])
    df = df[~df.index.duplicated(keep='last')].sort_index()
    df.to_csv(path)

    old_f = pd.read_csv(fpath, index_col=0, parse_dates=True) if os.path.exists(fpath) else None
    f_from = start_ms
    if old_f is not None and len(old_f):
        f_from = int(old_f.index[-1].tz_localize('UTC').timestamp() * 1000) + 1
    rates = client.funding_history(contract['symbol'], f_from, end_ms)
    new_f = pd.DataFrame(rates, columns=['time', 'rate'])
    new_f['time'] = pd.to_datetime(new_f['time'], unit='ms')
    new_f = new_f.set_index('time')
    fdf = new_f if old_f is None else pd.concat([old_f, new_f])
    fdf = fdf[~fdf.index.duplicated(keep='last')].sort_index()
    fdf.to_csv(fpath)
    return len(df), len(fdf)


def fetch_dataset(out_dir, timeframe, start, top_n=60, workers=3, end=None, client=None):
    """Download candles and funding for the top contracts."""
    client = client or KucoinFutures()
    os.makedirs(os.path.join(out_dir, 'funding'), exist_ok=True)
    meta_path = os.path.join(out_dir, 'metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        wanted = set(meta['symbols'])
        contracts = [c for c in client.active_contracts() if base_name(c) in wanted and c['symbol'] in meta['contract_ids']]
    else:
        contracts = candidate_contracts(client, top_n)

    step = TIMEFRAME_MINUTES[timeframe] * 60_000
    now = client.server_time_ms()
    end_ms = _ms(end) if end is not None else (now // step) * step
    start_ms = _ms(start)

    def job(c):
        n_bars, n_funding = fetch_symbol(client, c, out_dir, timeframe, start_ms, end_ms)
        print(f"{base_name(c)}: {n_bars} bars", flush=True)
        return c

    print(f"Fetching {len(contracts)} contracts", flush=True)
    with ThreadPoolExecutor(workers) as pool:
        done = list(pool.map(job, contracts))

    meta = {
        'source': 'KuCoin Futures public REST API (USDT-margined perpetuals)',
        'timeframe': timeframe,
        'start': str(pd.Timestamp(start_ms, unit='ms')),
        'end_exclusive': str(pd.Timestamp(end_ms, unit='ms')),
        'fetched_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'universe': f'top {top_n} USDT perps by 24h turnover at first fetch (survivorship-biased snapshot)',
        'symbols': [base_name(c) for c in done],
        'contract_ids': [c['symbol'] for c in done],
        'specs': {base_name(c): {k: c.get(k) for k in ('symbol', 'multiplier', 'makerFeeRate', 'takerFeeRate',
                                                        'maintainMargin', 'initialMargin', 'firstOpenDate',
                                                        'fundingRateGranularity')} for c in done},
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    return meta


def load_panel(data_dir, symbols=None):
    """Load a dataset as aligned DataFrames. funding is the summed rate within each bar."""
    with open(os.path.join(data_dir, 'metadata.json')) as f:
        meta = json.load(f)
    symbols = symbols or meta['symbols']
    step = pd.Timedelta(minutes=TIMEFRAME_MINUTES[meta['timeframe']])
    fields = {k: {} for k in ('open', 'high', 'low', 'close', 'turnover')}
    funding = {}
    for sym in symbols:
        path = os.path.join(data_dir, f'{sym}.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        for k in fields:
            fields[k][sym] = df[k]
        fpath = os.path.join(data_dir, 'funding', f'{sym}.csv')
        if os.path.exists(fpath):
            f = pd.read_csv(fpath, index_col=0, parse_dates=True)['rate']
            funding[sym] = f.groupby(f.index.floor(step)).sum()
    panel = {k: pd.DataFrame(v).sort_index() for k, v in fields.items()}
    index = panel['close'].index
    panel['funding'] = pd.DataFrame(funding).reindex(index).fillna(0.0).reindex(columns=panel['close'].columns, fill_value=0.0)
    panel['meta'] = meta
    return panel


def bars_per_day(index):
    step = pd.Series(index).diff().median()
    return pd.Timedelta(days=1) / step


def validate(data_dir):
    """Check each contract for gaps, duplicates and bad prices."""
    with open(os.path.join(data_dir, 'metadata.json')) as f:
        meta = json.load(f)
    step = pd.Timedelta(minutes=TIMEFRAME_MINUTES[meta['timeframe']])
    rows = []
    for sym in meta['symbols']:
        df = pd.read_csv(os.path.join(data_dir, f'{sym}.csv'), index_col=0, parse_dates=True)
        fr = pd.read_csv(os.path.join(data_dir, 'funding', f'{sym}.csv'), index_col=0, parse_dates=True)['rate']
        expected = int((df.index[-1] - df.index[0]) / step) + 1
        rows.append({
            'symbol': sym, 'first': df.index[0], 'last': df.index[-1], 'bars': len(df),
            'missing_bars': expected - len(df), 'duplicates': int(df.index.duplicated().sum()),
            'nonpositive_prices': int((df[['open', 'high', 'low', 'close']] <= 0).any(axis=1).sum()),
            'ohlc_violations': int(((df['high'] < df[['open', 'close']].max(axis=1)) |
                                    (df['low'] > df[['open', 'close']].min(axis=1))).sum()),
            'max_abs_bar_return': float(np.log(df['close']).diff().abs().max()),
            'zero_turnover_share': float((df['turnover'] <= 0).mean()),
            'funding_points': len(fr), 'funding_min': float(fr.min()) if len(fr) else np.nan,
            'funding_max': float(fr.max()) if len(fr) else np.nan,
        })
    return pd.DataFrame(rows)
