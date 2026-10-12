"""Paper trading: the walk-forward run on live bars."""
import json
import os
import pickle
import time

import numpy as np
import pandas as pd

from kalboy.data import KucoinFutures, fetch_symbol, load_panel, bars_per_day, TIMEFRAME_MINUTES
from kalboy.execution import PaperBroker, PortfolioSimulator
from kalboy.risk import RiskManager
from kalboy.walkforward import form, start_engines


class LiveSource:
    """Keeps the dataset files current from KuCoin and loads panels from them."""

    def __init__(self, data_dir, client=None):
        self.data_dir = data_dir
        self.client = client or KucoinFutures()
        with open(os.path.join(data_dir, 'metadata.json')) as f:
            meta = json.load(f)
        self.meta = meta
        self.start_ms = int(pd.Timestamp(meta['start'], tz='UTC').timestamp() * 1000)
        self.symbols = meta['symbols']
        self.step_ms = TIMEFRAME_MINUTES[meta['timeframe']] * 60_000
        self._contracts = None

    def now_ms(self):
        return self.client.server_time_ms()

    def refresh(self, symbols, end_ms):
        if self._contracts is None:
            ids = dict(zip(self.meta['symbols'], self.meta['contract_ids']))
            self._contracts = {s: {'symbol': ids[s], 'baseCurrency': 'XBT' if s == 'BTC' else s} for s in ids}
        for sym in symbols:
            fetch_symbol(self.client, self._contracts[sym], self.data_dir, self.meta['timeframe'],
                         self.start_ms, end_ms)

    def panel(self, symbols=None):
        return load_panel(self.data_dir, symbols)


class ReplaySource:
    """Replays a historical panel bar by bar, for tests."""

    def __init__(self, panel):
        self.full = panel
        self.symbols = list(panel['close'].columns)
        self.step_ms = int(pd.Timedelta(days=1) / bars_per_day(panel['close'].index) / pd.Timedelta(milliseconds=1))
        self.end_ms = None

    def now_ms(self):
        return self.end_ms

    def refresh(self, symbols, end_ms):
        self.end_ms = end_ms

    def panel(self, symbols=None):
        end = pd.Timestamp(self.end_ms, unit='ms')
        keep = self.full['close'].index < end
        return {k: v.loc[keep] for k, v in self.full.items() if isinstance(v, pd.DataFrame)}


class PaperTrader:
    def __init__(self, cfg, out_dir, source, alerts=None, dashboard=False):
        self.cfg = cfg
        self.out_dir = out_dir
        self.source = source
        self.alerts = alerts
        self.dashboard = dashboard
        os.makedirs(out_dir, exist_ok=True)
        self.state_path = os.path.join(out_dir, 'state.pkl')
        if os.path.exists(self.state_path):
            with open(self.state_path, 'rb') as f:
                self.state = pickle.load(f)
            if self.state['fingerprint'] != cfg.fingerprint():
                raise RuntimeError(f"{out_dir} was started with a different config; use a new label")
        else:
            broker = PaperBroker(cfg.walkforward.initial_cash, cfg.costs.taker_fee, cfg.costs.slippage_bps / 1e4)
            sim = PortfolioSimulator(broker, RiskManager(cfg.risk.max_concentration, cfg.risk.max_drawdown),
                                     cfg.risk.maintenance_margin)
            self.state = {'fingerprint': cfg.fingerprint(), 'sim': sim, 'last_bar': None,
                          'bars_in_period': None, 'window': -1, 'engines': [], 'equity': [],
                          'formations': [], 'decisions_written': 0}

    def process_new_bars(self):
        """Process every bar closed since the last pass and return the count."""
        st, cfg = self.state, self.cfg
        step = self.source.step_ms
        closed_until = (self.source.now_ms() // step) * step
        sim = st['sim']
        # refresh every symbol only when a re-formation is due
        arriving = 1 if st['last_bar'] is None else max(
            0, (closed_until - int(pd.Timestamp(st['last_bar']).value // 1_000_000)) // step - 1)
        reform_due = st['bars_in_period'] is None or st['bars_in_period'] + arriving >= self._trading_bars()
        # always refresh one symbol so time still advances with no pairs
        clock = 'BTC' if 'BTC' in self.source.symbols else self.source.symbols[0]
        self.source.refresh(self.source.symbols if reform_due else sorted(sim.symbols_needed() | {clock}), closed_until)
        panel = self.source.panel()
        index = panel['close'].index
        new = index[index > st['last_bar']] if st['last_bar'] is not None else index[-1:]
        bpd = bars_per_day(index)
        F = int(round(cfg.walkforward.formation_days * bpd))
        closes, opens = panel['close'].ffill(), panel['open'].ffill()
        funding = panel['funding'] if cfg.costs.funding else panel['funding'] * 0.0
        highs = panel['high'].reindex(columns=closes.columns) if 'high' in panel else None
        lows = panel['low'].reindex(columns=closes.columns) if 'low' in panel else None

        for t in new:
            i = index.get_loc(t)
            if st['bars_in_period'] is None or st['bars_in_period'] >= self._trading_bars(bpd):
                if i < F:
                    continue
                self._reform(panel, i, F, bpd, t)
            syms = sim.symbols_needed()
            equity = sim.step(t, {k: opens.iat[i, opens.columns.get_loc(k)] for k in syms},
                              {k: closes.iat[i, closes.columns.get_loc(k)] for k in syms},
                              {k: funding.iat[i, funding.columns.get_loc(k)] for k in syms},
                              highs={k: highs.iat[i, highs.columns.get_loc(k)] for k in syms} if highs is not None else None,
                              lows={k: lows.iat[i, lows.columns.get_loc(k)] for k in syms} if lows is not None else None)
            st['equity'].append((t, equity))
            st['bars_in_period'] += 1
            st['last_bar'] = t
        if len(new):
            if self.alerts:
                self.alerts.after_pass(st, cfg.walkforward.initial_cash)
            self._save()
            if self.dashboard:
                from kalboy import dashboard
                dashboard.write(self.out_dir, st, cfg)
        return len(new)

    def _trading_bars(self, bpd=None):
        if bpd is None:
            bpd = pd.Timedelta(days=1) / pd.Timedelta(milliseconds=self.source.step_ms)
        return int(round(self.cfg.walkforward.trading_days * bpd))

    def _reform(self, panel, i, F, bpd, t):
        st = self.state
        sim = st['sim']
        sim.retire_all(t)
        window = {k: panel[k].iloc[i - F:i] for k in ('open', 'close', 'turnover', 'funding')}
        result = form(window, self.cfg.to_dict(), bpd, self._trading_bars(bpd))
        st['window'] += 1
        engines = start_engines(result['specs'], self.cfg.strategy, panel['close'].iloc[i - F:i],
                                panel['funding'].iloc[i - F:i] if self.cfg.costs.funding else None)
        for e in engines:
            e.window = st['window']
            sim.add(e)
        st['engines'].extend(engines)
        st['formations'].append({'window': st['window'], 'start': t, **result['funnel'],
                                 'pairs': [f"{s['y']}/{s['x']}" for s in result['specs'] if s['tradeable']]})
        st['bars_in_period'] = 0

    def _save(self):
        tmp = self.state_path + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(self.state, f)
        os.replace(tmp, self.state_path)
        st = self.state
        # append new rows; the first save in a process rewrites the files
        written = getattr(self, '_written', None)
        fresh = written is None
        if fresh:
            written = self._written = {'equity': 0, 'decisions': 0, 'formations': -1, 'trades': -1}
        for name, rows, cols in (('equity', st['equity'], ['time', 'equity']),
                                 ('decisions', st['sim'].decision_log, ['time', 'pair', 'action', 'z'])):
            new = rows[written[name]:]
            if new or fresh:
                pd.DataFrame(new, columns=cols).to_csv(os.path.join(self.out_dir, f'{name}.csv'), index=False,
                                                       mode='w' if fresh else 'a', header=fresh,
                                                       date_format='%Y-%m-%d %H:%M:%S')
                written[name] = len(rows)
        if len(st['formations']) != written['formations']:
            pd.DataFrame(st['formations']).to_csv(os.path.join(self.out_dir, 'formations.csv'), index=False,
                                                  date_format='%Y-%m-%d %H:%M:%S')
            written['formations'] = len(st['formations'])
        n_trades = sum(len(e.trade_log) for e in st['engines'])
        if n_trades != written['trades']:
            trades = [{'window': e.window, 'pair': f"{e.y_name}/{e.x_name}", **{k: v for k, v in t.items() if k != 'features'}}
                      for e in st['engines'] for t in e.trade_log]
            pd.DataFrame(trades).to_csv(os.path.join(self.out_dir, 'trades.csv'), index=False,
                                        date_format='%Y-%m-%d %H:%M:%S')
            written['trades'] = n_trades

    def run_forever(self, poll_seconds=20):
        print(f"Paper trading {self.cfg.name} in {self.out_dir}")
        if self.alerts:
            self.alerts.notifier.send(f"Paper trader {self.alerts.name} started with config {self.cfg.name} "
                                      f"(fingerprint {self.cfg.fingerprint()}).")
        failing = False
        while True:
            try:
                n = self.process_new_bars()
                if failing and self.alerts:
                    self.alerts.notifier.send(f"Paper trader {self.alerts.name} is running again.")
                failing = False
                if n:
                    st = self.state
                    t, eq = st['equity'][-1] if st['equity'] else (None, None)
                    open_pairs = sum(1 for e in st['sim'].engines if e.status)
                    print(f"{t}: equity {eq:,.2f}, {len(st['sim'].engines)} pairs, {open_pairs} open", flush=True)
            except Exception as exc:
                print(f"Error: {exc!r}", flush=True)
                if not failing and self.alerts:
                    self.alerts.notifier.send(f"Paper trader {self.alerts.name} hit an error and will retry "
                                              f"every bar: {exc!r}")
                failing = True
            step = self.source.step_ms
            now = self.source.now_ms() if hasattr(self.source, 'client') else int(time.time() * 1000)
            wait = (step - now % step) / 1000 + poll_seconds
            time.sleep(max(5.0, wait))
