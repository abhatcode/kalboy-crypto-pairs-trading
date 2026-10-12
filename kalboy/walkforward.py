"""Walk-forward backtest with rolling formation and trading periods (Gatev et al. 2006)."""
import hashlib
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np
import pandas as pd

from kalboy import ledger
from kalboy import selection as selection_module
from kalboy.calibration import calibrate, make_engine, PairSpec
from kalboy.config import config_from_dict
from kalboy.data import load_panel, bars_per_day
from kalboy.execution import PaperBroker, PortfolioSimulator
from kalboy.risk import RiskManager
from kalboy.selection import select_pairs
from kalboy.stats import trade_stats, equity_stats, daily_returns, deflated_sharpe


def window_plan(index, wf, bpd, segment):
    F = int(round(wf.formation_days * bpd))
    T = int(round(wf.trading_days * bpd))
    end = int(index.searchsorted(pd.Timestamp(wf.end))) if wf.end else len(index)
    holdout = min(end, int(index.searchsorted(pd.Timestamp(wf.holdout_start)))) if wf.holdout_start else end
    if segment == 'dev':
        first = max(F, int(index.searchsorted(pd.Timestamp(wf.dev_start))) if wf.dev_start else F)
        last = holdout
    elif segment == 'holdout':
        first, last = holdout, end
        if first < F:
            raise ValueError('holdout starts before one full formation window')
    else:
        raise ValueError(f"segment must be 'dev' or 'holdout', not {segment!r}")
    return [(s, min(s + T, last)) for s in range(first, last, T)], F, T


_SELECTION_SOURCE = None


def _selection_key(window, sel, bpd, trading_bars):
    """Cache key from the data, the settings and the selection code."""
    global _SELECTION_SOURCE
    if _SELECTION_SOURCE is None:
        with open(selection_module.__file__, 'rb') as f:
            _SELECTION_SOURCE = f.read()
    h = hashlib.sha1(_SELECTION_SOURCE)
    h.update(json.dumps({'selection': asdict(sel), 'bpd': bpd, 'trading_bars': trading_bars}, sort_keys=True).encode())
    for k in ('close', 'turnover'):
        h.update(','.join(map(str, window[k].columns)).encode())
        h.update(pd.util.hash_pandas_object(window[k], index=True).values.tobytes())
    return h.hexdigest()[:24]


def cached_select(window, sel, bpd, trading_bars, cache_dir=None):
    """select_pairs() cached on disk."""
    if not cache_dir:
        return select_pairs(window['close'], window['turnover'], sel, bpd, trading_bars)
    path = os.path.join(cache_dir, _selection_key(window, sel, bpd, trading_bars) + '.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    result = select_pairs(window['close'], window['turnover'], sel, bpd, trading_bars)
    os.makedirs(cache_dir, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, 'wb') as f:
        pickle.dump(result, f)
    os.replace(tmp, path)
    return result


def form(window, cfg_dict, bpd, trading_bars, cache_dir=None):
    """Select and fit pairs on one formation window."""
    cfg = config_from_dict(cfg_dict)
    pairs, funnel = cached_select(window, cfg.selection, bpd, trading_bars, cache_dir)
    specs = [calibrate(row, window, cfg.strategy, cfg.costs, cfg.walkforward.initial_cash, cfg.kalman)
             for row in pairs.to_dict('records')] if len(pairs) else []
    funnel['n_tradeable'] = sum(s.tradeable for s in specs)
    return {'specs': [s.to_dict() for s in specs], 'funnel': funnel}


def panel_slice(panel, a, b):
    return {k: panel[k].iloc[a:b] for k in ('open', 'close', 'turnover', 'funding')}


def start_engines(specs, strat, warm_closes, warm_funding=None):
    """Engines for a new period, warmed up on the formation bars."""
    engines = []
    for d in specs:
        spec = PairSpec(**d)
        if not spec.tradeable:
            continue
        engine = make_engine(spec, strat)
        cy = warm_closes[spec.y].ffill().bfill().values
        cx = warm_closes[spec.x].ffill().bfill().values
        fy = warm_funding[spec.y].values if warm_funding is not None else None
        fx = warm_funding[spec.x].values if warm_funding is not None else None
        for i, (yv, xv) in enumerate(zip(cy, cx)):
            engine.update(yv, xv)
            if fy is not None:
                engine.accrue_funding(fy[i], fx[i])
        engine.spec = spec
        engines.append(engine)
    return engines


def run(cfg, segment='dev', workers=1, verbose=True, panel=None, cache_dir=None, purpose='trial'):
    """purpose is 'trial', 'sensitivity' or 'reproduction'. Only trials count toward the DSR."""
    t0 = time.time()
    wf = cfg.walkforward
    panel = panel or load_panel(wf.data_dir)
    index = panel['close'].index
    bpd = bars_per_day(index)
    plan, F, T = window_plan(index, wf, bpd, segment)
    if not plan:
        raise ValueError('no trading windows in this segment')
    if verbose:
        print(f"{cfg.name} {segment}: {len(plan)} windows, {panel['close'].shape[1]} contracts", flush=True)

    jobs = [(panel_slice(panel, s - F, s), cfg.to_dict(), bpd, T, cache_dir) for s, _ in plan]
    if workers > 1:
        # one BLAS thread per worker, or the workers oversubscribe the CPU
        for var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
            os.environ.setdefault(var, '1')
        with ProcessPoolExecutor(max_workers=workers) as pool:
            formations = list(pool.map(form, *zip(*jobs)))
    else:
        formations = [form(*job) for job in jobs]
    if verbose:
        print(f"Formation done in {time.time() - t0:.0f} s", flush=True)

    broker = PaperBroker(wf.initial_cash, cfg.costs.taker_fee, cfg.costs.slippage_bps / 1e4)
    risk = RiskManager(cfg.risk.max_concentration, cfg.risk.max_drawdown)
    sim = PortfolioSimulator(broker, risk, cfg.risk.maintenance_margin)
    closes = panel['close'].ffill()
    opens = panel['open'].ffill()
    closes_np, opens_np = closes.values, opens.values
    highs_np = panel['high'].reindex(columns=closes.columns).values if 'high' in panel else None
    lows_np = panel['low'].reindex(columns=closes.columns).values if 'low' in panel else None
    funding_np = panel['funding'].values if cfg.costs.funding else np.zeros_like(closes_np)
    col = {s: j for j, s in enumerate(closes.columns)}

    eq_t, eq_v, windows, all_engines = [], [], [], []
    for w, (s, e) in enumerate(plan):
        sim.retire_all(index[s])
        engines = start_engines(formations[w]['specs'], cfg.strategy, panel['close'].iloc[s - F:s],
                                panel['funding'].iloc[s - F:s] if cfg.costs.funding else None)
        for engine in engines:
            engine.window = w
            sim.add(engine)
        all_engines.extend(engines)
        start_value = eq_v[-1] if eq_v else wf.initial_cash
        for i in range(s, e):
            syms = sim.symbols_needed()
            eq_t.append(index[i])
            eq_v.append(sim.step(index[i], {k: opens_np[i, col[k]] for k in syms},
                                 {k: closes_np[i, col[k]] for k in syms},
                                 {k: funding_np[i, col[k]] for k in syms},
                                 highs={k: highs_np[i, col[k]] for k in syms} if highs_np is not None else None,
                                 lows={k: lows_np[i, col[k]] for k in syms} if lows_np is not None else None))
        f = formations[w]['funnel']
        windows.append({'window': w, 'start': index[s], 'end': index[e - 1], **f,
                        'cluster_sizes': ' '.join(map(str, f['cluster_sizes'])),
                        'pairs_traded': len(engines), 'return': eq_v[-1] / start_value - 1})
    sim.retire_all(index[plan[-1][1] - 1])
    last = plan[-1][1] - 1
    for engine, sym, qty in sim.pending:
        price, fee = broker.execute(sym, qty, closes_np[last, col[sym]])
        engine.record_fill(qty, price, fee, index[last])
    sim.pending = []
    eq_v[-1] = broker.equity({s: closes_np[last, col[s]] for s in broker.positions}) if broker.positions else broker.cash

    trades = trade_table(all_engines, closes, bpd)
    equity = pd.Series(eq_v, index=pd.DatetimeIndex(eq_t), name='equity')
    pairs = pd.DataFrame([{'window': w, **d} for w, fm in enumerate(formations) for d in fm['specs']])
    summary = summarize(cfg, segment, plan, index, bpd, windows, trades, equity, broker, risk, sim, t0, purpose)
    summary['funding_guard_skips'] = int(sum(e.funding_skips for e in all_engines))
    return {'summary': summary, 'windows': pd.DataFrame(windows), 'trades': trades,
            'equity': equity, 'pairs': pairs}


def trade_table(engines, closes, bpd):
    """All completed trades with their formation statistics."""
    btc = closes['BTC'] if 'BTC' in closes else None
    day = max(2, int(round(bpd)))
    btc_vol = np.log(btc).diff().rolling(day).std() * np.sqrt(day) if btc is not None else None
    rows = []
    for e in engines:
        for t in e.trade_log:
            row = {'window': e.window, 'pair': f"{e.y_name}/{e.x_name}", 'direction': t['direction'],
                   'entry_time': t['entry_time'], 'exit_time': t['exit_time'], 'exit_reason': t['exit_reason'],
                   'bars_held': t['bars_held'], 'entry_z': t['entry_z'], 'exit_z': t['exit_z'], 'exit_z_trade': t['exit_z_trade'],
                   'return': t['return'], 'pnl': t['pnl'], 'fees': t['fees'], 'funding': t['funding'],
                   'mae_y': t['mae_y'], 'mae_x': t['mae_x'],
                   'half_life': e.spec.half_life, 'zscore_window': e.spec.zscore_window,
                   'hurst': e.spec.hurst, 'p_fdr': e.spec.p_fdr, 'kelly': e.spec.kelly,
                   'formation_trades': e.spec.formation_trades,
                   'formation_mean_return': e.spec.formation_mean_return,
                   'hour_utc': pd.Timestamp(t['entry_time']).hour,
                   'btc_vol_1d': float(btc_vol.loc[t['entry_time']]) if btc_vol is not None else np.nan}
            row.update({f"f_{k}": v for k, v in t['features'].items()})
            rows.append(row)
    return (pd.DataFrame(rows).sort_values(['entry_time', 'pair'], kind='stable').reset_index(drop=True)
            if rows else pd.DataFrame())


def summarize(cfg, segment, plan, index, bpd, windows, trades, equity, broker, risk, sim, t0, purpose='trial'):
    wdf = pd.DataFrame(windows)
    tstats = trade_stats(trades['return'].values if len(trades) else [])
    estats = equity_stats(equity, cfg.walkforward.initial_cash)
    trials = ledger.dev_trials(cfg.walkforward.data_dir)
    if purpose == 'trial':
        trials[cfg.fingerprint()] = estats.get('daily_sharpe', 0.0)
    _, daily = daily_returns(equity, cfg.walkforward.initial_cash)
    return {
        'name': cfg.name, 'fingerprint': cfg.fingerprint(), 'segment': segment, 'purpose': purpose,
        'data_dir': cfg.walkforward.data_dir, 'bars_per_day': bpd,
        'oos_start': str(index[plan[0][0]]), 'oos_end': str(index[plan[-1][1] - 1]),
        'n_windows': len(plan),
        'funnel_avg': {k: float(wdf[k].mean()) for k in ('n_eligible', 'n_universe', 'n_tests', 'n_fdr_pass',
                                                          'n_filter_pass', 'n_tradeable')},
        'windows_without_trades': int((wdf['pairs_traded'] == 0).sum()),
        'trades': tstats,
        'exits_by_reason': trades['exit_reason'].value_counts().to_dict() if len(trades) else {},
        'mean_bars_held': float(trades['bars_held'].mean()) if len(trades) else None,
        'equity': estats,
        'deflated_sharpe': {'n_trials': len(trials),
                            'probability': deflated_sharpe(daily.values, len(trials), list(trials.values()))
                            if len(daily) > 2 else None},
        'costs_paid': {'fees': broker.fees_paid, 'funding': broker.funding_paid},
        'risk': {'drawdown_halt_at': str(risk.halted_at) if risk.halted else None,
                 'concentration_rejections': risk.rejections,
                 'liquidations': len(sim.liquidations),
                 'min_margin_ratio': None if sim.min_margin_ratio == float('inf') else float(sim.min_margin_ratio)},
        'leverage': leverage_stats(trades, sim, cfg.risk.maintenance_margin),
        'runtime_seconds': round(time.time() - t0, 1),
    }


def leverage_stats(trades, sim, maintenance_margin):
    """Peak leverage, lowest margin ratio and the safe isolated-margin leverage."""
    worst_leg = float(np.nanmax(trades[['mae_y', 'mae_x']].values)) if len(trades) else None
    return {'max_gross_leverage': float(sim.max_leverage),
            'min_margin_ratio_cross': None if sim.min_margin_ratio == float('inf') else float(sim.min_margin_ratio),
            'maintenance_margin': maintenance_margin,
            'worst_leg_adverse_move': worst_leg,
            'max_isolated_leverage_without_liquidation':
                None if worst_leg is None else float(1.0 / (worst_leg + maintenance_margin))}


def save(result, cfg, outdir):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'config.json'), 'w') as f:
        json.dump(cfg.to_dict(), f, indent=2)
    with open(os.path.join(outdir, 'summary.json'), 'w') as f:
        json.dump(result['summary'], f, indent=2, default=str)
    for name in ('windows', 'trades', 'pairs'):
        result[name].to_csv(os.path.join(outdir, f'{name}.csv'), index=False)
    result['equity'].to_csv(os.path.join(outdir, 'equity.csv'))


def log_run(result, outdir):
    s = result['summary']
    ledger.append({'label': os.path.basename(outdir), 'name': s['name'], 'fingerprint': s['fingerprint'],
                   'segment': s['segment'], 'purpose': s.get('purpose', 'trial'),
                   'data_dir': s['data_dir'], 'oos_start': s['oos_start'],
                   'oos_end': s['oos_end'], 'n_trades': s['trades']['n_trades'],
                   'mean_return': s['trades'].get('mean_return'), 't_stat': s['trades'].get('t_stat'),
                   'total_return': s['equity']['total_return'], 'sharpe': s['equity'].get('sharpe'),
                   'daily_sharpe': s['equity'].get('daily_sharpe')})


def print_summary(s):
    t, e = s['trades'], s['equity']
    nan = float('nan')
    print(f"{s['name']} {s['segment']}: {s['oos_start'][:10]} to {s['oos_end'][:10]}, {s['n_windows']} windows")
    print(f"Return {e['total_return']:.2%}, Sharpe {e.get('sharpe', nan):.2f}, "
          f"DSR {s['deflated_sharpe']['probability'] or nan:.2f}, max drawdown {e['max_drawdown']:.2%}")
    if t['n_trades']:
        print(f"{t['n_trades']} trades, win rate {t['win_rate']:.1%}, mean {t['mean_return']:.4%}, "
              f"t-stat {t.get('t_stat', nan):.2f}")
