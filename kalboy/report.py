"""Charts and summary pages for finished runs."""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd


def _load(run_dir):
    with open(os.path.join(run_dir, 'summary.json')) as f:
        summary = json.load(f)
    equity = pd.read_csv(os.path.join(run_dir, 'equity.csv'), index_col=0, parse_dates=True).iloc[:, 0]
    trades_path = os.path.join(run_dir, 'trades.csv')
    trades = pd.read_csv(trades_path) if os.path.getsize(trades_path) > 2 else pd.DataFrame()
    windows = pd.read_csv(os.path.join(run_dir, 'windows.csv'))
    return summary, equity, trades, windows


def charts(run_dir):
    summary, equity, trades, windows = _load(run_dir)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    daily = equity.resample('1D').last().dropna()
    ax.plot(daily.index, daily.values / daily.iloc[0] - 1, color='#1f6feb')
    ax.axhline(0, color='grey', lw=0.8)
    ax.set_title('Out-of-sample equity (cumulative return)')
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

    ax = axes[0, 1]
    dd = daily / daily.cummax() - 1
    ax.fill_between(dd.index, dd.values, 0, color='#d1242f', alpha=0.4)
    ax.set_title('Drawdown')
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

    ax = axes[1, 0]
    if len(trades):
        ax.hist(trades['return'] * 100, bins=50, color='#6e7781')
        ax.axvline(0, color='black', lw=0.8)
    ax.set_title('Trade returns (% of equity)')

    ax = axes[1, 1]
    cols = ['n_universe', 'n_tests', 'n_fdr_pass', 'n_filter_pass', 'n_tradeable']
    means = [windows[c].mean() for c in cols]
    ax.bar(['universe', 'tests', 'FDR pass', 'filters', 'tradeable'], means, color='#8250df')
    ax.set_yscale('log')
    ax.set_title('Selection funnel (mean per window, log scale)')
    for a in (axes[0, 0], axes[0, 1]):
        a.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator(maxticks=6))
        a.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m'))
    fig.suptitle(f"{summary['name']} [{summary['segment']}] {summary['oos_start'][:10]} to {summary['oos_end'][:10]}")
    fig.tight_layout()
    path = os.path.join(run_dir, 'report.png')
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def markdown(run_dir):
    s, equity, trades, windows = _load(run_dir)
    t, e, fa = s['trades'], s['equity'], s['funnel_avg']
    nan = float('nan')
    dsr = s['deflated_sharpe']['probability']
    lines = [f"# {s['name']}, {s['segment']} segment", '',
             f"Out of sample from {s['oos_start'][:10]} to {s['oos_end'][:10]}, {s['n_windows']} trading windows. "
             f"Config fingerprint {s['fingerprint']}.", '', '![charts](report.png)', '',
             f"The run returned {e['total_return']:.2%} after all costs, with a maximum drawdown of "
             f"{e['max_drawdown']:.2%}. The annualized Sharpe ratio is {e.get('sharpe', nan):.2f}, the probabilistic "
             f"Sharpe {e.get('probabilistic_sharpe', nan):.2f}, and the deflated Sharpe "
             f"{dsr if dsr is not None else nan:.2f} against {s['deflated_sharpe']['n_trials']} trials."]
    if t['n_trades'] > 1:
        exits = ', '.join(f"{n} {k.replace('_', ' ')}" for k, n in s['exits_by_reason'].items())
        lines += ['', f"It made {t['n_trades']} trades and {t['win_rate']:.1%} of them won. The mean trade returned "
                      f"{t['mean_return']:.4%} of equity, with a t-stat of {t['t_stat']:.2f} and a 95% confidence "
                      f"interval from {t['mean_ci95'][0]:.4%} to {t['mean_ci95'][1]:.4%}. The worst 5% of trades "
                      f"returned {t['worst_5pct_sum']:.2%} in total and the other 95% returned {t['rest_sum']:.2%}. "
                      f"Exits: {exits}. Trades were held for {s['mean_bars_held']:.0f} bars on average."]
    fund = s['costs_paid']['funding']
    lines += ['', f"Fees paid were ${s['costs_paid']['fees']:,.0f} and funding {'received' if fund < 0 else 'paid'} was "
                  f"${abs(fund):,.0f}. On average each window had {fa['n_universe']:.0f} contracts "
                  f"in the universe, {fa['n_tests']:.0f} pairs tested, {fa['n_fdr_pass']:.1f} passing the false "
                  f"discovery rate test, {fa['n_filter_pass']:.1f} passing the filters and {fa['n_tradeable']:.1f} "
                  f"passing the Kelly gate. {s['windows_without_trades']} of {s['n_windows']} windows had nothing to trade."]
    lv = s.get('leverage', {})
    if lv.get('worst_leg_adverse_move') is not None:
        lines += ['', f"Leverage peaked at {lv['max_gross_leverage']:.2f} times equity and the lowest margin ratio was "
                      f"{lv['min_margin_ratio_cross']:.2f}, against a maintenance requirement of "
                      f"{lv['maintenance_margin']:.2%}. There were {s['risk']['liquidations']} liquidations. The largest "
                      f"move of one leg against its position was {lv['worst_leg_adverse_move']:.2%}."]
    path = os.path.join(run_dir, 'report.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return path


def compare(run_dirs, out_path):
    """One row per run, with the DSR recomputed against every dev trial in the ledger."""
    from kalboy import ledger
    from kalboy.stats import daily_returns, deflated_sharpe
    trials = ledger.dev_trials()
    rows = []
    for d in run_dirs:
        s, equity, *_ = _load(d)
        t, e = s['trades'], s['equity']
        _, daily = daily_returns(equity, 100_000.0)
        dsr_all = deflated_sharpe(daily.values, len(trials), list(trials.values())) if len(daily) > 2 else None
        rows.append({'run': os.path.basename(d), 'segment': s['segment'], 'trades': t['n_trades'],
                     'win': t.get('win_rate'), 'mean': t.get('mean_return'), 't': t.get('t_stat'),
                     'total': e['total_return'], 'sharpe': e.get('sharpe'), 'max_dd': e['max_drawdown'],
                     'psr': e.get('probabilistic_sharpe'), 'dsr_all_trials': dsr_all, 'n_trials': len(trials),
                     'tradeable_per_window': s['funnel_avg']['n_tradeable']})
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


SERIES = ('#2a78d6', '#eb6834', '#1baf7a')


def sensitivity(run_dirs, out_path):
    """Net return against cost per fill, as a CSV and a chart."""
    rows = []
    for d in run_dirs:
        s, *_ = _load(d)
        with open(os.path.join(d, 'config.json')) as f:
            costs = json.load(f)['costs']
        t, e = s['trades'], s['equity']
        net = e['total_return'] * 100_000.0
        rows.append({'run': os.path.basename(d), 'config': s['name'], 'purpose': s.get('purpose', 'trial'),
                     'cost_bps_per_fill': round(costs['taker_fee'] * 1e4 + costs['slippage_bps'], 2),
                     'trades': t['n_trades'], 'mean_trade': t.get('mean_return'), 't': t.get('t_stat'),
                     'total_return': e['total_return'], 'fees': s['costs_paid']['fees'],
                     'funding': s['costs_paid']['funding'],
                     'gross_return': (net + s['costs_paid']['fees'] + s['costs_paid']['funding']) / 100_000.0,
                     'tradeable_per_window': s['funnel_avg']['n_tradeable']})
    df = pd.DataFrame(rows).sort_values(['config', 'cost_bps_per_fill'], ascending=[True, False])
    base = out_path[:-4] if out_path.endswith('.csv') else out_path
    df.to_csv(base + '.csv', index=False)

    df['tf'] = df['config'].str.split('_').str[0]
    df['variant'] = df['config'].str.split('_', n=1).str[1]
    variants = sorted(df['variant'].unique())
    fig, axes = plt.subplots(1, len(variants), figsize=(5.2 * len(variants), 4), sharey=True, squeeze=False)
    for ax, v in zip(axes[0], variants):
        for color, tf in zip(SERIES, ('h1', 'm15', 'm5')):
            g = df[(df['variant'] == v) & (df['tf'] == tf)].sort_values('cost_bps_per_fill')
            if not len(g):
                continue
            ax.plot(g['cost_bps_per_fill'], g['total_return'] * 100, color=color, lw=2, marker='o', ms=6,
                    mec='white', mew=2, solid_capstyle='round', label=tf)
        ax.axhline(0, color='#8a8984', lw=1)
        ax.set_title(f"{v}", fontsize=11, color='#0b0b0b')
        ax.set_xlabel('cost per fill (bps): fee + slippage', color='#52514e')
        ax.grid(axis='y', color='#e6e5e1', lw=1)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.set_xticks(sorted(df['cost_bps_per_fill'].unique()))
        ax.set_xlim(-0.5, df['cost_bps_per_fill'].max() + 0.5)
    axes[0][0].set_ylabel('dev net return (%)', color='#52514e')
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 0.93),
               title='timeframe', title_fontsize=9)
    fig.suptitle('Dev-period net return vs trading cost (9 bps = taker base case)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(base + '.png', dpi=130)
    plt.close(fig)
    return df


def paper_markdown(paper_dir):
    from kalboy.stats import trade_stats, equity_stats
    eq = pd.read_csv(os.path.join(paper_dir, 'equity.csv'), parse_dates=['time']).set_index('time')['equity']
    tpath = os.path.join(paper_dir, 'trades.csv')
    trades = pd.read_csv(tpath) if os.path.exists(tpath) and os.path.getsize(tpath) > 2 else pd.DataFrame()
    forms = pd.read_csv(os.path.join(paper_dir, 'formations.csv'))
    t = trade_stats(trades['return'].values if len(trades) else [])
    e = equity_stats(eq, initial=eq.iloc[0]) if len(eq) > 1 else {'total_return': 0.0, 'max_drawdown': 0.0}
    lines = [f"# Paper trading: {os.path.basename(os.path.normpath(paper_dir))}", '',
             f"From {eq.index[0]} to {eq.index[-1]}, {len(eq)} bars and {len(forms)} pair formations.", '',
             f"Equity is {eq.iloc[-1]:,.2f}, a return of {e['total_return']:.2%}, with a maximum drawdown of "
             f"{e['max_drawdown']:.2%}. Completed trades: {t['n_trades']}."
             + (f" {t['win_rate']:.1%} of them won and the mean trade returned {t['mean_return']:.4%} of equity."
                if t['n_trades'] else ''), '', '## Formations', '']
    for _, r in forms.iterrows():
        pairs = r['pairs'] if r['pairs'] not in ('[]', '') else 'none'
        lines.append(f"- Window {r['window']}, starting {r['start']}: {r['n_universe']} contracts, {r['n_tests']} pairs "
                     f"tested, {r['n_fdr_pass']} passed the false discovery rate test, {r['n_tradeable']} traded "
                     f"({pairs}).")
    path = os.path.join(paper_dir, 'report.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return path
