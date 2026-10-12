"""The result tables in REPORT.md, drawn as images."""
import json
import os
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

INK, INK_2, MUTED = '#0b0b0b', '#52514e', '#8a8984'
RULE, HAIRLINE, SURFACE, HIGHLIGHT = '#c9c8c3', '#e6e5e1', '#ffffff', '#eef4fc'
TIMEFRAMES = (('h1', '1 hour'), ('m15', '15 minutes'), ('m5', '5 minutes'))


def _summary(run, results_dir):
    with open(os.path.join(results_dir, run, 'summary.json')) as f:
        return json.load(f)


def _pct(v, digits=2):
    return f"{v * 100:.{digits}f}%"


def draw(path, title, subtitle, columns, rows, numeric, widths, note=None, highlight=()):
    row_h, pad = 0.34, 0.3
    widths = list(widths)
    probe = plt.figure(dpi=200)
    renderer = probe.canvas.get_renderer()
    text_w = max(probe.text(0, 0, s, fontsize=fs, weight=wt).get_window_extent(renderer).width / 200
                 for s, fs, wt in ((title, 12.5, 'bold'), (subtitle, 9, 'normal'), (note or '', 7.8, 'normal')))
    plt.close(probe)
    widths[0] += max(0.0, text_w + 0.1 - sum(widths))
    width = sum(widths) + 2 * pad
    note_lines = note.count('\n') + 1 if note else 0
    height = 1.15 + row_h * (len(rows) + 1) + 0.12 + 0.2 * note_lines + 0.2
    fig = plt.figure(figsize=(width, height), dpi=200)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis('off')
    ax.text(pad, 0.42, title, fontsize=12.5, weight='bold', color=INK, va='baseline')
    ax.text(pad, 0.74, subtitle, fontsize=9, color=INK_2, va='baseline')

    edges = [pad]
    for w in widths:
        edges.append(edges[-1] + w)

    def cell(text, i, y, **kw):
        if numeric[i]:
            ax.text(edges[i + 1] - 0.08, y, text, ha='right', va='center', **kw)
        else:
            ax.text(edges[i] + 0.04, y, text, ha='left', va='center', **kw)

    y = 1.15
    for i, c in enumerate(columns):
        cell(c, i, y + row_h / 2, fontsize=8.5, weight='bold', color=INK_2)
    y += row_h
    ax.plot([pad, width - pad], [y, y], color=RULE, lw=1.0)
    for r, row in enumerate(rows):
        if r in highlight:
            ax.add_patch(Rectangle((pad, y), width - 2 * pad, row_h, color=HIGHLIGHT, lw=0, zorder=0))
        for i, v in enumerate(row):
            cell(v, i, y + row_h / 2, fontsize=9, color=INK)
        y += row_h
        ax.plot([pad, width - pad], [y, y], color=HAIRLINE, lw=0.8)
    if note:
        ax.text(pad, y + 0.3, note, fontsize=7.8, color=MUTED, va='top', linespacing=1.5)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def dev_results(results_dir, out):
    from kalboy import ledger
    n_trials = len(ledger.dev_trials(path=os.path.join(results_dir, 'ledger.jsonl')))
    rows, variants = [], (('base', 'Base rules'), ('tail', 'With stop-loss'), ('allpairs', 'No Kelly gate'))
    for tf, tf_name in TIMEFRAMES:
        for v, v_name in variants:
            s = _summary(f"{tf}_{v}_dev", results_dir)
            t, e = s['trades'], s['equity']
            rows.append([tf_name, v_name, f"{t['n_trades']:,}", _pct(t['mean_return'], 3), f"{t['t_stat']:.2f}",
                         _pct(e['total_return']), _pct(e['max_drawdown'])])
    return draw(os.path.join(out, 'table1_dev_results.png'),
                'Walk-forward results on the dev segment',
                'Every trade is out of sample and net of fees, slippage and funding. Starting capital 100,000.',
                ['Bars', 'Strategy', 'Trades', 'Mean trade', 't-stat', 'Net return', 'Max drawdown'], rows,
                [False, False, True, True, True, True, True], [1.1, 1.35, 0.8, 1.0, 0.75, 1.0, 1.15],
                note='Dev segment: Nov 2024 to Jun 2026 for 1-hour bars, Jan 2025 to Jun 2026 for 15- and 5-minute bars.\n'
                     'Mean trade is profit or loss as a share of equity. With stop-loss adds a stop at a z-score of 3.5 and a time stop.\n'
                     f'No configuration is statistically significant after accounting for the {n_trials} configurations tried.')


def cost_sensitivity(results_dir, out):
    levels = (('', '9 bps (taker)'), ('_fee5', '5 bps'), ('_fee2', '2 bps'), ('_fee0', 'No costs'))
    rows = []
    for tf, tf_name in TIMEFRAMES:
        for v, v_name in (('base', 'Base rules'), ('tail', 'With stop-loss')):
            cells = [tf_name, v_name]
            for suffix, _ in levels:
                cells.append(_pct(_summary(f"{tf}_{v}{suffix}_dev", results_dir)['equity']['total_return']))
            rows.append(cells)
    m5 = _summary('m5_base_fee0_dev', results_dir)['trades']
    return draw(os.path.join(out, 'table2_cost_sensitivity.png'),
                'Net return on the dev segment at different trading costs',
                'Cost per fill is the exchange fee plus slippage. The base case pays the taker fee of 0.06% plus 3 bps slippage.',
                ['Bars', 'Strategy'] + [name for _, name in levels], rows,
                [False, False, True, True, True, True], [1.1, 1.35, 1.2, 0.85, 0.85, 0.95],
                highlight=(4, 5),
                note=f"At 1-hour and 15-minute bars every run loses money even with no costs. At 5-minute bars there is a real\n"
                     f"signal without costs ({m5['n_trades']:,} trades, t-stat {m5['t_stat']:.1f}), about 4 bps of notional per trade, "
                     f"while a round trip at\ntaker rates costs about 18 bps. Lower costs also let more pairs pass the Kelly gate.")


def holdout(results_dir, out):
    dev, ho = _summary('h1_tail_dev', results_dir), _summary('h1_tail_holdout', results_dir)
    repro = _summary('h1_tail_repro_dev', results_dir)

    def gross(s):
        return (s['equity']['total_return'] * 100_000.0 + s['costs_paid']['fees'] + s['costs_paid']['funding']) / 100_000.0

    def traded(s):
        return f"{s['n_windows'] - s['windows_without_trades']} of {s['n_windows']}"

    rows = [
        ['Net return', _pct(dev['equity']['total_return']), _pct(ho['equity']['total_return'])],
        ['Return before fees and funding', _pct(gross(dev)), _pct(gross(ho))],
        ['Trades', f"{dev['trades']['n_trades']}", f"{ho['trades']['n_trades']}"],
        ['Win rate', _pct(dev['trades']['win_rate'], 1), _pct(ho['trades']['win_rate'], 1)],
        ['Mean trade', _pct(dev['trades']['mean_return'], 3), _pct(ho['trades']['mean_return'], 3)],
        ['t-stat of the mean trade', f"{dev['trades']['t_stat']:.2f}", f"{ho['trades']['t_stat']:.2f}"],
        ['Probabilistic Sharpe', f"{dev['equity']['probabilistic_sharpe']:.2f}", f"{ho['equity']['probabilistic_sharpe']:.2f}"],
        ['Max drawdown', _pct(dev['equity']['max_drawdown']), _pct(ho['equity']['max_drawdown'])],
        ['Trading windows with trades', traded(dev), traded(ho)],
        ['Peak leverage', f"{repro['leverage']['max_gross_leverage']:.2f} times", f"{ho['leverage']['max_gross_leverage']:.2f} times"],
        ['Liquidations', f"{dev['risk']['liquidations']}", f"{ho['risk']['liquidations']}"],
    ]
    return draw(os.path.join(out, 'table3_holdout.png'),
                'Final configuration: development period against the sealed holdout',
                '1-hour bars with a stop-loss, frozen before the holdout was opened. The holdout was run exactly once.',
                ['', 'Dev, Nov 2024 to Jun 2026', 'Holdout, Jul to Oct 2026'], rows,
                [False, True, True], [2.6, 2.0, 2.0],
                note='Probabilistic Sharpe is the estimated probability that the true Sharpe ratio is above zero; 0.5 is a coin flip.\n'
                     'Peak leverage is gross notional divided by equity. Dev leverage comes from the reproduction run with the current code.')


def leverage(results_dir, out):
    rows = []
    frozen = _summary('h1_tail_repro_dev', results_dir)['leverage']
    for run, size in (('h1_tail_repro_dev', '10% (frozen)'), ('h1_tail_lev5_dev', '50%'), ('h1_tail_lev10_dev', '100%'),
                      ('h1_tail_lev20_dev', '200%'), ('h1_tail_lev50_dev', '500%')):
        s = _summary(run, results_dir)
        lv = s['leverage']['max_gross_leverage']
        rows.append([size, 'account wiped out' if lv > 100 else f"{lv:.2f} times", _pct(s['equity']['total_return']),
                     _pct(s['equity']['max_drawdown'], 1), f"{s['trades']['t_stat']:.2f}", f"{s['risk']['liquidations']}"])
    return draw(os.path.join(out, 'table4_leverage.png'),
                'The final configuration at larger position sizes',
                'Dev segment, 1-hour bars with a stop-loss. Size is the gross notional of one pair trade as a share of equity.',
                ['Size per pair', 'Peak leverage', 'Net return', 'Max drawdown', 't-stat', 'Liquidations'], rows,
                [False, True, True, True, True, True], [1.4, 1.6, 1.0, 1.15, 0.7, 1.0], highlight=(0,),
                note='Leverage adds no edge: the t-stat is the same at every size. Above about one times equity per pair, drawdowns\n'
                     f"outgrow the gains. The largest move of a single leg against its position was {_pct(frozen['worst_leg_adverse_move'], 1)}, "
                     f"so under isolated margin\nany leg levered above {frozen['max_isolated_leverage_without_liquidation']:.1f} times "
                     f"would have been liquidated even with the other leg hedging it.")


def write_all(results_dir='results', out='docs'):
    os.makedirs(out, exist_ok=True)
    paths = [dev_results(results_dir, out), cost_sensitivity(results_dir, out), holdout(results_dir, out),
             leverage(results_dir, out)]
    chart = os.path.join(results_dir, 'cost_sensitivity.png')
    if os.path.exists(chart):
        paths.append(shutil.copy(chart, os.path.join(out, 'chart_cost_sensitivity.png')))
    return paths
