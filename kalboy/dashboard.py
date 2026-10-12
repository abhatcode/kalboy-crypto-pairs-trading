"""HTML dashboard for a paper-trading run."""
import html
import json
import os

import numpy as np
import pandas as pd

from kalboy.stats import daily_returns


def backtest_reference(cfg, results_dir='results'):
    """Mean and std of daily returns from backtests of the same config."""
    rets, sources, seen = [], [], set()
    names = sorted(os.listdir(results_dir)) if os.path.isdir(results_dir) else []
    for label in names:
        d = os.path.join(results_dir, label)
        try:
            with open(os.path.join(d, 'summary.json')) as f:
                s = json.load(f)
            if s['fingerprint'] != cfg.fingerprint() or s['segment'] in seen \
                    or s.get('purpose', 'trial') not in ('trial', 'reproduction'):
                continue
            eq = pd.read_csv(os.path.join(d, 'equity.csv'), index_col=0, parse_dates=True).iloc[:, 0]
        except (OSError, KeyError, ValueError):
            continue
        seen.add(s['segment'])
        rets.append(daily_returns(eq, cfg.walkforward.initial_cash)[1])
        sources.append(label)
    if not rets:
        return None, None, []
    r = pd.concat(rets)
    return float(r.mean()), float(r.std(ddof=1)), sources


def _payload(state, cfg, mu, sigma):
    eq = pd.Series({pd.Timestamp(t): v for t, v in state['equity']}, dtype=float).sort_index()
    initial = cfg.walkforward.initial_cash
    if len(eq) > 1500:
        eq = eq.resample('1h').last().dropna()
    t0 = eq.index[0] if len(eq) else None
    points = []
    for t, v in eq.items():
        d = (t - t0) / pd.Timedelta(days=1)
        row = {'t': t.strftime('%Y-%m-%d %H:%M'), 'r': v / initial - 1}
        if mu is not None:
            row['lo'] = mu * d - 2 * sigma * np.sqrt(d)
            row['hi'] = mu * d + 2 * sigma * np.sqrt(d)
        points.append(row)
    return points


def _table(headers, rows):
    head = ''.join(f'<th>{html.escape(h)}</th>' for h in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(c))}</td>' for c in r) + '</tr>' for r in rows)
    if not rows:
        body = f'<tr><td colspan="{len(headers)}" class="empty">none yet</td></tr>'
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def build(state, cfg, generated_at=None, results_dir='results'):
    initial = cfg.walkforward.initial_cash
    mu, sigma, sources = backtest_reference(cfg, results_dir)
    points = _payload(state, cfg, mu, sigma)
    values = [v for _, v in state['equity']]
    equity = values[-1] if values else initial
    peak = np.maximum.accumulate(values) if values else np.array([initial])
    max_dd = float(np.min(np.array(values) / peak - 1)) if values else 0.0
    trades = sorted(((e, t) for e in state['engines'] for t in e.trade_log),
                    key=lambda et: str(et[1]['exit_fill_time']))
    wins = sum(t['return'] > 0 for _, t in trades)
    sim = state['sim']
    open_rows = [(f"{e.y_name}/{e.x_name}", 'long spread' if e.status > 0 else 'short spread',
                  str(e._trade['entry_time'])[:16] if e._trade else '', f"{e._trade['entry_z']:+.2f}" if e._trade else '',
                  f"{e.z:+.2f}") for e in sim.engines if e.status]
    trade_rows = [(f"{e.y_name}/{e.x_name}", str(t['entry_time'])[:16], str(t['exit_time'])[:16], t['exit_reason'],
                   t['bars_held'], f"{t['return']:+.3%}", f"{t['fees']:,.2f}", f"{t['funding']:+,.2f}")
                  for e, t in reversed(trades[-25:])]
    form_rows = [(f['window'], str(f['start'])[:16], f['n_universe'], f['n_tests'], f['n_fdr_pass'],
                  ', '.join(f['pairs']) or 'none') for f in reversed(state['formations'][-10:])]
    first = str(state['equity'][0][0])[:16] if state['equity'] else 'not started'
    last = str(state['equity'][-1][0])[:16] if state['equity'] else '-'
    generated_at = generated_at or pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')

    tiles = [('Equity', f"${equity:,.0f}"), ('Return since start', f"{equity / initial - 1:+.2%}"),
             ('Max drawdown', f"{max_dd:.2%}"), ('Closed trades', f"{len(trades)}"),
             ('Win rate', f"{wins / len(trades):.0%}" if trades else '-'),
             ('Pairs trading now', f"{len(sim.engines)} ({sum(1 for e in sim.engines if e.status)} open)")]
    tiles_html = ''.join(f'<div class="tile"><div class="label">{html.escape(a)}</div>'
                         f'<div class="value">{html.escape(b)}</div></div>' for a, b in tiles)
    if mu is not None:
        ref = (f"Backtest range: this config's walk-forward ({html.escape(', '.join(sources))}), "
               f"with a mean daily return of {mu:.4%} and a standard deviation of {sigma:.4%}. The band is the "
               f"mean times the number of days, plus or minus two standard deviations times the square root "
               f"of the number of days.")
    else:
        ref = 'No walk-forward run of this exact config found in results/, so no backtest range is drawn.'
    data = json.dumps(points).replace('</', '<\\/')

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kalboy paper trading</title>
<style>
:root {{ color-scheme: light; --surface:#fcfcfb; --card:#ffffff; --ink:#0b0b0b; --ink-2:#52514e; --muted:#8a8984;
  --line:#e6e5e1; --series:#2a78d6; --band:rgba(82,81,78,0.12); }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ color-scheme: dark; --surface:#1a1a19;
  --card:#222221; --ink:#f0efec; --ink-2:#c3c2b7; --muted:#8f8e86; --line:#34342f; --series:#3987e5;
  --band:rgba(195,194,183,0.14); }} }}
:root[data-theme="dark"] {{ color-scheme: dark; --surface:#1a1a19; --card:#222221; --ink:#f0efec; --ink-2:#c3c2b7;
  --muted:#8f8e86; --line:#34342f; --series:#3987e5; --band:rgba(195,194,183,0.14); }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--surface); color:var(--ink); font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1100px; margin:0 auto; padding:32px 16px 48px; }}
h1 {{ font-size:24px; margin:0 0 4px; font-weight:650; }}
h2 {{ font-size:16px; margin:32px 0 10px; font-weight:600; }}
.sub, .note {{ color:var(--ink-2); font-size:13px; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:24px 0; }}
.tile {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }}
.tile .label {{ color:var(--ink-2); font-size:13px; }}
.tile .value {{ font-size:22px; font-weight:600; font-variant-numeric:tabular-nums; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; }}
.legend {{ display:flex; gap:18px; font-size:13px; color:var(--ink-2); margin-bottom:8px; flex-wrap:wrap; }}
.key {{ display:inline-block; vertical-align:middle; margin-right:6px; }}
.key.line {{ width:16px; height:2px; background:var(--series); }}
.key.band {{ width:14px; height:10px; background:var(--band); border-radius:2px; }}
#chart {{ position:relative; }}
svg {{ display:block; width:100%; height:300px; }}
.tip {{ position:absolute; pointer-events:none; background:var(--card); border:1px solid var(--line); border-radius:8px;
  padding:6px 10px; font-size:12px; box-shadow:0 2px 8px rgba(0,0,0,.12); display:none; white-space:nowrap; }}
.tip b {{ font-size:14px; font-variant-numeric:tabular-nums; }}
.scroll {{ overflow-x:auto; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; font-variant-numeric:tabular-nums; }}
th {{ text-align:left; color:var(--ink-2); font-weight:500; border-bottom:1px solid var(--line); padding:6px 8px; white-space:nowrap; }}
td {{ border-bottom:1px solid var(--line); padding:6px 8px; white-space:nowrap; }}
td.empty {{ color:var(--muted); }}
</style></head><body><main>
<h1>Kalboy paper trading: {html.escape(cfg.name)}</h1>
<div class="sub">Simulated fills on live KuCoin Futures prices, no real orders. Config fingerprint
<code>{cfg.fingerprint()}</code>. Bars {html.escape(first)} to {html.escape(last)} UTC. Updated {html.escape(generated_at)}.</div>
<div class="tiles">{tiles_html}</div>
<div class="card">
  <div class="legend"><span><span class="key line"></span>Paper equity, cumulative return</span>
  {'<span><span class="key band"></span>Backtest range (about 95%)</span>' if mu is not None else ''}</div>
  <div id="chart"><svg id="svg" role="img" aria-label="Paper equity cumulative return over time"></svg><div class="tip" id="tip"></div></div>
  <div class="note">{ref}</div>
</div>
<h2>Open positions</h2><div class="card scroll">{_table(['Pair', 'Side', 'Entry (bar close)', 'Entry z', 'Current z'], open_rows)}</div>
<h2>Closed trades (latest 25)</h2><div class="card scroll">{_table(['Pair', 'Entry', 'Exit', 'Reason', 'Bars', 'Return (of equity)', 'Fees', 'Funding'], trade_rows)}</div>
<h2>Pair formations (latest 10)</h2><div class="card scroll">{_table(['Window', 'Start', 'Universe', 'Tests', 'FDR pass', 'Trading'], form_rows)}</div>
<p class="note">Each period the system re-selects pairs from the trailing {cfg.walkforward.formation_days:g} days only
(cointegration tests with Benjamini-Hochberg correction, then a quarter-Kelly gate on the pair's own formation trades),
and trades them for {cfg.walkforward.trading_days:g} days with taker fees ({cfg.costs.taker_fee:.2%}), {cfg.costs.slippage_bps:g} bps
slippage and actual funding. The code path is the one the walk-forward backtest uses.</p>
</main>
<script>
const pts = {data};
const svg = document.getElementById('svg'), tip = document.getElementById('tip');
const NS = 'http://www.w3.org/2000/svg';
function el(name, attrs) {{ const e = document.createElementNS(NS, name); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; }}
function pct(v) {{ return (v >= 0 ? '+' : '') + (v * 100).toFixed(2) + '%'; }}
function draw() {{
  svg.textContent = '';
  if (pts.length < 2) {{ const t = el('text', {{x: 12, y: 30, fill: 'var(--muted)', 'font-size': 13}}); t.textContent = 'Waiting for at least two bars.'; svg.appendChild(t); return; }}
  const W = svg.clientWidth, H = svg.clientHeight, L = 56, R = 12, T = 10, B = 26;
  const vals = pts.flatMap(p => [p.r, p.lo, p.hi].filter(v => v !== undefined));
  let lo = Math.min(0, ...vals), hi = Math.max(0, ...vals); const pad = (hi - lo) * 0.08 || 0.001; lo -= pad; hi += pad;
  const x = i => L + (W - L - R) * i / (pts.length - 1), y = v => T + (H - T - B) * (hi - v) / (hi - lo);
  const raw = (hi - lo) / 5, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 5, 10].find(m => m * mag >= raw) * mag, dec = Math.max(0, -Math.floor(Math.log10(step * 100)));
  for (let k = Math.ceil(lo / step); k * step <= hi; k++) {{
    const v = k * step;
    svg.appendChild(el('line', {{x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: 'var(--line)', 'stroke-width': 1}}));
    const t = el('text', {{x: L - 8, y: y(v) + 4, 'text-anchor': 'end', fill: 'var(--ink-2)', 'font-size': 11}}); t.textContent = (v * 100).toFixed(dec) + '%'; svg.appendChild(t);
  }}
  [0, pts.length - 1].forEach((i, k) => {{ const t = el('text', {{x: x(i), y: H - 6, 'text-anchor': k ? 'end' : 'start', fill: 'var(--ink-2)', 'font-size': 11}}); t.textContent = pts[i].t.slice(0, 10); svg.appendChild(t); }});
  if (pts[0].lo !== undefined) {{
    const up = pts.map((p, i) => `${{x(i)}},${{y(p.hi)}}`), dn = pts.map((p, i) => `${{x(i)}},${{y(p.lo)}}`).reverse();
    svg.appendChild(el('polygon', {{points: up.concat(dn).join(' '), fill: 'var(--band)'}}));
  }}
  svg.appendChild(el('path', {{d: 'M' + pts.map((p, i) => `${{x(i)}},${{y(p.r)}}`).join('L'), fill: 'none', stroke: 'var(--series)', 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round'}}));
  const n = pts.length - 1;
  svg.appendChild(el('circle', {{cx: x(n), cy: y(pts[n].r), r: 4, fill: 'var(--series)', stroke: 'var(--card)', 'stroke-width': 2}}));
  const cross = el('line', {{y1: T, y2: H - B, stroke: 'var(--muted)', 'stroke-width': 1, visibility: 'hidden'}}); svg.appendChild(cross);
  const dot = el('circle', {{r: 4, fill: 'var(--series)', stroke: 'var(--card)', 'stroke-width': 2, visibility: 'hidden'}}); svg.appendChild(dot);
  svg.onpointermove = ev => {{
    const b = svg.getBoundingClientRect(); const i = Math.max(0, Math.min(n, Math.round((ev.clientX - b.left - L) / (W - L - R) * n)));
    const p = pts[i]; cross.setAttribute('x1', x(i)); cross.setAttribute('x2', x(i)); dot.setAttribute('cx', x(i)); dot.setAttribute('cy', y(p.r));
    cross.setAttribute('visibility', 'visible'); dot.setAttribute('visibility', 'visible');
    tip.textContent = ''; const v = document.createElement('b'); v.textContent = pct(p.r); tip.appendChild(v);
    tip.appendChild(document.createTextNode(' paper, ' + p.t + ' UTC'));
    if (p.lo !== undefined) {{ tip.appendChild(document.createElement('br')); tip.appendChild(document.createTextNode('backtest range ' + pct(p.lo) + ' to ' + pct(p.hi))); }}
    tip.style.display = 'block'; const tx = x(i) + 12; tip.style.left = Math.min(tx, W - tip.offsetWidth - 4) + 'px'; tip.style.top = '8px';
  }};
  svg.onpointerleave = () => {{ cross.setAttribute('visibility', 'hidden'); dot.setAttribute('visibility', 'hidden'); tip.style.display = 'none'; }};
}}
draw(); addEventListener('resize', draw);
</script></body></html>
"""


def write(out_dir, state, cfg, results_dir='results'):
    path = os.path.join(out_dir, 'dashboard.html')
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(build(state, cfg, results_dir=results_dir))
    os.replace(tmp, path)
    return path
