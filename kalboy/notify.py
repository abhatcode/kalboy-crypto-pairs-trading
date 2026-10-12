"""Discord alerts for the paper trader. The webhook URL comes from KALBOY_WEBHOOK_URL."""
import os

import pandas as pd
import requests

ENV_VAR = 'KALBOY_WEBHOOK_URL'
ACTION_TEXT = {
    'enter_long': 'entered a long spread position', 'enter_short': 'entered a short spread position',
    'exit_long': 'exited the long spread', 'exit_short': 'exited the short spread',
    'stop_loss': 'stop-loss exit', 'time_stop': 'time-stop exit', 'breakdown': 'pair retired after a breakdown',
    'period_end': 'closed at the end of the trading period', 'liquidation': 'liquidated',
}
MAX_LEN = 1900  # Discord rejects messages over 2000 characters


class Notifier:
    def __init__(self, url=None, timeout=10, post=None):
        self.url = url if url is not None else os.environ.get(ENV_VAR, '')
        self.timeout = timeout
        self._post = post or requests.post
        self.sent, self.failed = 0, 0

    @property
    def enabled(self):
        return bool(self.url)

    def send(self, text):
        print(f"[alert] {text}", flush=True)
        if not self.enabled:
            return False
        for i in range(0, len(text), MAX_LEN):
            try:
                r = self._post(self.url, json={'content': text[i:i + MAX_LEN]}, timeout=self.timeout)
                if r.status_code >= 300:
                    raise RuntimeError(f"HTTP {r.status_code}")
            except Exception as exc:  # noqa: BLE001
                self.failed += 1
                print(f"[alert] webhook failed: {exc!r}", flush=True)
                return False
        self.sent += 1
        return True


class Alerts:
    """Builds alert messages from the trader state and remembers what was sent."""

    def __init__(self, notifier, name):
        self.notifier = notifier
        self.name = name

    def after_pass(self, state, initial_cash):
        marks = state.setdefault('alerts', {'decisions': 0, 'trades': 0, 'formations': 0, 'day': None})
        sim = state['sim']
        lines = []

        for f in state['formations'][marks['formations']:]:
            pairs = ', '.join(f['pairs']) if f['pairs'] else 'no pairs this period'
            lines.append(f"Window {f['window']} re-formed at {f['start']}: {f['n_universe']} coins, "
                         f"{f['n_tests']} pairs tested, {f['n_fdr_pass']} passed the false discovery rate test. "
                         f"Trading: {pairs}.")
        marks['formations'] = len(state['formations'])

        for t, pair, action, z in sim.decision_log[marks['decisions']:]:
            lines.append(f"{pair}: {ACTION_TEXT.get(action, action.replace('_', ' '))} at the {t} close, "
                         f"z-score {z:.2f}. The order fills at the next bar's open.")
        marks['decisions'] = len(sim.decision_log)
        trades = [(e, t) for e in state['engines'] for t in e.trade_log]
        trades.sort(key=lambda et: str(et[1]['exit_fill_time']))
        for e, t in trades[marks['trades']:]:
            lines.append(f"Closed {e.y_name}/{e.x_name} after {t['bars_held']} bars "
                         f"({ACTION_TEXT.get(t['exit_reason'], t['exit_reason'])}): {t['return']:.3%} of equity, "
                         f"fees {t['fees']:,.2f}, funding {t['funding']:,.2f}.")
        marks['trades'] = len(trades)

        if state['equity']:
            t, eq = state['equity'][-1]
            day = str(pd.Timestamp(t).date())
            if marks['day'] is not None and day != marks['day']:
                values = [v for _, v in state['equity']]
                peak = max(values)
                n_open = sum(1 for e in sim.engines if e.status)
                lines.append(f"Daily summary for {marks['day']}: equity {eq:,.2f}, {eq / initial_cash - 1:.2%} "
                             f"since the start, drawdown {eq / peak - 1:.2%}. {len(trades)} closed trades so far. "
                             f"{len(sim.engines)} pairs this period, {n_open} with open positions.")
            marks['day'] = day

        if lines:
            self.notifier.send('\n'.join([f"Paper trader {self.name}"] + lines))
