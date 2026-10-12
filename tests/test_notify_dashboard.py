import pandas as pd

from kalboy import dashboard
from kalboy.notify import Alerts, Notifier
from kalboy.paper import PaperTrader, ReplaySource
from tests.conftest import small_config

START, END = 480, 1200


class FakePost:
    def __init__(self, status=204, fail=False):
        self.messages, self.status, self.fail = [], status, fail

    def __call__(self, url, json, timeout):
        if self.fail:
            raise ConnectionError('discord is down')
        self.messages.append(json['content'])
        return type('R', (), {'status_code': self.status})()


def replay(panel, out_dir, post, restart_at=None, dash=False):
    cfg = small_config()
    index = panel['close'].index
    make = lambda: PaperTrader(cfg, out_dir, ReplaySource(panel), alerts=Alerts(Notifier('https://x', post=post), 't'),
                               dashboard=dash)
    trader = make()
    for i in range(START, END):
        if restart_at is not None and i == restart_at:
            trader = make()
        trader.source.end_ms = int((index[i] + pd.Timedelta(hours=1)).value // 1_000_000)
        trader.process_new_bars()
    return trader


def test_each_event_reported_exactly_once(panel, tmp_path):
    post = FakePost()
    trader = replay(panel, str(tmp_path / 'a'), post, restart_at=900)
    text = '\n'.join(post.messages)
    st = trader.state
    n_decisions = len(st['sim'].decision_log)
    n_trades = sum(len(e.trade_log) for e in st['engines'])
    assert n_trades >= 5
    assert text.count("fills at the next bar's open") == n_decisions
    assert text.count('Closed ') == n_trades
    assert text.count(' re-formed at ') == len(st['formations'])
    days = pd.Series([pd.Timestamp(t).date() for t, _ in st['equity']]).nunique()
    assert text.count('Daily summary') == days - 1


def test_failing_webhook_does_not_stop_trading(panel, tmp_path):
    ok = replay(panel, str(tmp_path / 'ok'), FakePost())
    bad = replay(panel, str(tmp_path / 'bad'), FakePost(fail=True))
    assert bad.alerts.notifier.failed > 0
    assert bad.state['equity'] == ok.state['equity']


def test_dashboard_shows_the_traders_numbers(panel, tmp_path):
    trader = replay(panel, str(tmp_path / 'd'), FakePost(), dash=True)
    page = (tmp_path / 'd' / 'dashboard.html').read_text(encoding='utf-8')
    st = trader.state
    equity = st['equity'][-1][1]
    n_trades = sum(len(e.trade_log) for e in st['engines'])
    assert f"${equity:,.0f}" in page
    assert f"{equity / 100_000 - 1:+.2%}" in page
    assert f'<div class="label">Closed trades</div><div class="value">{n_trades}</div>' in page
    assert 'No walk-forward run of this exact config' in page


def test_backtest_band_only_from_the_same_config(panel, tmp_path):
    from kalboy import walkforward
    cfg = small_config()
    res = walkforward.run(cfg, 'dev', workers=1, verbose=False, panel=panel)
    walkforward.save(res, cfg, str(tmp_path / 'test_dev'))
    mu, sigma, sources = dashboard.backtest_reference(cfg, str(tmp_path))
    assert sources == ['test_dev'] and sigma > 0
    other = small_config()
    other.strategy.entry_z = 2.5
    assert dashboard.backtest_reference(other, str(tmp_path)) == (None, None, [])
