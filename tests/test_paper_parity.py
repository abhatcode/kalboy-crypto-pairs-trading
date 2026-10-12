"""The paper trader must make the same decisions as the walk-forward."""
import pandas as pd

from kalboy import walkforward
from kalboy.paper import PaperTrader, ReplaySource
from tests.conftest import small_config

START, END = 480, 1200


def replay(panel, out_dir, start, end, restart_at=None):
    cfg = small_config()
    source = ReplaySource(panel)
    index = panel['close'].index
    step = pd.Timedelta(hours=1)
    trader = PaperTrader(cfg, out_dir, source)
    for i in range(start, end):
        if restart_at is not None and i == restart_at:
            trader = PaperTrader(cfg, out_dir, ReplaySource(panel))
            source = trader.source
        source.end_ms = int((index[i] + step).value // 1_000_000)
        trader.process_new_bars()
    return trader


def walk(panel):
    index = panel['close'].index
    cfg = small_config(dev_start=str(index[START]), holdout_start=str(index[END]))
    return walkforward.run(cfg, 'dev', workers=1, verbose=False, panel=panel)


def test_paper_replay_matches_walkforward(panel, tmp_path):
    wf = walk(panel)
    trader = replay(panel, str(tmp_path / 'p'), START, END)
    st = trader.state
    paper_eq = pd.Series(dict(st['equity']))
    pd.testing.assert_series_equal(paper_eq.iloc[:-1], wf['equity'].iloc[:-1], check_names=False, check_freq=False)
    last_bar = panel['close'].index[END - 1]
    wf_decisions = [d for d in wf_sim_decisions(panel) if d[0] < last_bar]
    paper_decisions = [(t, p, a, round(z, 9)) for t, p, a, z in st['sim'].decision_log if t < last_bar]
    assert len(paper_decisions) >= 10
    assert paper_decisions == wf_decisions
    assert len(wf['trades']) >= 5


def wf_sim_decisions(panel):
    import kalboy.execution as ex
    captured = []
    original = ex.PortfolioSimulator._submit
    def spy(self, engine, decision, bar_time):
        if decision is not None and isinstance(bar_time, pd.Timestamp):
            captured.append((bar_time, f"{engine.y_name}/{engine.x_name}", decision.action, round(decision.z, 9)))
        return original(self, engine, decision, bar_time)
    ex.PortfolioSimulator._submit = spy
    try:
        walk(panel)
    finally:
        ex.PortfolioSimulator._submit = original
    return captured


def test_restart_resumes_identically(panel, tmp_path):
    a = replay(panel, str(tmp_path / 'a'), START, END)
    b = replay(panel, str(tmp_path / 'b'), START, END, restart_at=800)
    assert a.state['equity'] == b.state['equity']
    assert a.state['sim'].decision_log == b.state['sim'].decision_log
    for name, rows, cols in (('equity', b.state['equity'], ['time', 'equity']),
                             ('decisions', b.state['sim'].decision_log, ['time', 'pair', 'action', 'z'])):
        expected = pd.DataFrame(rows, columns=cols).to_csv(index=False, date_format='%Y-%m-%d %H:%M:%S')
        with open(tmp_path / 'b' / f'{name}.csv', newline='') as f:
            assert f.read() == expected
    trades = pd.read_csv(tmp_path / 'b' / 'trades.csv')
    assert len(trades) == sum(len(e.trade_log) for e in b.state['engines'])
