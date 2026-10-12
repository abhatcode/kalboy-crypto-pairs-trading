"""Live data path with a fake KuCoin client instead of the network."""
import pandas as pd

from kalboy import walkforward
from kalboy.data import fetch_dataset, load_panel
from kalboy.paper import LiveSource, PaperTrader
from tests.conftest import make_panel, small_config

START, END = 480, 1000


class FakeKucoin:
    def __init__(self, panel):
        self.p = panel
        self.now = None
        self.times = panel['close'].index.values.astype('datetime64[ms]').astype('int64')

    def _ms(self, ts):
        return int(pd.Timestamp(ts).value // 1_000_000)

    def server_time_ms(self):
        return self.now

    def active_contracts(self):
        return [{'symbol': f'{s}USDTM', 'baseCurrency': s, 'type': 'FFWCSX', 'isInverse': False,
                 'quoteCurrency': 'USDT', 'status': 'Open', 'assetClass': 'CRYPTO', 'turnoverOf24h': 1.0}
                for s in self.p['close'].columns]

    def klines(self, symbol, timeframe, start_ms, end_ms):
        s = symbol[:-5]
        end_ms = min(end_ms, self.now - 3_600_000 + 1)
        lo, hi = self.times.searchsorted(start_ms), self.times.searchsorted(end_ms)
        cols = [self.p[k][s].values[lo:hi] for k in ('open', 'high', 'low', 'close', 'turnover')]
        return [[int(t), o, h, l, c, 0, v] for t, o, h, l, c, v in zip(self.times[lo:hi], *cols)]

    def funding_history(self, symbol, start_ms, end_ms):
        f = self.p['funding'][symbol[:-5]]
        return [(self._ms(ts), float(r)) for ts, r in f.items()
                if r != 0 and start_ms <= self._ms(ts) < min(end_ms, self.now)]


import pytest


@pytest.mark.parametrize('universe_size', [0, 1])
def test_paper_on_live_source_matches_walkforward(tmp_path, universe_size):
    synthetic = make_panel(n_bars=END + 10)
    index = synthetic['close'].index
    fake = FakeKucoin(synthetic)
    data_dir = str(tmp_path / 'data')
    step = 3_600_000
    fake.now = fake._ms(index[START]) + step
    fetch_dataset(data_dir, '1h', str(index[0]), top_n=50, workers=1, client=fake)

    cfg = small_config()
    cfg.walkforward.data_dir = data_dir
    cfg.selection.universe_size = universe_size
    trader = PaperTrader(cfg, str(tmp_path / 'paper'), LiveSource(data_dir, client=fake))
    for i in range(START, END):
        fake.now = fake._ms(index[i]) + step + 30_000
        trader.process_new_bars()

    panel = load_panel(data_dir)
    assert panel['close'].index[-1] == index[END - 1]
    wcfg = small_config(dev_start=str(index[START]), holdout_start=str(index[END]))
    wcfg.walkforward.data_dir = data_dir
    wcfg.selection.universe_size = universe_size
    wf = walkforward.run(wcfg, 'dev', workers=1, verbose=False, panel=panel)
    paper_eq = pd.Series(dict(trader.state['equity']))
    pd.testing.assert_series_equal(paper_eq.iloc[:-1], wf['equity'].iloc[:-1], check_names=False, check_freq=False)
    assert len(trader.state['formations']) == len(wf['windows'])
    n_trades = sum(len(e.trade_log) for e in trader.state['engines'])
    assert n_trades >= 3 if universe_size == 0 else n_trades == 0
