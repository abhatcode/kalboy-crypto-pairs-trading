import copy

import numpy as np
import pandas as pd
import pytest

from kalboy import walkforward
from tests.conftest import small_config


@pytest.fixture(scope='module')
def result(panel):
    return walkforward.run(small_config(), 'dev', workers=1, verbose=False, panel=panel)


def test_it_trades(result):
    assert result['summary']['trades']['n_trades'] >= 5


def test_equity_change_equals_sum_of_trade_pnl(result):
    s, trades = result['summary'], result['trades']
    change = s['equity']['end_value'] - s['equity']['start_value']
    final = result['equity'].iloc[-1] - 100_000.0
    assert final == pytest.approx(trades['pnl'].sum(), abs=1e-6)
    assert s['costs_paid']['fees'] == pytest.approx(trades['fees'].sum(), rel=1e-9)
    assert s['costs_paid']['funding'] == pytest.approx(trades['funding'].sum(), rel=1e-9, abs=1e-9)
    assert change == pytest.approx(final, abs=1e-6)


def test_formation_never_sees_its_trading_period(panel, result):
    cfg = small_config()
    index = panel['close'].index
    plan, F, T = walkforward.window_plan(index, cfg.walkforward, 24, 'dev')
    k = 2
    cut = plan[k][0]
    garbled = copy.deepcopy(panel)
    rng = np.random.default_rng(0)
    for key in ('open', 'close'):
        garbled[key].iloc[cut:] = garbled[key].iloc[cut:].values * rng.uniform(0.5, 1.5, garbled[key].iloc[cut:].shape)
    for w in range(k + 1):
        s = plan[w][0]
        a = walkforward.form(walkforward.panel_slice(panel, s - F, s), cfg.to_dict(), 24, T)
        b = walkforward.form(walkforward.panel_slice(garbled, s - F, s), cfg.to_dict(), 24, T)
        assert a == b


def test_decisions_before_t_ignore_bars_after_t(panel, result):
    cfg = small_config()
    index = panel['close'].index
    t_cut = index[1000]
    garbled = copy.deepcopy(panel)
    for key in ('open', 'close'):
        garbled[key].loc[index > t_cut] *= 1.3
    other = walkforward.run(cfg, 'dev', workers=1, verbose=False, panel=garbled)
    eq_a = result['equity'].loc[:t_cut]
    eq_b = other['equity'].loc[:t_cut]
    pd.testing.assert_series_equal(eq_a, eq_b)
    ta = result['trades'][pd.to_datetime(result['trades']['exit_time']) < t_cut]
    tb = other['trades'][pd.to_datetime(other['trades']['exit_time']) < t_cut]
    pd.testing.assert_frame_equal(ta.reset_index(drop=True), tb.reset_index(drop=True))


def test_dev_segment_stops_at_the_holdout(panel):
    cfg = small_config(holdout_start=str(panel['close'].index[1100]))
    index = panel['close'].index
    plan, _, _ = walkforward.window_plan(index, cfg.walkforward, 24, 'dev')
    assert plan[-1][1] <= 1100
    hplan, _, _ = walkforward.window_plan(index, cfg.walkforward, 24, 'holdout')
    assert hplan[0][0] == 1100
    res = walkforward.run(cfg, 'dev', workers=1, verbose=False, panel=panel)
    assert res['equity'].index.max() < index[1100]


def test_holdout_needs_explicit_confirmation(tmp_path, monkeypatch):
    from kalboy.__main__ import main
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / 'c.json'
    cfg_path.write_text('{"walkforward": {"data_dir": "nowhere"}}')
    with pytest.raises(SystemExit) as exc:
        main(['walkforward', '--config', str(cfg_path), '--segment', 'holdout'])
    assert 'sealed' in str(exc.value)


def test_selection_cache_gives_identical_results(panel, result, tmp_path):
    cfg = small_config()
    cold = walkforward.run(cfg, 'dev', workers=1, verbose=False, panel=panel, cache_dir=str(tmp_path))
    n_files = len(list(tmp_path.glob('*.pkl')))
    warm = walkforward.run(cfg, 'dev', workers=1, verbose=False, panel=panel, cache_dir=str(tmp_path))
    for other in (cold, warm):
        pd.testing.assert_series_equal(other['equity'], result['equity'])
        pd.testing.assert_frame_equal(other['trades'], result['trades'])
        pd.testing.assert_frame_equal(other['windows'], result['windows'])
    assert n_files == len(cold['windows']) and len(list(tmp_path.glob('*.pkl'))) == n_files

    garbled = copy.deepcopy(panel)
    garbled['close'].iloc[:200] *= 1.01
    index = panel['close'].index
    plan, F, T = walkforward.window_plan(index, cfg.walkforward, 24, 'dev')
    s = plan[0][0]
    walkforward.cached_select(walkforward.panel_slice(garbled, s - F, s), cfg.selection, 24, T, str(tmp_path))
    assert len(list(tmp_path.glob('*.pkl'))) == n_files + 1


def test_sensitivity_runs_are_not_trials(tmp_path):
    from kalboy import ledger
    path = str(tmp_path / 'ledger.jsonl')
    base = {'segment': 'dev', 'data_dir': 'd', 'daily_sharpe': 0.1}
    ledger.append({**base, 'fingerprint': 'a'}, path)
    ledger.append({**base, 'fingerprint': 'b', 'purpose': 'trial'}, path)
    ledger.append({**base, 'fingerprint': 'c', 'purpose': 'sensitivity'}, path)
    ledger.append({**base, 'fingerprint': 'd', 'segment': 'holdout'}, path)
    assert set(ledger.dev_trials('d', path)) == {'a', 'b'}
    assert set(ledger.dev_trials(None, path)) == {'a', 'b'}
