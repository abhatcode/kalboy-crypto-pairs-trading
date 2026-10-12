import numpy as np
import pytest

from kalboy.config import Config, load_config
from kalboy.stats import probabilistic_sharpe, expected_max_sharpe, deflated_sharpe, trade_stats


def test_psr_and_dsr_behave():
    rng = np.random.default_rng(0)
    good = rng.normal(0.002, 0.01, 500)
    noise = rng.normal(0.0, 0.01, 500)
    assert probabilistic_sharpe(good) > 0.95
    assert 0.05 < probabilistic_sharpe(noise) < 0.95
    assert expected_max_sharpe(1, 0.01) == 0.0
    assert expected_max_sharpe(100, 0.01) > expected_max_sharpe(10, 0.01) > 0
    trials = rng.normal(0, 0.05, 20)
    assert deflated_sharpe(good, 20, trials) <= probabilistic_sharpe(good)


def test_trade_stats_tail_split():
    r = np.r_[np.full(95, 0.001), np.full(5, -0.05)]
    s = trade_stats(r)
    assert s['worst_5pct_sum'] == pytest.approx(-0.25)
    assert s['rest_sum'] == pytest.approx(0.095)
    assert s['win_rate'] == pytest.approx(0.95)


def test_config_overrides_and_fingerprint(tmp_path):
    p = tmp_path / 'c.json'
    p.write_text('{"name": "x", "strategy": {"stop_z": 3.5}}')
    cfg = load_config(str(p), ['costs.slippage_bps=5', 'strategy.require_positive_kelly=false'])
    assert cfg.strategy.stop_z == 3.5 and cfg.costs.slippage_bps == 5.0
    assert cfg.strategy.require_positive_kelly is False
    renamed = load_config(str(p), ['costs.slippage_bps=5', 'strategy.require_positive_kelly=false'])
    renamed.name = 'other'
    assert renamed.fingerprint() == cfg.fingerprint()
    assert Config().fingerprint() != cfg.fingerprint()
    with pytest.raises(KeyError):
        load_config(None, ['strategy.not_a_key=1'])
