"""Synthetic market with two cointegrated pairs and four random walks."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kalboy.config import Config  # noqa: E402


def make_panel(n_bars=1300, seed=7):
    rng = np.random.default_rng(seed)
    index = pd.date_range('2025-01-01', periods=n_bars, freq='1h')
    market = np.cumsum(rng.normal(0, 0.004, n_bars))
    prices = {}

    def walk(scale=0.006):
        return np.cumsum(rng.normal(0, scale, n_bars)) + market

    def ou(theta=0.12, sigma=0.012):
        s = np.zeros(n_bars)
        for t in range(1, n_bars):
            s[t] = s[t - 1] - theta * s[t - 1] + rng.normal(0, sigma)
        return s

    for name, base, beta, const in (('A', 100.0, 0.8, 20.0), ('B', 5.0, 1.5, 1.0)):
        x = base * np.exp(walk())
        prices[f'{name}2'] = x
        prices[f'{name}1'] = beta * x + const + ou() * base
    for k, base in enumerate((50.0, 2.0, 300.0, 0.5)):
        prices[f'R{k}'] = base * np.exp(walk())

    close = pd.DataFrame(prices, index=index)
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.0005, close.shape))
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    turnover = pd.DataFrame(rng.uniform(1e5, 1e6, close.shape), index=index, columns=close.columns)
    funding = pd.DataFrame(0.0, index=index, columns=close.columns)
    on_funding = index.hour % 8 == 0
    funding.loc[on_funding] = rng.normal(1e-4, 2e-4, (on_funding.sum(), close.shape[1]))
    return {'open': open_, 'high': high, 'low': low, 'close': close, 'turnover': turnover, 'funding': funding,
            'meta': {'timeframe': '1h', 'symbols': list(close.columns)}}


@pytest.fixture(scope='session')
def panel():
    return make_panel()


def small_config(**walkforward):
    cfg = Config(name='test')
    cfg.walkforward.formation_days = 20
    cfg.walkforward.trading_days = 5
    cfg.walkforward.holdout_start = ''
    cfg.selection.universe_size = 0
    cfg.strategy.require_positive_kelly = False
    cfg.risk.max_concentration = 1.0
    for k, v in walkforward.items():
        setattr(cfg.walkforward, k, v)
    return cfg


@pytest.fixture
def cfg():
    return small_config()
