"""Settings for every stage. JSON files in configs/ override these defaults."""
import copy
import hashlib
import json
from dataclasses import dataclass, field, asdict, fields, is_dataclass


@dataclass
class Selection:
    universe_size: int = 30
    min_coverage: float = 0.98
    n_market_factors: int = 1
    max_cluster_size: int = 8
    fdr_alpha: float = 0.05
    max_hurst: float = 0.5
    min_half_life_bars: float = 2.0
    max_half_life_frac: float = 0.25


@dataclass
class Kalman:
    separate_q: bool = False
    adaptive_r: bool = False


@dataclass
class Strategy:
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 0.0  # 0 turns it off
    time_stop_windows: float = 0.0  # 0 turns it off
    anchor_exits: bool = False
    zscore_window_half_lives: float = 2.0
    min_zscore_window: int = 10
    max_zscore_window: int = 500
    breakdown_vol_mult: float = 3.0
    position_pct: float = 0.10
    require_positive_kelly: bool = True
    min_formation_trades: int = 5
    max_funding_carry_bps: float = 0.0  # 0 turns it off


@dataclass
class Costs:
    taker_fee: float = 0.0006  # KuCoin base tier
    slippage_bps: float = 3.0
    funding: bool = True


@dataclass
class Risk:
    max_concentration: float = 0.30
    max_drawdown: float = 1.0  # 1.0 turns it off
    maintenance_margin: float = 0.01


@dataclass
class WalkForward:
    data_dir: str = 'data/kucoinf_15m'
    formation_days: float = 60.0
    trading_days: float = 7.0
    dev_start: str = ''
    holdout_start: str = '2026-07-01'  # dev runs stop here
    end: str = ''  # fixed, so new data cannot extend the holdout
    initial_cash: float = 100_000.0


@dataclass
class Config:
    name: str = 'default'
    selection: Selection = field(default_factory=Selection)
    kalman: Kalman = field(default_factory=Kalman)
    strategy: Strategy = field(default_factory=Strategy)
    costs: Costs = field(default_factory=Costs)
    risk: Risk = field(default_factory=Risk)
    walkforward: WalkForward = field(default_factory=WalkForward)

    def to_dict(self):
        return asdict(self)

    def fingerprint(self):
        """Hash of the config without its name."""
        d = self.to_dict()
        d.pop('name')
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:10]


def _merge(obj, overrides, path=''):
    for key, value in overrides.items():
        if not hasattr(obj, key):
            raise KeyError(f"unknown config key {path}{key}")
        current = getattr(obj, key)
        if is_dataclass(current):
            _merge(current, value, f"{path}{key}.")
        else:
            setattr(obj, key, type(current)(value) if current is not None and not isinstance(current, bool) else value)


def _parse_scalar(text, like):
    if isinstance(like, bool):
        return text.lower() in ('1', 'true', 'yes')
    return type(like)(text)


def load_config(path=None, sets=()):
    cfg = Config()
    if path:
        with open(path) as f:
            _merge(cfg, json.load(f))
    for item in sets:
        key, value = item.split('=', 1)
        *parents, leaf = key.split('.')
        obj = cfg
        for p in parents:
            obj = getattr(obj, p)
        if not hasattr(obj, leaf):
            raise KeyError(f"unknown config key {key}")
        setattr(obj, leaf, _parse_scalar(value, getattr(obj, leaf)))
    return cfg


def config_from_dict(d):
    cfg = Config()
    d = copy.deepcopy(d)
    cfg.name = d.pop('name', cfg.name)
    _merge(cfg, d)
    return cfg
