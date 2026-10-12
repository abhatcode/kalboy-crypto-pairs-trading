"""Per-pair Kalman noise, z-score window and Kelly gate, fitted on the formation window."""
from dataclasses import dataclass, asdict

import numpy as np

from kalboy.engine import PairEngine
from kalboy.execution import simulate_single
from kalboy.kalman import log_likelihood

Q_GRID = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
R_MULT_GRID = (0.1, 0.3, 1.0)
R_FORGETTING_GRID = (0.0, 0.005, 0.02, 0.05)


@dataclass
class PairSpec:
    y: str
    x: str
    beta: float
    alpha: float
    half_life: float
    hurst: float
    p_fdr: float
    cluster: int
    process_noise: float
    measurement_noise: float
    zscore_window: int
    burn_in: int
    max_holding_bars: int
    price_scale_y: float
    price_scale_x: float
    formation_spread_std: float
    formation_trades: int
    formation_mean_return: float
    kelly: float
    tradeable: bool
    process_noise_alpha: float = None
    r_forgetting: float = 0.0

    def to_dict(self):
        return asdict(self)


def fit_noise(y_n, x_n, beta0, alpha0, separate_q=False, adaptive_r=False):
    """Grid search for the Kalman noise values with the highest likelihood."""
    resid_var = float(np.var(y_n - (beta0 * x_n + alpha0)))
    best = (-np.inf, {'q': 1e-4, 'r': max(resid_var, 1e-12), 'q_alpha': None, 'r_forgetting': 0.0})
    for q in Q_GRID:
        for qa in (Q_GRID if separate_q else (None,)):
            for mult in R_MULT_GRID:
                r = max(resid_var * mult, 1e-12)
                for lam in (R_FORGETTING_GRID if adaptive_r else (0.0,)):
                    ll = log_likelihood(y_n, x_n, beta0, alpha0, q, r, q_alpha=qa, r_forgetting=lam)
                    if ll > best[0]:
                        best = (ll, {'q': q, 'r': r, 'q_alpha': None if qa == q else qa, 'r_forgetting': lam})
    return best[1]


def quarter_kelly(returns, min_trades):
    """Quarter Kelly fraction from trade returns, 0 if there are too few trades."""
    n = len(returns)
    if n < max(2, min_trades):
        return 0.0
    var = float(np.var(returns, ddof=1))
    raw = float(np.mean(returns)) / var if var > 0 else 0.0
    return min(max(0.0, raw), 1.0) * 0.25 * min(1.0, n / 30.0)


def make_engine(spec, strat, tradeable=None):
    return PairEngine(
        spec.y, spec.x, spec.beta, spec.alpha, spec.process_noise, spec.measurement_noise,
        spec.zscore_window, spec.price_scale_y, spec.price_scale_x,
        entry_z=strat.entry_z, exit_z=strat.exit_z, stop_z=strat.stop_z,
        max_holding_bars=spec.max_holding_bars, burn_in=spec.burn_in,
        formation_spread_std=spec.formation_spread_std, breakdown_vol_mult=strat.breakdown_vol_mult,
        position_pct=strat.position_pct, tradeable=spec.tradeable if tradeable is None else tradeable,
        anchor_exits=strat.anchor_exits, process_noise_alpha=spec.process_noise_alpha,
        r_forgetting=spec.r_forgetting, max_funding_carry_bps=strat.max_funding_carry_bps)


def calibrate(row, window, strat, costs, initial_cash, kalman=None):
    """Fit one selected pair on its formation window."""
    y, x = row['asset_y'], row['asset_x']
    cy = window['close'][y].ffill().bfill().values
    cx = window['close'][x].ffill().bfill().values
    sy, sx = float(cy.mean()), float(cx.mean())
    noise = fit_noise(cy / sy, cx / sx, row['beta'] * sx / sy, row['alpha'] / sy,
                      separate_q=bool(kalman and kalman.separate_q), adaptive_r=bool(kalman and kalman.adaptive_r))
    q, r = noise['q'], noise['r']
    zwin = int(np.clip(round(strat.zscore_window_half_lives * row['half_life']),
                       strat.min_zscore_window, strat.max_zscore_window))
    spec = PairSpec(
        y=y, x=x, beta=row['beta'], alpha=row['alpha'], half_life=row['half_life'], hurst=row['hurst'],
        p_fdr=row['p_fdr'], cluster=int(row['cluster']), process_noise=q, measurement_noise=r,
        zscore_window=zwin, burn_in=int(max(100, 3 * zwin)),
        max_holding_bars=int(round(strat.time_stop_windows * zwin)),
        price_scale_y=sy, price_scale_x=sx, formation_spread_std=float('nan'),
        formation_trades=0, formation_mean_return=float('nan'), kelly=0.0, tradeable=False,
        process_noise_alpha=noise['q_alpha'], r_forgetting=noise['r_forgetting'])

    engine = make_engine(spec, strat, tradeable=True)
    stds = []
    original_update = engine.update
    def update(yv, xv):
        z = original_update(yv, xv)
        if engine.warmed_up:
            stds.append(engine.spread_std)
        return z
    engine.update = update
    simulate_single(engine, window['open'][y].ffill().bfill().values, window['open'][x].ffill().bfill().values,
                    cy, cx, window['funding'][y].values if costs.funding else None,
                    window['funding'][x].values if costs.funding else None,
                    initial_cash=initial_cash, fee=costs.taker_fee, slippage=costs.slippage_bps / 1e4)
    rets = engine.trade_returns
    spec.formation_spread_std = float(np.mean(stds)) if stds else float('nan')
    spec.formation_trades = len(rets)
    spec.formation_mean_return = float(np.mean(rets)) if rets else float('nan')
    spec.kelly = quarter_kelly(rets, strat.min_formation_trades)
    spec.tradeable = spec.kelly > 0 if strat.require_positive_kelly else True
    return spec
