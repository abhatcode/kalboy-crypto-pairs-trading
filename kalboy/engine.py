"""Trading rules for one pair, run bar by bar."""
import collections
import math
from dataclasses import dataclass, field

import numpy as np

from kalboy.kalman import KalmanState

ENTRY_ACTIONS = ('enter_long', 'enter_short')
EXIT_ACTIONS = ('exit_long', 'exit_short', 'stop_loss', 'time_stop', 'breakdown', 'period_end', 'liquidation')


@dataclass
class Decision:
    action: str
    z: float
    orders: list = field(default_factory=list)  # (symbol, signed quantity)


class PairEngine:
    """Long the spread is long y and short hedge_ratio units of x."""

    def __init__(self, y, x, beta, alpha, process_noise, measurement_noise, zscore_window,
                 price_scale_y=1.0, price_scale_x=1.0, entry_z=2.0, exit_z=0.5, stop_z=0.0,
                 max_holding_bars=0, burn_in=0, formation_spread_std=None, breakdown_vol_mult=3.0,
                 position_pct=0.10, tradeable=True, anchor_exits=False, process_noise_alpha=None,
                 r_forgetting=0.0, max_funding_carry_bps=0.0):
        self.y_name, self.x_name = y, x
        # anchored exits judge the trade on the spread frozen at entry
        self.anchor_exits = anchor_exits
        self._anchor = None
        self.z_trade = 0.0
        self.entry_z, self.exit_z, self.stop_z = entry_z, exit_z, stop_z
        self.zscore_window = int(zscore_window)
        self.max_holding_bars = int(max_holding_bars)
        self.burn_in = int(burn_in)
        self.formation_spread_std = (formation_spread_std if formation_spread_std and not math.isnan(formation_spread_std)
                                     else None)
        self.breakdown_vol_mult = breakdown_vol_mult
        self.position_pct = position_pct
        self.tradeable = tradeable

        # the filter runs on prices scaled to their formation means
        self.price_scale_y, self.price_scale_x = price_scale_y, price_scale_x
        self.kf = KalmanState(beta * price_scale_x / price_scale_y, alpha / price_scale_y,
                              process_noise, measurement_noise, process_noise_alpha, r_forgetting)
        self.max_funding_carry_bps = max_funding_carry_bps
        self._funding_seen = collections.deque(maxlen=max(200, 4 * int(zscore_window)))
        self.funding_skips = 0
        self._beta0 = self.kf.beta
        self.spreads = collections.deque(maxlen=self.zscore_window)
        self.z_history = collections.deque(maxlen=self.zscore_window)

        self.bar_count = 0
        self.y_val = self.x_val = None
        self.F = 0.0
        self.spread = 0.0
        self.z = 0.0
        self.spread_std = 0.0
        self.warmed_up = False
        self.broken = False

        self.status = 0  # +1 long spread, -1 short, 0 flat
        # this pair's own holdings
        self.held_y = 0.0
        self.held_x = 0.0
        self.pending_fills = 0

        self.trade_log = []
        self._trade = None

    @property
    def hedge_ratio(self):
        return self.kf.beta * self.price_scale_y / self.price_scale_x

    @property
    def busy(self):
        return self.status != 0 or self.pending_fills > 0

    @property
    def trade_returns(self):
        return [t['return'] for t in self.trade_log]

    def update(self, y_val, x_val):
        self.bar_count += 1
        self.y_val, self.x_val = y_val, x_val
        yn, xn = y_val / self.price_scale_y, x_val / self.price_scale_x
        _, self.F = self.kf.update(yn, xn)
        self.spread = yn - (self.kf.beta * xn + self.kf.alpha)
        self.spreads.append(self.spread)
        if len(self.spreads) < self.zscore_window:
            self.warmed_up, self.z, self.spread_std, self.spread_mean = False, 0.0, 0.0, 0.0
        else:
            self.warmed_up = True
            self.spread_mean = float(np.mean(self.spreads))
            self.spread_std = float(np.std(self.spreads, ddof=1))
            self.z = (self.spread - self.spread_mean) / self.spread_std if self.spread_std > 0 else 0.0
        self.z_history.append(self.z)
        if self._anchor is not None:
            b, a, m, sd = self._anchor
            self.z_trade = ((yn - (b * xn + a)) - m) / sd if sd > 0 else 0.0
        return self.z

    def observe_range(self, high_y, low_y, high_x, low_x):
        """Track the largest move against each held leg during a trade."""
        t = self._trade
        if t is None or t['closing'] or not self.held_y:
            return
        vals = (high_y, low_y, high_x, low_x)
        if any(v is None or math.isnan(v) for v in vals):
            return
        ey, ex = t['entry_close_y'], t['entry_close_x']
        t['mae_y'] = max(t['mae_y'], (ey - low_y) / ey if self.held_y > 0 else (high_y - ey) / ey)
        t['mae_x'] = max(t['mae_x'], (ex - low_x) / ex if self.held_x > 0 else (high_x - ex) / ex)

    def accrue_funding(self, rate_y, rate_x):
        """Funding owed this bar on the legs this pair holds."""
        if self.max_funding_carry_bps:
            self._funding_seen.append((rate_y, rate_x))
        cost = 0.0
        if self.held_y:
            cost += self.held_y * self.y_val * rate_y
        if self.held_x:
            cost += self.held_x * self.x_val * rate_x
        if self._trade is not None:
            self._trade['funding'] += cost
            self._trade['cash'] -= cost
        return cost

    def record_fill(self, qty, price, fee, bar_time=None):
        self.pending_fills -= 1
        t = self._trade
        t['cash'] += -qty * price - fee
        t['fees'] += fee
        if t['closing'] and self.pending_fills == 0:
            t['exit_fill_time'] = bar_time
            t['return'] = t['cash'] / t['entry_equity']
            t['pnl'] = t['cash']
            del t['cash'], t['closing']
            self.trade_log.append(t)
            self._trade = None

    def decide(self, equity, trading_allowed=True, risk=None, exposures=None, bar_time=None):
        if self.bar_count < self.burn_in or not trading_allowed or not self.warmed_up:
            return None

        if (self.formation_spread_std and not self.broken
                and self.spread_std > self.formation_spread_std * self.breakdown_vol_mult):
            self.broken = True
            return self._close('breakdown', bar_time) if self.status else Decision('breakdown', self.z)
        if self.broken:
            return None

        z = self.z
        if self.status == 0:
            if not self.tradeable or (risk is not None and risk.halted):
                return None
            if z < -self.entry_z:
                direction = 1
            elif z > self.entry_z:
                direction = -1
            else:
                return None
            gross = equity * self.position_pct
            n = gross / (self.y_val + abs(self.hedge_ratio) * self.x_val)
            qty_y, qty_x = direction * n, -direction * n * self.hedge_ratio
            if self.max_funding_carry_bps and self._funding_seen:
                fy = sum(r[0] for r in self._funding_seen) / len(self._funding_seen)
                fx = sum(r[1] for r in self._funding_seen) / len(self._funding_seen)
                carry = (qty_y * self.y_val * fy + qty_x * self.x_val * fx) * self.zscore_window
                if carry > gross * self.max_funding_carry_bps / 1e4:
                    self.funding_skips += 1
                    return None
            if risk is not None and not risk.can_open({self.y_name: abs(qty_y * self.y_val),
                                                        self.x_name: abs(qty_x * self.x_val)},
                                                       exposures or {}, equity):
                return None
            self.status = direction
            self.held_y, self.held_x = qty_y, qty_x
            if self.anchor_exits:
                self._anchor = (self.kf.beta, self.kf.alpha, self.spread_mean, self.spread_std)
                self.z_trade = z
            self.pending_fills += 2
            self._trade = {'direction': direction, 'entry_bar': self.bar_count, 'entry_time': bar_time,
                           'entry_z': z, 'entry_equity': equity, 'entry_close_y': self.y_val,
                           'entry_close_x': self.x_val, 'hedge_ratio': self.hedge_ratio,
                           'features': self._features(), 'cash': 0.0, 'fees': 0.0, 'funding': 0.0,
                           'mae_y': 0.0, 'mae_x': 0.0, 'closing': False}
            return Decision('enter_long' if direction == 1 else 'enter_short', z,
                            [(self.y_name, qty_y), (self.x_name, qty_x)])

        if self.anchor_exits:
            z = self.z_trade
        if self.stop_z and ((self.status == 1 and z < -self.stop_z) or (self.status == -1 and z > self.stop_z)):
            return self._close('stop_loss', bar_time)
        if self.max_holding_bars and self.bar_count - self._trade['entry_bar'] >= self.max_holding_bars:
            return self._close('time_stop', bar_time)
        if (self.status == 1 and z >= -self.exit_z) or (self.status == -1 and z <= self.exit_z):
            return self._close('exit_long' if self.status == 1 else 'exit_short', bar_time)
        return None

    def force_close(self, reason='period_end', bar_time=None):
        return self._close(reason, bar_time) if self.status else None

    def _close(self, reason, bar_time):
        orders = [(s, -q) for s, q in ((self.y_name, self.held_y), (self.x_name, self.held_x)) if q]
        self.held_y = self.held_x = 0.0
        self.status = 0
        self.pending_fills += len(orders)
        t = self._trade
        t.update(closing=True, exit_reason=reason, exit_bar=self.bar_count, exit_time=bar_time, exit_z=self.z,
                 exit_z_trade=self.z_trade if self._anchor is not None else self.z,
                 bars_held=self.bar_count - t['entry_bar'])
        self._anchor = None
        return Decision(reason, self.z, orders)

    def _features(self):
        """Entry-time features for the meta-labeling experiment."""
        zh = list(self.z_history)
        lookback = max(1, self.zscore_window // 4)
        return {
            'abs_z': abs(self.z),
            'z_velocity': (zh[-1] - zh[-1 - lookback]) * np.sign(-self.z) if len(zh) > lookback else 0.0,
            'vol_ratio': self.spread_std / self.formation_spread_std if self.formation_spread_std else 1.0,
            'innovation_sd': math.sqrt(self.F) / self.spread_std if self.F > 0 and self.spread_std > 0 else 0.0,
            'beta_drift': self.kf.beta / self._beta0 - 1.0 if self._beta0 else 0.0,
            'bars_since_start': self.bar_count,
        }
