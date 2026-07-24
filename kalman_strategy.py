import backtrader as bt
import numpy as np
import collections

class KalmanState:
    def __init__(self, initial_beta, initial_alpha, process_noise=1e-4, measurement_noise=1e-3):
        self.beta = initial_beta
        self.alpha = initial_alpha
        
        self.P = np.zeros((2, 2))
        self.P[0, 0] = (0.5 * abs(initial_beta)) ** 2 if initial_beta != 0 else 1.0
        self.P[1, 1] = (0.5 * abs(initial_alpha)) ** 2 if initial_alpha != 0 else 1.0
        
        self.Q = np.eye(2) * process_noise
        self.R = measurement_noise

    def update(self, y_val, x_val):
        P_pred = self.P + self.Q
        
        H = np.array([[x_val, 1.0]])
        y_pred = x_val * self.beta + self.alpha
        v = y_val - y_pred
        
        F = np.dot(H, np.dot(P_pred, H.T))[0, 0] + self.R
        
        K = np.dot(P_pred, H.T) / F
        
        self.beta += K[0, 0] * v
        self.alpha += K[1, 0] * v
        
        self.P = np.dot(np.eye(2) - np.dot(K, H), P_pred)
        
        return v, F

class KalmanPairs(bt.Strategy):
    params = (
        ('entry_threshold', 2.0),
        ('exit_threshold', 0.5),
        ('process_noise', 1e-4),
        ('measurement_noise', 1e-3),
        ('initial_beta', 1.0),
        ('initial_alpha', 0.0),
        ('allocation', 10000.0),
        ('kelly_fraction', 0.0),
        ('max_position_cap_pct', 0.10),
        ('is_spread_vol_avg', None),
        ('breakdown_vol_multiplier', 3.0),
        ('portfolio_manager', None),
        ('data_y_name', None),
        ('data_x_name', None),
        ('burn_in', 7),
        ('trade_start_date', None),
        ('zscore_window', 20),
    )

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()}: {txt}")

    def __init__(self):
        if self.p.data_y_name and self.p.data_x_name:
            self.y = self.getdatabyname(self.p.data_y_name)
            self.x = self.getdatabyname(self.p.data_x_name)
        else:
            self.y = self.datas[0]
            self.x = self.datas[1]

        self.kf = KalmanState(
            initial_beta=self.p.initial_beta,
            initial_alpha=self.p.initial_alpha,
            process_noise=self.p.process_noise,
            measurement_noise=self.p.measurement_noise
        )
        self.convergence_log = []

        self.status = 0
        self.order_y = None
        self.order_x = None
        self.bar_count = 0
        self.spread_history = collections.deque(maxlen=self.p.zscore_window)
        self.zscore_history = []
        
        self.trade_returns = []
        self.all_spread_stds = []
        self.entry_portfolio_value = 0.0
        self.cointegration_broken = False

    def next(self):
        self.bar_count += 1
        
        y_val = self.y.close[0]
        x_val = self.x.close[0]

        if np.isnan(y_val) or np.isnan(x_val):
            return

        spread, F = self.kf.update(y_val, x_val)
        
        if self.bar_count <= 100:
            self.convergence_log.append((self.bar_count, self.kf.beta, self.kf.alpha))
        
        posterior_spread = y_val - (self.kf.beta * x_val + self.kf.alpha)
        self.spread_history.append(posterior_spread)
        
        if len(self.spread_history) < self.p.zscore_window:
            z_score = 0.0
            rolling_std = 0.0
        else:
            rolling_mean = np.mean(self.spread_history)
            rolling_std = np.std(self.spread_history, ddof=1)
            z_score = (posterior_spread - rolling_mean) / rolling_std if rolling_std > 0 else 0.0
            self.all_spread_stds.append(rolling_std)
        
        self.current_z_score = z_score
        self.zscore_history.append(z_score)
        
        if self.bar_count < self.p.burn_in:
            return

        current_date = self.datas[0].datetime.date(0)
        if self.p.trade_start_date and current_date < self.p.trade_start_date:
            return

        if self.p.is_spread_vol_avg is not None and not self.cointegration_broken:
            if rolling_std > self.p.is_spread_vol_avg * self.p.breakdown_vol_multiplier:
                self.log(f"COINTEGRATION BREAKDOWN! Spread Vol ({rolling_std:.4f}) > {self.p.breakdown_vol_multiplier}x IS Vol ({self.p.is_spread_vol_avg:.4f}). Halting pair.")
                self.cointegration_broken = True
                if self.status != 0:
                    self.close(data=self.y)
                    self.close(data=self.x)
                    self.status = 0
        
        if self.cointegration_broken:
            return

        portfolio_value = self.broker.getvalue()
        if self.p.portfolio_manager:
            self.p.portfolio_manager.check_and_update(portfolio_value)
            if self.p.portfolio_manager.trading_halted:
                return

        cost_of_spread = y_val + abs(self.kf.beta) * x_val
        
        if self.p.kelly_fraction > 0 and F > 0:
            raw_target_alloc = portfolio_value * self.p.kelly_fraction * (abs(posterior_spread) / F)
            max_alloc = portfolio_value * self.p.max_position_cap_pct
            target_alloc = min(raw_target_alloc, max_alloc)
            num_spreads = target_alloc / cost_of_spread
        else:
            num_spreads = self.p.allocation / cost_of_spread
        
        qty_y = num_spreads
        qty_x = num_spreads * self.kf.beta

        current_positions = {}
        for d in self.broker.positions:
            pos = self.broker.getposition(d)
            if pos.size != 0:
                current_positions[d._name] = abs(pos.size * d.close[0])

        if self.status == 0:
            if z_score < -self.p.entry_threshold:
                if self.p.portfolio_manager:
                    assets = [self.y._name, self.x._name]
                    proposed_cost_y = qty_y * y_val
                    proposed_cost_x = qty_x * x_val
                    max_cost = max(proposed_cost_y, proposed_cost_x)
                    if not self.p.portfolio_manager.can_trade(assets, max_cost, current_positions, portfolio_value):
                        return
                        
                self.log(f"BUY SPREAD (Z-Score: {z_score:.2f}) | Long {qty_y:.4f} Y at {y_val:.2f}, Short {qty_x:.4f} X at {x_val:.2f}")
                self.entry_portfolio_value = portfolio_value
                self.order_y = self.buy(data=self.y, size=qty_y)
                self.order_x = self.sell(data=self.x, size=qty_x)
                self.status = 1
            elif z_score > self.p.entry_threshold:
                if self.p.portfolio_manager:
                    assets = [self.y._name, self.x._name]
                    proposed_cost_y = qty_y * y_val
                    proposed_cost_x = qty_x * x_val
                    max_cost = max(proposed_cost_y, proposed_cost_x)
                    if not self.p.portfolio_manager.can_trade(assets, max_cost, current_positions, portfolio_value):
                        return
                        
                self.log(f"SELL SPREAD (Z-Score: {z_score:.2f}) | Short {qty_y:.4f} Y at {y_val:.2f}, Long {qty_x:.4f} X at {x_val:.2f}")
                self.entry_portfolio_value = portfolio_value
                self.order_y = self.sell(data=self.y, size=qty_y)
                self.order_x = self.buy(data=self.x, size=qty_x)
                self.status = -1
        
        elif self.status == 1:
            if z_score >= -self.p.exit_threshold:
                self.log(f"EXIT LONG SPREAD (Z-Score: {z_score:.2f}) | Closing positions.")
                trade_ret = (portfolio_value - self.entry_portfolio_value) / self.entry_portfolio_value
                self.trade_returns.append(trade_ret)
                self.close(data=self.y)
                self.close(data=self.x)
                self.status = 0
                
        elif self.status == -1:
            if z_score <= self.p.exit_threshold:
                self.log(f"EXIT SHORT SPREAD (Z-Score: {z_score:.2f}) | Closing positions.")
                trade_ret = (portfolio_value - self.entry_portfolio_value) / self.entry_portfolio_value
                self.trade_returns.append(trade_ret)
                self.close(data=self.y)
                self.close(data=self.x)
                self.status = 0

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED, Price: {order.executed.price:.4f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}")
            else:
                self.log(f"SELL EXECUTED, Price: {order.executed.price:.4f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}")
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"ORDER FAILED, Status: {order.getstatusname()}")

        self.order_y = None
        self.order_x = None
