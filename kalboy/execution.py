"""Fills, fees, funding, margin checks and accounting."""
import math


class PaperBroker:
    def __init__(self, initial_cash=100_000.0, fee=0.0006, slippage=0.0):
        self.cash = initial_cash
        self.fee = fee
        self.slippage = slippage
        self.positions = {}
        self.fees_paid = 0.0
        self.funding_paid = 0.0
        self.fills = 0

    def execute(self, symbol, qty, price):
        fill = price * (1.0 + self.slippage) if qty > 0 else price * (1.0 - self.slippage)
        fee = abs(qty) * fill * self.fee
        self.cash -= qty * fill + fee
        pos = self.positions.get(symbol, 0.0) + qty
        if abs(pos) < 1e-12:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = pos
        self.fees_paid += fee
        self.fills += 1
        return fill, fee

    def equity(self, prices):
        return self.cash + sum(q * prices[s] for s, q in self.positions.items())

    def exposures(self, prices):
        return {s: abs(q * prices[s]) for s, q in self.positions.items()}


class PortfolioSimulator:
    """Runs any number of pair engines against one broker."""

    def __init__(self, broker, risk=None, maintenance_margin=0.0):
        self.broker = broker
        self.risk = risk
        self.maintenance_margin = maintenance_margin
        self.engines = []
        self.retiring = []
        self.pending = []
        self.last_close = {}
        self.decision_log = []
        self.liquidations = []
        self.min_margin_ratio = math.inf
        self.max_leverage = 0.0

    def add(self, engine):
        self.engines.append(engine)

    def retire_all(self, bar_time=None):
        for engine in self.engines:
            self._submit(engine, engine.force_close('period_end', bar_time), bar_time)
        self.retiring.extend(e for e in self.engines if e.busy)
        self.engines = []

    def symbols_needed(self):
        syms = {s for e in self.engines + self.retiring for s in (e.y_name, e.x_name)}
        syms.update(s for _, s, _ in self.pending)
        syms.update(self.broker.positions)
        return syms

    def step(self, bar_time, opens, closes, funding=None, trading_allowed=True, highs=None, lows=None):
        """Advance one bar. Highs and lows are only used for the intrabar margin check."""
        # fill last bar's decisions at this open
        for engine, sym, qty in self.pending:
            price, fee = self.broker.execute(sym, qty, opens[sym])
            engine.record_fill(qty, price, fee, bar_time)
        self.pending = []
        self.retiring = [e for e in self.retiring if e.busy]
        self.last_close.update(closes)

        equity = self.broker.equity(self.last_close)
        exposures = self.broker.exposures(self.last_close)
        if self.risk is not None:
            self.risk.mark(equity, bar_time)

        # cross-margin check at the close and the intrabar worst case
        gross = sum(exposures.values())
        if gross > 0:
            ratio = equity / gross
            self.max_leverage = max(self.max_leverage, gross / equity if equity > 0 else math.inf)
            if highs is not None and lows is not None:
                worst = {}
                for sym, q in self.broker.positions.items():
                    v = (lows if q > 0 else highs).get(sym)
                    worst[sym] = self.last_close[sym] if v is None or math.isnan(v) else v
                w_gross = sum(abs(q * worst[sym]) for sym, q in self.broker.positions.items())
                ratio = min(ratio, self.broker.equity(worst) / w_gross)
            self.min_margin_ratio = min(self.min_margin_ratio, ratio)
            if self.maintenance_margin and ratio < self.maintenance_margin:
                self.liquidations.append(bar_time)
                for engine in self.engines:
                    self._submit(engine, engine.force_close('liquidation', bar_time), bar_time)

        # each pair updates, pays funding, then decides
        funding = funding or {}
        for engine in self.engines:
            y, x = closes.get(engine.y_name), closes.get(engine.x_name)
            if y is None or x is None or math.isnan(y) or math.isnan(x):
                continue
            if highs is not None and lows is not None:
                engine.observe_range(highs.get(engine.y_name), lows.get(engine.y_name),
                                     highs.get(engine.x_name), lows.get(engine.x_name))
            engine.update(y, x)
            cost = engine.accrue_funding(funding.get(engine.y_name, 0.0), funding.get(engine.x_name, 0.0))
            self.broker.cash -= cost
            self.broker.funding_paid += cost
            decision = engine.decide(equity, trading_allowed, self.risk, exposures, bar_time)
            self._submit(engine, decision, bar_time)
        return equity

    def _submit(self, engine, decision, bar_time):
        if decision is None:
            return
        self.decision_log.append((bar_time, f"{engine.y_name}/{engine.x_name}", decision.action, decision.z))
        for sym, qty in decision.orders:
            self.pending.append((engine, sym, qty))


def simulate_single(engine, opens_y, opens_x, closes_y, closes_x, funding_y=None, funding_x=None,
                    trade_from=0, initial_cash=100_000.0, fee=0.0006, slippage=0.0):
    """Run one engine alone, for the formation-window trades behind the Kelly gate."""
    broker = PaperBroker(initial_cash, fee, slippage)
    sim = PortfolioSimulator(broker)
    sim.add(engine)
    y, x = engine.y_name, engine.x_name
    for i in range(len(closes_y)):
        f = {y: funding_y[i], x: funding_x[i]} if funding_y is not None else None
        sim.step(i, {y: opens_y[i], x: opens_x[i]}, {y: closes_y[i], x: closes_x[i]}, f, i >= trade_from)
    return engine
