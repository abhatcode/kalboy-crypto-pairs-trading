"""Portfolio risk limits shared by every pair."""


class RiskManager:
    def __init__(self, max_concentration=0.30, max_drawdown=1.0):
        self.max_concentration = max_concentration
        self.max_drawdown = max_drawdown
        self.peak = 0.0
        self.halted = False
        self.halted_at = None
        self.rejections = 0

    def mark(self, equity, bar_time=None):
        """Halt new entries once drawdown passes the limit."""
        self.peak = max(self.peak, equity)
        if not self.halted and self.peak > 0 and (self.peak - equity) / self.peak >= self.max_drawdown:
            self.halted, self.halted_at = True, bar_time

    def can_open(self, new_legs, exposures, equity):
        """Reject a trade that would push one coin over the concentration limit."""
        for sym, notional in new_legs.items():
            if exposures.get(sym, 0.0) + notional > self.max_concentration * equity:
                self.rejections += 1
                return False
        return True
