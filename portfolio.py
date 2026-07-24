class PortfolioManager:
    def __init__(self, max_concentration=0.30, max_drawdown=0.15):
        self.max_concentration = max_concentration
        self.max_drawdown = max_drawdown
        self.peak_value = 0.0
        self.trading_halted = False
        
    def check_and_update(self, portfolio_value):
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
            
        if self.peak_value > 0:
            current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
            if current_drawdown >= self.max_drawdown and not self.trading_halted:
                self.trading_halted = True
                print(f"CRITICAL: Portfolio Max Drawdown ({current_drawdown*100:.2f}%) exceeded {self.max_drawdown*100:.2f}%. Trading Halted.")
                
    def can_trade(self, assets, proposed_cost_per_leg, current_positions, portfolio_value):
        if self.trading_halted:
            return False
            
        for asset in assets:
            base_asset = asset.split('/')[0] if '/' in asset else asset
            
            current_exposure = 0.0
            for pos_symbol, pos_value in current_positions.items():
                pos_base = pos_symbol.split('/')[0] if '/' in pos_symbol else pos_symbol
                if pos_base == base_asset:
                    current_exposure += pos_value
                    
            if (current_exposure + proposed_cost_per_leg) > (portfolio_value * self.max_concentration):
                print(f"PORTFOLIO CAP REJECT: Trade would push {base_asset} exposure ({(current_exposure + proposed_cost_per_leg)/portfolio_value*100:.1f}%) over max cap ({self.max_concentration*100:.1f}%).")
                return False
                
        return True
