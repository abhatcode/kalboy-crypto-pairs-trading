import asyncio
import os
import argparse
import logging
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt
import collections

os.makedirs('results', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('results', 'live_bot.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('live_bot')

from kalman_strategy import KalmanState
from portfolio import PortfolioManager

class AsyncPaperBroker:
    def __init__(self, portfolio_manager, initial_cash=100000.0, commission=0.001):
        self.cash = initial_cash
        self.commission = commission
        self.positions = {}
        self.pm = portfolio_manager
        self.lock = asyncio.Lock()

    def get_position(self, symbol):
        return self.positions.get(symbol, 0.0)

    def execute_order(self, symbol, side, qty, price):
        value = qty * price
        comm_fee = value * self.commission
        
        if side == 'buy':
            cost = value + comm_fee
            self.cash -= cost
            self.positions[symbol] = self.positions.get(symbol, 0.0) + qty
            logger.info(f"BUY {qty:.6f} {symbol} @ {price:.4f} | Cost: ${value:.2f} + Comm: ${comm_fee:.2f}")
        elif side == 'sell':
            revenue = value - comm_fee
            self.cash += revenue
            self.positions[symbol] = self.positions.get(symbol, 0.0) - qty
            logger.info(f"SELL {qty:.6f} {symbol} @ {price:.4f} | Rev: ${value:.2f} - Comm: ${comm_fee:.2f}")
            
    def get_total_value(self, current_prices):
        val = self.cash
        for sym, qty in self.positions.items():
            if qty != 0:
                price = current_prices.get(sym, 0.0)
                val += qty * price
        return val

async def pair_loop(exchange, broker, row, interval, total_iterations, shared_prices):
    symbol_y = row['asset_a']
    symbol_x = row['asset_b']
    kelly = row['kelly_fraction']
    is_spread_vol_avg = row.get('is_spread_vol_avg', None)
    
    logger.info(f"Starting loop for {symbol_y} vs {symbol_x} | Kelly: {kelly:.4f}")
    
    kf = KalmanState(initial_beta=row['beta'], initial_alpha=row['alpha'])
    
    status = 0
    entry_threshold = 2.0
    exit_threshold = 0.5
    max_position_cap_pct = 0.10
    breakdown_vol_multiplier = 3.0
    
    spread_history = collections.deque(maxlen=20)
    cointegration_broken = False
    
    iteration = 0
    while total_iterations is None or iteration < total_iterations:
        iteration += 1
        
        if cointegration_broken:
            await asyncio.sleep(interval)
            continue
            
        try:
            ticker_y = await exchange.fetch_ticker(symbol_y)
            ticker_x = await exchange.fetch_ticker(symbol_x)
            
            price_y = ticker_y['last']
            price_x = ticker_x['last']
            
            if price_y is None or price_x is None:
                await asyncio.sleep(interval)
                continue
                
            async with broker.lock:
                shared_prices[symbol_y] = price_y
                shared_prices[symbol_x] = price_x
                
            spread, F = kf.update(price_y, price_x)
            posterior_spread = price_y - (kf.beta * price_x + kf.alpha)
            spread_history.append(posterior_spread)
            
            if len(spread_history) < 20:
                z_score = 0.0
                rolling_std = 0.0
            else:
                rolling_mean = np.mean(spread_history)
                rolling_std = np.std(spread_history, ddof=1)
                z_score = (posterior_spread - rolling_mean) / rolling_std if rolling_std > 0 else 0.0
                
            if is_spread_vol_avg is not None:
                if rolling_std > is_spread_vol_avg * breakdown_vol_multiplier:
                    logger.critical(f"BREAKDOWN: {symbol_y} vs {symbol_x} Spread Vol ({rolling_std:.4f}) > {breakdown_vol_multiplier}x IS Vol. Halting pair.")
                    cointegration_broken = True
                    async with broker.lock:
                        if status != 0:
                            current_pos_y = broker.get_position(symbol_y)
                            current_pos_x = broker.get_position(symbol_x)
                            if current_pos_y > 0: broker.execute_order(symbol_y, 'sell', abs(current_pos_y), price_y)
                            elif current_pos_y < 0: broker.execute_order(symbol_y, 'buy', abs(current_pos_y), price_y)
                            if current_pos_x > 0: broker.execute_order(symbol_x, 'sell', abs(current_pos_x), price_x)
                            elif current_pos_x < 0: broker.execute_order(symbol_x, 'buy', abs(current_pos_x), price_x)
                            status = 0
                    continue

            async with broker.lock:
                portfolio_value = broker.get_total_value(shared_prices)
                broker.pm.check_and_update(portfolio_value)
                
                if broker.pm.trading_halted:
                    pass
                
                cost_of_spread = price_y + abs(kf.beta) * price_x
                if kelly > 0 and F > 0:
                    raw_target_alloc = portfolio_value * kelly * (abs(posterior_spread) / F)
                    max_alloc = portfolio_value * max_position_cap_pct
                    target_alloc = min(raw_target_alloc, max_alloc)
                    num_spreads = target_alloc / cost_of_spread
                else:
                    num_spreads = 0.0
                
                qty_y = num_spreads
                qty_x = num_spreads * kf.beta
                
                current_positions = {}
                for sym, qty in broker.positions.items():
                    if qty != 0:
                        current_positions[sym] = abs(qty * shared_prices.get(sym, 0.0))
                
                if status == 0 and num_spreads > 0 and not broker.pm.trading_halted:
                    assets = [symbol_y, symbol_x]
                    proposed_cost_y = qty_y * price_y
                    proposed_cost_x = qty_x * price_x
                    max_cost = max(proposed_cost_y, proposed_cost_x)
                    
                    if z_score < -entry_threshold:
                        if broker.pm.can_trade(assets, max_cost, current_positions, portfolio_value):
                            logger.info(f"{symbol_y} vs {symbol_x} Buy Spread at Z: {z_score:.2f} | Port: ${portfolio_value:.2f}")
                            broker.execute_order(symbol_y, 'buy', qty_y, price_y)
                            broker.execute_order(symbol_x, 'sell', qty_x, price_x)
                            status = 1
                    elif z_score > entry_threshold:
                        if broker.pm.can_trade(assets, max_cost, current_positions, portfolio_value):
                            logger.info(f"{symbol_y} vs {symbol_x} Sell Spread at Z: {z_score:.2f} | Port: ${portfolio_value:.2f}")
                            broker.execute_order(symbol_y, 'sell', qty_y, price_y)
                            broker.execute_order(symbol_x, 'buy', qty_x, price_x)
                            status = -1
                            
                elif status == 1:
                    if z_score >= -exit_threshold:
                        logger.info(f"{symbol_y} vs {symbol_x} Exit Long Spread at Z: {z_score:.2f}")
                        current_pos_y = broker.get_position(symbol_y)
                        current_pos_x = broker.get_position(symbol_x)
                        broker.execute_order(symbol_y, 'sell', abs(current_pos_y), price_y)
                        broker.execute_order(symbol_x, 'buy', abs(current_pos_x), price_x)
                        status = 0
                        
                elif status == -1:
                    if z_score <= exit_threshold:
                        logger.info(f"{symbol_y} vs {symbol_x} Exit Short Spread at Z: {z_score:.2f}")
                        current_pos_y = broker.get_position(symbol_y)
                        current_pos_x = broker.get_position(symbol_x)
                        broker.execute_order(symbol_y, 'buy', abs(current_pos_y), price_y)
                        broker.execute_order(symbol_x, 'sell', abs(current_pos_x), price_x)
                        status = 0
                        
        except Exception as e:
            logger.error(f"Error in {symbol_y} vs {symbol_x} loop: {e}")
            
        await asyncio.sleep(interval)


async def main_bot(exchange_id, pairs_csv, interval, total_iterations):
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({'enableRateLimit': True})
    
    if not os.path.exists(pairs_csv):
        logger.error(f"Pairs file {pairs_csv} not found!")
        return
        
    pairs_df = pd.read_csv(pairs_csv)
    pairs_df = pairs_df[pairs_df['kelly_fraction'] > 0]
    
    pm = PortfolioManager(max_concentration=0.30, max_drawdown=0.15)
    broker = AsyncPaperBroker(portfolio_manager=pm, initial_cash=100000.0)
    
    shared_prices = {}
    
    logger.info(f"Starting Multi-Pair Live Bot on {exchange_id} for {len(pairs_df)} pairs.")
    
    tasks = []
    for idx, row in pairs_df.iterrows():
        task = asyncio.create_task(pair_loop(exchange, broker, row, interval, total_iterations, shared_prices))
        tasks.append(task)
        
    try:
        await asyncio.gather(*tasks)
    finally:
        await exchange.close()
        logger.info("Bot shut down cleanly.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-Pair Async Kalman Live Bot')
    parser.add_argument('--exchange', type=str, default='binance', help='CCXT exchange ID')
    parser.add_argument('--pairs', type=str, default='results/selected_pairs_with_kelly.csv', help='Pairs CSV')
    parser.add_argument('--interval', type=int, default=5, help='Polling interval in seconds')
    parser.add_argument('--iterations', type=int, default=5, help='Max loop iterations for demo/testing')
    
    args = parser.parse_args()
    
    asyncio.run(main_bot(
        exchange_id=args.exchange,
        pairs_csv=args.pairs,
        interval=args.interval,
        total_iterations=args.iterations
    ))
