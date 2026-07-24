import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_mock_crypto_data(output_dir='data', days=730):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Generating mock crypto data for {days} days...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days-1)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    formatted_dates = dates.strftime('%Y-%m-%d')
    
    np.random.seed(42)
    x_returns = np.random.normal(0.0005, 0.02, size=len(dates))
    x_prices = 50000.0 * np.exp(np.cumsum(x_returns))
    
    beta = 0.06
    alpha = 500.0
    rho = 0.85
    spread = np.zeros(len(dates))
    spread_noise = np.random.normal(0, 150, size=len(dates))
    for t in range(1, len(dates)):
        spread[t] = rho * spread[t-1] + spread_noise[t]
        
    y_prices = beta * x_prices + alpha + spread
    
    sol_returns = np.random.normal(0.001, 0.04, size=len(dates))
    sol_prices = 100.0 * np.exp(np.cumsum(sol_returns))
    
    ada_returns = np.random.normal(0.0002, 0.03, size=len(dates))
    ada_prices = 1.0 * np.exp(np.cumsum(ada_returns))
    
    assets = {
        'BTC_USDT': x_prices,
        'ETH_USDT': y_prices,
        'SOL_USDT': sol_prices,
        'ADA_USDT': ada_prices
    }
    
    close_prices = {}
    
    for name, prices in assets.items():
        df = pd.DataFrame(index=formatted_dates)
        df['close'] = prices
        df['open'] = df['close'] * (1.0 + np.random.normal(0, 0.002, size=len(dates)))
        df['high'] = df[['open', 'close']].max(axis=1) * (1.0 + abs(np.random.normal(0, 0.005, size=len(dates))))
        df['low'] = df[['open', 'close']].min(axis=1) * (1.0 - abs(np.random.normal(0, 0.005, size=len(dates))))
        df['volume'] = np.random.uniform(1000, 50000, size=len(dates))
        
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        csv_path = os.path.join(output_dir, f"{name}.csv")
        df.index.name = 'datetime'
        df.to_csv(csv_path)
        print(f"Generated mock data for {name} saved to {csv_path}")
        
        close_prices[name.replace('_', '/')] = df['close']
        
    combined_df = pd.DataFrame(close_prices, index=formatted_dates)
    combined_df.index.name = 'datetime'
    combined_df.dropna(inplace=True)
    combined_csv_path = os.path.join(output_dir, "combined_close.csv")
    combined_df.to_csv(combined_csv_path)
    print(f"Generated combined close prices saved to {combined_csv_path}")

if __name__ == "__main__":
    generate_mock_crypto_data()
