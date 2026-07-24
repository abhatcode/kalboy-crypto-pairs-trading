import os
import argparse
import pandas as pd
import ccxt
import time

def fetch_historical_data(exchange_id='binance', symbols=None, timeframe='1d', days=730, output_dir='data'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({
        'enableRateLimit': True,
    })

    if symbols is None:
        symbols = [
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT',
            'XRP/USDT', 'LTC/USDT', 'LINK/USDT', 'DOGE/USDT', 'AVAX/USDT',
            'MATIC/USDT', 'UNI/USDT', 'ETC/USDT', 'ICP/USDT', 'FIL/USDT',
            'TRX/USDT', 'BCH/USDT', 'XLM/USDT', 'ATOM/USDT', 'VET/USDT',
            'EOS/USDT', 'THETA/USDT', 'FTM/USDT', 'ALGO/USDT', 'NEAR/USDT',
            'SAND/USDT', 'MANA/USDT', 'AXS/USDT', 'GRT/USDT', 'SHIB/USDT'
        ]

    print(f"Initializing data fetch from {exchange_id} for {len(symbols)} symbols...")
    print(f"Timeframe: {timeframe}, Days: {days}")

    since = exchange.milliseconds() - days * 24 * 60 * 60 * 1000
    
    close_prices = {}
    
    for symbol in symbols:
        safe_symbol = symbol.replace('/', '_')
        csv_path = os.path.join(output_dir, f"{safe_symbol}.csv")
        
        print(f"Fetching {symbol}...")
        all_ohlcv = []
        current_since = since
        
        while True:
            try:
        while True:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
                if not ohlcv:
                    break
                
                all_ohlcv.extend(ohlcv)
                
                last_timestamp = ohlcv[-1][0]
                if last_timestamp == current_since or len(ohlcv) < 1000:
                    break
                
                current_since = last_timestamp + 1
                time.sleep(exchange.rateLimit / 1000.0)
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                break
        
        if len(all_ohlcv) == 0:
            print(f"No data fetched for {symbol}. Skipping.")
            continue
            
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        df.drop(columns=['timestamp'], inplace=True)
        
        if timeframe.endswith('d'):
            df.index = df.index.strftime('%Y-%m-%d')
        else:
            df.index = df.index.strftime('%Y-%m-%d %H:%M:%S')
            
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        df.to_csv(csv_path)
        print(f"Saved {len(df)} rows for {symbol} to {csv_path}")
        
        close_prices[symbol] = df['close']
        
    if close_prices:
        combined_df = pd.DataFrame(close_prices)
        combined_df.dropna(inplace=True)
        combined_csv_path = os.path.join(output_dir, "combined_close.csv")
        combined_df.to_csv(combined_csv_path)
        print(f"Saved combined closing prices to {combined_csv_path} (shape: {combined_df.shape})")
        return combined_df
    else:
        print("No data was fetched successfully.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch OHLCV data using CCXT')
    parser.add_argument('--exchange', type=str, default='binance', help='CCXT exchange ID (default: binance)')
    parser.add_argument('--symbols', type=str, nargs='+', help='List of symbols (default: top 30)')
    parser.add_argument('--timeframe', type=str, default='1d', help='Timeframe (e.g., 1d, 1h)')
    parser.add_argument('--days', type=int, default=730, help='Number of days of history')
    parser.add_argument('--outdir', type=str, default='data', help='Output directory')
    
    args = parser.parse_args()
    
    fetch_historical_data(
        exchange_id=args.exchange,
        symbols=args.symbols,
        timeframe=args.timeframe,
        days=args.days,
        output_dir=args.outdir
    )
