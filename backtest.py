import os
import argparse
import pandas as pd
import numpy as np
import collections
from datetime import datetime

try:
    import collections.abc
    collections.Iterable = collections.abc.Iterable
except AttributeError:
    pass

import backtrader as bt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from kalman_strategy import KalmanPairs
from portfolio import PortfolioManager

class PortfolioTracker(bt.Analyzer):
    def __init__(self):
        self.portfolio_values = []
        self.dates = []
        self.zscores = []

    def next(self):
        self.portfolio_values.append(self.strategy.broker.getvalue())
        self.dates.append(self.strategy.data0.datetime.date(0))
        self.zscores.append(getattr(self.strategy, 'current_z_score', np.nan))

    def get_analysis(self):
        return {
            'dates': self.dates,
            'values': self.portfolio_values,
            'zscores': self.zscores
        }

class KalmanPairsWithCosts(KalmanPairs):
    params = (
        ('margin_borrow_rate', 0.0), 
    )

    def __init__(self):
        super().__init__()
        self.cumulative_costs = 0.0

    def next(self):
        super().next()
        
        for data in self.datas:
            pos = self.getposition(data)
            if pos.size < 0:
                short_val = abs(pos.size) * data.close[0]
                borrow_fee = short_val * self.p.margin_borrow_rate
                self.broker.set_cash(self.broker.get_cash() - borrow_fee)
                self.cumulative_costs += borrow_fee

    def notify_order(self, order):
        super().notify_order(order)
        if order.status in [order.Completed]:
            self.cumulative_costs += order.executed.comm

def run_test(asset_y, asset_x, initial_beta, initial_alpha, burn_in=7, data_dir='data', results_dir='results', initial_cash=100000.0, commission=0.001, margin_borrow_rate=0.0, start_date=None, end_date=None, trade_start_date=None, run_label=''):
    cerebro = bt.Cerebro()
    
    safe_y = asset_y.replace('/', '_')
    safe_x = asset_x.replace('/', '_')
    
    file_y = os.path.join(data_dir, f"{safe_y}.csv")
    file_x = os.path.join(data_dir, f"{safe_x}.csv")
    
    if not os.path.exists(file_y) or not os.path.exists(file_x):
        print(f"Data files missing for {asset_y} or {asset_x}. Fetching data first is required.")
        return None
        
    data0 = bt.feeds.GenericCSVData(
        dataname=file_y,
        datetime=0, open=1, high=2, low=3, close=4, volume=5, openinterest=-1,
        headers=True,
        dtformat=lambda x: datetime.strptime(x, '%Y-%m-%d' if len(x) <= 10 else '%Y-%m-%d %H:%M:%S'),
        fromdate=pd.to_datetime(start_date) if start_date else datetime.min,
        todate=pd.to_datetime(end_date) if end_date else datetime.max
    )
    cerebro.adddata(data0, name=asset_y)
    
    data1 = bt.feeds.GenericCSVData(
        dataname=file_x,
        datetime=0, open=1, high=2, low=3, close=4, volume=5, openinterest=-1,
        headers=True,
        dtformat=lambda x: datetime.strptime(x, '%Y-%m-%d' if len(x) <= 10 else '%Y-%m-%d %H:%M:%S'),
        fromdate=pd.to_datetime(start_date) if start_date else datetime.min,
        todate=pd.to_datetime(end_date) if end_date else datetime.max
    )
    cerebro.adddata(data1, name=asset_x)
    
    cerebro.addstrategy(
        KalmanPairsWithCosts,
        initial_beta=initial_beta,
        initial_alpha=initial_alpha,
        allocation=initial_cash / 2.0,
        margin_borrow_rate=margin_borrow_rate,
        burn_in=burn_in,
        trade_start_date=trade_start_date
    )
    
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(PortfolioTracker, _name='portfolio')
    
    print(f"Starting Backtest: {asset_y} vs {asset_x}")
    print(f"Initial Portfolio Value: {initial_cash:.2f}")
    
    results = cerebro.run()
    strat = results[0]
    
    final_value = cerebro.broker.getvalue()
    net_return = (final_value - initial_cash) / initial_cash * 100.0
    
    total_costs = strat.cumulative_costs
    gross_return = net_return + (total_costs / initial_cash * 100.0)
    
    sharpe_analysis = strat.analyzers.sharpe.get_analysis()
    sharpe = sharpe_analysis.get('sharpe_ratio', np.nan)
    
    dd_analysis = strat.analyzers.drawdown.get_analysis()
    max_dd = dd_analysis.get('max', {}).get('drawdown', 0.0)
    
    trade_analysis = strat.analyzers.trades.get_analysis()
    total_trades = trade_analysis.total.total if 'total' in trade_analysis else 0
    
    print(f"Final Portfolio Value: {final_value:.2f}")
    print(f"Gross Return: {gross_return:.2f}%")
    print(f"Net Return: {net_return:.2f}%")
    print(f"Total Costs: {total_costs:.2f}")
    print(f"Sharpe Ratio: {sharpe if sharpe is not None and not np.isnan(sharpe) else 0.0:.2f}")
    print(f"Max Drawdown: {max_dd:.2f}%")
    print(f"Total Trades: {total_trades}")
    
    port_data = strat.analyzers.portfolio.get_analysis()
    df_equity = pd.DataFrame({
        'Date': port_data['dates'],
        'Equity': port_data['values'],
        'ZScore': port_data['zscores']
    })
    df_equity.set_index('Date', inplace=True)
    
    pair_name = f"{run_label}{safe_y}_vs_{safe_x}"
    df_equity.to_csv(os.path.join(results_dir, f"equity_{pair_name}.csv"))
    
    entry_threshold = strat.p.entry_threshold
    exit_threshold = strat.p.exit_threshold
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    ax1.plot(df_equity.index, df_equity['Equity'], color='#8884d8', linewidth=2, label='Portfolio Value')
    ax1.set_title(f"Equity Curve - {asset_y} vs {asset_x} (Kalman Pairs)")
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()
    
    ax2.plot(df_equity.index, df_equity['ZScore'], color='#ff7300', linewidth=1.5, label='Z-Score')
    ax2.axhline(entry_threshold, color='r', linestyle='--', alpha=0.6, label=f'Entry (+{entry_threshold})')
    ax2.axhline(-entry_threshold, color='r', linestyle='--', alpha=0.6, label=f'Entry (-{entry_threshold})')
    ax2.axhline(exit_threshold, color='g', linestyle='--', alpha=0.6, label=f'Exit (+{exit_threshold})')
    ax2.axhline(-exit_threshold, color='g', linestyle='--', alpha=0.6, label=f'Exit (-{exit_threshold})')
    ax2.set_title("Z-Score Trajectory")
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Z-Score')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"tearsheet_{pair_name}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved tearsheet plot to {plot_path}")
    
    return {
        'pair': f"{asset_y}_vs_{asset_x}",
        'asset_y': asset_y,
        'asset_x': asset_x,
        'initial_cash': initial_cash,
        'final_value': final_value,
        'gross_return': gross_return,
        'net_return': net_return,
        'total_costs': total_costs,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'total_trades': total_trades,
        'strat': strat
    }

def run_all_backtests(selected_pairs_csv='results/selected_pairs.csv', data_dir='data', results_dir='results', borrow_rate=0.0, split_date=None):
    if not os.path.exists(selected_pairs_csv):
        print(f"Selected pairs file not found: {selected_pairs_csv}")
        return
        
    pairs_df = pd.read_csv(selected_pairs_csv)
    if pairs_df.empty:
        print("No pairs available for backtesting.")
        return
        
    summary_results = []
    
    if split_date is not None:
        is_end = pd.to_datetime(split_date) - pd.Timedelta(days=1)
        oos_start = pd.to_datetime(split_date)
    else:
        is_end = None
        oos_start = None
    
    for idx, row in pairs_df.iterrows():
        burn_in_val = int(row.get('burn_in', 7))
        print(f"RUNNING IS (IN-SAMPLE) BACKTEST FOR {row['asset_a']} vs {row['asset_b']} (Burn-in: {burn_in_val})")
        res_is = run_test(
            asset_y=row['asset_a'],
            asset_x=row['asset_b'],
            initial_beta=row['beta'],
            initial_alpha=row['alpha'],
            burn_in=burn_in_val,
            data_dir=data_dir,
            results_dir=results_dir,
            margin_borrow_rate=borrow_rate,
            start_date=None,
            end_date=is_end,
            run_label='IS_'
        )
        
        print(f"RUNNING OOS (OUT-OF-SAMPLE) BACKTEST FOR {row['asset_a']} vs {row['asset_b']} (Burn-in: {burn_in_val})")
        res_oos = run_test(
            asset_y=row['asset_a'],
            asset_x=row['asset_b'],
            initial_beta=row['beta'],
            initial_alpha=row['alpha'],
            burn_in=burn_in_val,
            data_dir=data_dir,
            results_dir=results_dir,
            margin_borrow_rate=borrow_rate,
            start_date=None,
            end_date=None,
            trade_start_date=oos_start.date() if oos_start else None,
            run_label='OOS_'
        )
        
        if res_is and res_oos:
            summary_results.append({
                'pair': res_is['pair'],
                'asset_y': res_is['asset_y'],
                'asset_x': res_is['asset_x'],
                'initial_cash': res_is['initial_cash'],
                'is_gross_ret': res_is['gross_return'],
                'oos_gross_ret': res_oos['gross_return'],
                'is_net_ret': res_is['net_return'],
                'oos_net_ret': res_oos['net_return'],
                'is_sharpe': res_is['sharpe'],
                'oos_sharpe': res_oos['sharpe'],
                'is_max_dd': res_is['max_dd'],
                'oos_max_dd': res_oos['max_dd'],
                'is_trades': res_is['total_trades'],
                'oos_trades': res_oos['total_trades']
            })
            
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_path = os.path.join(results_dir, "backtest_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved overall backtest summary to {summary_path}")
        print(summary_df.to_string())

def run_portfolio_backtest(selected_pairs_csv='results/selected_pairs_with_kelly.csv', data_dir='data', results_dir='results', initial_cash=100000.0, commission=0.001, borrow_rate=0.0, split_date=None):
    if not os.path.exists(selected_pairs_csv):
        print(f"Selected pairs file not found: {selected_pairs_csv}")
        return
        
    pairs_df = pd.read_csv(selected_pairs_csv)
    if pairs_df.empty:
        print("No pairs available for backtesting.")
        return
        
    if split_date is not None:
        oos_start = pd.to_datetime(split_date)
    else:
        oos_start = None
        
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    
    pm = PortfolioManager(max_concentration=0.30, max_drawdown=0.15)
    added_assets = set()
    
    for idx, row in pairs_df.iterrows():
        kelly = row.get('kelly_fraction', 0.0)
        if kelly <= 0:
            print(f"Skipping {row['asset_a']} vs {row['asset_b']} - Kelly fraction is 0.")
            continue
            
        for asset in [row['asset_a'], row['asset_b']]:
            if asset not in added_assets:
                safe_name = asset.replace('/', '_')
                file_path = os.path.join(data_dir, f"{safe_name}.csv")
                data = bt.feeds.GenericCSVData(
                    dataname=file_path,
                    datetime=0, open=1, high=2, low=3, close=4, volume=5, openinterest=-1,
                    headers=True,
                    dtformat=lambda x: datetime.strptime(x, '%Y-%m-%d' if len(x) <= 10 else '%Y-%m-%d %H:%M:%S'),
                    fromdate=datetime.min,
                    todate=datetime.max
                )
                cerebro.adddata(data, name=asset)
                added_assets.add(asset)
                
        cerebro.addstrategy(
            KalmanPairsWithCosts,
            initial_beta=row['beta'],
            initial_alpha=row['alpha'],
            kelly_fraction=kelly,
            is_spread_vol_avg=row.get('is_spread_vol_avg', None),
            max_position_cap_pct=0.10,
            breakdown_vol_multiplier=3.0,
            portfolio_manager=pm,
            margin_borrow_rate=borrow_rate,
            burn_in=int(row.get('burn_in', 7)),
            trade_start_date=oos_start.date() if oos_start else None,
            data_y_name=row['asset_a'],
            data_x_name=row['asset_b']
        )
        print(f"Added {row['asset_a']} vs {row['asset_b']} (Kelly: {kelly:.4f})")
        
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    print("Starting Portfolio OOS Backtest")
    results = cerebro.run()
    
    final_value = cerebro.broker.getvalue()
    net_return = (final_value - initial_cash) / initial_cash * 100.0
    
    print(f"Portfolio Final Value: {final_value:.2f}")
    print(f"Portfolio Net Return: {net_return:.2f}%")
    print(f"Trading Halted via Drawdown Limit: {pm.trading_halted}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Backtest Kalman pairs strategy on cointegrated pairs.')
    parser.add_argument('--pairs', type=str, default='results/selected_pairs.csv', help='Path to selected pairs CSV')
    parser.add_argument('--portfolio', action='store_true', help='Run multi-pair portfolio OOS backtest using kelly sizing')
    parser.add_argument('--datadir', type=str, default='data', help='Data directory')
    parser.add_argument('--resdir', type=str, default='results', help='Results directory')
    parser.add_argument('--borrow_rate', type=float, default=0.0, help='Daily margin borrow rate on shorts (e.g. 0.0 for spot with no fees)')
    parser.add_argument('--test_days', type=int, default=180, help='Number of days for Out-of-Sample testing')

    args = parser.parse_args()
    if args.portfolio:
        df_combined = pd.read_csv(os.path.join(args.datadir, "combined_close.csv"), index_col=0, parse_dates=True)
        split_idx = max(1, len(df_combined) - args.test_days)
        split_date = df_combined.index[split_idx]
        print(f"Portfolio OOS trading starts {split_date.date()} (OOS Window: {args.test_days} days)")
        run_portfolio_backtest('results/selected_pairs_with_kelly.csv', args.datadir, args.resdir, borrow_rate=args.borrow_rate, split_date=split_date)
    else:
        run_all_backtests(args.pairs, args.datadir, args.resdir, args.borrow_rate)
