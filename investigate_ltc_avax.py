import os
import pandas as pd
import backtrader as bt
from datetime import datetime
import numpy as np

summary_path = 'results/backtest_summary.csv'
if os.path.exists(summary_path):
    df = pd.read_csv(summary_path)
    row = df[df['pair'] == 'LTC/USDT_vs_AVAX/USDT'].iloc[0]
    print("Summary Checks:")
    print(f"OOS Gross Return: {row['oos_gross_ret']:.2f}%")
    print(f"OOS Net Return:   {row['oos_net_ret']:.2f}%")
    print(f"OOS Max Drawdown: {row['oos_max_dd']:.2f}%")
    print(f"OOS Sharpe:       {row['oos_sharpe']}")
    print(f"OOS Trades:       {row['oos_trades']}")
    print(f"Gross >= Net holds: {row['oos_gross_ret'] >= row['oos_net_ret']}")

data_dir = 'data'
y_file = os.path.join(data_dir, 'LTC_USDT.csv')
x_file = os.path.join(data_dir, 'AVAX_USDT.csv')

df_y = pd.read_csv(y_file, parse_dates=['datetime']).set_index('datetime')
df_x = pd.read_csv(x_file, parse_dates=['datetime']).set_index('datetime')

df_combined = pd.read_csv(os.path.join(data_dir, 'combined_close.csv'), index_col=0, parse_dates=True)
split_idx = int(len(df_combined) * 0.75)
split_date = df_combined.index[split_idx]

oos_y = df_y.loc[split_date:]
oos_x = df_x.loc[split_date:]

print("Data Anomaly Checks (OOS Window):")
print(f"OOS Window: {oos_y.index[0].date()} to {oos_y.index[-1].date()} ({len(oos_y)} bars)")

for name, df_oos in [('LTC', oos_y), ('AVAX', oos_x)]:
    pct_change = df_oos['close'].pct_change().abs()
    max_move = pct_change.max()
    max_move_date = pct_change.idxmax()
    stale_bars = (pct_change == 0).sum()
    print(f"{name}: Max Daily Move = {max_move*100:.2f}% on {max_move_date.date() if pd.notnull(max_move_date) else 'N/A'}, Stale Bars = {stale_bars}")

print("Individual Trade Analysis:")
from backtest import run_test
pairs_df = pd.read_csv('results/selected_pairs.csv')
pair_row = pairs_df[(pairs_df['asset_a'] == 'LTC/USDT') & (pairs_df['asset_b'] == 'AVAX/USDT')].iloc[0]

from kalman_strategy import KalmanPairs
class KalmanPairsWithCosts(KalmanPairs):
    params = (('margin_borrow_rate', 0.0),)
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

cerebro = bt.Cerebro()
data0 = bt.feeds.GenericCSVData(dataname=y_file, datetime=0, open=1, high=2, low=3, close=4, volume=5, openinterest=-1, headers=True, dtformat=lambda x: datetime.strptime(x, '%Y-%m-%d' if len(x) <= 10 else '%Y-%m-%d %H:%M:%S'))
data1 = bt.feeds.GenericCSVData(dataname=x_file, datetime=0, open=1, high=2, low=3, close=4, volume=5, openinterest=-1, headers=True, dtformat=lambda x: datetime.strptime(x, '%Y-%m-%d' if len(x) <= 10 else '%Y-%m-%d %H:%M:%S'))
cerebro.adddata(data0, name='LTC')
cerebro.adddata(data1, name='AVAX')

cerebro.addstrategy(KalmanPairsWithCosts, initial_beta=pair_row['beta'], initial_alpha=pair_row['alpha'], allocation=50000.0, margin_borrow_rate=0.0, burn_in=pair_row['burn_in'], trade_start_date=split_date.date())
cerebro.broker.setcash(100000.0)
cerebro.broker.setcommission(commission=0.001)
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

results = cerebro.run()
strat = results[0]
trade_analysis = strat.analyzers.trades.get_analysis()

pnl_list = []
if 'closed' in trade_analysis and len(trade_analysis.closed) > 0:
    for trade in trade_analysis.closed:
        pnl_list.append(trade.pnlcomm)

pnl_list.sort(reverse=True)
print(f"Total trades analyzed: {len(pnl_list)}")
print("Top 5 Winning Trades (Net PnL):")
for pnl in pnl_list[:5]:
    print(f"  +${pnl:.2f}")
print("Top 5 Losing Trades (Net PnL):")
for pnl in pnl_list[-5:]:
    print(f"  ${pnl:.2f}")

total_net_pnl = sum(pnl_list)
print(f"Total Net PnL from trades: ${total_net_pnl:.2f}")
print(f"Max single trade contribution to total PnL: {max(pnl_list) / total_net_pnl * 100 if total_net_pnl > 0 else 0:.2f}%")
