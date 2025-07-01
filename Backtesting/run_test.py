import sys, os
sys.path.append(os.path.abspath('.'))

from Backtesting import DSPBacktester

# Set parameters
tickers = ['spy', 'xlb', 'xle', 'xlf', 'xli', 'xlk', 'xlu', 'xlv', 'xly']
J = 10000
df = 6
rebalance_every = 10  # daily for test
decay = 0.95
n_days = 100

# Initialize and run
bt = DSPBacktester(tickers=tickers, J=J, df=df, rebalance_every=rebalance_every, decay=decay)
bt.run_backtest(start_idx=252, end_idx=252 + n_days)

# Plot performance
bt.plot_performance()

# Optional: print weights and PnL
print(bt.get_weights_df())
print(bt.get_pnl_df())
