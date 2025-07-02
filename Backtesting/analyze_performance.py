import sys, os
sys.path.append(os.path.abspath('.'))
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'Results')
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_PATH = os.path.join(SAVE_DIR, "backtest_results_0_4330_75.npy")


import numpy as np
from Backtesting import DSPBacktester

# Set parameters
tickers = ['spy', 'xlb', 'xle', 'xlf', 'xli', 'xlk', 'xlu', 'xlv', 'xly']
J = 10000
df = 6
rebalance_every = 10  # daily for test
decay = 0.95
n_days = 4330
start_idx = 0
end_idx = start_idx + n_days

# Initialize and load or run
bt = DSPBacktester(tickers=tickers, J=J, df=df, rebalance_every=rebalance_every, decay=decay)
try:
    results = np.load(SAVE_PATH, allow_pickle=True)
    bt.store_backtest_results(results, start_idx=start_idx, end_idx=end_idx)
    # Plot performance
    bt.performance()
except:
    print("Data not found, first run 'run_parallel_backtest.py' file")
    #bt.run_backtest(start_idx=start_idx, end_idx=end_idx)


