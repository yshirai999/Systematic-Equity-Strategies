import sys, os
sys.path.append(os.path.abspath('.'))
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'Results')
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_PATH = os.path.join(SAVE_DIR, "backtest_results_0_4330_50.npy")


import numpy as np
from Backtesting import DSPBacktester
import matplotlib.pyplot as plt

# Set parameters
tickers = ['spy', 'xlb', 'xle', 'xlf', 'xli', 'xlk', 'xlu', 'xlv', 'xly']
J = 10000
df = 6
rebalance_every = 20  # daily for test
decay = 0.5
n_days = 3*252  # 5 years of trading days (252 days/year * 5 years)
k0 = range(0, 4330-n_days, 252)
k1 = range(252, 4330, 252)
sharpe_ratios = []

for k in k0:
    start_idx = k
    end_idx = k + n_days  # n_days days for Sharpe ratio comparison

    # Initialize and load or run
    bt = DSPBacktester(tickers=tickers, J=J, df=df, rebalance_every=rebalance_every, decay=decay)
    try:
        results = np.load(SAVE_PATH, allow_pickle=True)
        bt.store_backtest_results(results, start_idx=start_idx, end_idx=end_idx)
        print(f"Performance of backtest results over period {start_idx // 252 + 2007} to {end_idx // 252 + 2007}")
        sharpe_ratio = bt.performance()
        sharpe_ratios.append(sharpe_ratio)
    except:
        print("Data not found, first run 'run_parallel_backtest.py' file")
        #bt.run_backtest(start_idx=start_idx, end_idx=end_idx)

years = [k // 252 + 2007 + n_days // 252 for k in k0]
plt.figure(figsize=(10,5))
plt.plot(years, sharpe_ratios, label="Sharpe")
plt.grid(True)
plt.title(f"Rolling {n_days//252}-Year Sharpe Ratio Over Time")
plt.xlabel(f"End of {n_days//252}-Year Period")
plt.ylabel("Sharpe Ratio")
plt.legend(['DSP Portfolio','SPY'])
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "rolling_sharpe_ratio_comparison.png"))
plt.show()
plt.close()

