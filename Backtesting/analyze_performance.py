import sys, os
sys.path.append(os.path.abspath('.'))
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'Results')
SAVE_DIR_DSP = os.path.join(SAVE_DIR, 'DSP')
SAVE_DIR_MCVAR = os.path.join(SAVE_DIR, 'MCVAR')
SAVE_PATH_DSP = os.path.join(SAVE_DIR, "backtest_results_DSP_0_4330_50.npy")
SAVE_PATH_MCVAR = os.path.join(SAVE_DIR, "backtest_results_MCVAR_0_4330_95.npy")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR_DSP, exist_ok=True)
os.makedirs(SAVE_DIR_MCVAR, exist_ok=True)

import numpy as np
from Backtesting import DSPBacktester, MeanCVaRBacktester, Backtester
import matplotlib.pyplot as plt

# Set parameters
Backtester_class = "MCVAR"  # or "DSP"
tickers = ['spy', 'xlb', 'xle', 'xlf', 'xli', 'xlk', 'xlu', 'xlv', 'xly']
J = 10000
df = 6
rebalance_every = 20  # daily for test
decay = 0.5
n_days = 3*252  # 5 years of trading days (252 days/year * 5 years)
k0 = range(0, 4330-n_days, 252)
k1 = range(252, 4330, 252)
sharpe_ratios = []
sortino = []
CVaR = []
MaxDrawdown = []

for k in k0:
    start_idx = k
    end_idx = k + n_days  # n_days days for Sharpe ratio comparison

    # Initialize and load or run
    if Backtester_class == "DSP":
        bt = DSPBacktester(tickers=tickers, J=J, df=df,

                                rebalance_every=rebalance_every, decay=decay)
        SAVE_PATH = SAVE_PATH_DSP
        SAVE_DIR = SAVE_DIR_DSP
    else:
        bt = MeanCVaRBacktester(tickers=tickers, J=J, df=df,
                                 rebalance_every=rebalance_every, decay=decay)
        SAVE_PATH = SAVE_PATH_MCVAR
        SAVE_DIR = SAVE_DIR_MCVAR

    bt = Backtester(bt)   
    results = np.load(SAVE_PATH, allow_pickle=True)
    bt.store_backtest_results(results, start_idx=start_idx, end_idx=end_idx)
    print(f"Performance of backtest results over period {start_idx // 252 + 2007} to {end_idx // 252 + 2007}")
    metrics = bt.performance()
    sharpe_ratios.append(metrics[0])
    MaxDrawdown.append(metrics[1])
    CVaR.append(metrics[2])
    sortino.append(metrics[3])
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

plt.figure(figsize=(10,5))
plt.plot(years, sortino, label="Sortino")
plt.grid(True)
plt.title(f"Rolling {n_days//252}-Year Sortino Ratio Over Time")
plt.xlabel(f"End of {n_days//252}-Year Period")
plt.ylabel("Sortino Ratio")
plt.legend(['DSP Portfolio','SPY'])
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "rolling_sortino_ratio_comparison.png"))
plt.show()
plt.close()

plt.figure(figsize=(10,5))
plt.plot(years, CVaR, label="CVaR")
plt.grid(True)
plt.title(f"Rolling {n_days//252}-Year CVaR Over Time")
plt.xlabel(f"End of {n_days//252}-Year Period")
plt.ylabel("CVaR")
plt.legend(['DSP Portfolio','SPY'])
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "rolling_cvar_comparison.png"))
plt.show()
plt.close()

plt.figure(figsize=(10,5))
plt.plot(years, [(-mdd[0],-mdd[1]) for mdd in MaxDrawdown], label="Max Drawdown")
plt.grid(True)
plt.title(f"Rolling {n_days//252}-Year Max Drawdown Over Time")
plt.xlabel(f"End of {n_days//252}-Year Period")
plt.ylabel("Max Drawdown")
plt.legend(['DSP Portfolio','SPY'])
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "rolling_max_drawdown_comparison.png"))
plt.show()
plt.close()
