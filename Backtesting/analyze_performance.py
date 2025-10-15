import sys, os
sys.path.append(os.path.abspath('.'))
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'Results')
SAVE_DIR_DSP = os.path.join(SAVE_DIR, 'DSP')
SAVE_DIR_MCVAR = os.path.join(SAVE_DIR, 'MCVAR')
SAVE_PATH_DSP = os.path.join(SAVE_DIR, "backtest_results_DSP_0_4330_75_Rebal20_5Short_160Gross.npy")
SAVE_PATH_MCVAR = os.path.join(SAVE_DIR, "backtest_results_MCVAR_0_4330_95_Rebal20_5Short_160Gross.npy")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR_DSP, exist_ok=True)
os.makedirs(SAVE_DIR_MCVAR, exist_ok=True)

import numpy as np
from Backtesting import DSPBacktester, MeanCVaRBacktester, Backtester
import matplotlib.pyplot as plt

# Set parameters
tickers = ['spy', 'xlb', 'xle', 'xlf', 'xli', 'xlk', 'xlu', 'xlv', 'xly']
J = 10000
df = 6
rebalance_every = 20  # day for test
decay = 0.95
n_days = int(1*252)  # n years of trading days (252 days/year * n years)
k0 = range(0, 4330-n_days, 252)
k1 = range(252, 4330, 252)
sharpe_ratios_DSP = []
sortino_DSP = []
CVaR_DSP = []
MaxDrawdown_DSP = []
sharpe_ratios_MCVAR = []
sortino_MCVAR = []
CVaR_MCVAR = []
MaxDrawdown_MCVAR = []
sharpe_ratios = []
sortino = []
CVaR = []
MaxDrawdown = []

for k in k0:
    start_idx = k
    end_idx = k + n_days  # n_days days for Sharpe ratio comparison

    # Initialize and load or run
    btDSP = DSPBacktester(tickers=tickers, J=J, df=df,
                            rebalance_every=rebalance_every, decay=decay)

    btMCVAR = MeanCVaRBacktester(tickers=tickers, J=J, df=df,
                                rebalance_every=rebalance_every, decay=decay)

    btDSP = Backtester(btDSP)
    resultsDSP = np.load(SAVE_PATH_DSP, allow_pickle=True)
    btDSP.store_backtest_results(resultsDSP, start_idx=start_idx, end_idx=end_idx)
    #print(f"Performance of DSP backtest results over period {start_idx // 252 + 2007} to {end_idx // 252 + 2007}")
    metrics = btDSP.performance()
    sharpe_ratios_DSP.append(metrics[0])
    MaxDrawdown_DSP.append(metrics[1])
    CVaR_DSP.append(metrics[2])
    sortino_DSP.append(metrics[3])

    btMCVAR = Backtester(btMCVAR)
    resultsMCVAR = np.load(SAVE_PATH_MCVAR, allow_pickle=True)
    btMCVAR.store_backtest_results(resultsMCVAR, start_idx=start_idx, end_idx=end_idx)
    #print(f"Performance of Mean-CVaR backtest results over period {start_idx // 252 + 2007} to {end_idx // 252 + 2007}")
    metrics = btMCVAR.performance()
    sharpe_ratios_MCVAR.append(metrics[0])
    MaxDrawdown_MCVAR.append(metrics[1])
    CVaR_MCVAR.append(metrics[2])
    sortino_MCVAR.append(metrics[3])
    
    # Convert lists to numpy arrays for easier manipulation
    # Each metrics tuple contains: (strategy_sharpe, spy_sharpe), (strategy_mdd, spy_mdd), etc.
    dsp_sharpe, spy_sharpe = sharpe_ratios_DSP[-1]  # Extract DSP and SPY Sharpe ratios (current iteration)
    mcvar_sharpe, _ = sharpe_ratios_MCVAR[-1]       # Extract Mean-CVaR Sharpe ratio (current iteration)
    
    sharpe_ratios.append([dsp_sharpe, spy_sharpe, mcvar_sharpe])
    
    dsp_sortino, spy_sortino = sortino_DSP[-1]
    mcvar_sortino, _ = sortino_MCVAR[-1]
    sortino.append([dsp_sortino, spy_sortino, mcvar_sortino])
    
    dsp_cvar, spy_cvar = CVaR_DSP[-1]
    mcvar_cvar, _ = CVaR_MCVAR[-1]
    CVaR.append([dsp_cvar, spy_cvar, mcvar_cvar])
    
    dsp_mdd, spy_mdd = MaxDrawdown_DSP[-1]
    mcvar_mdd, _ = MaxDrawdown_MCVAR[-1]
    MaxDrawdown.append([dsp_mdd, spy_mdd, mcvar_mdd])

# Convert to numpy arrays for plotting
sharpe_ratios = np.array(sharpe_ratios)
sortino = np.array(sortino)
CVaR = np.array(CVaR)
MaxDrawdown = np.array(MaxDrawdown)

years = [k // 252 + 2007 + n_days // 252 for k in k0[:len(sharpe_ratios)]]
plt.figure(figsize=(10,5))
plt.plot(years, sharpe_ratios[:, 0], label="DSP")
plt.plot(years, sharpe_ratios[:, 1], label="SPY")
plt.plot(years, sharpe_ratios[:, 2], label="Mean-CVaR")
plt.grid(True)
plt.title(f"Rolling {n_days//252}-Year Sharpe Ratio Over Time")
plt.xlabel(f"End of {n_days//252}-Year Period")
plt.ylabel("Sharpe Ratio")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "rolling_sharpe_ratio_comparison.png"))
plt.show()
plt.close()

plt.figure(figsize=(10,5))
plt.plot(years, sortino[:, 0], label="DSP")
plt.plot(years, sortino[:, 1], label="SPY")
plt.plot(years, sortino[:, 2], label="Mean-CVaR")
plt.grid(True)
plt.title(f"Rolling {n_days//252}-Year Sortino Ratio Over Time")
plt.xlabel(f"End of {n_days//252}-Year Period")
plt.ylabel("Sortino Ratio")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "rolling_sortino_ratio_comparison.png"))
plt.show()
plt.close()

plt.figure(figsize=(10,5))
plt.plot(years, CVaR[:, 0], label="DSP")
plt.plot(years, CVaR[:, 1], label="SPY")
plt.plot(years, CVaR[:, 2], label="Mean-CVaR")
plt.grid(True)
plt.title(f"Rolling {n_days//252}-Year CVaR Over Time")
plt.xlabel(f"End of {n_days//252}-Year Period")
plt.ylabel("CVaR")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "rolling_cvar_comparison.png"))
plt.show()
plt.close()

plt.figure(figsize=(10,5))
plt.plot(years, -MaxDrawdown[:, 0], label="DSP")  # Convert to positive for plotting
plt.plot(years, -MaxDrawdown[:, 1], label="SPY")
plt.plot(years, -MaxDrawdown[:, 2], label="Mean-CVaR")
plt.grid(True)
plt.title(f"Rolling {n_days//252}-Year Max Drawdown Over Time")
plt.xlabel(f"End of {n_days//252}-Year Period")
plt.ylabel("Max Drawdown")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "rolling_max_drawdown_comparison.png"))
plt.show()
plt.close()


# Plot weight evolution for the full period (outside the rolling window loop)
print("Generating weight evolution plots for DSP portfolio...")

# Load full period results for weight analysis
btDSP_full = DSPBacktester(tickers=tickers, J=J, df=df,
                          rebalance_every=rebalance_every, decay=decay)
btDSP_full = Backtester(btDSP_full)
resultsDSP_full = np.load(SAVE_PATH_DSP, allow_pickle=True)
btDSP_full.store_backtest_results(resultsDSP_full, start_idx=0, end_idx=4330)

# Get weights dataframe
weights_df = btDSP_full.get_weights_df()
dates = btDSP_full.rebalance_dates

# Simple line chart showing all weights over time
plt.figure(figsize=(14, 8))
for col in weights_df.columns:
    plt.plot(dates, weights_df[col], label=col.upper(), linewidth=2)

plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.title('DSP Portfolio Weights Evolution Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio Weight')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "DSP_weight_evolution.png"), dpi=300, bbox_inches='tight')
plt.show()
plt.close()


# Plot weight evolution for the full period (outside the rolling window loop)
print("Generating weight evolution plots for Mean-CVaR portfolio...")

# Load full period results for weight analysis
btMeanCVaR_full = MeanCVaRBacktester(tickers=tickers, J=J, df=df,
                                      rebalance_every=rebalance_every, decay=decay)
btMeanCVaR_full = Backtester(btMeanCVaR_full)
resultsMeanCVaR_full = np.load(SAVE_PATH_MCVAR, allow_pickle=True)
btMeanCVaR_full.store_backtest_results(resultsMeanCVaR_full, start_idx=0, end_idx=4330)

# Get weights dataframe
weights_df = btMeanCVaR_full.get_weights_df()
dates = btMeanCVaR_full.rebalance_dates

# Simple line chart showing all weights over time
plt.figure(figsize=(14, 8))
for col in weights_df.columns:
    plt.plot(dates, weights_df[col], label=col.upper(), linewidth=2)

plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.title('Mean-CVaR Portfolio Weights Evolution Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio Weight')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "MCVAR_weight_evolution.png"), dpi=300, bbox_inches='tight')
plt.show()
plt.close()


