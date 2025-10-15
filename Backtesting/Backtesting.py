import sys, os
sys.path.append(os.path.abspath('.'))

import numpy as np
import matplotlib.pyplot as plt
from Optimizers.DSP.dsp_solver import DSPOptimizer
from Optimizers.MeanCVaR.mean_CVaR_solver import MeanCVaROptimizer

class Backtester:
    def __init__(self, Optimizer):
        self.optimizer = Optimizer  # Instance of DSPBacktester, MeanCVaRBacktester, etc.
        self.rebalance_every = self.optimizer.rebalance_every
        self.decay = self.optimizer.decay

        self.w_total_history = []
        self.pnl_history = []
        self.rebalance_dates = []
        self.results = []
        self.total_wealth = []
        self.total_wealth_spy = [] 
        self.pnl_history_spy = []

        self.valid_returns = Optimizer.valid_returns
        self.valid_dates = Optimizer.valid_dates
        self.tickers = Optimizer.tickers

    def compute_weights_on_date(self, date_idx):
        date_str = str(self.valid_dates[date_idx])[:10]
        weights = self.optimizer.solve(date_str)
        return (date_str, weights)

    def store_backtest_results(self, results, start_idx, end_idx, save_path=None):
        """ Store the results of the backtest run. """

        # Save results using object dtype to avoid inhomogeneous shape errors
        if save_path:
            np.save(save_path, np.array(results, dtype=object), allow_pickle=True)
            print(f"Results saved to {save_path}")

        self.results = results
        w_total = np.zeros(len(self.tickers))

        for i, t in enumerate(range(start_idx, end_idx, self.rebalance_every)):
            # Compute optimal weights from simulated R
            _, w_opt = results[i]
            w_new = w_opt[0]

            # Update total positions

            if isinstance(self.optimizer, DSPOptimizer) or isinstance(self.optimizer, MeanCVaROptimizer):
                # Method 2: Simple decay - assume optimizer enforces sum(w) = 1
                # No normalization needed if optimization has budget constraint
                if i == 0:
                    w_total = w_new
                else:
                    # m = min(self.decay, 1 - self.decay)
                    # M = max(self.decay, 1 - self.decay)
                    # decay = np.clip(sum(w_total != 0)/sum(w_new != 0),m,M)
                    decay = self.decay
                    w_total = decay * w_total + ( 1 - decay ) * w_new
            else:
                # Method 3: Simple decay and normalize for other optimizers
                w_total = w_new

            # Compute return from observed market returns
            R_forward = self.valid_returns[t+1:t+1+self.rebalance_every]
            if len(R_forward) < self.rebalance_every:
                break
            pnl = R_forward @ w_total
            pnl = np.sum(pnl)
            if i == 0:
                self.total_wealth.append(100 * (1 + pnl))
            else:
                self.total_wealth.append(self.total_wealth[-1] * (1 + pnl))
            self.w_total_history.append(w_total.copy())
            self.pnl_history.append(pnl)
            self.rebalance_dates.append(self.valid_dates[t])

            # Comparison with buy and hold SPY
            pnl_spy = R_forward[:, 0] # Assuming SPY is the first ticker in self.valid_returns
            pnl_spy = np.sum(pnl_spy)

            self.pnl_history_spy.append(pnl_spy)
            self.total_wealth_spy.append(self.total_wealth_spy[-1] * (1 + pnl_spy)) if i > 0 else self.total_wealth_spy.append(100 * (1 + pnl_spy))

    def performance(self, plot=False):
        
        if isinstance(self.optimizer, DSPOptimizer):
            label = "DSP"
        else:
            label = "Mean-CVaR"

        pnl = np.array(self.pnl_history)
        pnl_spy = np.array(self.pnl_history_spy)
    
        mean_return = np.mean(pnl)
        volatility = np.std(pnl)
        mean_return_spy = np.mean(pnl_spy)
        volatility_spy = np.std(pnl_spy)

        annual_factor = 252 // self.rebalance_every  # Assuming daily data, rebalance every k days
        annual_return = (1 + mean_return) ** annual_factor - 1
        annual_vol = volatility * np.sqrt(annual_factor)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else np.nan

        annual_return_spy = (1 + mean_return_spy) ** annual_factor - 1
        annual_vol_spy = volatility_spy * np.sqrt(annual_factor)
        sharpe_ratio_spy = annual_return_spy / annual_vol_spy if annual_vol_spy > 0 else np.nan
        
        #print(f"Annual Return: {annual_return:.2%}")
        #print(f"Annual Volatility: {annual_vol:.2%}")
        # print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        mdd, cvar, sortino = self.compute_risk_metrics(pnl, label=label)

        #print(f"SPY Annual Return: {annual_return_spy:.2%}")
        #print(f"SPY Annual Volatility: {annual_vol_spy:.2%}")
        # print(f"SPY Sharpe Ratio: {sharpe_ratio_spy:.2f}")
        mdd_spy, cvar_spy, sortino_spy = self.compute_risk_metrics(pnl_spy, label="SPY")

        if plot:
            plt.figure(figsize=(10,5))
            plt.plot(self.rebalance_dates, pnl, label="Period Return")
            plt.grid(True)
            plt.title("DSP Strategy Period Return")
            plt.xlabel("Date")
            plt.ylabel("Period Return")
            plt.legend()
            plt.tight_layout()
            plt.show()
            plt.close()

            total_wealth = np.array(self.total_wealth)
            plt.figure(figsize=(10,5))
            plt.plot(self.rebalance_dates, total_wealth, label="Total Wealth", color='orange')
            plt.grid(True)
            plt.title("DSP Strategy Total Wealth")
            plt.xlabel("Date")
            plt.ylabel("Total Wealth")
            plt.legend()
            plt.tight_layout()
            plt.show()
            plt.close()

            # Plot SPY performance
            plt.figure(figsize=(10,5))
            plt.plot(self.rebalance_dates, pnl_spy, label="SPY Period Return", color='green')
            plt.grid(True)
            plt.title("SPY Period Return")
            plt.xlabel("Date")
            plt.ylabel("Period Return")
            plt.legend()
            plt.tight_layout()
            plt.show()
            plt.close()

            total_wealth_spy = np.array(self.total_wealth_spy)
            plt.figure(figsize=(10,5))
            plt.plot(self.rebalance_dates, total_wealth_spy, label="SPY Total Wealth", color='red')
            plt.grid(True)
            plt.title("SPY Total Wealth")
            plt.xlabel("Date")
            plt.ylabel("Total Wealth")
            plt.legend()
            plt.tight_layout()
            plt.show()
            plt.close()

        return [(sharpe_ratio, sharpe_ratio_spy), (mdd, mdd_spy), (cvar, cvar_spy), (sortino, sortino_spy)]

    def get_weights_df(self):
        import pandas as pd
        return pd.DataFrame(self.w_total_history, index=self.rebalance_dates, columns=self.tickers)

    def get_pnl_df(self):
        import pandas as pd
        return pd.DataFrame({"date": self.rebalance_dates, "pnl": self.pnl_history})
    
    def summary_stats(self):
        pnl = np.array(self.pnl_history)
        ann_return = np.mean(pnl) * (252 / self.rebalance_every)
        ann_vol = np.std(pnl) * np.sqrt(252 / self.rebalance_every)
        sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
        return {"Annual Return": ann_return, "Annual Volatility": ann_vol, "Sharpe Ratio": sharpe}
    
    def compute_risk_metrics(self, pnl_array, label=""):

        #print(f"Min {label} PnL:", pnl_array.min())
        rebalance_every = self.rebalance_every
        annual_factor = 252 / rebalance_every

        wealth = np.cumprod(1 + pnl_array)
        peak = np.maximum.accumulate(wealth)
        dd = (wealth - peak) / peak
        mdd = dd.min()

        losses = np.sort(pnl_array[pnl_array < 0])
        n = int(np.ceil(0.05 * len(losses)))
        cvar_val = losses[:n].mean() if n > 0 else 0.0
        cvar_annual = (1 + cvar_val) ** annual_factor - 1

        downside = pnl_array[pnl_array < 0]
        downside_dev = np.sqrt(np.mean((downside) ** 2))
        sortino_period = np.mean(pnl_array) / downside_dev if downside_dev > 0 else np.nan
        sortino_annual = sortino_period * np.sqrt(annual_factor)

        # print(f"{label} Max Drawdown: {mdd:.2%}")
        # print(f"{label} Annualized CVaR (5%): {cvar_annual:.2%}")
        # print(f"{label} Annualized Sortino: {sortino_annual:.3f}")

        return mdd, cvar_val, sortino_annual

class MeanCVaRBacktester(MeanCVaROptimizer):
    def __init__(self,
                tickers=['spy', 'xlb', 'xle', 'xlf', 'xli', 'xlk', 'xlu', 'xlv', 'xly'],
                window=100,
                J=10000,
                df=6,
                lam=0.75,
                target_return=None,
                **kwargs):
        super().__init__(tickers=tickers, window=window, J=J, df=df, lam=lam, target_return=target_return, **kwargs)
        self.rebalance_every = kwargs.get("rebalance_every", 10)
        self.decay = kwargs.get("decay", 0.95)
        self.results = []
        self.total_wealth = []
        self.w_total_history = []
        self.pnl_history = []
        self.rebalance_dates = []
        self.valid_returns = self.data_handler.DataETFsReturns.iloc[self.window:].drop(columns="days").values
        self.valid_dates = self.dates[self.window:]

    def compute_weights_on_date(self, date_idx):
        date_str = str(self.valid_dates[date_idx])[:10]
        weights = self.solve(date_str=date_str)
        # print(f"Computed weights for {date_str}: {weights[0]} given target return {self.target_return}")
        return (date_str, weights)

class DSPBacktester(DSPOptimizer):
    def __init__(self,
                tickers=['spy', 'xlb', 'xle', 'xlf', 'xli', 'xlk', 'xlu', 'xlv', 'xly'],
                window=100,
                J=10000,
                df=6,
                lam=0.75,
                theta=0.5,
                alpha=1.25,
                beta=0.25,
                rebalance_every=10,
                decay=0.95):
        super().__init__(tickers=tickers, window=window, J=J, df=df, lam=lam, 
                         theta=theta, alpha=alpha, beta=beta)
        self.rebalance_every = rebalance_every
        self.decay = decay
        self.w_total_history = []
        self.pnl_history = []
        self.rebalance_dates = []
        self.results = []
        self.total_wealth = []
        self.total_wealth_spy = [] 
        self.pnl_history_spy = []

    def compute_weights_on_date(self, date_idx):
        """
        Compute optimal weights for a given date index.
        Returns: (date_str, weights) tuple
        """
        try:
            date_str = str(self.valid_dates[date_idx])[:10]
        except IndexError:
            raise ValueError(f"Invalid date_idx={date_idx}, valid range is 0 to {len(self.valid_dates)-1}")

        w_opt = self.solve(date_str=date_str)

        return (date_str, w_opt)
    
    # def run_backtest(self, start_idx=100, end_idx=None):
    #     if end_idx is None:
    #         end_idx = len(self.valid_dates) - self.rebalance_every  # leave space for forward return window

    #     w_total = np.zeros(len(self.tickers))

    #     for t in range(start_idx, end_idx, self.rebalance_every):
    #         date_str = str(self.valid_dates[t])[:10]

    #         # Compute optimal weights from simulated R
    #         w_opt = self.solve(date_str=date_str)
    #         w_new = w_opt[0]

    #     # Update total positions
    #     w_total = self.decay * w_total + w_new

    #     # Compute return from observed market returns
    #     R_forward = self.valid_returns[t+1:t+1+self.rebalance_every]
    #     if len(R_forward) < self.rebalance_every:
    #         break
    #     pnl = R_forward @ w_total
    #     cum_pnl = np.sum(pnl)

    #     self.results[date_str] = w_opt
    #     self.w_total_history.append(w_total.copy())
    #     self.pnl_history.append(cum_pnl)
    #     self.rebalance_dates.append(self.valid_dates[t])

    # def store_backtest_results(self, results, start_idx, end_idx, save_path=None):
    #     """ Store the results of the backtest run. """

    #     # Save results using object dtype to avoid inhomogeneous shape errors
    #     if save_path:
    #         np.save(save_path, np.array(results, dtype=object), allow_pickle=True)
    #         print(f"Results saved to {save_path}")

    #     self.results = results
    #     w_total = np.zeros(len(self.tickers))

    #     for i, t in enumerate(range(start_idx, end_idx, self.rebalance_every)):
    #         # Compute optimal weights from simulated R
    #         _, w_opt = results[i]
    #         w_new = w_opt[0]

    #         # Update total positions
    #         # Method 1: Decay and accumulate weights (original in the paper)
    #         #cost = (w_new - (1 - self.decay ) * w_total ) @ self.valid_returns[t]
    #         # w_total = self.decay * w_total + w_new

    #         # Method 2: Normalize weights and accumulate with weight-dependent decay
    #         w_new = w_new / np.sum(np.abs(w_new))
    #         if i == 0:
    #             #print(w_new)
    #             w_total = w_new
    #         else:
    #             m = min(self.decay, 1 - self.decay)
    #             M = max(self.decay, 1 - self.decay)
    #             decay = np.clip(sum(w_total != 0)/sum(w_new != 0),m,M)  # Adjust decay based on previous weights
    #             w_total = decay * w_total + ( 1 - decay ) * w_new

    #         # Method 3: Simple decay and normalize
    #         # w_total = w_new / np.sum(np.abs(w_new))

    #         # Compute return from observed market returns
    #         R_forward = self.valid_returns[t+1:t+1+self.rebalance_every]
    #         if len(R_forward) < self.rebalance_every:
    #             break
    #         pnl = R_forward @ w_total
    #         pnl = np.sum(pnl)
    #         if i == 0:
    #             self.total_wealth.append(100 * (1 + pnl))
    #         else:
    #             self.total_wealth.append(self.total_wealth[-1] * (1 + pnl))
    #         self.w_total_history.append(w_total.copy())
    #         self.pnl_history.append(pnl)
    #         self.rebalance_dates.append(self.valid_dates[t])

    #         # Comparison with buy and hold SPY
    #         pnl_spy = R_forward[:, 0] # Assuming SPY is the first ticker in self.valid_returns
    #         pnl_spy = np.sum(pnl_spy)

    #         self.pnl_history_spy.append(pnl_spy)
    #         self.total_wealth_spy.append(self.total_wealth_spy[-1] * (1 + pnl_spy)) if i > 0 else self.total_wealth_spy.append(100 * (1 + pnl_spy))

    # def performance(self, plot=False):
    #     pnl = np.array(self.pnl_history)
    #     pnl_spy = np.array(self.pnl_history_spy)
    
    #     mean_return = np.mean(pnl)
    #     volatility = np.std(pnl)
    #     mean_return_spy = np.mean(pnl_spy)
    #     volatility_spy = np.std(pnl_spy)

    #     annual_factor = 252 // self.rebalance_every  # Assuming daily data, rebalance every k days
    #     annual_return = (1 + mean_return) ** annual_factor - 1
    #     annual_vol = volatility * np.sqrt(annual_factor)
    #     sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else np.nan

    #     annual_return_spy = (1 + mean_return_spy) ** annual_factor - 1
    #     annual_vol_spy = volatility_spy * np.sqrt(annual_factor)
    #     sharpe_ratio_spy = annual_return_spy / annual_vol_spy if annual_vol_spy > 0 else np.nan
        
    #     #print(f"Annual Return: {annual_return:.2%}")
    #     #print(f"Annual Volatility: {annual_vol:.2%}")
    #     # print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    #     mdd, cvar, sortino = self.compute_risk_metrics(pnl, label="DSP")

    #     #print(f"SPY Annual Return: {annual_return_spy:.2%}")
    #     #print(f"SPY Annual Volatility: {annual_vol_spy:.2%}")
    #     # print(f"SPY Sharpe Ratio: {sharpe_ratio_spy:.2f}")
    #     mdd_spy, cvar_spy, sortino_spy = self.compute_risk_metrics(pnl_spy, label="SPY")

    #     if plot:
    #         plt.figure(figsize=(10,5))
    #         plt.plot(self.rebalance_dates, pnl, label="Period Return")
    #         plt.grid(True)
    #         plt.title("DSP Strategy Period Return")
    #         plt.xlabel("Date")
    #         plt.ylabel("Period Return")
    #         plt.legend()
    #         plt.tight_layout()
    #         plt.show()
    #         plt.close()

    #         total_wealth = np.array(self.total_wealth)
    #         plt.figure(figsize=(10,5))
    #         plt.plot(self.rebalance_dates, total_wealth, label="Total Wealth", color='orange')
    #         plt.grid(True)
    #         plt.title("DSP Strategy Total Wealth")
    #         plt.xlabel("Date")
    #         plt.ylabel("Total Wealth")
    #         plt.legend()
    #         plt.tight_layout()
    #         plt.show()
    #         plt.close()

    #         # Plot SPY performance
    #         plt.figure(figsize=(10,5))
    #         plt.plot(self.rebalance_dates, pnl_spy, label="SPY Period Return", color='green')
    #         plt.grid(True)
    #         plt.title("SPY Period Return")
    #         plt.xlabel("Date")
    #         plt.ylabel("Period Return")
    #         plt.legend()
    #         plt.tight_layout()
    #         plt.show()
    #         plt.close()

    #         total_wealth_spy = np.array(self.total_wealth_spy)
    #         plt.figure(figsize=(10,5))
    #         plt.plot(self.rebalance_dates, total_wealth_spy, label="SPY Total Wealth", color='red')
    #         plt.grid(True)
    #         plt.title("SPY Total Wealth")
    #         plt.xlabel("Date")
    #         plt.ylabel("Total Wealth")
    #         plt.legend()
    #         plt.tight_layout()
    #         plt.show()
    #         plt.close()

    #     return [(sharpe_ratio, sharpe_ratio_spy), (mdd, mdd_spy), (cvar, cvar_spy), (sortino, sortino_spy)]

    # def get_weights_df(self):
    #     import pandas as pd
    #     return pd.DataFrame(self.w_total_history, index=self.rebalance_dates, columns=self.tickers)

    # def get_pnl_df(self):
    #     import pandas as pd
    #     return pd.DataFrame({"date": self.rebalance_dates, "pnl": self.pnl_history})
    
    # def summary_stats(self):
    #     pnl = np.array(self.pnl_history)
    #     ann_return = np.mean(pnl) * (252 / self.rebalance_every)
    #     ann_vol = np.std(pnl) * np.sqrt(252 / self.rebalance_every)
    #     sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    #     return {"Annual Return": ann_return, "Annual Volatility": ann_vol, "Sharpe Ratio": sharpe}
    
    # def compute_risk_metrics(self, pnl_array, label=""):

    #     print(f"Min {label} PnL:", pnl_array.min())
    #     rebalance_every = self.rebalance_every
    #     annual_factor = 252 / rebalance_every

    #     wealth = np.cumprod(1 + pnl_array)
    #     peak = np.maximum.accumulate(wealth)
    #     dd = (wealth - peak) / peak
    #     mdd = dd.min()

    #     losses = np.sort(pnl_array[pnl_array < 0])
    #     n = int(np.ceil(0.05 * len(losses)))
    #     cvar_val = losses[:n].mean() if n > 0 else 0.0
    #     cvar_annual = (1 + cvar_val) ** annual_factor - 1

    #     downside = pnl_array[pnl_array < 0]
    #     downside_dev = np.sqrt(np.mean((downside) ** 2))
    #     sortino_period = np.mean(pnl_array) / downside_dev if downside_dev > 0 else np.nan
    #     sortino_annual = sortino_period * np.sqrt(annual_factor)

    #     # print(f"{label} Max Drawdown: {mdd:.2%}")
    #     # print(f"{label} Annualized CVaR (5%): {cvar_annual:.2%}")
    #     # print(f"{label} Annualized Sortino: {sortino_annual:.3f}")

    #     return mdd, cvar_val, sortino_annual


