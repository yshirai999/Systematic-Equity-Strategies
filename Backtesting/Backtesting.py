import sys, os
sys.path.append(os.path.abspath('.'))

import numpy as np
import matplotlib.pyplot as plt
from Optimizers.DSP.dsp_solver import DSPOptimizer

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
        self.dsp_sols = []
        self.results = []
        self.total_wealth = []

    def run_backtest(self, start_idx=100, end_idx=None):
        if end_idx is None:
            end_idx = len(self.valid_dates) - self.rebalance_every  # leave space for forward return window

        w_total = np.zeros(len(self.tickers))

        for t in range(start_idx, end_idx, self.rebalance_every):
            date_str = str(self.valid_dates[t])[:10]

            # Compute optimal weights from simulated R
            w_opt = self.solve(date_str=date_str)
            w_new = w_opt[0]

            # Update total positions
            w_total = self.decay * w_total + w_new

            # Compute return from observed market returns
            R_forward = self.valid_returns[t+1:t+1+self.rebalance_every]
            if len(R_forward) < self.rebalance_every:
                break
            pnl = R_forward @ w_total
            cum_pnl = np.sum(pnl)

            self.results[date_str] = w_opt
            self.w_total_history.append(w_total.copy())
            self.pnl_history.append(cum_pnl)
            self.rebalance_dates.append(self.valid_dates[t])

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
            # w_total = self.decay * w_total + w_new
            # w_total = w_new  # <-- replaces, instead of accumulating
            w_total = w_new / np.sum(np.abs(w_new))

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

    def performance(self):
        pnl = np.array(self.pnl_history)

        mean_return = np.mean(pnl)
        volatility = np.std(pnl)

        annual_factor = 252 // self.rebalance_every  # Assuming daily data, rebalance every k days
        annual_return = (1 + mean_return) ** annual_factor - 1
        annual_vol = volatility * np.sqrt(annual_factor)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else np.nan

        print(f"Annual Return: {annual_return:.2%}")
        print(f"Annual Volatility: {annual_vol:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

        plt.figure(figsize=(10,5))
        plt.plot(self.rebalance_dates, pnl, label="Period Return")
        plt.grid(True)
        plt.title("DSP Strategy Period Return")
        plt.xlabel("Date")
        plt.ylabel("Period Return")
        plt.legend()
        plt.tight_layout()
        plt.show()

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


