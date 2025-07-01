import os
import numpy as np
from scipy.stats import t, gamma
from Data.DataProcessing import data  # lowercase 'data' class

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT_DIR, "BG_Modeling", "estimates", "FINAL")
CORR_DIR = os.path.join(ROOT_DIR, "t_Copula_Modeling", "results", "correlation_matrices")

class JointReturnSimulator:
    def __init__(self,
                tickers=['spy', 'xlb', 'xle', 'xlf', 'xli', 'xlk', 'xlu', 'xlv', 'xly'],
                window=100,
                J=10000,
                df=6):
        self.tickers = tickers
        self.J = J
        self.df = df
        self.M = len(tickers)

        self.path_params = DATA_DIR
        self.path_corr = CORR_DIR

        self.bp, self.cp, self.bn, self.cn = [], [], [], []
        self.C = None

        # Load data object (lowercase class: data) and match index
        self.data_handler = data(tickers=self.tickers)
        self.dates = self.data_handler.DataETFsReturns['days'].values
        self.window = window
        self.valid_returns = self.data_handler.DataETFsReturns.iloc[self.window:].drop(columns="days").values
        self.valid_dates = self.dates[self.window:]  # Valid dates after the window

    def simulate_t_Copula(self, date_str=None, window=None):
        """
        Generate J samples from the t-Copula + BG marginal distribution.

        Returns:
            np.ndarray: shape (J, M) simulated return matrix
        """
        if not window:
            window = self.window

        # Step 0: Load parameters
        target_date = np.datetime64(date_str)

        if target_date not in self.valid_dates:
            raise ValueError(f"Date '{target_date}' not found in available data range "
                            f"({self.valid_dates[0]} to {self.valid_dates[-1]})")

        idx = int(np.where(self.valid_dates == target_date)[0][0])

        if idx < 0:
            raise ValueError(f"Date '{date_str}' is too early to have model parameters "
                            f"(needs at least {window} prior days)")

        if idx is None:
            raise ValueError("Index not set. Call 'simulate_t_Copula' with a valid date_str first.")
        for ticker in self.tickers:
            file = os.path.join(self.path_params, f"theta_{ticker.upper()}_FINAL.npy")
            theta_matrix = np.load(file)  # shape: (4330, 4)
            bp, cp, bn, cn = theta_matrix[idx]
            self.bp.append(bp)
            self.cp.append(cp)
            self.bn.append(bn)
            self.cn.append(cn)

        # Load full time series of correlation matrices and extract correct day
        corr_cube = np.load(os.path.join(self.path_corr, f"corr_matrix_w{window}.npy"))  # shape: (4330, M, M)
        self.C = corr_cube[idx]

        print(self.C.shape)

        # Step 1: simulate t-copula samples
        np.random.seed(42)
        z = np.random.multivariate_normal(np.zeros(self.M), self.C, size=self.J)
        chi2 = np.random.chisquare(self.df, size=self.J)
        t_samples = z / np.sqrt(chi2[:, None] / self.df)

        # Step 2: map to uniform
        u = t.cdf(t_samples, df=self.df)

        # Step 3: inverse transform to BG marginals
        returns = np.zeros_like(u)
        for m in range(self.M):
            up = u[:, m]
            split = self.cp[m] / (self.cp[m] + self.cn[m])
            mask_pos = up > (1 - split)

            pos = np.zeros_like(up)
            neg = np.zeros_like(up)

            # Positive part
            pos[mask_pos] = gamma.ppf((up[mask_pos] - (1 - split)) / split, a=self.cp[m], scale=self.bp[m])
            # Negative part
            neg[~mask_pos] = gamma.ppf(up[~mask_pos] / (1 - split), a=self.cn[m], scale=self.bn[m])

            returns[:, m] = pos - neg

        return returns  # shape (J, M)

