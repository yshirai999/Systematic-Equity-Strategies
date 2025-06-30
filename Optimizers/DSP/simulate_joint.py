import os
import numpy as np
from scipy.stats import t, gamma
from Data.DataProcessing import data  # lowercase 'data' class

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT_DIR, "BG_Modeling", "estimates", "FINAL")
CORR_DIR = os.path.join(ROOT_DIR, "t_Copula_Modeling", "results", "correlation_matrices")

class JointReturnSimulator:
    def __init__(self, date_str, tickers=['spy', 'xlb', 'xle', 'xlf', 'xli', 'xlk', 'xlu', 'xlv', 'xly'], window=100, J=10000, df=6):
        self.date_str = date_str
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

        # Convert date_str and check if it's in the data
        target_date = np.datetime64(self.date_str)
        if target_date not in self.dates:
            raise ValueError(f"Date '{self.date_str}' not found in available data range "
                            f"({self.dates[0]} to {self.dates[-1]})")

        self.idx = int(np.where(self.dates == target_date)[0][0])
        self.window = window  # Or whatever rolling length you used
        self.idx = int(np.where(self.dates == target_date)[0][0]) - self.window

        if self.idx < 0:
            raise ValueError(f"Date '{self.date_str}' is too early to have model parameters "
                            f"(needs at least {self.window} prior days)")


        self.load_parameters()


    def load_parameters(self):
        """Load per-ticker marginal BG parameters and daily correlation matrix."""
        for ticker in self.tickers:
            file = os.path.join(self.path_params, f"theta_{ticker.upper()}_FINAL.npy")
            theta_matrix = np.load(file)  # shape: (4330, 4)
            bp, cp, bn, cn = theta_matrix[self.idx]
            self.bp.append(bp)
            self.cp.append(cp)
            self.bn.append(bn)
            self.cn.append(cn)

        # Load full time series of correlation matrices and extract correct day
        corr_cube = np.load(os.path.join(self.path_corr, f"corr_matrix_nu{self.df}.npy"))  # shape: (4330, M, M)
        self.C = corr_cube[self.idx]

    def simulate_t_Copula(self):
        """
        Generate J samples from the t-Copula + BG marginal distribution.

        Returns:
            np.ndarray: shape (J, M) simulated return matrix
        """
        # Step 1: simulate t-copula samples
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

