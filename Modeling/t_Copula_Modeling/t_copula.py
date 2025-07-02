# t_copula.py

import numpy as np
from scipy.stats import t, multivariate_t

class TCopula:
    def __init__(self, nu, corr):
        self.nu = nu  # degrees of freedom
        self.corr = corr  # correlation matrix (d x d)
        self.dim = corr.shape[0]

    def pseudo_obs(self, u):
        """Inverse t-CDF transform to map PITs to t-distributed values"""
        return t.ppf(u, df=self.nu)

    def log_pdf(self, z):
        """Log-density of multivariate t with zero mean"""
        return multivariate_t.logpdf(z, df=self.nu, shape=self.corr)

    def sample(self, n_samples):
        """Generate samples from the t-copula (returns u-values)"""
        z = multivariate_t.rvs(df=self.nu, shape=self.corr, size=n_samples)
        u = t.cdf(z, df=self.nu)
        return u
