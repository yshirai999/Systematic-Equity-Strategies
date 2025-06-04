import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata
from scipy.special import kv, gamma

def bilateral_gamma_pdf(x, bp, cp, bn, cn):
    """
    PDF of the bilateral Gamma distribution
    """
    z = np.abs(x)
    scale = (G * M) ** C / (2 * gamma(C) * np.sqrt(2 * np.pi))
    base = ((G + M) / 2) ** (C - 0.5)
    exp_term = np.exp((G - M) / 2 * x)
    bessel_term = np.abs(x) ** (C - 0.5) * kv(C - 0.5, (G + M) / 2 * np.abs(x))
    return scale * base * exp_term * bessel_term

def bilateral_gamma_cdf_grid(x, C, G, M):
    """
    Numerically integrate PDF to approximate CDF on grid of x
    """
    dx = x[1] - x[0]
    pdf_vals = bilateral_gamma_pdf(x, C, G, M)
    return np.cumsum(pdf_vals) * dx

def tail_error_loss(params, x, probs, y_vals):
    """
    Loss function for minimizing squared relative error of tail probabilities
    """
    C, G, M = params
    x_grid = np.linspace(min(y_vals) - 0.1, max(y_vals) + 0.1, 1000)
    cdf_vals = bilateral_gamma_cdf_grid(x_grid, C, G, M)
    cdf_interp = np.interp(y_vals, x_grid, cdf_vals)
    # Compute predicted π_i
    pi_hat = np.where(y_vals < 0, cdf_interp, 1 - cdf_interp)
    return np.sum(((probs - pi_hat) / probs) ** 2)

def estimate_bilateral_gamma_params(x, quantiles=np.linspace(0.01, 0.99, 20)):
    """
    Estimate (C, G, M) parameters for one asset via tail probability matching
    """
    x_sorted = np.sort(x)
    y_vals = np.quantile(x_sorted, quantiles)
    probs = np.where(y_vals < 0, quantiles, 1 - quantiles)

    result = minimize(
        tail_error_loss,
        x0=[1.0, 1.0, 1.0],  # Initial guess
        args=(x, probs, y_vals),
        method='L-BFGS-B',
        bounds=[(1e-3, 10), (1e-3, 10), (1e-3, 10)]
    )
    return result.x  # C, G, M
