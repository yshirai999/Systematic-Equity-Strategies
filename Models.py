from joblib import Parallel, delayed
import concurrent.futures

import numpy as np
from scipy.fft import fft, fftfreq, fftshift, ifftshift
from scipy.optimize import minimize
import scipy.interpolate as interp

import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))  # Optional: Shows GPU name


# -----------------------------------------------------------------------------
# Module-level helpers (must be top-level so they can be pickled):
# -----------------------------------------------------------------------------

def _fit_window_bg(args):
    """
    Args:
      args = (t_idx, m_sub, bg_instance)

    Returns:
      (t_idx, theta_hat) where theta_hat = bg_instance.fit_bilateral_gamma(m_sub)
    """
    t_idx, m_sub, bg_instance = args
    theta_hat = bg_instance.fit_bilateral_gamma(m_sub)
    return t_idx, theta_hat


def _fit_column_bg(args):
    """
    Args:
      args = (j_idx, X_col, bg_instance, window)

    Returns:
      (j_idx, params_j) where params_j = bg_instance.fit_series(X_col, window)
    """
    j_idx, X_col, bg_instance, window = args
    params_j = bg_instance.fit_series(X_col, window=window)
    return j_idx, params_j


# -----------------------------------------------------------------------------
# BG class:
# -----------------------------------------------------------------------------

class BG:
    """
    Bilateral-Gamma tail-matching and rolling-window fitting utilities.

    - __init__(N, Xmax): build FFT grid once.
    - pdf(theta): return PDF array f(x) for theta=[bp,cp,bn,cn].
    - theoretical_tails(theta, s_i): compute tail-probs at s_i.
    - fit_bilateral_gamma(m): fit one 1D array by tail-matching.
    - fit_series(series, window): rolling-window fits on one series (serial).
    - fit_multiple(X, window, n_workers): fits each column of X in parallel.
    """

    def __init__(self, N=4096, B=0.1, device=None):
        """
        Initialize the BG class for bilateral gamma PDF estimation using FFT.
        Parameters
        ----------
        N      : int     - Number of FFT points
        B      : float   - Half-width of support; domain is [-pi*N/B, pi*N/B), frequency spacing is B/N.
        device : torch.device or str (optional) - 'cpu' or 'cuda'
        """
        self.N = N
        self.B = B
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.eta = B / N
        self.lambda_ = 2 * np.pi / B
        self.bb = self.lambda_ * N / 2

        # Real-space grid
        self.x = torch.tensor(-self.bb + self.lambda_ * np.arange(N), dtype=torch.float32, device=self.device)

        # Frequency-space grid
        self.u = torch.tensor(np.arange(N) * self.eta, dtype=torch.float32, device=self.device)

        # Trapezoidal weights
        self.w = torch.ones(N, dtype=torch.float32, device=self.device)
        self.w[0] = 0.5

        # Precompute the target quantile levels
        self.Pi_target = np.linspace(0.01, 0.99, 99)
        self.Pi_target_torch = torch.tensor(self.Pi_target, dtype=torch.float32, device=self.device)

    def phi(self, theta):
        """
        Compute the characteristic function for bilateral gamma distribution with given theta.
        theta : Tensor of shape (..., 4) - (bp, cp, bn, cn)
        Returns: Tensor of shape (..., N)
        """
        bp, cp, bn, cn = theta[..., 0], theta[..., 1], theta[..., 2], theta[..., 3]
        u = self.u[None, :] if theta.ndim == 2 else self.u

        phi_p = (1 - 1j * u * bp[..., None]) ** (-cp[..., None])
        phi_n = (1 + 1j * u * bn[..., None]) ** (-cn[..., None])
        return phi_p * phi_n

    def pdf(self, theta):
        """
        Compute the PDF from the characteristic function using FFT.
        theta : Tensor of shape (4,) or (batch_size, 4)
        Returns: PDF values (same shape as self.x or (batch_size, N))
        """
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)

        phi_vals = self.phi(theta) * self.w
        fft_input = torch.exp(1j * self.u * self.bb) * phi_vals
        pdf_vals = torch.real(torch.fft.fft(fft_input, dim=-1)) / np.pi
        return pdf_vals.squeeze()  # remove batch dim if batch_size = 1
    
    def theoretical_quantiles(self, theta):
        """
        Torch differentiable theoretical quantiles:
        - Accepts theta (T, 4) directly
        - Computes PDF → CDF → interpolated quantiles at self.s_batch
        - Returns: Q_model (T, K)
        """
        if not hasattr(self, "s_batch"):
            raise ValueError("self.s_batch not found. Load it using torch.load(...)")

        # Step 1: Compute PDF from theta (requires grad)
        pdf = self.pdf(theta)  # (T, N)

        # Step 2: Compute normalized CDF
        dx = self.lambda_
        cdf = torch.cumsum(pdf, dim=1) * dx  # (T, N)
        cdf = cdf / cdf[:, -1:].clamp(min=1e-8)

        # Step 3: Interpolate at self.s_batch
        T, K = self.s_batch.shape # T = # of days, K = # of quantiles
        N = self.x.shape[0] # N = space grids of the PDF

        x_grid = self.x.view(1, N).expand(T, N)
        s_batch_clamped = self.s_batch.clamp(min=self.x[0].item(), max=self.x[-1].item())

        idx = torch.sum(x_grid <= s_batch_clamped.unsqueeze(-1), dim=1) - 1  # For each t,k, idx satisfies x_{idx} <= s_{t,k} <= x_{idx+1}
        idx = idx.clamp(min=0, max=N - 2)

        x_lower = torch.gather(x_grid, 1, idx)
        x_upper = torch.gather(x_grid, 1, idx + 1)
        cdf_lower = torch.gather(cdf, 1, idx)
        cdf_upper = torch.gather(cdf, 1, idx + 1)

        weight = (s_batch_clamped - x_lower) / (x_upper - x_lower + 1e-8)
        Q_model = cdf_lower + weight * (cdf_upper - cdf_lower) # Linear interpolation

        return Q_model  # (T, K), fully differentiable w.r.t. theta


    def empirical_quantiles(self, batch_returns, save_path='estimates/empirical_quantiles.pt'):
        """
        Compute empirical quantiles for a batch of return series.

        Parameters
        ----------
        batch_returns : np.ndarray
            Shape (n_days, n_obs) where each row is a return series for one day.
        save_path : str, optional
            If provided, saves s_batch in the specified path.

        Returns
        -------
        Pi_target : (99,) array of quantile levels
        s_batch : (n_days, 99) array of empirical quantiles
        """
        n_days, n_obs = batch_returns.shape
        s_sorted = np.sort(batch_returns, axis=1)
        pi_emp = np.arange(1, n_obs + 1) / n_obs
        Pi_target = self.Pi_target

        # Interpolate quantiles for each day
        s_batch = np.array([
            np.interp(Pi_target, pi_emp, s_sorted[i], left=s_sorted[i, 0], right=s_sorted[i, -1])
            for i in range(n_days)
        ])

        # Optional saving
        if save_path is not None:
            # Optional: save torch.cuda version for GPU pipeline
            s_batch_torch = torch.tensor(s_batch, dtype=torch.float32).to('cuda')
            torch.save(s_batch_torch, save_path)

        return Pi_target, s_batch

    def loss (self, Q_model, Q_emp, x_grid, quantile_levels):
        
        # Q_model = self.compute_theoretical_quantiles(cdf, x_grid, quantile_levels)
        
        # Anderson-Darling weights
        w = 1.0 / (quantile_levels * (1 - quantile_levels) + 1e-6)
        loss = ((Q_emp - Q_model) ** 2 * w.view(1, -1)).mean()
        
        return loss

    ####################################################################
    ## Fitting methods for rolling windows and multiple series
    ####################################################################

    def fit_series(self, series, window=100):
        """
        Rolling-window fit (serial) on a single 1D series of length T:
          - For t < window: row stays [nan,nan,nan,nan].
          - For t ≥ window: fit on series[t-window : t].

        Returns
        -------
        params : array of shape (T,4).
        """
        T = len(series)
        params = np.full((T, 4), np.nan)

        for t in range(window, T):
            m_window = series[t - window : t]
            theta_hat = self.fit_bilateral_gamma(m_window)
            params[t, :] = theta_hat

        return params

    def fit_multiple(self, X, window=100, n_workers=1):
        """
        Fit each column of X (shape T x num_assets) via rolling-window. If n_workers>1,
        parallelize across *assets* (columns). 

        Returns
        -------
        all_params : array of shape (T, num_assets, 4).
        """
        T, num_assets = X.shape
        all_params = np.full((T, num_assets, 4), np.nan)

        # Prepare a list of (j_idx, X_col, self, window) for each asset
        tasks = [(j, X[:, j], self, window) for j in range(num_assets)]

        if n_workers is None or n_workers <= 1:
            # Serial over assets
            for j_idx, X_col, bg_inst, win in tasks:
                _, params_j = _fit_column_bg((j_idx, X_col, bg_inst, win))
                all_params[:, j_idx, :] = params_j
        else:
            # Parallel across assets
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as exe:
                for j_idx, params_j in exe.map(_fit_column_bg, tasks):
                    all_params[:, j_idx, :] = params_j

        return all_params  # shape (T, num_assets, 4)

    def fit_series_parallel(series, window=100, bg_model=None, n_jobs=-1):
        """
        Parallelized version of fit_series using joblib. Applies BG fitting in parallel across time windows.

        Parameters
        ----------
        series : np.ndarray
            1D array of time series data of length T.
        window : int
            Rolling window size.
        bg_model : object
            An instance of the BG class with a `fit_bilateral_gamma` method.
        n_jobs : int
            Number of jobs for parallel processing. -1 means using all available cores.

        Returns
        -------
        params : np.ndarray
            Array of shape (T, 4) with fitted BG parameters.
        """
        T = len(series)
        params = np.full((T, 4), np.nan)

        def fit_at_time(t):
            window_data = series[t - window:t]
            return bg_model.fit_bilateral_gamma(window_data)

        # Apply in parallel only for valid indices
        results = Parallel(n_jobs=n_jobs)(
            delayed(fit_at_time)(t) for t in range(window, T)
        )

        # Assign results
        for i, t in enumerate(range(window, T)):
            params[t, :] = results[i]

        return params


# -----------------------------------------
# Example usage:
# -----------------------------------------
if __name__ == "__main__":
    # Suppose X is (4430, 11) daily returns for 11 tickers:
    np.random.seed(0)
    X = np.random.randn(4430, 11) * 0.015  # ~±1.5% daily returns

    # 1) Build one BG instance with FFT grid on [−10%, +10%]:
    bg = BG(N=4096, Xmax=0.1)

    # 2) Fit SPY (column 0) *serially* (no per-window parallel):
    spy_params = bg.fit_series(X[:, 0], window=100)
    print("SPY params shape:", spy_params.shape)
    print("Example SPY at t=150:", spy_params[150, :])

    # 3) Fit all 11 assets *in parallel across assets* (1 process per asset):
    all_params = bg.fit_multiple(X, window=100, n_workers=11)
    print("All params shape:", all_params.shape)
    print("Example (t=150, asset=3):", all_params[150, 3, :])




    # def __init__(self, N=2**14, B=300000):
    #     """
    #     Initialize FFT grids for the BG density based on MATLAB-style FFT inversion.
    #     Parameters
    #     ----------
    #     N : int   - Number of FFT points (power of 2)
    #     B : float - Total domain width (used to compute spacing eta)
    #     """
        
    #     self.N = N
    #     self.B = B

    #     self.eta = B / N
    #     self.lambda_ = 2 * np.pi / B
    #     self.bb = self.lambda_ * N / 2

    #     self.u = np.arange(N) * self.eta
    #     self.w = np.ones(N)
    #     self.w[0] = 0.5  # for trapezoidal rule
    #     self.x = -self.bb + self.lambda_ * np.arange(N)  # real-space grid

    # def pdf(self, theta):
    #     """
    #     Compute BG PDF using FFT of the characteristic function as in the user's MATLAB code.
    #     Parameters
    #     ----------
    #     theta : list or array-like [bp, cp, bn, cn]
    #     Returns
    #     -------
    #     pdf_vals : array - PDF evaluated over self.x
    #     """
    #     bp, cp, bn, cn = theta

    #     phi = ((1 - 1j * self.u * bp) ** -cp) * ((1 + 1j * self.u * bn) ** -cn)
    #     phi *= self.w

    #     fft_input = np.exp(1j * self.u * self.bb) * phi
    #     f = np.fft.fft(fft_input) / np.pi
    #     pdf_vals = np.real(f)

    #     return pdf_vals

    # @staticmethod
    # def empirical_quantiles(m):
    #     """
    #     Given a 1D array m of returns, return:
    #       Pi_target : 99-vector [0.01..0.99]
    #       s_i       : 99-vector of empirical quantiles so CDF_emp(s_i) ≈ Pi_target[i]
    #     """
    #     n = len(m)
    #     s_sorted = np.sort(m)
    #     pi_emp = np.arange(1, n + 1) / n
    #     Pi_target = np.linspace(0.01, 0.99, 99)
    #     s_i = np.interp(
    #         Pi_target, pi_emp, s_sorted,
    #         left=s_sorted[0], right=s_sorted[-1]
    #     )
    #     return Pi_target, s_i

    # def loss_fn(self, theta, s_i, Pi_target):
    #     """
    #     Weighted squared error between Pi_target and theoretical tails at s_i.
    #     If any component of θ ≤ 0, return a large penalty.
    #     """
    #     bp, cp, bn, cn = theta
    #     if (bp <= 0) or (cp <= 0) or (bn <= 0) or (cn <= 0):
    #         return 1e6 + np.sum(np.abs(theta))

    #     Pihat = self.theoretical_tails(theta, s_i)
    #     eps = 1e-12
    #     Pi_clipped = np.clip(Pi_target, eps, 1.0 - eps)
    #     W = Pi_clipped * (1.0 - Pi_clipped)
    #     sq = (Pi_target - Pihat) ** 2 / W
    #     return np.sum(sq)

    # def theoretical_tails(self, theta, s_i):
    #     """
    #     Given theta=[bp,cp,bn,cn] and quantile points s_i (length 99),
    #     compute Pihat[i] = F_theta(s_i) if s_i ≤ 0, else 1 - F_theta(s_i).

    #     Uses the PDF from self.pdf(theta) and a trapezoidal CDF.
    #     """
    #     pdf_vals = self.pdf(theta)
    #     xgrid = self.x
    #     dx = self.lambda_

    #     # Build CDF midpoints and cumulative sums
    #     x_mid = (xgrid[:-1] + xgrid[1:]) / 2           # (N-1,)
    #     cdf_vals = np.cumsum((pdf_vals[:-1] + pdf_vals[1:]) / 2 * dx)  # (N-1,)

    #     # Interpolator from x_mid → cdf_vals
    #     cdf_interp = interp.interp1d(
    #         x_mid, cdf_vals, kind='linear',
    #         bounds_error=False, fill_value=(0.0, 1.0)
    #     )

    #     Pihat = np.empty_like(s_i)
    #     for idx, sv in enumerate(s_i):
    #         if sv <= x_mid[0]:
    #             Fsv = 0.0
    #         elif sv >= x_mid[-1]:
    #             Fsv = 1.0
    #         else:
    #             Fsv = float(cdf_interp(sv))
    #         Pihat[idx] = Fsv if (sv <= 0.0) else (1.0 - Fsv)

    #     return Pihat


    # def fit_bilateral_gamma(self, m, initial_theta=None):
    #     """
    #     Fit theta=[bp,cp,bn,cn] to a 1D array of returns m by matching tails.

    #     Returns
    #     -------
    #     theta_hat : fitted 4-vector
    #     """
    #     Pi_target, s_i = BG.empirical_quantiles(m)
    #     if initial_theta is None:
    #         initial_theta = np.array([0.02, 2.0, 0.02, 2.0])

    #     bounds = [(1e-6, None), (1e-6, None), (1e-6, None), (1e-6, None)]
    #     result = minimize(
    #         fun=self.loss_fn,
    #         x0=initial_theta,
    #         args=(s_i, Pi_target),
    #         method='L-BFGS-B',
    #         bounds=bounds,
    #         options={'maxiter': 200, 'ftol': 1e-9}
    #     )
    #     if not result.success:
    #         print("WARNING: fit did not converge:", result.message)
    #     return result.x  # [bp, cp, bn, cn]