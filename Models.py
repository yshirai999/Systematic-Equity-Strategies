import numpy as np
from scipy.fft import fft, fftfreq, fftshift, ifftshift
from scipy.optimize import minimize
import scipy.interpolate as interp
import concurrent.futures


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
    - pdf(theta): return PDF array f(x) for θ=[bp,cp,bn,cn].
    - theoretical_tails(theta, s_i): compute tail-probs at s_i.
    - fit_bilateral_gamma(m): fit one 1D array by tail-matching.
    - fit_series(series, window): rolling-window fits on one series (serial).
    - fit_multiple(X, window, n_workers): fits each column of X in parallel.
    """

    def __init__(self, N=4096, Xmax=2.0):
        """
        Build and cache the FFT grid (x, t, Hann window) once.

        Parameters
        ----------
        N    : int     # FFT size (power-of-2)
        Xmax : float   # half-width of x range; PDF built on [−Xmax, +Xmax)
        """
        self.N = N
        self.Xmax = Xmax

        # 1) grid spacing in x
        self.dx = (2.0 * Xmax) / N

        # 2) frequency grid (cycles per unit x) → angular frequencies t
        freq = fftfreq(N, d=self.dx)
        t = 2.0 * np.pi * freq  # in [−π/dx, +π/dx)

        # 3) Hann window so φ(±π/dx)=0
        T = np.pi / self.dx
        self.window = 0.5 * (1 + np.cos((np.pi / T) * t))

        # 4) store t and x grids
        self.t = t
        self.x = np.linspace(-Xmax, Xmax - self.dx, N)

        # 5) scaling factor Δt/(2π)
        self.dt_over_2pi = (2.0 * np.pi / (N * self.dx)) / (2.0 * np.pi)

    def pdf(self, theta):
        """
        Build the Bilateral-Gamma PDF on self.x for θ = [bp,cp,bn,cn].

        Returns
        -------
        pdf_vals : 1D array of length N, normalized so ∫pdf(x)dx = 1 on [−Xmax, +Xmax).
        """
        bp, cp, bn, cn = theta

        # 1) characteristic function φ(t) = (1 - i t bp)^(-cp)*(1 + i t bn)^(-cn)
        phi_vals = (1.0 - 1j * self.t * bp) ** (-cp) * (1.0 + 1j * self.t * bn) ** (-cn)

        # 2) apply Hann window
        phi_vals *= self.window

        # 3) shift so x=−Xmax ↔ index 0
        phi_shifted = phi_vals * np.exp(1j * self.t * self.Xmax)

        # 4) invert via FFT + scale
        C = fft(ifftshift(phi_shifted))
        unscaled = np.real(C) * (self.dt_over_2pi * self.N)

        # 5) shift back so index j ↔ x_j = −Xmax + j·dx
        pdf_vals = fftshift(unscaled)

        # 6) clamp negatives and normalize
        pdf_vals[pdf_vals < 0] = 0.0
        pdf_vals /= np.trapz(pdf_vals, self.x)

        return pdf_vals

    def theoretical_tails(self, theta, s_i):
        """
        Given θ=[bp,cp,bn,cn] and quantile points s_i (length 99),
        compute Pihat[i] = F_θ(s_i) if s_i ≤ 0, else 1 - F_θ(s_i).

        Uses the PDF from self.pdf(theta) and a trapezoidal CDF.
        """
        pdf_vals = self.pdf(theta)
        xgrid = self.x
        dx = self.dx

        # Build CDF midpoints and cumulative sums
        x_mid = (xgrid[:-1] + xgrid[1:]) / 2           # (N-1,)
        cdf_vals = np.cumsum((pdf_vals[:-1] + pdf_vals[1:]) / 2 * dx)  # (N-1,)

        # Interpolator from x_mid → cdf_vals
        cdf_interp = interp.interp1d(
            x_mid, cdf_vals, kind='linear',
            bounds_error=False, fill_value=(0.0, 1.0)
        )

        Pihat = np.empty_like(s_i)
        for idx, sv in enumerate(s_i):
            if sv <= x_mid[0]:
                Fsv = 0.0
            elif sv >= x_mid[-1]:
                Fsv = 1.0
            else:
                Fsv = float(cdf_interp(sv))
            Pihat[idx] = Fsv if (sv <= 0.0) else (1.0 - Fsv)

        return Pihat

    @staticmethod
    def empirical_quantiles(m):
        """
        Given a 1D array m of returns, return:
          Pi_target : 99‐vector [0.01..0.99]
          s_i       : 99‐vector of empirical quantiles so CDF_emp(s_i) ≈ Pi_target[i]
        """
        n = len(m)
        s_sorted = np.sort(m)
        pi_emp = np.arange(1, n + 1) / n
        Pi_target = np.linspace(0.01, 0.99, 99)
        s_i = np.interp(
            Pi_target, pi_emp, s_sorted,
            left=s_sorted[0], right=s_sorted[-1]
        )
        return Pi_target, s_i

    def loss_fn(self, theta, s_i, Pi_target):
        """
        Weighted squared error between Pi_target and theoretical tails at s_i.
        If any component of θ ≤ 0, return a large penalty.
        """
        bp, cp, bn, cn = theta
        if (bp <= 0) or (cp <= 0) or (bn <= 0) or (cn <= 0):
            return 1e6 + np.sum(np.abs(theta))

        Pihat = self.theoretical_tails(theta, s_i)
        eps = 1e-12
        Pi_clipped = np.clip(Pi_target, eps, 1.0 - eps)
        W = Pi_clipped * (1.0 - Pi_clipped)
        sq = (Pi_target - Pihat) ** 2 / W
        return np.sum(sq)

    def fit_bilateral_gamma(self, m, initial_theta=None):
        """
        Fit θ=[bp,cp,bn,cn] to a 1D array of returns m by matching tails.

        Returns
        -------
        theta_hat : fitted 4‐vector
        """
        Pi_target, s_i = BG.empirical_quantiles(m)
        if initial_theta is None:
            initial_theta = np.array([0.02, 2.0, 0.02, 2.0])

        bounds = [(1e-6, None), (1e-6, None), (1e-6, None), (1e-6, None)]
        result = minimize(
            fun=self.loss_fn,
            x0=initial_theta,
            args=(s_i, Pi_target),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-9}
        )
        if not result.success:
            print("WARNING: fit did not converge:", result.message)
        return result.x  # [bp, cp, bn, cn]

    def fit_series(self, series, window=100):
        """
        Rolling‐window fit (serial) on a single 1D series of length T:
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
        Fit each column of X (shape T×num_assets) via rolling‐window. If n_workers>1,
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


