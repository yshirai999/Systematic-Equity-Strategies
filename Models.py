import numpy as np
import pandas as pd
from scipy.io import loadmat

import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))  # Optional: Shows GPU name

import os

checkpoint_dir = "theta_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

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

    def __init__(self,
            N=2**14,
            B=300000,
            device=None,
            batch_size=256,
            ticker='spy',
            window=100,
            fit_day_indices=None):
        """
        Initialize the BG class for bilateral gamma PDF estimation using FFT.
        Parameters
        ----------
        N      : int     - Number of FFT points
        B      : float   - Half-width of support; domain is [-pi*N/B, pi*N/B), frequency spacing is B/N.
        device : torch.device or str (optional) - 'cpu' or 'cuda'
        """

        # --------------------------------------------------------------
        # Basic parameters
        # --------------------------------------------------------------
        self.batch_size = batch_size
        self.window = window
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---------------------------------------------------------------
        # Load data and prepare returns
        # ---------------------------------------------------------------

        mat = loadmat('tsd180.mat')
        d = mat['days']
        p = mat['pm']
        n = mat['nmss']
        [_,M] = np.shape(p)

        Data = pd.DataFrame({'days': d[:,0]})
        for i in range(M):
            Datai = pd.DataFrame({n[i,0][0] : p[:,i]})
            Data = pd.concat([Data,Datai],axis=1)

        DataETFs = Data[[ticker]]
        DataETFsReturns = DataETFs.pct_change()
        DataETFsReturns = DataETFsReturns.drop(index = 0)
        DataETFsReturns.insert(0, 'days', d[1:])
        DataETFs.insert(0, 'days', d)
        DataETFsReturns['days'] = pd.to_datetime(DataETFsReturns['days'], format='%Y%m%d')
        self.days = DataETFsReturns['days'].values

        X_full = DataETFsReturns.iloc[:, 1:].values  # exclude 'days' column
        T_full, _ = X_full.shape

        # Determine which days to include for fitting
        if fit_day_indices is not None:
            self.fit_day_indices = np.array([t+window for t in fit_day_indices])
        else:
            self.fit_day_indices = np.arange(window, T_full)

        # Now build the returns_matrix with shape (T, window)
        self.returns_matrix = np.array([
            X_full[t-window:t, 0] for t in self.fit_day_indices
        ])
        self.T = len(self.fit_day_indices)  # number of training samples

        self.all_params = np.full((self.T, 4), np.nan) 
        self.batch_losses = np.full((self.T,), np.nan)

        # --------------------------------------------------------------
        # Precompute the target quantile levels
        # --------------------------------------------------------------

        self.empirical_quantiles()
        self.Pi_target_torch = torch.tensor(self.Pi_target, dtype=torch.float32, device=self.device)

        # --------------------------------------------------------------
        # FFT routine parameters
        # --------------------------------------------------------------
        self.N = N
        self.B = B

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
    
    def empirical_quantiles(self):
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
        s_batch : (n_days, 99) array of empirical quantiles
        """

        batch_returns = self.returns_matrix  # shape (T, window)
        W = self.window
        s_batch = np.sort(batch_returns, axis=1)
        pi = np.arange(1, W + 1) / W
        pi_broadcast = np.tile(pi, (s_batch.shape[0], 1))  # shape (T, K)
        # Flip to tail probability when s_batch >= 0
        Pi_target = np.where(
            s_batch < 0,
            pi_broadcast,
            1.0 - pi_broadcast
        )
        self.Pi_target = Pi_target
        self.s_batch = torch.tensor(s_batch, dtype=torch.float32).to(self.device) # Move to GPU

    def theoretical_quantiles(self, theta_batch, s_batch=None):
        """
        Torch differentiable theoretical quantiles:
        - Accepts batches of theta (B, 4) directly
        - Computes PDF → CDF → interpolated quantiles at self.s_batch
        - Returns: Q_model (T, K)
        """
        if s_batch is None:
            raise ValueError("s_batch must be provided")

        # Step 1: Compute PDF from theta (requires grad)
        pdf = self.pdf(theta_batch)  # (B, N)

        # Step 2: Compute normalized CDF
        dx = self.lambda_
        cdf = torch.cumsum(pdf, dim=1) * dx  # (B, N)
        cdf = cdf / cdf[:, -1:].clamp(min=1e-8)

        # Step 3: Interpolate at s_batch
        B, K = s_batch.shape # T = # of days, K = # of quantiles
        N = self.x.shape[0] # N = space grids of the PDF

        x_grid = self.x.view(1, 1, N).expand(B, K, N)  # expand x from (1, 1, N) to (B, K, N) for broadcasting
        s_grid = s_batch.unsqueeze(-1) # reshape s_batch to (B, K, 1) for comparison
        s_clamped = s_batch.clamp(min=self.x[0].item(), max=self.x[-1].item()) # Clamp s_batch to [x[0], x[-1]]

        idx = torch.sum(x_grid <= s_grid, dim=2) - 1  # For each t,k, idx satisfies x_{idx} <= s_{t,k} <= x_{idx+1}
        #idx = torch.searchsorted(self.x, s_clamped, right=True) - 1

        idx = idx.clamp(min=0, max=N - 2)      # Ensure idx is within bounds [0, N-2]
        idx_unsq = idx.unsqueeze(-1)  # (B, K, 1) for gathering

        #cdf_grid = cdf.unsqueeze(1).expand(B, K, N)
        cdf_grid = cdf.unsqueeze(1).expand(-1, K, -1)

        x_lower = torch.gather(x_grid, 2, idx_unsq).squeeze(-1) 
        x_upper = torch.gather(x_grid, 2, idx_unsq + 1).squeeze(-1) 
        cdf_lower = torch.gather(cdf_grid, 2, idx_unsq).squeeze(-1) 
        cdf_upper = torch.gather(cdf_grid, 2, idx_unsq + 1).squeeze(-1) 

        weight = (s_clamped - x_lower) / (x_upper - x_lower + 1e-8)
        Pi_hat = cdf_lower + weight * (cdf_upper - cdf_lower) # Linear interpolation

        # Flip to tail probability when s_batch >= 0    
        Pi_hat = torch.where(
            s_batch < 0,
            Pi_hat,
            1.0 - Pi_hat
        )

        return Pi_hat  # (B, K), fully differentiable w.r.t. theta

    def theoretical_quantiles_batched(self, theta):
        """
        Computes theoretical quantiles for each row of theta in batches.
        Each theta[t] corresponds to a parameter vector for day t.
        
        Args:
            theta: Tensor of shape (T, 4), requires_grad=True.
            batch_size: Number of samples per batch.
            
        Returns:
            Q_model: Tensor of shape (T, K), differentiable w.r.t. theta.
        """
        T = self.T  # Number of days after the window = len(theta)
        
        Q_model_list = []
        batch_size = self.batch_size

        for t0 in range(0, T, batch_size):
            t1 = min(t0 + batch_size, T)
            if t1 <= t0:
                continue  # Skip empty batches
            theta_batch = theta[t0:t1]  # shape (B, 4)
            s_batch = self.s_batch[t0:t1]  # shape (B, K)

            # Compute Q_model for this batch
            Q_batch = self.theoretical_quantiles(theta_batch,s_batch)  # (B, K)
            Q_model_list.append(Q_batch)

        Q_model = torch.cat(Q_model_list, dim=0)  # shape (T, K)
        return Q_model

    def quantile_loss_AD(self, theta_batch, s_batch, pi_batch, return_per_day=False):
        """
        Anderson-Darling weighted loss between theoretical and empirical quantiles.
        Fully differentiable w.r.t. theta_batch.
        """
        # Transform parameters to avoid negative values
        eps = 1e-6 
        bp = torch.sqrt(theta_batch[:, 0]**2 + eps).unsqueeze(1)
        cp = torch.sqrt(theta_batch[:, 1]**2 + eps).unsqueeze(1)
        bn = torch.sqrt(theta_batch[:, 2]**2 + eps).unsqueeze(1)
        cn = torch.sqrt(theta_batch[:, 3]**2 + eps).unsqueeze(1)

        theta_transformed = torch.cat([bp, cp, bn, cn], dim=1)

        Q_model = self.theoretical_quantiles(theta_transformed, s_batch)  # shape (B, K)
        Q_emp = pi_batch  # shape (B, K)

        
        # Anderson-Darling weights (tail-emphasizing)
        eps = 1e-6
        w = 1.0 / (Q_emp * (1 - Q_emp) + eps)  # shape (B,K)

        weighted_sq_diff = ((Q_model - Q_emp) ** 2) * w  # shape (B, K)
        per_day_loss = weighted_sq_diff.mean(dim=1) # shape (B,)

        # Replace any NaNs with a large penalty
        per_day_loss = torch.where(
            torch.isnan(per_day_loss),
            torch.full_like(per_day_loss, 1e5),
            per_day_loss
        )
        
        return per_day_loss if return_per_day else per_day_loss.mean()


    ####################################################################
    ## Fitting methods for rolling windows and multiple series
    ####################################################################

    # Assume: theta_batch (T_small, 4) on CUDA, requires_grad=True
    # Assume: s_batch (T_small, K) already set

    def fit_theta_in_batches(self, initial_theta=None, max_iter=200, verbose=True, backtracking=False):
        """
        Fit theta_batch over all time steps using Adams with mini-batches and (optionally) backtracking line search.
        # LBFGS introduces across days dependence, so it is better to use Adam for per-day independence
        Parameters:
        - initial_theta: optional numpy array of shape (T, 4), default = constant guess
        - batch_size: number of days per optimization batch
        - max_iter: Adam maximum iterations per batch
        - verbose: if True, print loss at each batch

        Stores:
        - self.all_params: numpy array (T, 4) with optimized parameters
        """
        T = self.T  # Number of days after the window
        batch_size = self.batch_size

        if initial_theta is None:
            # Default: repeat a reasonable starting guess
            initial_theta = np.tile(np.array([0.0075, 1.55, 0.0181, 0.6308]), (T, 1))
        if isinstance(initial_theta, np.ndarray):
            if initial_theta.shape != (1, 4):
                # Assume: initial theta is a numpy array of shape (1,4)
                initial_theta = np.tile(initial_theta, (min(batch_size,T), 1))
        
        apply_warm_start = (initial_theta is None) or (len(initial_theta) < T)

        # Split s_batch into CUDA batches
        for t0 in range(0, T, batch_size):
            t1 = min(t0 + batch_size, T)

            # Optionally warm-start batch from previous day
            if apply_warm_start and t0 > 0:
                initial_theta[t0:t1] = self.all_params[t0 - 1]

            # Get theta_batch
            theta_batch = initial_theta[t0:t1]
            if isinstance(theta_batch, torch.Tensor):
                if not theta_batch.is_cuda or not theta_batch.requires_grad:
                    theta_batch = theta_batch.detach().clone().to(self.device).requires_grad_()
            else:
                theta_batch = torch.tensor(theta_batch, dtype=torch.float32, device=self.device)
                theta_batch = theta_batch.detach().clone().requires_grad_()

            s_batch = self.s_batch[t0:t1]  # narrow s_batch to batch window
            Pi_batch = self.Pi_target_torch[t0:t1]  # narrow Pi_target to batch window  

            # Use Adam optimizer instead of LBFGS for per-day independence
            optimizer = torch.optim.Adam([theta_batch], lr=1e-3, amsgrad=True)  # Use Adam optimizer
            for s in range(max_iter):
                optimizer.zero_grad()
                loss = self.quantile_loss_AD(theta_batch, s_batch, Pi_batch)
                loss.backward()
                if backtracking:
                    grad_snapshot = theta_batch.grad.detach().clone()
                    optimizer.step()
                    # Optional backtracking line search
                    with torch.no_grad():
                        theta_bt, alpha_bt = self.backtracking_step(
                            theta=theta_batch, grad=grad_snapshot, loss_fn=self.quantile_loss_AD, s_batch=s_batch
                        )
                        theta_batch[:] = theta_bt  # update in place
                else: 
                    optimizer.step() 
                s += 1

            if verbose:
                opt_str = "Adam"
                print(f"[{t0:4d}:{t1:4d}] {opt_str} Loss = {loss.item():.6f} Total steps = {s}")

            # Finalize theta_batch to ensure non-negativity
            eps = 1e-6  # small value to prevent sqrt(0)
            theta_final = torch.stack([
                torch.sqrt(theta_batch[:, 0]**2 + eps),
                torch.sqrt(theta_batch[:, 1]**2 + eps),
                torch.sqrt(theta_batch[:, 2]**2 + eps),
                torch.sqrt(theta_batch[:, 3]**2 + eps)
            ], dim=1).detach()

            # Save to all_params
            self.all_params[t0:t1] = theta_final.detach().cpu().numpy()
            # Recompute per-day loss using final theta
            with torch.no_grad():
                per_day_loss = self.quantile_loss_AD(theta_final, s_batch, Pi_batch, return_per_day=True)
            self.batch_losses[t0:t1] = per_day_loss.cpu().numpy()

            # Save checkpoint
            torch.save(
                theta_batch.detach().cpu(),
                os.path.join(checkpoint_dir, f"theta_{t0:04d}_{t1:04d}.pt")
            )

        return self.all_params  # Return after each batch for debugging

    def backtracking_step(self, theta, grad, loss_fn, s_batch, alpha_init=1.0, beta=0.5, c=1e-4):
        with torch.no_grad():
            alpha = alpha_init
            loss_0 = loss_fn(theta, s_batch)
            grad_norm_sq = (grad ** 2).sum()

            while alpha > 1e-5:
                theta_new = theta - alpha * grad
                loss_new = loss_fn(theta_new, s_batch)
                if loss_new <= loss_0 - c * alpha * grad_norm_sq:
                    return theta_new, alpha
                alpha *= beta

        return theta, 0.0  # fallback

# ------------------------------------------
# Example usage
# ------------------------------------------
if __name__ == "__main__":
    import numpy as np
    import torch
    from Models import BG

    # 1) Simulate returns (4430 days, 11 assets)
    np.random.seed(0)
    X = np.random.randn(4430, 11)

    # 2) Build BG instance with appropriate x-grid
    bg = BG(N=4096, Xmax=0.1, device='cuda')  # Ensure it uses GPU

    # 3) Fit theta over all time with batching
    print("Fitting theta in batches...")
    theta_spy = bg.fit_theta_in_batches(batch_size=200, max_iter=20, verbose=True)

    print("Final theta shape:", theta_spy.shape)
    print("Example SPY params at t=150:", theta_spy[150])



    # # -----------------------------------------------------------------------------
    # # Module-level helpers (must be top-level so they can be pickled):
    # # -----------------------------------------------------------------------------

    # def _fit_window_bg(args):
    #     """
    #     Args:
    #       args = (t_idx, m_sub, bg_instance)

    #     Returns:
    #       (t_idx, theta_hat) where theta_hat = bg_instance.fit_bilateral_gamma(m_sub)
    #     """
    #     t_idx, m_sub, bg_instance = args
    #     theta_hat = bg_instance.fit_bilateral_gamma(m_sub)
    #     return t_idx, theta_hat


    # def _fit_column_bg(args):
    #     """
    #     Args:
    #       args = (j_idx, X_col, bg_instance, window)

    #     Returns:
    #       (j_idx, params_j) where params_j = bg_instance.fit_series(X_col, window)
    #     """
    #     j_idx, X_col, bg_instance, window = args
    #     params_j = bg_instance.fit_series(X_col, window=window)
    #     return j_idx, params_j

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

    # def fit_series(self, series, window=100):
    #     """
    #     Rolling-window fit (serial) on a single 1D series of length T:
    #         - For t < window: row stays [nan,nan,nan,nan].
    #         - For t ≥ window: fit on series[t-window : t].

    #     Returns
    #     -------
    #     params : array of shape (T,4).
    #     """
    #     T = len(series)
    #     params = np.full((T, 4), np.nan)

    #     for t in range(window, T):
    #         m_window = series[t - window : t]
    #         theta_hat = self.fit_bilateral_gamma(m_window)
    #         params[t, :] = theta_hat

    #     return params

    # def fit_multiple(self, X, window=100, n_workers=1):
    #     """
    #     Fit each column of X (shape T x num_assets) via rolling-window. If n_workers>1,
    #     parallelize across *assets* (columns). 

    #     Returns
    #     -------
    #     all_params : array of shape (T, num_assets, 4).
    #     """
    #     T, num_assets = X.shape
    #     all_params = np.full((T, num_assets, 4), np.nan)

    #     # Prepare a list of (j_idx, X_col, self, window) for each asset
    #     tasks = [(j, X[:, j], self, window) for j in range(num_assets)]

    #     if n_workers is None or n_workers <= 1:
    #         # Serial over assets
    #         for j_idx, X_col, bg_inst, win in tasks:
    #             _, params_j = _fit_column_bg((j_idx, X_col, bg_inst, win))
    #             all_params[:, j_idx, :] = params_j
    #     else:
    #         # Parallel across assets
    #         with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as exe:
    #             for j_idx, params_j in exe.map(_fit_column_bg, tasks):
    #                 all_params[:, j_idx, :] = params_j

    #     return all_params  # shape (T, num_assets, 4)

    # def fit_series_parallel(series, window=100, bg_model=None, n_jobs=-1):
    #     """
    #     Parallelized version of fit_series using joblib. Applies BG fitting in parallel across time windows.

    #     Parameters
    #     ----------
    #     series : np.ndarray
    #         1D array of time series data of length T.
    #     window : int
    #         Rolling window size.
    #     bg_model : object
    #         An instance of the BG class with a `fit_bilateral_gamma` method.
    #     n_jobs : int
    #         Number of jobs for parallel processing. -1 means using all available cores.

    #     Returns
    #     -------
    #     params : np.ndarray
    #         Array of shape (T, 4) with fitted BG parameters.
    #     """
    #     T = len(series)
    #     params = np.full((T, 4), np.nan)

    #     def fit_at_time(t):
    #         window_data = series[t - window:t]
    #         return bg_model.fit_bilateral_gamma(window_data)

    #     # Apply in parallel only for valid indices
    #     results = Parallel(n_jobs=n_jobs)(
    #         delayed(fit_at_time)(t) for t in range(window, T)
    #     )

    #     # Assign results
    #     for i, t in enumerate(range(window, T)):
    #         params[t, :] = results[i]
