import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib

matplotlib.use('TkAgg')  # or try 'QtAgg' if you have it
import matplotlib.pyplot as plt
import random
import torch

import sys

import os
print(os.getcwd())

from Data.DataProcessing import data

print("cuda is available: ", torch.cuda.is_available())
print("GPU name: ", torch.cuda.get_device_name(0))  # Optional: Shows GPU name

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "estimates", "Checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# BG class:
# -----------------------------------------------------------------------------

class BG(data):
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
            tickers = ['spy', 'xle', 'xlf', 'xli', 'xlk', 'xlp', 'xlu', 'xlv', 'xly', 'xom', 'xrx'],
            window=100,
            fit_day_indices=None,
            save_path=None,
            plot_path=None):
        """
        Initialize the BG class for bilateral gamma PDF estimation using FFT.
        Parameters
        ----------
        N      : int     - Number of FFT points
        B      : float   - Half-width of support; domain is [-pi*N/B, pi*N/B), frequency spacing is B/N.
        device : torch.device or str (optional) - 'cpu' or 'cuda'
        """
        super().__init__(tickers=tickers)
        # --------------------------------------------------------------
        # Basic parameters
        # --------------------------------------------------------------
        self.batch_size = batch_size
        self.window = window
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---------------------------------------------------------------
        # Load data and prepare returns
        # ---------------------------------------------------------------

        self.days = self.DataETFsReturns['days'].values
        X_full = self.DataETFsReturns[ticker].values.reshape(-1, 1)  # Exclude 'days' column
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
        self.save_path = save_path
        self.plot_path = plot_path
        self.ticker = ticker

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
        # if isinstance(initial_theta, np.ndarray):
        #     if initial_theta.shape != (1, 4):
        #         # Assume: initial theta is a numpy array of shape (1,4)
        #         initial_theta = np.tile(initial_theta, (min(batch_size,T), 1))
        
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

            # Use Adam optimizer instead of LBFGS to preserve inter-day independence
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
                            theta=theta_batch, grad=grad_snapshot, loss_fn=self.quantile_loss_AD, s_batch=s_batch, Pi_batch=Pi_batch
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
                os.path.join(CHECKPOINT_DIR, f"theta_{t0:04d}_{t1:04d}.pt")
            )
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            np.save(self.save_path, self.all_params)

        return  # Return after each batch for debugging

    def backtracking_step(self, theta, grad, loss_fn, s_batch, Pi_batch, alpha_init=1.0, beta=0.5, c=1e-4):
        with torch.no_grad():
            alpha = alpha_init
            loss_0 = loss_fn(theta, s_batch, Pi_batch)
            grad_norm_sq = (grad ** 2).sum()

            while alpha > 1e-5:
                theta_new = theta - alpha * grad
                loss_new = loss_fn(theta_new, s_batch, Pi_batch)
                if loss_new <= loss_0 - c * alpha * grad_norm_sq:
                    return theta_new, alpha
                alpha *= beta

        return theta, 0.0  # fallback
    
    # -----------------------------------------------------------------------------
    # Visualization and utility methods
    # -----------------------------------------------------------------------------
    def plot_bg_pdfs(self, theta_batch, pdfs=None, show=False, returnfig=False):
        """
        Plot a batch of BG PDFs.
        
        Parameters:
        - bg_model: the BG model object (already instantiated)
        - theta_batch: torch.Tensor of shape (B, 4)
        - pdfs: optional, precomputed tensor of shape (B, N)
        """
        B = theta_batch.shape[0]

        if pdfs is None:
            pdfs = self.pdf(theta_batch)

        x = self.x.cpu().numpy() if isinstance(self.x, torch.Tensor) else self.x
        figures = []

        for i in range(B):
            theta = theta_batch[i].cpu().numpy()
            label = f"bp={theta[0]:.2e}, cp={theta[1]:.2e}, bn={theta[2]:.2e}, cn={theta[3]:.2e}"

            fig = plt.figure(figsize=(8, 4))
            plt.plot(x, pdfs[i].cpu().numpy(), label=label)
            plt.title(f'BG PDF Estimate - Sample {i}')
            plt.xlabel('Return')
            plt.ylabel('Density')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            figures.append(fig)
            if show:
                plt.show()
            else:
                plt.close(fig)

        if returnfig:
            return figures

    def plot_empirical_vs_theoretical(self, theta_spy, n=3, seed=42, days=None, save_path=None, returnfig=False):
        """
        Plots empirical vs theoretical CDFs for `n` randomly selected days.
        Now splits upper and lower tails into separate subplots.
        """
        s_batch = self.s_batch          # shape (T, K)
        Pi_target = self.Pi_target    # shape (T,K)
        cal_days = self.days          # List of days estimated

        # Select valid days
        total_days = len(s_batch)
        if days is None:
            random.seed(seed)
            days = random.sample(range(total_days), min(n, total_days))
        else:
            days = list(days)[:n]

        theta_batch = torch.tensor(theta_spy[days], dtype=torch.float32).to(self.device)
        s_selected = s_batch[days]  # Select directly, no offset math
        Q_model = self.theoretical_quantiles(theta_batch, s_selected).cpu().detach().numpy()

        K = len(Pi_target)

        figures = []

        for i, day in enumerate(days):
            s = s_selected[i].cpu().numpy()
            pi_emp = Pi_target[day]
            pi_model = Q_model[i]

            fig = plt.figure()
            plt.plot(s, pi_emp, label='Empirical tails')
            plt.plot(s, pi_model, label='Theoretical tails')
            plt.title(f'Tails Fitting (day {cal_days[day]})')
            plt.xlabel('s')
            plt.grid(True)
            plt.legend()
            figures.append(fig)

            plt.tight_layout()
            if returnfig:
                plt.close(fig)  # Close the figure to avoid displaying it immediately
                pass
            elif self.plot_path and save_path and i == 0:
                plt.savefig(os.path.join(self.plot_path, f"{self.ticker}_empirical_vs_theoretical_day_{save_path}.png"))
                plt.close(fig)
            else:
                plt.show()
        
        if returnfig:
            return figures

    def plot_loss_per_day(self, logscale=False, returnfig=False):
        """
        Plot per-day loss using BG.batch_losses and BG.days.

        Parameters
        ----------
        bg : instance of BG class
            Must contain attributes:
            - batch_losses: (T,) array of losses
            - fit_day_indices: list of indices into bg.days
            - days: full list of datetime64 days
        logscale : bool
            If True, plot y-axis on log scale
        """
        assert hasattr(self, 'batch_losses') and hasattr(self, 'fit_day_indices') and hasattr(self, 'days')

        # Map fit_day_indices to actual calendar days
        day_labels = pd.to_datetime(self.days[self.fit_day_indices])
        
        # Build plot
        fig = plt.figure(figsize=(10, 4))
        plt.plot(day_labels, self.batch_losses, marker='.', linestyle='-', alpha=0.8)
        plt.xlabel("Date")
        plt.ylabel("Loss per day")
        if logscale:
            plt.yscale("log")
            plt.ylabel("Loss per day (log)")
        plt.title("Anderson-Darling Loss Per Day")
        plt.grid(True)
        plt.tight_layout()
        if self.plot_path:
            plt.savefig(os.path.join(self.plot_path, f"{self.ticker}_loss_evolution.png"))
        if returnfig:
            plt.close()
            return fig
        else:
            return None

    def plot_params(self, idx, show_comps=False):
        idx_0 = [id-idx[0] for id in idx]
        bp, cp, bn, cn = self.all_params[idx_0].T
        T = self.days[idx_0]

        # 1. Plot each parameter over time
        fig1, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        params = ['bp', 'cp', 'bn', 'cn']
        for i, param in enumerate([bp, cp, bn, cn]):
            axs[i].plot(T, param, label=params[i])
            axs[i].set_ylabel(params[i])
            axs[i].legend()
            axs[i].grid(True)
        axs[-1].set_xlabel('Time')
        plt.suptitle('Parameter Evolution Over Time')
        plt.tight_layout()
        if self.plot_path:
            plt.savefig(os.path.join(self.plot_path, f"{self.ticker}_params_evolution.png"))

        if not show_comps:
            return fig1
        # 2. Plot parameter pairs: (mup, sigmap) and (mun, sigman)
        fig2, ax = plt.subplots(1, 3, figsize=(12, 5))
        ax[0].scatter(bp*cp, np.sqrt(bp)*cp, alpha=0.6)
        ax[0].set_xlabel('mup')
        ax[0].set_ylabel('sigmap')
        ax[0].set_title('mup vs sigmap')

        ax[1].scatter(bp*cp, bn*cn, alpha=0.6, color='orange')
        ax[1].set_xlabel('mup')
        ax[1].set_ylabel('mun')
        ax[1].set_title('mup vs mun')

        ax[2].scatter(bp*cp, np.sqrt(bn)*cn, alpha=0.6, color='green')
        ax[2].set_xlabel('mup')
        ax[2].set_ylabel('sigman')
        ax[2].set_title('mup vs sigman')

        plt.suptitle('Parameter Pair Relationships')
        plt.tight_layout()

        # 3. Plot parameter pairs: (bp, bn) and (cp, cn)
        fig3, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].scatter(cp, cn, alpha=0.6)
        ax[0].set_xlabel('cp')
        ax[0].set_ylabel('bcn')
        ax[0].set_title('cp vs cn')

        ax[1].scatter(bp, bn, alpha=0.6)
        ax[1].set_xlabel('bp')
        ax[1].set_ylabel('bn')
        ax[1].set_title('bp vs bn')

        plt.tight_layout()

        # 4. Plot parameter pairs: (bp, bn) and (cp, cn)
        fig4, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].scatter(cp, bp, alpha=0.6)
        ax[0].set_xlabel('cp')
        ax[0].set_ylabel('bp')
        ax[0].set_title('cp vs bp')

        ax[1].scatter(cn, bn, alpha=0.6)
        ax[1].set_xlabel('cn')
        ax[1].set_ylabel('bn')
        ax[1].set_title('bn vs cn')

        plt.tight_layout()

        return fig1

    def plot_diagnostics(self, theta_batch, save_path_empirical_vs_theoretical=False):
        self.plot_loss_per_day()
        print(f"Average loss per day: {self.batch_losses.mean():.4f}")
        print(f"Max loss per day: {self.batch_losses.max():.4f}")
        print(f"Day with max loss: {self.days[self.fit_day_indices][self.batch_losses.argmax()]}")

        # Plot empirical vs theoretical quantiles for SPY on day with max loss
        idx_worst = self.batch_losses.argmax()+self.window
        idx = [idx_worst,idx_worst+1]
        self.plot_empirical_vs_theoretical(theta_batch, n=2, days=idx, save_path=save_path_empirical_vs_theoretical)