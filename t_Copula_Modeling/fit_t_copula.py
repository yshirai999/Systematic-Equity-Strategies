import numpy as np
import torch
from scipy.stats import t
import os
from BG_Modeling.Models import BG 
from ArchakovHansen import ArchakovHansen
from scipy.linalg import logm

# === Settings ===
tickers = ["SPY", "XLB", "XLE", "XLF", "XLI", "XLK", "XLU", "XLV", "XLY"]
data_dir = "../BG_Modelling/estimates"
nu = 6  # degrees of freedom for t-copula
save_path = "../Copula_Modeling/results/correlation_matrices"
theta_path = "../BG_Modelling/estimates/theta_{}_FINAL.npy"
os.makedirs(save_path, exist_ok=True)

# === Create a BG instance per ticker ===
bg_dict = {}
# === Load BG objects with theta ===
for ticker in tickers:
    theta = np.load(theta_path.format(ticker.lower()))
    bg = BG(device="cpu")
    bg.ticker = ticker
    bg.theta_batch = torch.tensor(theta, dtype=torch.float32)
    bg_dict[ticker] = bg

# === Get return shape from one object ===
T, W = next(iter(bg_dict.values())).returns.shape
N = len(tickers)

# === Build 3D return tensor ===
returns_tensor = torch.zeros((T, W, N))
for i, ticker in enumerate(tickers):
    returns_tensor[:, :, i] = bg_dict[ticker].returns

# === Compute PIT (u) and inverse t CDF (z) ===
u_tensor = torch.zeros_like(returns_tensor)
z_tensor = torch.zeros_like(returns_tensor)

for i, ticker in enumerate(tickers):
    bg = bg_dict[ticker]
    returns_i = returns_tensor[:, :, i]  # shape (T, W)
    # Compute CDF per return value
    u_i = bg.compute_cdf(returns_i.unsqueeze(-1)).squeeze(-1)  # shape (T, W)
    u_i = torch.clamp(u_i, 1e-6, 1 - 1e-6)  # prevent infs in tails

    u_tensor[:, :, i] = u_i
    z_tensor[:, :, i] = torch.tensor(t.ppf(u_i.numpy(), df=nu))

# Clip u to avoid infs in tails
eps = 1e-6
u_tensor = torch.clip(u_tensor, eps, 1 - eps)

# Estimate correlation matrix
Sigma_sequence = []
epsilon = 1e-6  # Ridge parameter for numerical stability
for t in range(T):
    Z_t = z_tensor[t]  # shape (W, N)
    Z_np = Z_t.cpu().numpy()
    empirical_corr = np.corrcoef(Z_np.T)
    
    try:
        A = logm(empirical_corr)
    except ValueError:
        # Add ridge and retry
        print(f"logm failed at t={t}, adding ridge")
        empirical_corr += epsilon * np.eye(N)
        A = logm(empirical_corr)
    
    v = A[np.tril_indices(N, -1)]
    Sigma_t = ArchakovHansen(N, v, eps=1e-6)
    
    Sigma_sequence.append(Sigma_t)


# Save result
np.save(os.path.join(save_path, f"..."), np.array(Sigma_sequence))
print("Saved correlation matrix to:", os.path.join(save_path, f"corr_matrix_nu{nu}.npy"))
