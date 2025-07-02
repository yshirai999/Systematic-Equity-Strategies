import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

import numpy as np
import torch
from scipy.stats import t
import os
from tqdm import tqdm
import time
from scipy.linalg import logm

from Modeling.BG_Modeling.Models import BG 
from Modeling.t_Copula_Modeling.utils import ArchakovHansen, estimate_copula_corr_kendall


# === Settings ===
tickers = ["spy", "xlb", "xle", "xlf", "xli", "xlk", "xlu", "xlv", "xly"]
nu = 6  # degrees of freedom for t-copula
window = 252  # rolling window size in days
save_path = os.path.abspath(os.path.join(PROJECT_ROOT,"Modeling","t_Copula_Modeling","results","correlation_matrices"))
theta_path = os.path.abspath(os.path.join(PROJECT_ROOT, "Modeling", "BG_Modeling", "estimates", "FINAL", "theta_{}_FINAL.npy"))
os.makedirs(save_path, exist_ok=True)

# === Create a BG instance per ticker ===
bg_dict = {}
# === Load BG objects with theta ===
for ticker in tickers:
    theta = np.load(theta_path.format(ticker))
    print(f"loading model for {ticker}")
    bg = BG(device="cpu", window=window, ticker=ticker, tickers=tickers)
    bg.ticker = ticker
    bg.theta_batch = torch.tensor(theta, dtype=torch.float32)
    bg_dict[ticker] = bg

# === Get return shape from one object ===
T, W = next(iter(bg_dict.values())).returns_matrix.shape
N = len(tickers)

# === Build 3D return tensor ===
returns_tensor = np.zeros((T, W, N), dtype=np.float32)
for i, ticker in enumerate(tickers):
    print(f"Loading returns for {bg_dict[ticker].ticker}...")
    returns_tensor[:, :, i] = bg_dict[ticker].returns_matrix

# # === Compute PIT (u) and inverse t CDF (z) ===
# u_tensor = np.zeros_like(returns_tensor)  # (T, W, N)
# z_tensor = np.zeros_like(returns_tensor)

# for i, ticker in enumerate(tickers):
#     bg = bg_dict[ticker]
    
#     # Step 1: Compute PDF and CDF from theta_batch
#     pdf = bg.pdf(bg.theta_batch)                    # (T, N_grid)
#     dx = bg.lambda_
#     cdf = torch.cumsum(pdf, dim=1) * dx             # (T, N_grid)
#     cdf = cdf / cdf[:, -1:].clamp(min=1e-8)

#     # Step 2: Interpolate at s_batch
#     s_batch = torch.tensor(bg.returns_matrix, dtype=torch.float32)
#     u_i = torch.zeros_like(s_batch)
#     for h in range(len(s_batch)):
#         u_i[h] = torch.tensor(np.interp(s_batch[h], bg.x.numpy(), cdf[h].numpy())) # Manual loop or vectorized interp if supported

#     # Step 3: Store clipped PIT and z values
#     u_i = np.clip(u_i.numpy(), 1e-6, 1 - 1e-6)
#     u_tensor[:, :, i] = u_i
#     z_tensor[:, :, i] = t.ppf(u_i, df=nu)

# Estimate correlation matrix
corr_sequence = []
#epsilon = 1e-6  # Ridge parameter for numerical stability
start_time = time.time()
for h in tqdm(range(T), desc="Estimating daily correlation"):
    Z_h = returns_tensor[h]  # shape (W, N)
    # empirical_corr = np.corrcoef(Z_h.T)
    # try:
    #     A = logm(empirical_corr)
    # except ValueError:
    #     epsilon = 1e-4
    #     empirical_corr += epsilon * np.eye(N)
    #     A = logm(empirical_corr)

    # v = A[np.tril_indices(N, -1)]
    #Sigma_t = ArchakovHansen(N, v, eps=1e-15, kmax=100000)
    corr = estimate_copula_corr_kendall(returns=Z_h)
    corr_sequence.append(corr)

print(f"\nCompleted in {time.time() - start_time:.2f} seconds.")

# Save result
np.save(os.path.join(save_path, f"corr_matrix_w{window}.npy"), np.array(corr_sequence))
print("Saved correlation matrix to:", os.path.join(save_path, f"corr_matrix_w{window}.npy"))
