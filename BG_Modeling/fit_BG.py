import yaml
from Models import BG

import numpy as np
from scipy.io import loadmat
import torch

import os

# Load config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Build BG instance
bg = BG(
    N=cfg["N"],
    B=cfg["B"],
    device=cfg["device"],
    batch_size=cfg["batch_size"],
    ticker=cfg["ticker"],
    window=cfg["window"],
    save_path=cfg["save_path"]
)

# Fit or load
if cfg["train"]:
    try:
        initial_theta = loadmat(cfg["initial_theta_path"])["theta"]
    except FileNotFoundError:
        initial_theta = [[0.0075, 1.55, 0.0181, 0.6308]]
    bg.fit_theta_in_batches(max_iter=cfg["max_iter"], verbose=True, backtracking=cfg["backtracking"], backtracking_params=cfg["backtracking_params"], initial_theta=initial_theta)
else:
    if os.path.exists(cfg["save_path"]):
        bg.all_params = np.load(cfg["save_path"])
        print("Loaded pre-trained parameters.")
    else:
        raise FileNotFoundError(f"File {cfg['save_path']} not found.")

theta_batch = torch.tensor(bg.all_params[:min(5,bg.T)], dtype=torch.float32).to('cuda')
pdfs = bg.pdf(theta_batch)  # Expected shape: [5, 4096]
bg.plot_bg_pdfs(theta_batch, pdfs)


# Split s_batch into CUDA batches
if isinstance(bg.batch_losses, np.full((bg.T,), np.nan)):
    for t0 in range(0, bg.T, bg.batch_size):
        t1 = min(t0 + bg.batch_size, bg.T)

        # Get theta_batch
        theta_batch_t = theta_batch[t0:t1]
        s_batch = bg.s_batch[t0:t1]  # narrow s_batch to batch window
        Pi_batch = bg.Pi_target_torch[t0:t1]  # narrow Pi_target to batch window

        with torch.no_grad():
            per_day_loss = bg.quantile_loss_AD(theta_batch_t, s_batch, Pi_batch, return_per_day=True)
        bg.batch_losses[t0:t1] = per_day_loss.cpu().numpy()

bg.plot_loss_per_day()
print(f"Average loss per day: {bg.batch_losses.mean():.4f}")
print(f"Max loss per day: {bg.batch_losses.max():.4f}")
print(f"Day with max loss: {bg.days[bg.fit_day_indices][bg.batch_losses.argmax()]}")

# Plot empirical vs theoretical quantiles for SPY on day with max loss
idx_worst = bg.batch_losses.argmax()+bg.window
idx = [idx_worst-100,idx_worst,idx_worst+100]
bg.plot_empirical_vs_theoretical(theta_batch, pdfs, n=3, days=idx)
