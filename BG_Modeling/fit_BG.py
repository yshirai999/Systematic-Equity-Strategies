import yaml
from Models import BG

import numpy as np
from scipy.io import loadmat
import torch

import os

# Run this script in PowerShell to update the config.yaml file:
# (Get-Content BG_Modeling\config.yaml) -replace 'ticker:.*', 'ticker: spy' -replace 'train:.*', 'train: True' | Set-Content BG_Modeling\config.yaml

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.yaml")

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

ticker = cfg["ticker"]
save_path = cfg["save_path"]
params_path = cfg["params_path"]
initial_theta_path = cfg["initial_theta_path"]
save_path = os.path.join('estimates', ticker, f"{ticker}_{save_path}")
params_path = os.path.join('estimates', ticker, f"{ticker}_{params_path}")
initial_theta_path = os.path.join('estimates', ticker, f"{ticker}_{initial_theta_path}")

# Build BG instance
bg = BG(
    N=cfg["N"],
    B=cfg["B"],
    device=cfg["device"],
    batch_size=cfg["batch_size"],
    ticker=cfg["ticker"],
    window=cfg["window"],
    save_path=save_path
)

# Fit or load
if cfg["train"]:
    try:
        initial_theta = loadmat(initial_theta_path)["theta"]
    except FileNotFoundError:
        initial_theta = [[0.0075, 1.55, 0.0181, 0.6308]]
    bg.fit_theta_in_batches(max_iter=cfg["max_iter"], verbose=True, backtracking=cfg["backtracking"], backtracking_params=cfg["backtracking_params"], initial_theta=initial_theta)
else:
    param_path = os.path.join(script_dir, params_path)
    if os.path.exists(param_path):
        bg.all_params = np.load(param_path)
    else:
        raise FileNotFoundError(f"File {params_path} not found.")

theta_batch = torch.tensor(bg.all_params[:min(5,bg.T)], dtype=torch.float32).to('cuda')
pdfs = bg.pdf(theta_batch)  # Expected shape: [5, 4096]
# bg.plot_bg_pdfs(theta_batch, pdfs)

theta_batch = torch.tensor(bg.all_params, dtype=torch.float32).to('cuda')
# Split s_batch into CUDA batches
if np.isnan(bg.batch_losses).all():
    for t0 in range(0, bg.T, bg.batch_size):
        t1 = min(t0 + bg.batch_size, bg.T)

        # Get theta_batch
        theta_batch_t = theta_batch[t0:t1]
        s_batch = bg.s_batch[t0:t1]  # narrow s_batch to batch window
        Pi_batch = bg.Pi_target_torch[t0:t1]  # narrow Pi_target to batch window

        with torch.no_grad():
            per_day_loss = bg.quantile_loss_AD(theta_batch_t, s_batch, Pi_batch, return_per_day=True)
        bg.batch_losses[t0:t1] = per_day_loss.cpu().numpy()

bg.plot_diagnostics(theta_batch)
