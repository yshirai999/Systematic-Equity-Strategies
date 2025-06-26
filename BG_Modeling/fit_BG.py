import yaml
from Models import BG

import numpy as np
from scipy.io import loadmat
import torch

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Run this script in PowerShell to update the config.yaml file:
# (Get-Content BG_Modeling\config.yaml) -replace 'ticker:.*', 'ticker: xlb' | Set-Content BG_Modeling\config.yaml
# (Get-Content BG_Modeling\config.yaml) -replace 'train:.*', 'train: True' | Set-Content BG_Modeling\config.yaml
# (Get-Content BG_Modeling\config.yaml) -replace 'max_iter:.*', 'max_iter: 500' | Set-Content BG_Modeling\config.yaml
# (Get-Content BG_Modeling\config.yaml) -replace 'backtracking:.*', 'backtracking: False' | Set-Content BG_Modeling\config.yaml
# Tickers: spy, xlb, xle, xlf, xli, xlk, xlp, xlu, xlv, xly

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.yaml")

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

# Set paths and parameters
ticker = cfg["ticker"]
save_path = cfg["save_path"]
params_path = cfg["params_path"]
max_iter = cfg["max_iter"]
backtracking = cfg["backtracking"]
initial_theta_prefix = cfg["initial_theta_prefix"]
init_max_iter = cfg["initial_theta_max_iter"]
init_backtracking = cfg["initial_theta_backtracking"]
init_filename = f"{ticker}_{initial_theta_prefix}_{init_max_iter}_BT_{init_backtracking}.npy"
initial_theta_path = os.path.join(SCRIPT_DIR, 'estimates', ticker, init_filename)

save_path = os.path.join(SCRIPT_DIR,"estimates", ticker, f"{ticker}_{save_path}_{max_iter}_BT_{backtracking}.npy")
params_path = os.path.join(SCRIPT_DIR,"estimates", ticker, f"{ticker}_{params_path}_{max_iter}_BT_{backtracking}.npy")
if cfg["train"] or cfg["load_initial_theta"]:
    try:
        initial_theta = np.load(initial_theta_path)
        initial_theta_loaded = True
        filename = f"{ticker}_{initial_theta_prefix}_{init_max_iter}_BT_{init_backtracking}_{max_iter}_BT_{backtracking}.npy"
        save_path = os.path.join(SCRIPT_DIR, 'estimates', ticker, filename)
        params_path = os.path.join(SCRIPT_DIR, 'estimates', ticker, filename)
    except:
        print(f"Initial theta file {initial_theta_path} not found. Using default initial theta.")
        initial_theta = None
        initial_theta_loaded = False

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
    if os.path.exists(save_path):
        print(f"Directory {save_path} already exists. Skipping training.")
        bg.all_params = np.load(params_path)
    else:
        bg.fit_theta_in_batches(max_iter=cfg["max_iter"], verbose=True, backtracking=cfg["backtracking"], initial_theta=initial_theta)
else:
    if os.path.exists(params_path):
        bg.all_params = np.load(params_path)
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
bg.plot_params(range(bg.T))
