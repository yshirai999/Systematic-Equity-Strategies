import os
import shutil

# === Settings ===
target_base_dir = os.path.dirname(os.path.abspath(__file__))
ticker_list = ["SPY", "XLF", "XLK", "XLI", "XLV", "XLY", "XLE", "XLC", "XLRE", "XLB", "XLU"]

theta_spy = 'spy_theta_500_BT_False_500_BT_True.npy'
theta_xlf = 'xlf_theta_500_BT_False.npy'
theta_xlk = 'xlk_theta_500_BT_False.npy'
theta_xli = 'xli_theta_500_BT_False.npy'
theta_xlv = 'xlv_theta_500_BT_False.npy'
theta_xly = 'xly_theta_500_BT_False_500_BT_True.npy'
theta_xle = 'xle_theta_500_BT_False.npy'
theta_xlb = 'xlb_theta_500_BT_False.npy'
theta_xlu = 'xlu_theta_500_BT_False_500_BT_True.npy'

# === Copy and rename theta files ===
for ticker in ticker_list:
    if ticker == "SPY":
        theta_file = theta_spy
    elif ticker == "XLF":
        theta_file = theta_xlf
    elif ticker == "XLK":
        theta_file = theta_xlk
    elif ticker == "XLI":
        theta_file = theta_xli
    elif ticker == "XLV":
        theta_file = theta_xlv
    elif ticker == "XLY":
        theta_file = theta_xly
    elif ticker == "XLE":
        theta_file = theta_xle
    elif ticker == "XLB":
        theta_file = theta_xlb
    elif ticker == "XLU":
        theta_file = theta_xlu
    else:
        continue  # Skip if ticker is not in the list
    source_path = os.path.join(target_base_dir, f"{ticker}", f"{theta_file}")
    target_path = os.path.join(target_base_dir, f"{ticker}", f"theta_{ticker}_FINAL.npy")

    print("Checking source file...")
    if not os.path.exists(source_path):
        print(f"Source file {source_path} does not exist. Skipping...")
        continue
    shutil.copy2(source_path, target_path)
    print(f"Copied {source_path} to {target_path}")
