import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Data.DataProcessing import data

SAVE_PATH = os.path.join(PROJECT_ROOT, "t_Copula_Modeling", "results", "correlation_matrices")
PLOT_DIR = os.path.join(PROJECT_ROOT, "t_Copula_Modeling", "results", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
PLOT_PATH = os.path.join(PLOT_DIR, "corr_over_time.png")

# Load days and returns data
data_obj = data()
days = data_obj.DataETFsReturns['days'].values

# Load correlation tensor
corr_tensor = np.load(os.path.join(SAVE_PATH, "corr_matrix_nu6.npy"))  # shape (T, N, N)
tickers = ['spy', 'xle', 'xlf', 'xli', 'xlk', 'xlp', 'xlu', 'xly']
pairs = [('spy', 'xle'), ('spy', 'xlp'), ('xlf', 'xlu')]
indices = [(tickers.index(i), tickers.index(j)) for i, j in pairs]



T = corr_tensor.shape[0]
for (i, j), label in zip(indices, pairs):
    plt.plot(range(T), corr_tensor[:, i, j], label=f"{label[0]}-{label[1]}")

plt.xticks(ticks=np.linspace(0, len(days)-1, 8, dtype=int), labels=pd.Series(days).iloc[np.linspace(0, len(days)-1, 8, dtype=int)].dt.strftime('%Y-%m'))
plt.xlabel("Date")
plt.title("Time-Varying Correlation")
plt.ylabel("Correlation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
plt.show()
