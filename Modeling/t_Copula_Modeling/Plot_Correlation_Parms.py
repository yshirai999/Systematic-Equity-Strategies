import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import permutations

from Data.DataProcessing import data

SAVE_PATH = os.path.join(PROJECT_ROOT, "Modeling", "t_Copula_Modeling", "results", "correlation_matrices")
PLOT_DIR = os.path.join(PROJECT_ROOT, "Modeling", "t_Copula_Modeling", "results", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Load days and returns data
window = 100
nu = 6
data_obj = data()
days = data_obj.DataETFsReturns['days'].values
days = days[window:]

PLOT_PATH = os.path.join(PLOT_DIR, f"corr_over_time_w{window}.png")

# Load correlation tensor
corr_tensor = np.load(os.path.join(SAVE_PATH, f"corr_matrix_w{window}.npy"))  # shape (T, N, N)

# Check positive semidefiniteness of each correlation matrix
def is_positive_semidefinite(matrix, tol=1e-8):
    eigvals = np.linalg.eigvalsh(matrix)
    return np.all(eigvals >= -tol), eigvals

non_psd_days = []

for t in range(corr_tensor.shape[0]):
    is_psd, eigvals = is_positive_semidefinite(corr_tensor[t])
    if not is_psd:
        non_psd_days.append((t, eigvals.min()))

if non_psd_days:
    print(f"{len(non_psd_days)} non-PSD matrices found:")
    for day, min_eig in non_psd_days:
        print(f" - Day {day}: min eigenvalue = {min_eig}")
else:
    print("All correlation matrices are positive semidefinite.")

# Plot correlations over time
tickers = ['spy', 'xle', 'xlf', 'xli', 'xlk', 'xlp', 'xlu', 'xly']
pairs = [('spy', 'xle')]
#pairs = np.array(list(permutations(tickers, 2)))
indices = [(tickers.index(i), tickers.index(j)) for i, j in pairs]
#print(f"Plotting {len(indices)} pairs: {pairs}, indices: {indices}")
T = corr_tensor.shape[0]
for (i, j), label in zip(indices, pairs):
    print(f"Plotting pair: {tickers[i]}, {tickers[j]}")
    plt.plot(range(T), (2/np.pi)*np.arcsin(corr_tensor[:, i, j]), label=f"{label[0]}-{label[1]}")

plt.xticks(ticks=np.linspace(0, len(days)-1, 8, dtype=int), labels=pd.Series(days).iloc[np.linspace(0, len(days)-1, 8, dtype=int)].dt.strftime('%Y-%m'))
plt.xlabel("Date")
plt.title(f"Time-Varying Correlation with window = {window}")
plt.ylabel("Correlation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
plt.show()
