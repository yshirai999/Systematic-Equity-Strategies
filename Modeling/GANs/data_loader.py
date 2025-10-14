import torch
from torch.utils.data import Dataset
import numpy as np
    
class MacroGANData(Dataset):
    def __init__(self, data_object, ell=9, latent_dim=30, window=60):
        self.ell = ell
        self.latent_dim = latent_dim
        self.window = window  # total window size = ell + latent_dim

        self.returns = data_object.DataETFsReturns.set_index('days')
        self.macro = data_object.DataMacro
        self.all_dates = data_object.DataETFsReturns['days'].values
        self.assets = self.returns.columns
        self.macro_vars = self.macro.columns

        # Join and align data
        joined = self.returns.join(self.macro, how='inner')

        self.X = []  # [ell, num_assets]
        self.Y = []  # [latent_dim, num_assets]
        self.M = []  # macro vector
        self.dates = []  # date corresponding to last day in X

        for i in range(window, len(joined) - latent_dim):
            x = joined[self.assets].iloc[i - window:i - latent_dim].values.T  # shape [num_assets, window]
            y = joined[self.assets].iloc[i:i + latent_dim].values.T          # shape [num_assets, latent_dim]
            m = joined[self.macro_vars].iloc[i].values                       # shape [macro_dim]

            self.X.append(torch.tensor(x, dtype=torch.float32).T)  # [window, num_assets]
            self.Y.append(torch.tensor(y, dtype=torch.float32).T)  # [latent_dim, num_assets]
            self.M.append(torch.tensor(m, dtype=torch.float32))    # [macro_dim]
            self.dates.append(joined.index[i])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.M[idx], self.Y[idx], self.dates[idx]


