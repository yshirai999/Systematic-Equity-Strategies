import numpy as np
import os
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt

class data():

    def __init__(self, tickers = ['spy', 'xlb', 'xle', 'xlf', 'xli', 'xlk', 'xlu', 'xlv', 'xly']):
        
        # === Load ETF Data ===
        self.tickers = tickers
        base_path = os.path.dirname(__file__)
        mat_path = os.path.join(base_path, 'tsd180.mat')
        mat = loadmat(mat_path)
        self.d = mat['days']
        self.p = mat['pm']
        self.n = mat['nmss']
        [N,M] = np.shape(self.p)
        self.N = N
        self.M = M
        self.Data = pd.DataFrame({'days': self.d[:,0]})
        for i in range(M):
            Datai = pd.DataFrame({self.n[i,0][0] : self.p[:,i]})
            self.Data = pd.concat([self.Data,Datai],axis=1)

        self.DataETFs = self.Data[self.tickers]
        self.DataETFsReturns = self.DataETFs.pct_change()
        self.DataETFsReturns = self.DataETFsReturns.drop(index = 0)
        self.DataETFsReturns.insert(0, 'days', self.d[1:])
        self.DataETFs.insert(0, 'days', self.d)
        self.DataETFsReturns['days'] = pd.to_datetime(self.DataETFsReturns['days'], format='%Y%m%d')

        # === Load Macro Data ===
        macro_path = os.path.join(base_path, 'macro_factors_fred.mat')
        macro_mat = loadmat(macro_path)

        macro_dates = pd.to_datetime(macro_mat['dates'].ravel(), format='%Y-%m-%d')
        macro_df = pd.DataFrame({'days': macro_dates})
        for key in macro_mat:
            if key not in ['__header__', '__version__', '__globals__', 'dates']:
                macro_df[key] = macro_mat[key].ravel()

        self.DataMacro = macro_df.set_index('days')

    def Returns_Visualization(self, ticker: str):
        if ticker not in self.tickers:
             print('Error: ticker should be one of:', self.tickers)
             return
        x = self.DataETFsReturns['days']
        y = self.DataETFsReturns[ticker]

        # Create the plot
        plt.figure(figsize=(10, 5))
        plt.plot(x, y, label='spy', color='blue', linestyle='-', linewidth=0.1, marker='o', markersize=1)

        # Add labels and title
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')
        plt.title('SPY daily return')

        # Add legend
        plt.legend()

        # Add grid
        plt.grid(True)

        # Show the plot
        plt.show()

    def plot_macro_data(self):

        if not hasattr(self, 'DataMacro') or self.DataMacro is None:
            raise ValueError("DataMacro not loaded. Please initialize the class correctly with macro data.")

        for col in self.DataMacro.columns:
            plt.figure(figsize=(10, 4))
            plt.plot(self.DataMacro.index, self.DataMacro[col], label=col, linewidth=1.5)
            plt.title(f"{col.upper()} over Time")
            plt.xlabel("Date")
            plt.ylabel(col)
            plt.grid(True)
            plt.tight_layout()
            plt.legend()
            plt.show()



