import numpy as np
from scipy.io import loadmat
import pandas as pd
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

class data():

    def __init__(self, tickers = ['spy', 'xle', 'xlf', 'xli', 'xlk', 'xlp', 'xlu', 'xlv', 'xly', 'xom', 'xrx']):
        self.tickers = tickers
        mat = loadmat('tsd180.mat')
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
        self.DataETFsReturns = self.DataETFs.diff()
        self.DataETFsReturns = self.DataETFsReturns.drop(index = 0)
        self.DataETFsReturns.insert(0, 'days', self.d[1:])
        self.DataETFsReturns['days'] = pd.to_datetime(self.DataETFsReturns['days'], format='%Y%m%d')
    
    def ICA(self, max_iter = 500, tol = 0.1):
        X = self.DataETFsReturns.iloc[:,1:].values
        transformer = FastICA(n_components=10,
            random_state=0,
            whiten='unit-variance',
            max_iter = max_iter,
            tol = tol)
        X_transformed = {}
        W = {}
        A = {}
        lookbackperiod = 63
        Failed = list()
        for i in range(lookbackperiod,self.N):
            X_transformed[self.DataETFsReturns['days'][i]] = transformer.fit_transform(X[i-lookbackperiod:i])
            W[self.DataETFsReturns['days'][i]] = transformer.components_
            A[self.DataETFsReturns['days'][i]] = transformer.mixing_
            if transformer.n_iter_ >= max_iter:
                Failed.append(self.d[i])
        return [W,A, Failed]

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

    def BasicInvesting_CumReturn(self, ticker: str):
        x = self.DataETFsReturns['days']
        y = self.DataETFsReturns[ticker].cumsum()
        for i in range(1,self.N):
            y[i:] += self.DataETFsReturns[ticker][i:].cumsum()
        # Create the plot
        plt.figure(figsize=(10, 5))
        plt.plot(x, y, label='spy', color='blue', linestyle='-', linewidth=0.1, marker='o', markersize=1)

        # Add labels and title
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')
        plt.title('Cummulated returns on basic strategy on '+ticker)

        # Add legend
        plt.legend()

        # Add grid
        plt.grid(True)

        # Show the plot
        plt.show()



