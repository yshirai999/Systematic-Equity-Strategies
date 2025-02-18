import numpy as np
from scipy.io import loadmat
import pandas as pd
from sklearn.decomposition import FastICA
# import matlab.engine
# eng = matlab.engine.start_matlab()

mat = loadmat('tsd180.mat')
d = mat['days']
p = mat['pm']
n = mat['nmss']

Data = pd.DataFrame({'days': d[:,0]})
for i in range(np.shape(p)[1]):
    Datai = pd.DataFrame({n[i,0][0] : p[:,i]})
    Data = pd.concat([Data,Datai],axis=1)

DataETFs = Data[['spy', 'xle', 'xlf', 'xli', 'xlk', 'xlp', 'xlu', 'xlv', 'xly', 'xom', 'xrx']]
DataETFsReturns = DataETFs.diff()
DataETFsReturns = DataETFsReturns.drop(index = 0)
DataETFsReturns.insert(0, 'days', d[1:])
DataETFsReturns['days'] = pd.to_datetime(DataETFsReturns['days'], format='%Y%m%d')

X = DataETFsReturns.iloc[:,1:].values


transformer = FastICA(n_components=6,
        random_state=0,
        whiten='unit-variance')
X_transformed = transformer.fit_transform(X[:21])