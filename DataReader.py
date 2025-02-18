import numpy as np
from scipy.io import loadmat
import pandas as pd
import matlab.engine
eng = matlab.engine.start_matlab()
eng.BGDataReader(nargout = 0)
mat = loadmat('tsd180.mat')

d = mat['days']
p = mat['pm']
n = mat['nmss']
