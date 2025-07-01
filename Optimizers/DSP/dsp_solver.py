import sys, os
sys.path.append(os.path.abspath('../..'))

import numpy as np
import cvxpy as cp
from dsp import SaddlePointProblem, MinimizeMaximize, inner

from Optimizers.DSP.constraints_utils import MMV_Phi
from Optimizers.DSP.simulate_joint import JointReturnSimulator

class DSPOptimizer(JointReturnSimulator):
    """DSP-style optimizer for joint returns."""
    def __init__(self,
                tickers=['spy', 'xlb', 'xle', 'xlf', 'xli', 'xlk', 'xlu', 'xlv', 'xly'],
                window=100,
                J=10000,
                df=6,
                lam=0.75,
                theta=0.5,
                alpha=1.25,
                beta=0.25):
        super().__init__(tickers=tickers, window=window, J=J, df=df)

        self.theta = theta
        self.alpha = alpha
        self.beta = beta
        self.lam = lam
        self.a = np.linspace(0.01, 0.99, 25)
        self.Phi = MMV_Phi(self.a, lam)


    def rebate(self, z):
        """r(z) = theta z^{1+alpha} + (1-theta) z^{-(1+beta)}"""
        return self.theta * cp.power(z, 1 + self.alpha) + (1 - self.theta) * cp.power(z, -(1 + self.beta))

    def build_problem(self, date_str='2020-03-15'):
        """Build DSP-style saddle point problem."""
        self.R = self.simulate_t_Copula(date_str=date_str)  # shape: (J, M)
        self.J, self.M = self.R.shape
        self.p = cp.Variable(self.M)
        self.z = cp.Variable(self.J)

        payoff = self.R @ self.p                  # (J,)
        expected = inner(self.z, payoff) / self.J
        rebate = cp.sum(self.rebate(self.z)) / self.J
        objective = MinimizeMaximize(expected + rebate)

        constraints = [
            cp.sum(self.z) / self.J == 1,
            self.z >= 0
        ]
        for ak, phik in zip(self.a, self.Phi):
            constraints.append(cp.sum(cp.pos(self.z - ak)) / self.J <= phik)

        return SaddlePointProblem(objective, constraints)

    def solve(self, verbose=False, date_str='2020-03-15'):
        problem = self.build_problem(date_str=date_str)
        problem.solve(verbose=verbose)
        return self.p.value, self.z.value, problem.value


