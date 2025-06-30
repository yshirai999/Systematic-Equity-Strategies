import numpy as np
import cvxpy as cp
from dsp import SaddlePointProblem, MinimizeMaximize, inner
from constraints_utils import MMV_Phi


class DSPOptimizer:
    def __init__(self, R, lam=2.0, theta=0.75, alpha=1.25, beta=0.25):
        self.R = R
        self.J, self.M = R.shape
        self.theta = theta
        self.alpha = alpha
        self.beta = beta
        self.lam = lam

        self.p = cp.Variable(self.M)
        self.z = cp.Variable(self.J)

        self.a = np.linspace(0.01, 0.99, 25)
        self.Phi = MMV_Phi(self.a, lam)


    def rebate(self, z):
        """r(z) = theta z^{1+alpha} + (1-theta) z^{-(1+beta)}"""
        return self.theta * cp.power(z, 1 + self.alpha) + (1 - self.theta) * cp.power(z, -(1 + self.beta))

    def build_problem(self):
        """Build DSP-style saddle point problem."""
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

    def solve(self, verbose=False):
        problem = self.build_problem()
        problem.solve(verbose=verbose)
        return self.p.value, self.z.value, problem.value


