import sys, os
sys.path.append(os.path.abspath('../..'))

import cvxpy as cp

from Optimizers.DSP.simulate_joint import JointReturnSimulator

class MeanCVaROptimizer(JointReturnSimulator):
    def __init__(self,
                tickers=['spy', 'xlb', 'xle', 'xlf', 'xli', 'xlk', 'xlu', 'xlv', 'xly'],
                window=100,
                J=10000,
                df=6,
                lam=0.75,
                target_return=0.10,
                **kwargs):
        super().__init__(tickers=tickers, window=window, J=J, df=df)
        self.lam = lam
        self.target_return = target_return
        self.weights_ = None

    def solve(self, verbose=False, date_str='2020-03-15'):
        # Step 1: simulate joint returns via inherited method
        R = self.simulate_t_Copula(date_str=date_str)  # shape: (J, M)
        T, N = R.shape

        # Step 2: define variables
        w = cp.Variable(N)
        eta = cp.Variable()
        z = cp.Variable(T)

        # Step 3: CVaR loss and constraints
        portfolio_loss = -R @ w
        constraints = [
            cp.sum(w) == 1,
            w >= -0.05,  # Max 5% short per position (institutional limit)
            cp.sum(cp.abs(w)) <= 1.6,  # 160% gross exposure (130/30 strategy)
            z >= portfolio_loss - eta,
            z >= 0,
        ]

        if self.target_return is not None:
            constraints.append(R.mean(axis=0) @ w >= self.target_return)

        objective = cp.Minimize(eta + (1 / ((1 - self.lam) * T)) * cp.sum(z))
        prob = cp.Problem(objective, constraints)
        prob.solve(verbose=verbose)

        # self.weights_ = w.value
        # return ([],[],self.weights_)
    
        self.p = w  # rename to match DSP style
        self.z = z  # z is used in both
        self.target_val = prob.value

        return self.p.value, self.z.value, self.target_val

