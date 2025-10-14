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
        self.a = np.linspace(0.1, 1, 10)
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
            self.z >= 0, 
            cp.sum(self.p) == 1, 
            self.p >= 0                   # Budget constraint
            # cp.sum(cp.abs(self.p)) <= 2.0         # Max 200% gross exposure
        ]
        for ak, phik in zip(self.a, self.Phi):
            constraints.append(cp.sum(cp.pos(self.z - ak)) / self.J <= phik)

        return SaddlePointProblem(objective, constraints)

    def solve(self, verbose=False, date_str='2020-03-15', **solver_kwargs):
        """
        Solve the DSP optimization problem with configurable solver parameters.
        
        Parameters:
        -----------
        verbose : bool, default=True
            Print solver output
        date_str : str, default='2020-03-15'  
            Date for return simulation
        solver : cvxpy solver, optional
            Specific solver to use (cp.CLARABEL, cp.ECOS, cp.SCS, etc.)
        **solver_kwargs : dict
            Solver-specific parameters
            
        Available solvers and their key parameters:
        ------------------------------------------
        CLARABEL (default, best for DSP):
            - eps: Primary/dual infeasibility tolerance (default: 1e-8)
            - max_iter: Maximum iterations (default: 200)
            - time_limit: Time limit in seconds
            
        ECOS (good alternative):
            - abstol: Absolute tolerance (default: 1e-7)
            - reltol: Relative tolerance (default: 1e-6) 
            - feastol: Feasibility tolerance (default: 1e-7)
            - max_iters: Maximum iterations (default: 100)
            
        SCS (robust for difficult problems):
            - eps: Convergence tolerance (default: 1e-4)
            - max_iters: Maximum iterations (default: 2500)
            - alpha: Relaxation parameter (default: 1.5)
            
        Example usage:
        --------------
        # High accuracy with CLARABEL
        w, z, val = solver.solve(eps=1e-10, max_iter=1000)
        
        # Use ECOS solver  
        w, z, val = solver.solve(solver=cp.ECOS, abstol=1e-9, max_iters=500)
        
        # Use SCS for robustness
        w, z, val = solver.solve(solver=cp.SCS, eps=1e-6, max_iters=5000)
        """

        # Warnings about ECOS solver: see C:\Users\yoshi\anaconda3\envs\dsp_equity_strategy\lib\site-packages\cvxpy\reductions\solvers\solving_chain.py:353

        problem = self.build_problem(date_str=date_str)
        
        # Set default parameters to balance accuracy and convergence
        # eps=1e-1 gives excellent accuracy while avoiding convergence issues
        default_solver_kwargs = {
            'solver': cp.ECOS#,
            #'eps': 1e-4,      # Relaxed from 1e-8 to avoid "inaccurate solution" warnings
        }
        
        # Update with any user-provided parameters
        default_solver_kwargs.update(solver_kwargs)
        
        problem.solve(verbose=verbose, **default_solver_kwargs)
        return self.p.value, self.z.value, problem.value


