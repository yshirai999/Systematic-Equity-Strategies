# Optimizers

This folder contains the core optimization logic used for allocating portfolio weights based on distorted return scenarios. It includes tools for simulating returns under a joint distribution model and solving a max-min optimization problem designed to capture worst-case performance scenarios.

## Directory Structure

```bash
Optimizers/
│
├── readme.md                     # This file - optimization overview
└── DSP/                          # Dynamic Saddle Programming implementation
    ├── dsp_solver.py             # DSPOptimizer class for max-min optimization
    ├── simulate_joint.py         # JointReturnDistribution class for return simulation
    ├── constraints_utils.py      # Utility functions for optimization constraints
    ├── Tests.ipynb               # Testing and validation notebook
    └── __pycache__/              # Python cache files
```

## Contents

- **`simulate_joint.py`**  
  Contains the `JointReturnDistribution` class, responsible for simulating multivariate return vectors using a fitted joint distribution (e.g., t-Copula, Multivariate Bilateral Gamma, GAN-based conditional samplers, etc.).

- **`dsp_solver.py`**  
  Contains the `DSPOptimizer` Inherits from `JointReturnDistribution` and implements the core DSP (Disciplined Saddle Programming) solver. It:

  1. **Builds a max-min optimization problem**:
     - The goal is to compute portfolio weights that maximize the worst-case expected return.
     - The worst-case distribution is derived by distorting the empirical distribution of simulated returns. The degree and type of distortion are configurable (e.g., minmaxvar).

  2. **Solves the optimization**:
     - Uses numerical solvers based on the cvxpy library to compute optimal portfolio weights.
     - Enforces constraints that ensure lower weights are given to extreme scenarios

## Usage

This module is called during backtesting to:

1. Fit a joint return model using recent return history.

2. Simulate a large number of return scenarios.

3. Solve for portfolio weights using the DSP procedure.

4. Apply the resulting weights to compute realized portfolio returns.

See the `Backtesting/` folder for examples of how this optimizer is integrated into the full pipeline.

## Notes

- The optimizer is modular and designed to accommodate different return simulation models (copulas, GANs, factor models).

- Future extensions include regularized versions of DSP, and a PyTorch implementation of the maxmin problem solution via Lagrangian duality
