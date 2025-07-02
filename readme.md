# An End-to-End Pipeline for Systematic Equity Investing

This repo implements a t-copula with Bilateral Gamma (BG) marginals for joint ETFs return distribution, with a focus on robust, systematic signal extraction.

## Project Structure

```bash
Systematic-Equity-Strategies/
│
├── readme.md                       # This file - project overview
├── environment.yml                 # Conda environment specification
├── .gitignore                      # Git ignore patterns
├── .git/                           # Git repository files
│
├── Data/                           # Data processing and source files
│   ├── DataProcessing.py           # Basic data class inherited by BG class
│   ├── tsd180.mat                  # Source return data (MATLAB format)
│
├── Modeling/                        # Return distribution modeling
│   ├── readme.md                  # Modeling overview and methodology
│   ├── BG_Modeling/               # Bilateral Gamma distribution fitting
│   │   ├── fit_BG.py              # Main fitting script
│   │   ├── fit_BG.ipynb           # Interactive notebook for BG fitting
│   │   ├── Models.py              # BG class definition and methods
│   │   ├── config.yaml            # Configuration parameters
│   │   ├── estimates/             # Fitted model parameters
│   │   └── theta_checkpoints/     # Additional training checkpoints
│   ├── GANs/                      # Generative Adversarial Networks (experimental)
│   │   └── CMacro-GAN             # Conditional Macro GAN implementation
│   └── t_Copula_Modeling/         # t-Copula dependence modeling
│       ├── fit_t_Copula.py        # Main correlation estimation script
│       ├── Plot_Correlation_Parms.py  # Visualization of correlations
│       ├── t_copula.py            # t-Copula class implementation
│       ├── utils.py               # Utility functions (Archakov-Hansen, etc.)
│       └── results/               # Fitted correlation matrices and plots
│
├── Optimizers/                     # Portfolio optimization methods
│   └── DSP/                       # Dynamic Saddle Programming
│       ├── dsp_solver.py          # General DSP problem class
│       ├── constraints_utils.py   # Builds Phi(a) and a-grid
│       ├── simulate_joint.py      # Joint return simulation from BG + t-Copula
│       └── Tests.ipynb            # Testing and validation notebook
│
├── Backtesting/                    # Strategy backtesting framework
│   ├── Backtesting.py             # Main backtesting class
│   ├── run_parallel_backtest.py   # Parallel backtest execution
│   ├── analyze_performance.py     # Performance analysis and metrics
│   └── Results/                   # Backtest results storage
│       ├── backtest_results_0_4330_50.npy
│       ├── backtest_results_0_4330_75.npy
│       └── ... (other result files)
│
└── Deprecated/                     # Archived experimental code
```

## Methodology

We model the marginal distribution of each ETF's return using the Bilateral Gamma distribution to capture skewed, heavy-tailed behavior.

Optimal portfolio weights are then constructed by maximizing the worst case expected return with rebates for scenarios that are far from the base case

A systematic strategy is then implemented by rebalancing the portfolio weights every 20 business days based on the maxmin solution

## Results

So far, no conclusive evidence that investing in the portfolio thus constructed provides superior performance with respect to a simple buy and hold strategy on SPY

![Rolling Sharpe Ratio Comparison](Backtesting/Results/rolling_sharpe_ratio_comparison.png)

## Running the Code

### Set up environment

```bash
conda env create -f environment.yml
conda activate mbg-env
```

## Future Work

Planned extensions include:

Implement Generative Adversarial Networks conditional on macro factors to improve the estimate of the joint distribution of return and to capture volatility clustering phenomena
