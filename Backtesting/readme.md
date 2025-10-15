# Backtesting Module

This folder contains the code responsible for evaluating strategy performance using historical data. It includes the core logic for executing dynamic strategies and measuring relevant performance metrics.

## Directory Structure

```bash
Backtesting/
│
├── readme.md                        # This file - backtesting overview
├── Backtesting.py                   # DSPBacktester class implementation
├── run_parallel_backtest.py         # Parallel backtest execution logic
├── analyze_performance.py           # Performance analysis and metrics
└── Results/                         # Backtest results and visualizations
    ├── backtest_results_0_4330_50.npy  # Backtest results (lambda=50)
    ├── backtest_results_0_4330_75.npy  # Backtest results (lambda=75)
    └── rolling_sharpe_ratio_comparison.png  # Performance visualization
```

## Contents

- **Backteststing.py**: Implements the DSPBacktester class, which inherits from the DSPOptimizer class in the Optimizers folder; applies the optimized weights from the DSP procedure to historical asset returns to simulate portfolio performance over time.
- **run_parallel_backtest.py**: uses parallel computation logic to isntantiate several DSPBacktester objects and compute optimized weights for each trading day
- **analyze_performance.py**: Outputs key performance data and plots (e.g., rolling Sharpe ratios) for post-analysis.

## Key Features

- Supports dynamic allocation strategies (e.g., from DSPOptimizer)

- Incorporates a rolling window for simulating realistic strategy updates

- Generates comparison benchmarks (e.g., SPY Buy-and-Hold)

## Notes

The backtests assume daily rebalancing and self-financing strategies (weights sum to 1 and no cash injection). The strategy's 1-year rolling Sharpe ratio is reported in the root folder README and is visualized to assess the strategy’s stability over time.

## Future Enhancements

- Add transaction cost modeling

- Incorporate multiple strategies with parallel tracking

- Allow benchmark flexibility beyond SPY

---
For a full pipeline overview, see the root-level [`README`](../README.md). For modeling logic, refer to [`Modeling/README.md`](../Modeling/readme.md).
