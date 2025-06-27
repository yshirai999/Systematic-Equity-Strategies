# Multivariate Investments — BG Modeling and Systematic Analysis

This repo implements a Clayton's copula with Bilateral Gamma marginals for joint ETFs return distribution, with a focus on robust, systematic signal extraction.

## Project Structure

```bash
Multivariate-Investments/
│
├── BG_Modeling/
│   ├── fit_BG.py, Models.py      # Core MBG fitting logic
│   ├── config.yaml               # Model + asset config
│   ├── estimates/                # Calibrated MBG parameter files
│   ├── theta_checkpoints/       # Model training checkpoints
│
├── Data/
│   └── tsd180.mat                # Source return data
│
├── Deprecated/                  # Archived experimental code
│
├── MutlivariateFIR.ipynb        # FIR-based signal extraction
├── readme.md                    # This file
```

## Methodology

We model the marginal distribution of each ETF's return using the Bilateral Gamma distribution to capture skewed, heavy-tailed behavior.

Estimation is done via a differentiable tail-matching loss with Anderson–Darling weights.

### Assets Modeled

Calibrated ETFs include:

SPY, XLB, XLE, XLF, XLI, XLK, XLU, XLV, XLY

Each asset’s return distribution is modeled using a Bilateral Gamma density, calibrated to daily returns. Parameters are stored under BG_Modeling/estimates/.

The estimation procedure features:

+ Autograd-differentiable loss defined in PyTorch

+ Loss is computed via quantile-based scoring

+ Theoretical quantiles are derived from the BG pdf, reconstructed using torch.ifft from analytically known characteristic function

+ Tail-sensitive weighting based on the Anderson–Darling criterion

+ Optimization via Adam, combined with backtracking line search for stability

+ Batch GPU implementation enabling high-throughput calibration:

+ Over 15 years of daily data can be fit in under 15 minutes on standard GPU setups for most ETFs

+ Achieves accuracy on par with and often exceeding classical Nelder–Mead optimization

### Visuals (Optional Section)

(You can drop in a few thumbnails here if you want to showcase fit quality or tail calibration — let me know if you'd like me to prepare this.)

## Running the Code

### Set up environment

conda env create -f BG_Modeling/environment.yml
conda activate mbg-env

### Run calibration for SPY

python BG_Modeling/fit_BG.py --asset SPY --config config.yaml
🔮 Future Work
Planned extensions include:

Copula-based joint dependence modeling (Gaussian, t, Clayton)

Implementation of Dynamic Saddle Programming to solve max-min optimization problem

