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
We model the marginal distribution of each ETF's return using the Bilateral Gamma distribution. This captures:

Skewed, heavy-tailed behavior

Non-Gaussian joint dependence structures

Regime shifts via bootstrapped or filtered data

Estimation is done via a differentiable quantile-matching loss, weighted by an Anderson–Darling-style tail emphasis.

### Assets Modeled
Calibrated ETFs include:

SPY, XLB, XLE, XLF, XLI, XLK, XLU, XLV, XLY

Each has its own parameter file under BG_Modeling/estimates/.

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

