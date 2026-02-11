```markdown
# B4 - Macro Regime Detection 📊

**Status:** ✅ Complete  
**Version:** 1.0.0  
**Date:** February 12, 2026

---

## Overview

B4 detects market regimes (Bull/Normal/Crisis) using Hidden Markov Models on FRED macroeconomic data.

## What It Does

- Downloads 6 FRED economic indicators (VIX, GDP, Unemployment, Fed Funds, Treasury Yields, CPI)
- Engineers 4 features from macro data
- Fits a 3-state Hidden Markov Model
- Predicts regime labels for each month (2015-2024)
- Saves regime predictions to CSV

## Results

**Data Processed:** 120 months (Jan 2015 - Dec 2024)

**Regime Distribution:**
- **Regime 0 (Expansion):** 21.7% of time
- **Regime 1 (Transition):** 10.0% of time  
- **Regime 2 (Normal/Stable):** 68.3% of time

## How to Run

```bash
# Activate environment
conda activate quantforge

# Run pipeline
python projects/b4_macro_regimes/pipeline.py

# Results saved to:
# data/regimes/regime_labels_YYYYMMDD_HHMMSS.csv
```

## Files

- `config.yaml` - Configuration
- `data_loader.py` - FRED data downloader
- `hmm_model.py` - Hidden Markov Model
- `pipeline.py` - Main orchestrator

## Next Steps

**B5:** Integrate regimes with B3 baseline strategies (regime-conditioned trading)

---

**Author:** Joshua Jose  
**Email:** joshua.jose2002@gmail.com
```