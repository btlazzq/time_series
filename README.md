# TempoStackNet - Cryptocurrency Price Prediction Framework

This repository contains the code implementation for *TempoStackNet*, a novel framework designed for multi-step cryptocurrency price prediction. The research introduces an advanced stacking-based ensemble model with multi-timescale temporal attention, aimed at enhancing the accuracy of long-term financial forecasts. By integrating a cross-time window strategy, TempoStackNet efficiently processes historical cryptocurrency data to forecast price trends over a 32-day horizon.

### Features:
- **Multi-Timescale Temporal Attention**: Leverages time windows of 192, 96, and 48 intervals to capture diverse temporal patterns.
- **Stacking Ensemble Learning**: Combines predictions from multiple models to improve robustness and accuracy.
- **Validated Across Datasets**: Tested on the *Global Tech and Crypto Volatility* dataset (Bitcoin, Ether,Dogecoin) and the S&P 500 dataset, achieving high R² and low RMSE values.

### Usage:
1. Clone the repository.
2. Install dependencies listed in `requirements.txt`.
3. Run `main.py` to start training and evaluation.

### Requirements:
Ensure that the following dependencies are installed before running the framework:
1. Clone the repository:
   ```bash
   git clone https://github.com/btlazzq/time_series.git
   ```
2. Install dependencies from the requirements.txt file:
   ```bash
   pip install -r requirements.txt
   ```
