# Finding an Edge in Prediction Markets
### ML-based Jump Detection on Kalshi

![Project Status: Complete](https://img.shields.io/badge/Status-Complete-brightgreen)
![Python](https://img.shields.io/badge/Made%20with-Python-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

This project studies whether short-horizon price movements in Kalshi prediction markets can be predicted from trade-level microstructure data. The problem is framed as a three-class classification task: given a sequence of trades, predict whether the contract `yes_price` will move **down**, stay **flat**, or move **up** over 5, 15, 30, and 60 minute horizons.

📄 [Read the paper](finding_an_edge_kalshi.pdf)

**Authors:** Lennard Pische (Harvard SEAS), Gianluca Pisa, Andres Blanco Prada, Moritz Wassermann, Vishwesh Venkatramani (MIT Sloan)

---

## Motivation

Prediction markets aggregate beliefs about future events, but their microstructure is sparse, noisy, and highly imbalanced. Most trades do not precede a large move, while directional jumps are rare and often concentrated in specific tickers or event regimes. The central question is whether modern time-series and tabular models can extract useful signal from this setting, and whether combining models via a learned mixture improves robustness.

---

## Data

The pipeline starts from Kalshi trade records covering January 2025 onward. After feature engineering and windowing, the final Mixture of Experts dataset contains precomputed trade features, labels, and base-model probability outputs. Data is split chronologically: **34,956,173 training rows** and **8,593,950 held-out test rows**. A validation fold is carved from the end of the training period for early stopping.

Target labels are built from forward price movements. For each horizon, a price jump is classified as down, flat, or up based on whether the future return crosses training-fold directional thresholds.

---

## Modeling Approach

The report evaluates a progression of models:

- **Baseline** — 1D CNN over trade-level windows (32 trades × 6 features)
- **Individual experts** — LightGBM, LSTM, Mamba SSM, Moirai backbone classifier, FT-Transformer, CNN and Transformer Time-Series (CTTS)
- **Mixture of Experts (MoE)** — a PyTorch gating network that learns how much to trust each expert per trade and per horizon, combining the six expert probability vectors with trade-level context features into a final three-class prediction

---

## Main Results

The final MoE model improves as the prediction horizon lengthens. On the held-out test set:

| Horizon | Balanced Accuracy |
|---|---:|
| 5 minutes  | 0.495 |
| 15 minutes | 0.512 |
| 30 minutes | 0.540 |
| 60 minutes | 0.548 |

At the 5-minute horizon the model reaches 0.676 overall accuracy and 0.473 macro F1 on 8.59M test trades. The flat class is easiest to identify because it dominates the data; directional jumps remain substantially harder, as visible in the per-class classification reports.

The learned gate is not uniform. At the 5-minute horizon, average test-set gate weights are approximately **0.503 for LightGBM** and **0.274 for CTTS**, with smaller weights for LSTM, Mamba, FT-Transformer, and Moirai. The gating network leans heavily on the strongest tabular and sequence experts while preserving model diversity.

---

## Interpretation

Three main conclusions emerge:

1. Longer horizons contain more learnable signal than very short horizons — consistent across every model family.
2. Balanced metrics are more informative than raw accuracy given the severe class imbalance (~80% flat trades).
3. The MoE design is useful: different model families contribute differently across trades and horizons, and the gate weights make the ensemble more interpretable than a simple average.

---

## Limitations

Directional jumps are rare, labels depend on threshold choices, and liquidity varies sharply across markets. Sparse trading can create misleading windows, and models may learn price-level or liquidity heuristics rather than robust event-level structure. These are important caveats for applying the model outside the historical evaluation window.

---

## Repository Structure

```
kalshi-jump-detection/
├── main.py                  # Entry point — runs full pipeline
├── models/                  # Model paths (LightGBM, LSTM, Mamba, Moirai, CTTS, FT-Transformer, MoE) (**IN DRIVE**)
├── notebooks/               # Exploratory analysis and per-model training notebooks
├── finding_an_edge_kalshi.pdf 
└── README.md
```

---

## Artifacts

Large data files, trained model checkpoints, and preprocessed feature matrices are stored outside Git due to file size:

📁 [Google Drive — Data & Checkpoints](https://drive.google.com/drive/folders/1bwPVZxxEhNfPodRlXFBZV-RKXiuSnlkq?usp=sharing)

Contents include:
- `mamba_best.pt` — Mamba SSM classifier checkpoint
- `moirai_classifier_best.pt` — Moirai backbone classifier checkpoint
- `kalshi_trades_2025.parquet` — Preprocessed trade-level dataset
- `features_train.parquet` / `features_test.parquet` — Feature matrices
