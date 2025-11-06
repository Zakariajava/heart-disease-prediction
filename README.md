# Heart Disease Prediction — Logistic Regression (PyTorch)

A transparent, reproducible **baseline model** for predicting 10-year coronary heart disease (CHD) risk using the **Framingham Heart Study** dataset.

This project demonstrates a complete end-to-end pipeline:
data preprocessing → feature scaling → logistic regression training → evaluation → metrics saving.

---

## Project structure

heart-disease-prediction/
├── data/
│   └── framingham.csv           # Original dataset (or see instructions below)
├── notebooks/
│   └── 01_train_logistic_regression.ipynb
├── artifacts/                   # Generated outputs (ignored in Git)
│   ├── model.pt                 # Trained model weights
│   ├── metrics.json             # Metrics summary
│   ├── config.json              # Preprocessing + threshold
│   └── plots/                   # Visualization outputs
├── requirements.txt
└── README.md

---

## Environment setup

### Option 1 — Virtualenv (recommended)
python -m venv .venv
source .venv/bin/activate          # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

### Option 2 — Conda
conda create -n heart python=3.11
conda activate heart
pip install -r requirements.txt

---

## Dataset

**Framingham Heart Study** — a classic dataset for cardiovascular risk prediction.

| Column | Description |
|---------|--------------|
| `age`, `totChol`, `sysBP`, `BMI`, `glucose` | Continuous medical measures |
| `male`, `currentSmoker`, `prevalentHyp`, etc. | Binary indicators |
| `education` | Ordinal (1–4) |
| `TenYearCHD` | Target — 1 if CHD event within 10 years, else 0 |

---

## Preprocessing overview

All preprocessing is fitted **only on the training split** to avoid data leakage:
- **Missing values:** imputed (median for numeric, mode for categorical)
- **Outliers:** winsorized at 1st/99th percentiles
- **Skewed features:** log1p transform (`cigsPerDay`, `glucose`, `totChol`)
- **Scaling:** robust scaling → `(x − median) / IQR`
- **Missing indicators:** extra binary columns (e.g., `glucose_is_missing`)

Parameters are stored in `artifacts/preprocessing_params.json`.

---

## Model

Simple **logistic regression** implemented in PyTorch:
self.linear = nn.Linear(in_features, 1, bias=True)

Loss: **BCEWithLogitsLoss** with class imbalance weighting  
Optimizer: **Adam** (lr=1e-3, weight_decay=1e-4)  
Early stopping based on **validation PR-AUC**  
Threshold tuned on validation to **maximize F1-score**

---

## Metrics

Example baseline:

| Metric | Validation | Test |
|---------|-------------|------|
| ROC-AUC | ~0.72 | ~0.67 |
| PR-AUC  | ~0.40 | ~0.29 |
| Accuracy | ~0.68 | ~0.69 |
| F1-score | ~0.31 | ~0.30 |

Artifacts:
- model.pt → trained model weights  
- metrics.json → all metrics (train/val/test)  
- config.json → includes best threshold and preprocessing info  

---

## Visualizations

All plots (before and after scaling) are stored in:
artifacts/plots/
artifacts/plots_scaled_only/

They include:
- Class balance  
- Histograms (numeric features)  
- KDE distributions (by target)  
- Correlation heatmap  

---

## Run the notebook

Open Jupyter or VS Code and run:
notebooks/01_train_logistic_regression.ipynb

The notebook will:
1. Load the dataset  
2. Split train/val/test  
3. Fit preprocessing  
4. Train logistic regression  
5. Save all artifacts  

---

## Requirements

pandas>=2.0
numpy>=1.25
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.13
pyarrow>=14.0
torch>=2.1

---

## License

MIT License © 2025  
Feel free to use and modify for educational or research purposes.

---