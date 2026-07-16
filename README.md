# Vehicle Insurance Fraud Detection

An end-to-end machine learning project to detect fraudulent vehicle insurance claims, built to help insurance claims teams prioritize manual review effort instead of screening every claim equally.

## Business Problem

Insurance claims teams currently review every incoming claim manually to check for fraud — slow, costly, and inconsistent. This project builds a model that flags claims with a high likelihood of fraud, so review effort can be focused on high-risk cases.

A key business constraint: falsely flagging an honest customer as fraudulent is more costly than missing some fraud cases. So the model is designed to feed into a **human review queue**, not an automated rejection system, and precision is weighed carefully against recall rather than optimizing accuracy alone.

## Dataset

- **Source**: `fraud_oracle.csv`
- **Size**: 15,420 claims, 32 raw features
- **Target**: `FraudFound_P` (binary) — highly imbalanced, ~94% non-fraud vs ~6% fraud

## Project Workflow

1. **Business Problem** — defined the objective and the precision-vs-recall trade-off based on business cost.
2. **Data Cleaning** — removed 1 invalid row, identified 320 placeholder `Age=0` values for imputation, preserved pre-binned range columns as ordinal categories.
3. **Exploratory Data Analysis** — uncovered key fraud drivers: fault type, policy coverage level (All Perils vs Liability), and policyholder age. Also identified noisy/low-signal features (rare vehicle makes, numeric columns with near-zero correlation).
4. **Feature Engineering** — grouped rare categories, imputed missing ages, ordinal-encoded ranked categories, one-hot encoded nominal categories, dropped ID columns. Final feature set: 88 columns.
5. **Feature Selection & Scaling** — stratified train/test split, standard scaling for linear models, class-weighting to address imbalance.
6. **Model Training & Evaluation** — compared Logistic Regression, Random Forest, and XGBoost with decision threshold tuning.
7. **Business Recommendation** — final model deployed as a flagging system for manual review, with fraud-driver insights to help prioritize the review queue.

## Model Results

| Model | Precision (Fraud) | Recall (Fraud) | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Logistic Regression (thr 0.5) | 0.13 | 0.88 | 0.81 | – |
| Random Forest (thr 0.2) | 0.31 | 0.19 | 0.84 | 0.24 |
| **XGBoost (thr 0.5) — Final** | 0.20 | 0.52 | 0.84 | 0.24 |

**Final model: XGBoost (threshold 0.5)** — chosen for the best balance between catching fraud and controlling false alarms, aligned with the business priority defined above.

## Key Fraud Indicators Found

- Claims where the **policy holder is at fault** (7.89% fraud rate vs 0.88% for third-party fault)
- **High-coverage policies** ("All Perils" — 10.16% fraud rate vs 0.72% for "Liability")
- **Younger policyholders** (18–25 age group had the highest fraud rates)

## Business Recommendation

Deploy the XGBoost model as a claim-flagging system feeding a human review queue, not an automated decision system. The model catches roughly half of all fraud cases while keeping the false-alarm rate low enough to meaningfully reduce manual review workload. Review queues should be prioritized using the fraud indicators above, and the model should be retrained periodically as fraud patterns evolve.

## Tech Stack

- Python, Pandas, NumPy
- Scikit-learn (Logistic Regression, Random Forest, preprocessing)
- XGBoost
- Matplotlib / Seaborn (EDA)

## Repository Structure

```
├── fraud_oracle.csv                # dataset
├── Fraud Detection.ipynb                  # full analysis and modeling
├── Vehicle_Insurance_Fraud_Detection_Report.docx   # written project report
└── README.md
```

