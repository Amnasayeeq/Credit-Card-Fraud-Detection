# Credit-Card-Fraud-Detection-Kaggle
Data-Mining 
---
A complete data mining pipeline for detecting fraudulent credit card transactions, built on a real-world, highly imbalanced dataset of European cardholder transactions.

---
## Overview

Fraud is a costly problem for financial services firms — both in direct monetary losses and in customer trust. This project walks through an end-to-end approach to spotting fraudulent transactions:

- Loading and preprocessing the data
- Exploratory Data Analysis (EDA)
- Handling extreme class imbalance with **SMOTE**
- Training and comparing three models: **Logistic Regression**, **Random Forest**, and **XGBoost**
- Evaluating models with metrics suited to imbalanced data (Precision, Recall, F1, ROC-AUC, PR-AUC)
- Threshold optimisation and an unsupervised anomaly detection baseline (Isolation Forest)
- Business recommendations based on the findings
---
## Dataset

[Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
---
- 284,807 transactions from European cardholders over two days in September 2013
- Only 492 transactions (0.172%) are fraudulent — an extremely imbalanced dataset
- Most features (`V1`–`V28`) are anonymised via PCA for confidentiality
- Only three original features remain: `Time`, `Amount`, and `Class` (target: `1` = fraud, `0` = genuine)

The notebook downloads the dataset automatically via `kagglehub`, so no manual download is required.
---
## Pipeline

1. **Setup** — import libraries (pandas, numpy, scikit-learn, imbalanced-learn, XGBoost, matplotlib/seaborn/plotly)
2. **Load data** — pull the dataset from Kaggle via `kagglehub`
3. **Exploratory Data Analysis**
   - Class imbalance visualisation
   - Distribution of transaction amount and time
   - Feature distributions: fraud vs. genuine (V1–V28)
   - Correlation heatmap
4. **Preprocessing** — remove duplicates, scale `Amount`/`Time`, stratified 80/20 train/test split
5. **Class imbalance handling** — SMOTE applied to the training set only (test set stays imbalanced, as in real-world conditions)
6. **Model training** — Logistic Regression, Random Forest, XGBoost
7. **Model evaluation** — confusion matrices, ROC curves, Precision-Recall curves, summary metrics table
8. **Feature importance analysis** — comparing Random Forest and XGBoost
9. **Threshold optimisation** — sweeping decision thresholds to balance precision and recall
10. **Anomaly detection baseline** — Isolation Forest as an unsupervised comparison
11. **Summary dashboard** — consolidated visualisation of key results
12. **Conclusions and business recommendations**
---
## Key Findings

- **XGBoost was the best-performing model**, achieving the highest F1-Score and PR-AUC.
- **V14, V4, V12, and V17** were consistently the most important features across tree-based models.
- **SMOTE substantially improved model performance** by preventing models from defaulting to "predict everything as genuine."
- Fraudulent transactions tend to be **smaller in amount** and more likely to occur **during off-peak hours**.
- **Lowering the decision threshold to ~0.3** (from the default 0.5) meaningfully improves recall (catches more fraud) at an acceptable cost to precision.
---
## Business Recommendations

- Deploy **XGBoost** as the real-time transaction scoring model.
- Use a decision threshold of **~0.3** rather than the default 0.5.
- Apply a **tiered response**: auto-decline (>0.8 fraud probability), human review (0.3–0.8), auto-approve (<0.3).
- Retrain the model **quarterly** to account for model drift as fraud patterns evolve.
- Investigate **V14 and V4** with domain experts to better understand the underlying business drivers.
- Run the model **alongside existing rule-based systems** (e.g. velocity checks, geographic anomaly detection) rather than as a replacement.
---
## Limitations

- The dataset covers only two days of transactions from a single geographic region (Europe), which may limit generalisability.
- PCA-anonymised features limit direct business interpretability of the model's drivers.
- SMOTE-generated synthetic samples may not perfectly reflect real-world fraud patterns; alternatives like ADASYN or class weighting could be explored.
---
## Tech Stack

- **Data manipulation:** pandas, numpy
- **Visualisation:** matplotlib, seaborn, plotly
- **Modelling:** scikit-learn (Logistic Regression, Random Forest, Isolation Forest), XGBoost
- **Imbalance handling:** imbalanced-learn (SMOTE)
- **Dataset access:** kagglehub
---
## Getting Started

### Requirements

```bash
pip install numpy pandas matplotlib seaborn plotly tabulate scikit-learn imbalanced-learn xgboost kagglehub
```
---
### Running the notebook

1. Clone this repository
2. Open `Data_Mining_GH1039590_commented.ipynb` in Jupyter or Google Colab
3. Run all cells — the dataset will be downloaded automatically via `kagglehub` on first run
---
## References

1. [Credit Card Fraud Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. [SMOTE: Synthetic Minority Over-sampling Technique](https://doi.org/10.1613/jair.953)
3. [XGBoost: A Scalable Tree Boosting System](https://doi.org/10.1145/2939672.2939785)
4. [Random Forests](https://doi.org/10.1023/A:1010933404324)
5. [Scikit-learn: Machine Learning in Python](https://scikit-learn.org)
6. [Imbalanced-learn documentation](https://imbalanced-learn.org)
7. [Calibrating Probability with Undersampling for Unbalanced Classification](https://doi.org/10.1109/ssci.2015.33)
---
## Author

**Amna** — Data Science, AI and Digital Business, Gisma University of Applied Sciences
*Data Mining (SS0326) — Individual Project*
