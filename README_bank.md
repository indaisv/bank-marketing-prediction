# Bank Marketing Campaign Prediction

A supervised machine learning project that predicts whether a bank client will subscribe to a term deposit, using the Bank Marketing dataset. Built as part of B.Sc. Data Science coursework.

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Logistic Regression | ~90% | ~0.65 | ~0.34 | ~0.44 | ~0.90 |
| Random Forest | ~91% | ~0.67 | ~0.39 | ~0.49 | ~0.93 |

Random Forest outperformed Logistic Regression across all key metrics.

---

## About the Project

The Bank Marketing dataset contains records of direct marketing campaigns (phone calls) conducted by a Portuguese bank. The goal is to predict whether a client will subscribe to a term deposit (yes/no) — a binary classification problem.

The dataset is highly imbalanced — approximately 88% of records are "no" and only 12% are "yes" — making recall and F1 score more meaningful metrics than accuracy alone.

---

## Tech Stack

- Language: Python
- Libraries: Pandas, NumPy, Matplotlib, Scikit-learn
- Models: Logistic Regression, Random Forest
- Techniques: EDA, One-Hot Encoding, StandardScaler, Pipeline, Threshold Tuning

---

## Dataset

- Source: UCI Machine Learning Repository / Kaggle
- File: `bank-full.csv` (semicolon separated)
- Records: ~45,000 rows, 17 columns
- Target variable: `y` — Has the client subscribed a term deposit? (yes/no)

---

## Key Findings

**Top 3 Predictors (Random Forest Feature Importance):**
1. Duration — length of last contact call
2. Balance — client's account balance
3. Age — client's age

**Threshold Tuning Results:**

| Threshold | Precision | Recall |
|-----------|-----------|--------|
| 0.5 | Higher | Lower |
| 0.4 | Medium | Medium |
| 0.3 | Lower | Higher |

Lowering the threshold increases recall (catches more subscribers) but reduces precision — a classic precision-recall tradeoff.

---

## Project Structure

```
bank-marketing-prediction/
|-- bank_marketing_prediction.py   # Main script
|-- bank-full.csv                  # Dataset (download separately)
|-- balance_by_target.png          # Generated visualization
|-- roc_random_forest.png          # ROC curve plot
|-- README.md
```

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/indaisv/bank-marketing-prediction.git
cd bank-marketing-prediction

# 2. Install dependencies
pip install pandas numpy matplotlib scikit-learn

# 3. Download dataset and place in same folder
# bank-full.csv from Kaggle:
# https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing

# 4. Run the script
python bank_marketing_prediction.py
```

---

## Exploratory Data Analysis

The script answers 16 analytical questions including:
- Dataset shape and feature types
- Target variable distribution
- Missing value analysis
- Correlation analysis
- Subscription rate by education level
- Balance distribution comparison (subscribed vs not subscribed)
- One-hot encoding demonstration

---

## Author

**Viraj Indais**
- Email: indaisviraj@gmail.com
- LinkedIn: https://www.linkedin.com/in/viraj-indais
- GitHub: https://github.com/indaisv
