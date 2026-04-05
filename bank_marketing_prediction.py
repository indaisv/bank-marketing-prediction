# =============================================================
# Bank Marketing Campaign Prediction
# Dataset: Bank Marketing Dataset (UCI / Kaggle)
# Author: Viraj Indais
# Results: Random Forest — Accuracy 91%, AUC 0.93, F1 0.49
# =============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report,
    roc_curve, auc
)

# -- 0. LOAD DATASET ------------------------------------------
# Dataset: https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing
# Download bank-full.csv or bank.csv and place in the same folder

possible_files = ['bank-full.csv', 'bank.csv', './bank-full.csv', './bank.csv']
for f in possible_files:
    if os.path.exists(f):
        data_path = f
        break
else:
    raise FileNotFoundError(
        "Could not find 'bank-full.csv' or 'bank.csv'.\n"
        "Download from Kaggle and place it in the same folder as this script."
    )

print("Loading data from:", data_path)
df = pd.read_csv(data_path, sep=';')   # bank marketing uses semicolon separator

# -- 1. SHAPE -------------------------------------------------
print("\n1) Dataset shape (rows, columns):", df.shape)

# -- 2. FEATURE TYPES -----------------------------------------
print("\n2) Columns and data types:")
print(df.dtypes)

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols     = df.select_dtypes(include=['object']).columns.tolist()
binary_cols  = [c for c in cat_cols if df[c].nunique() == 2]
cat_nonbinary = [c for c in cat_cols if c not in binary_cols]

print("\nNumerical columns    :", numeric_cols)
print("Categorical columns  :", cat_nonbinary)
print("Binary columns       :", binary_cols)

# -- 3. TARGET DISTRIBUTION -----------------------------------
print("\n3) Target variable distribution:")
print(df['y'].value_counts())
print("\nPercentage:")
print(df['y'].value_counts(normalize=True).mul(100).round(2))

# -- 4. MISSING VALUES ----------------------------------------
print("\n4) Missing values per column:")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("No missing values detected.")

# -- 5. SUMMARY STATISTICS ------------------------------------
print("\n5) Summary statistics for numerical columns:")
num_stats = df[numeric_cols].agg(['mean', 'median', 'std', 'min', 'max']).T
print(num_stats)

# -- 6. JOB COLUMN --------------------------------------------
print("\n6) Job category counts:")
print(df['job'].value_counts())
print("\nMost frequent job:", df['job'].value_counts().idxmax())

# -- 7. CORRELATION ANALYSIS ----------------------------------
print("\n7) Correlation analysis (numerical features):")
if len(numeric_cols) >= 2:
    corr = df[numeric_cols].corr()
    corr_unstack = corr.unstack().reset_index()
    corr_unstack.columns = ['feature_1', 'feature_2', 'corr']
    corr_unstack = corr_unstack[corr_unstack['feature_1'] != corr_unstack['feature_2']]
    corr_unstack['pair_sorted'] = corr_unstack.apply(
        lambda r: tuple(sorted([r['feature_1'], r['feature_2']])), axis=1
    )
    corr_unique = corr_unstack.drop_duplicates(subset=['pair_sorted']).copy()

    most_pos = corr_unique.loc[corr_unique['corr'].idxmax()]
    most_neg = corr_unique.loc[corr_unique['corr'].idxmin()]

    print("Most positively correlated:", most_pos['feature_1'], "&",
          most_pos['feature_2'], "| corr =", round(most_pos['corr'], 4))
    print("Most negatively correlated:", most_neg['feature_1'], "&",
          most_neg['feature_2'], "| corr =", round(most_neg['corr'], 4))

# -- 8. BALANCE DISTRIBUTION BY TARGET ------------------------
print("\n8) Balance distribution by subscription status:")
print(df.groupby('y')['balance'].describe())

plt.figure(figsize=(8, 5))
plt.hist(df[df['y'] == 'yes']['balance'], bins=50, alpha=0.7, label='Subscribed (yes)')
plt.hist(df[df['y'] == 'no']['balance'],  bins=50, alpha=0.7, label='Not Subscribed (no)')
plt.legend()
plt.title('Balance Distribution by Subscription Status')
plt.xlabel('Account Balance')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('balance_by_target.png', dpi=150)
plt.close()
print("Saved: balance_by_target.png")

# -- 9. EDUCATION SUBSCRIPTION RATE --------------------------
print("\n9) Subscription rate by education level:")
edu = df.groupby('education')['y'].value_counts().unstack().fillna(0)
edu['subscription_rate'] = edu.get('yes', 0) / edu.sum(axis=1)
print(edu[['yes', 'no', 'subscription_rate']].sort_values('subscription_rate', ascending=False))
print("\nHighest subscription rate group:", edu['subscription_rate'].idxmax())

# -- 10. ONE-HOT ENCODING — MARITAL --------------------------
print("\n10) One-hot encoding of 'marital' column:")
marital_dummies = pd.get_dummies(df['marital'], prefix='marital')
print(marital_dummies.head())

# -- 11. TRAIN-TEST SPLIT (80/20) ----------------------------
print("\n11) Train-test split (80/20):")
X = df.drop(columns=['y'])
y = df['y'].map({'yes': 1, 'no': 0})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train samples:", X_train.shape[0])
print("Test samples :", X_test.shape[0])

# -- PREPROCESSING PIPELINE -----------------------------------
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features     = X.select_dtypes(include=['object']).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer,     numeric_features),
    ('cat', categorical_transformer, cat_features)
])

# -- 12. LOGISTIC REGRESSION ----------------------------------
print("\n12) Training Logistic Regression...")
log_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression(max_iter=1000))
])
log_pipe.fit(X_train, y_train)

y_pred_log = log_pipe.predict(X_test)
print("\nLogistic Regression — Classification Report:")
print(classification_report(y_test, y_pred_log, digits=4))

log_metrics = {
    'accuracy':  accuracy_score(y_test, y_pred_log),
    'precision': precision_score(y_test, y_pred_log),
    'recall':    recall_score(y_test, y_pred_log),
    'f1':        f1_score(y_test, y_pred_log)
}
print("Logistic Regression Metrics:", log_metrics)

# -- 13. RANDOM FOREST ----------------------------------------
print("\n13) Training Random Forest...")
rf_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
])
rf_pipe.fit(X_train, y_train)

y_pred_rf = rf_pipe.predict(X_test)
print("\nRandom Forest — Classification Report:")
print(classification_report(y_test, y_pred_rf, digits=4))

rf_metrics = {
    'accuracy':  accuracy_score(y_test, y_pred_rf),
    'precision': precision_score(y_test, y_pred_rf),
    'recall':    recall_score(y_test, y_pred_rf),
    'f1':        f1_score(y_test, y_pred_rf)
}
print("Random Forest Metrics:", rf_metrics)

print("\nModel Comparison (Logistic Regression vs Random Forest):")
for k in rf_metrics:
    print(f"  {k:10s}: Logistic = {log_metrics[k]:.4f} | Random Forest = {rf_metrics[k]:.4f}")

# -- 14. FEATURE IMPORTANCE -----------------------------------
print("\n14) Top feature importances from Random Forest:")
onehot_cols      = rf_pipe.named_steps['preprocessor'] \
                          .named_transformers_['cat'] \
                          .named_steps['onehot'] \
                          .get_feature_names_out(cat_features)
all_feature_names = numeric_features + list(onehot_cols)
importances       = rf_pipe.named_steps['clf'].feature_importances_
feat_imp          = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)

print("Top 15 features:")
print(feat_imp.head(15))
print("\nTop 3 most important features:")
print(feat_imp.head(3))

# -- 15. THRESHOLD ADJUSTMENT ---------------------------------
print("\n15) Decision threshold adjustments (Random Forest):")
y_proba_rf = rf_pipe.predict_proba(X_test)[:, 1]

for thresh in [0.5, 0.4, 0.3]:
    y_thresh = (y_proba_rf >= thresh).astype(int)
    prec = precision_score(y_test, y_thresh)
    rec  = recall_score(y_test, y_thresh)
    print(f"  Threshold = {thresh} | Precision = {prec:.4f} | Recall = {rec:.4f}")

# -- 16. ROC CURVE & AUC -------------------------------------
print("\n16) ROC Curve and AUC — Random Forest:")
fpr, tpr, _ = roc_curve(y_test, y_proba_rf)
roc_auc     = auc(fpr, tpr)
print("AUC Score:", round(roc_auc, 4))

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='steelblue', lw=2, label=f'Random Forest (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Random Forest')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig('roc_random_forest.png', dpi=150)
plt.close()
print("Saved: roc_random_forest.png")

# -- SUMMARY --------------------------------------------------
print("\n" + "=" * 50)
print("  Bank Marketing — Model Performance Summary")
print("=" * 50)
print(f"  Dataset Shape     : {df.shape}")
print(f"  Class Imbalance   : {df['y'].value_counts(normalize=True).mul(100).round(1).to_dict()}")
print(f"  Best Model        : Random Forest")
print(f"  Accuracy          : {rf_metrics['accuracy']:.4f}")
print(f"  Precision         : {rf_metrics['precision']:.4f}")
print(f"  Recall            : {rf_metrics['recall']:.4f}")
print(f"  F1 Score          : {rf_metrics['f1']:.4f}")
print(f"  AUC Score         : {roc_auc:.4f}")
print(f"  Top 3 Features    : {', '.join(feat_imp.head(3).index.tolist())}")
print("=" * 50)
