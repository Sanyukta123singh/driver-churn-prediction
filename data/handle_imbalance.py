import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, precision_score,
                             recall_score, f1_score,
                             average_precision_score)
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 5 — HANDLING CLASS IMBALANCE
# ============================================================
# Problem: 80.4% non-churners vs 19.6% churners
# Approaches to compare:
# 1. No handling (baseline)
# 2. Class weights
# 3. Random oversampling
# 4. SMOTE oversampling
# 5. Random undersampling
# 6. XGBoost scale_pos_weight
# ============================================================

print("Loading feature matrix...")
df = pd.read_csv('data/feature_matrix.csv')

FEATURES = [col for col in df.columns
            if col not in ['driver_id', 'will_churn', 'is_noise_churn']]

X = df[FEATURES]
y = df['will_churn']

print(f"Features: {len(FEATURES)}")
print(f"Class distribution:")
print(f"  Non-churners: {(y==0).sum():,} ({(y==0).mean():.1%})")
print(f"  Churners: {(y==1).sum():,} ({(y==1).mean():.1%})")
print(f"  Imbalance ratio: {(y==0).sum()/(y==1).sum():.1f}:1")

# Time based train/test split
# Important: split by driver_id order to prevent data leakage
# Earlier drivers = train, later drivers = test
split_idx = int(len(df) * 0.8)
train_ids = df['driver_id'].iloc[:split_idx]
test_ids = df['driver_id'].iloc[split_idx:]

X_train = X[df['driver_id'].isin(train_ids)]
X_test = X[df['driver_id'].isin(test_ids)]
y_train = y[df['driver_id'].isin(train_ids)]
y_test = y[df['driver_id'].isin(test_ids)]

print(f"\nTime-based train/test split:")
print(f"  Train: {len(X_train):,} drivers")
print(f"  Test:  {len(X_test):,} drivers")

# Scale features for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, X_test, y_test, model_name,
                   is_scaled=False, X_test_scaled=None):
    if is_scaled and X_test_scaled is not None:
        probs = model.predict_proba(X_test_scaled)[:, 1]
        preds = model.predict(X_test_scaled)
    else:
        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

    auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"\n  {model_name}:")
    print(f"    AUC:       {auc:.4f}")
    print(f"    PR-AUC:    {pr_auc:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1:        {f1:.4f}")

    return {
        'model': model_name,
        'auc': auc,
        'pr_auc': pr_auc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

results = []

# ============================================================
# APPROACH 1 — BASELINE (NO HANDLING)
# ============================================================

print(f"\n=== APPROACH 1: BASELINE (NO HANDLING) ===")

lr_baseline = LogisticRegression(max_iter=1000, random_state=42)
lr_baseline.fit(X_train_scaled, y_train)
r = evaluate_model(lr_baseline, X_test, y_test,
                   "LR Baseline", True, X_test_scaled)
results.append(r)

# ============================================================
# APPROACH 2 — CLASS WEIGHTS
# ============================================================

print(f"\n=== APPROACH 2: CLASS WEIGHTS ===")

lr_weighted = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
lr_weighted.fit(X_train_scaled, y_train)
r = evaluate_model(lr_weighted, X_test, y_test,
                   "LR Class Weights", True, X_test_scaled)
results.append(r)

# ============================================================
# APPROACH 3 — RANDOM OVERSAMPLING
# ============================================================

print(f"\n=== APPROACH 3: RANDOM OVERSAMPLING ===")

ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train_scaled, y_train)
print(f"  After oversampling: {pd.Series(y_train_ros).value_counts().to_dict()}")

lr_ros = LogisticRegression(max_iter=1000, random_state=42)
lr_ros.fit(X_train_ros, y_train_ros)
r = evaluate_model(lr_ros, X_test, y_test,
                   "LR Random Oversample", True, X_test_scaled)
results.append(r)

# ============================================================
# APPROACH 4 — SMOTE
# ============================================================

print(f"\n=== APPROACH 4: SMOTE ===")

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
print(f"  After SMOTE: {pd.Series(y_train_smote).value_counts().to_dict()}")

lr_smote = LogisticRegression(max_iter=1000, random_state=42)
lr_smote.fit(X_train_smote, y_train_smote)
r = evaluate_model(lr_smote, X_test, y_test,
                   "LR SMOTE", True, X_test_scaled)
results.append(r)

# ============================================================
# APPROACH 5 — RANDOM UNDERSAMPLING
# ============================================================

print(f"\n=== APPROACH 5: RANDOM UNDERSAMPLING ===")

rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train_scaled, y_train)
print(f"  After undersampling: {pd.Series(y_train_rus).value_counts().to_dict()}")

lr_rus = LogisticRegression(max_iter=1000, random_state=42)
lr_rus.fit(X_train_rus, y_train_rus)
r = evaluate_model(lr_rus, X_test, y_test,
                   "LR Undersample", True, X_test_scaled)
results.append(r)

# ============================================================
# APPROACH 6 — XGBOOST WITH SCALE_POS_WEIGHT
# ============================================================

print(f"\n=== APPROACH 6: XGBOOST SCALE_POS_WEIGHT ===")

# scale_pos_weight = non-churners / churners
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"  scale_pos_weight: {pos_weight:.2f}")

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=pos_weight,
    random_state=42,
    eval_metric='auc',
    verbosity=0
)
xgb_model.fit(X_train, y_train)
r = evaluate_model(xgb_model, X_test, y_test, "XGBoost scale_pos_weight")
results.append(r)

# ============================================================
# COMPARISON TABLE
# ============================================================

print(f"\n=== FINAL COMPARISON ===")
results_df = pd.DataFrame(results)
results_df = results_df.round(4)
print(results_df.to_string(index=False))

best_recall = results_df.loc[results_df['recall'].idxmax()]
best_f1 = results_df.loc[results_df['f1'].idxmax()]
best_auc = results_df.loc[results_df['auc'].idxmax()]

print(f"\nBest Recall:  {best_recall['model']} ({best_recall['recall']:.4f})")
print(f"Best F1:      {best_f1['model']} ({best_f1['f1']:.4f})")
print(f"Best AUC:     {best_auc['model']} ({best_auc['auc']:.4f})")

print(f"\n=== KEY INSIGHT ===")
baseline_recall = results_df[
    results_df['model'] == 'LR Baseline'
]['recall'].values[0]
best_recall_val = best_recall['recall']
print(f"Recall improved from {baseline_recall:.4f} to {best_recall_val:.4f}")
print(f"by handling class imbalance")
print(f"This means we catch {best_recall_val:.1%} of churners vs {baseline_recall:.1%} without handling")