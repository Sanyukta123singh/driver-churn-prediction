import pandas as pd
import numpy as np
from sklearn.model_selection import (RandomizedSearchCV,
                                     StratifiedKFold,
                                     cross_val_score)
from sklearn.metrics import (roc_auc_score, recall_score,
                             precision_score, f1_score,
                             average_precision_score)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 8 — HYPERPARAMETER TUNING
# ============================================================
# Why tune?
# Default hyperparameters are rarely optimal
# Small tuning can meaningfully improve recall
#
# Method: RandomizedSearchCV
# Why random over grid search?
# - Grid search: tests every combination (too slow)
# - Random search: samples from distributions (faster)
# - Research shows random search finds good params
#   with far fewer iterations than grid search
#
# Parameters to tune:
# - n_estimators: number of trees
# - max_depth: tree depth (overfitting control)
# - learning_rate: step size
# - subsample: row sampling per tree
# - colsample_bytree: feature sampling per tree
# - min_child_weight: minimum samples in leaf
# - gamma: minimum loss reduction to split
# - reg_alpha: L1 regularization
# - reg_lambda: L2 regularization
# ============================================================

print("Loading data...")
df = pd.read_csv('data/feature_matrix.csv')
final_features = joblib.load('models/final_features.pkl')

X = df[final_features]
y = df['will_churn']

# Time based split
split_idx = int(len(df) * 0.8)
train_ids = df['driver_id'].iloc[:split_idx]
test_ids = df['driver_id'].iloc[split_idx:]

X_train = X[df['driver_id'].isin(train_ids)]
X_test = X[df['driver_id'].isin(test_ids)]
y_train = y[df['driver_id'].isin(train_ids)]
y_test = y[df['driver_id'].isin(test_ids)]

print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

# Apply SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {pd.Series(y_train_smote).value_counts().to_dict()}")

pos_weight = (y_train_smote==0).sum() / (y_train_smote==1).sum()

# ============================================================
# BASELINE XGBOOST (before tuning)
# ============================================================

print(f"\n=== BASELINE XGBOOST (DEFAULT PARAMS) ===")

xgb_baseline = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=pos_weight,
    random_state=42,
    verbosity=0
)
xgb_baseline.fit(X_train_smote, y_train_smote)
baseline_probs = xgb_baseline.predict_proba(X_test)[:, 1]
baseline_preds = xgb_baseline.predict(X_test)

baseline_auc = roc_auc_score(y_test, baseline_probs)
baseline_recall = recall_score(y_test, baseline_preds)
baseline_pr_auc = average_precision_score(y_test, baseline_probs)

print(f"Baseline AUC:    {baseline_auc:.4f}")
print(f"Baseline PR-AUC: {baseline_pr_auc:.4f}")
print(f"Baseline Recall: {baseline_recall:.4f}")

# ============================================================
# HYPERPARAMETER SEARCH SPACE
# ============================================================

param_distributions = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5, 7, 10],
    'gamma': [0, 0.1, 0.2, 0.3, 0.5],
    'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
    'reg_lambda': [0.5, 1.0, 1.5, 2.0, 5.0]
}

print(f"\n=== HYPERPARAMETER SEARCH ===")
print(f"Search space size: {5*6*5*5*5*5*5*5*5:,} combinations")
print(f"Random search iterations: 50 (efficient sampling)")
print(f"Cross validation folds: 5")
print(f"This will take 3-5 minutes...")

# ============================================================
# RANDOMIZED SEARCH
# ============================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

xgb_tuned = XGBClassifier(
    scale_pos_weight=pos_weight,
    random_state=42,
    eval_metric='auc',
    verbosity=0
)

random_search = RandomizedSearchCV(
    estimator=xgb_tuned,
    param_distributions=param_distributions,
    n_iter=50,
    scoring='roc_auc',
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train_smote, y_train_smote)

print(f"\nBest parameters found:")
for param, value in random_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nBest CV AUC: {random_search.best_score_:.4f}")

# ============================================================
# EVALUATE TUNED MODEL
# ============================================================

print(f"\n=== TUNED XGBOOST RESULTS ===")

best_xgb = random_search.best_estimator_
tuned_probs = best_xgb.predict_proba(X_test)[:, 1]
tuned_preds = best_xgb.predict(X_test)

tuned_auc = roc_auc_score(y_test, tuned_probs)
tuned_recall = recall_score(y_test, tuned_preds)
tuned_precision = precision_score(y_test, tuned_preds)
tuned_f1 = f1_score(y_test, tuned_preds)
tuned_pr_auc = average_precision_score(y_test, tuned_probs)

print(f"Tuned AUC:       {tuned_auc:.4f}")
print(f"Tuned PR-AUC:    {tuned_pr_auc:.4f}")
print(f"Tuned Precision: {tuned_precision:.4f}")
print(f"Tuned Recall:    {tuned_recall:.4f}")
print(f"Tuned F1:        {tuned_f1:.4f}")

# ============================================================
# BEFORE VS AFTER COMPARISON
# ============================================================

print(f"\n=== BEFORE VS AFTER TUNING ===")
print(f"{'Metric':<15} {'Before':>10} {'After':>10} {'Change':>10}")
print(f"{'-'*45}")
print(f"{'AUC':<15} {baseline_auc:>10.4f} {tuned_auc:>10.4f} {tuned_auc-baseline_auc:>+10.4f}")
print(f"{'PR-AUC':<15} {baseline_pr_auc:>10.4f} {tuned_pr_auc:>10.4f} {tuned_pr_auc-baseline_pr_auc:>+10.4f}")
print(f"{'Recall':<15} {baseline_recall:>10.4f} {tuned_recall:>10.4f} {tuned_recall-baseline_recall:>+10.4f}")

# ============================================================
# THRESHOLD OPTIMISATION
# ============================================================

print(f"\n=== THRESHOLD OPTIMISATION ===")
print(f"Default threshold: 0.5")
print(f"Testing different thresholds to maximise recall...")

thresholds = np.arange(0.1, 0.9, 0.05)
threshold_results = []

for threshold in thresholds:
    preds = (tuned_probs >= threshold).astype(int)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds, zero_division=0)
    threshold_results.append({
        'threshold': round(threshold, 2),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4)
    })

threshold_df = pd.DataFrame(threshold_results)
print(f"\nThreshold analysis:")
print(threshold_df.to_string(index=False))

# Find optimal threshold for recall > 0.75
high_recall = threshold_df[threshold_df['recall'] >= 0.75]
if len(high_recall) > 0:
    optimal = high_recall.loc[high_recall['f1'].idxmax()]
    print(f"\nOptimal threshold (recall >= 75%): {optimal['threshold']}")
    print(f"  Precision: {optimal['precision']}")
    print(f"  Recall:    {optimal['recall']}")
    print(f"  F1:        {optimal['f1']}")
    optimal_threshold = optimal['threshold']
else:
    optimal_threshold = 0.5
    print(f"Using default threshold: 0.5")

# ============================================================
# SAVE TUNED MODEL
# ============================================================

joblib.dump(best_xgb, 'models/tuned_xgb_model.pkl')
joblib.dump(optimal_threshold, 'models/optimal_threshold.pkl')

print(f"\nSaved:")
print(f"  models/tuned_xgb_model.pkl")
print(f"  models/optimal_threshold.pkl")
print(f"\n=== STEP 8 COMPLETE ===")
print(f"Best model: XGBoost (tuned)")
print(f"Optimal threshold: {optimal_threshold}")
print(f"Expected recall in production: {tuned_recall:.1%}")