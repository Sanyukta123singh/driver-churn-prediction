import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import (StratifiedKFold,
                                     learning_curve,
                                     cross_validate)
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 11 — CROSS VALIDATION & OVERFITTING CHECK
# ============================================================
# Why this matters:
# 1. Single train/test split can be lucky or unlucky
# 2. Cross validation gives more reliable estimate
# 3. Learning curves show if model needs more data
# 4. Train vs validation gap reveals overfitting
#
# What we check:
# 1. Stratified K-Fold CV — stable performance?
# 2. Learning curves — does more data help?
# 3. Train vs test gap — are we overfitting?
# 4. Variance analysis — how stable are predictions?
# ============================================================

print("Loading data and model...")
df = pd.read_csv('data/feature_matrix.csv')
final_features = joblib.load('models/final_features.pkl')

X = df[final_features]
y = df['will_churn']

split_idx = int(len(df) * 0.8)
train_ids = df['driver_id'].iloc[:split_idx]
test_ids = df['driver_id'].iloc[split_idx:]

X_train = X[df['driver_id'].isin(train_ids)]
X_test = X[df['driver_id'].isin(test_ids)]
y_train = y[df['driver_id'].isin(train_ids)]
y_test = y[df['driver_id'].isin(test_ids)]

print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

# Apply SMOTE to training data
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

pos_weight = (y_train_smote==0).sum() / (y_train_smote==1).sum()

# ============================================================
# CHECK 1 — STRATIFIED K-FOLD CROSS VALIDATION
# ============================================================

print(f"\n=== CHECK 1: STRATIFIED K-FOLD CV ===")

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=pos_weight,
    random_state=42,
    verbosity=0
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = cross_validate(
    xgb,
    X_train_smote,
    y_train_smote,
    cv=cv,
    scoring=['roc_auc', 'average_precision',
             'recall', 'precision'],
    return_train_score=True,
    n_jobs=-1
)

print(f"\n5-Fold Cross Validation Results:")
print(f"{'Metric':<20} {'Train Mean':>12} {'Val Mean':>12} {'Val Std':>10} {'Gap':>10}")
print(f"{'-'*64}")

metrics = [
    ('roc_auc', 'AUC'),
    ('average_precision', 'PR-AUC'),
    ('recall', 'Recall'),
    ('precision', 'Precision')
]

for metric_key, metric_name in metrics:
    train_scores = cv_results[f'train_{metric_key}']
    val_scores = cv_results[f'test_{metric_key}']
    gap = train_scores.mean() - val_scores.mean()
    print(f"{metric_name:<20} "
          f"{train_scores.mean():>12.4f} "
          f"{val_scores.mean():>12.4f} "
          f"{val_scores.std():>10.4f} "
          f"{gap:>+10.4f}")

print(f"\nIndividual fold AUC scores:")
for i, score in enumerate(cv_results['test_roc_auc'], 1):
    bar = '█' * int(score * 20)
    print(f"  Fold {i}: {score:.4f} {bar}")

print(f"\nStability check:")
auc_std = cv_results['test_roc_auc'].std()
if auc_std < 0.005:
    print(f"  ✓ Very stable (std={auc_std:.4f}) — model generalizes consistently")
elif auc_std < 0.01:
    print(f"  ✓ Stable (std={auc_std:.4f}) — acceptable variance across folds")
else:
    print(f"  ⚠ Unstable (std={auc_std:.4f}) — high variance, consider more data")

# ============================================================
# CHECK 2 — TRAIN VS TEST GAP
# ============================================================

print(f"\n=== CHECK 2: TRAIN VS TEST GAP ===")

xgb_final = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=pos_weight,
    random_state=42,
    verbosity=0
)
xgb_final.fit(X_train_smote, y_train_smote)

train_probs = xgb_final.predict_proba(X_train_smote)[:, 1]
test_probs = xgb_final.predict_proba(X_test)[:, 1]

train_auc = roc_auc_score(y_train_smote, train_probs)
test_auc = roc_auc_score(y_test, test_probs)
gap = train_auc - test_auc

print(f"\nTrain AUC: {train_auc:.4f}")
print(f"Test AUC:  {test_auc:.4f}")
print(f"Gap:       {gap:.4f}")

if gap < 0.05:
    print(f"✓ Small gap — model is not overfitting")
elif gap < 0.10:
    print(f"~ Moderate gap — slight overfitting, acceptable")
else:
    print(f"✗ Large gap — model is overfitting, needs regularization")

# ============================================================
# CHECK 3 — LEARNING CURVES
# ============================================================

print(f"\n=== CHECK 3: LEARNING CURVES ===")
print(f"Does performance improve with more data?")
print(f"This takes 2-3 minutes...")

xgb_lc = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=pos_weight,
    random_state=42,
    verbosity=0
)

train_sizes = np.linspace(0.1, 1.0, 8)

train_sizes_abs, train_scores, val_scores = learning_curve(
    xgb_lc,
    X_train_smote,
    y_train_smote,
    train_sizes=train_sizes,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42
)

print(f"\nLearning curve results:")
print(f"{'Training Size':>15} {'Train AUC':>12} {'Val AUC':>12} {'Gap':>10}")
print(f"{'-'*50}")

for size, tr, val in zip(
    train_sizes_abs,
    train_scores.mean(axis=1),
    val_scores.mean(axis=1)
):
    gap_lc = tr - val
    print(f"{size:>15,.0f} {tr:>12.4f} {val:>12.4f} {gap_lc:>+10.4f}")

# Is model still improving at full data?
final_val = val_scores.mean(axis=1)[-1]
prev_val = val_scores.mean(axis=1)[-2]
improvement = final_val - prev_val

print(f"\nImprovement from 89% to 100% data: {improvement:+.4f}")
if improvement > 0.002:
    print(f"→ Model still improving — MORE DATA would help")
elif improvement > 0:
    print(f"→ Model nearly plateaued — diminishing returns from more data")
else:
    print(f"→ Model plateaued — more data won't help, focus on better features")

# ============================================================
# CHECK 4 — PREDICTION STABILITY
# ============================================================

print(f"\n=== CHECK 4: PREDICTION STABILITY ===")
print(f"How stable are predictions across different random seeds?")

auc_scores = []
for seed in [42, 123, 456, 789, 999]:
    xgb_seed = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        random_state=seed,
        verbosity=0
    )
    xgb_seed.fit(X_train_smote, y_train_smote)
    probs = xgb_seed.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    auc_scores.append(auc)
    print(f"  Seed {seed}: AUC = {auc:.4f}")

print(f"\nMean AUC: {np.mean(auc_scores):.4f}")
print(f"Std AUC:  {np.std(auc_scores):.4f}")
if np.std(auc_scores) < 0.003:
    print(f"✓ Very stable across seeds — results are reliable")
else:
    print(f"~ Some variance across seeds — report mean ± std")

# ============================================================
# VISUALISATIONS
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Cross Validation & Overfitting Analysis',
             fontsize=14, fontweight='bold')

# Chart 1 — Learning Curves
ax1 = axes[0]
train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

ax1.plot(train_sizes_abs, train_mean, 'o-',
         color='#4A90E2', linewidth=2, label='Training AUC')
ax1.fill_between(train_sizes_abs,
                 train_mean - train_std,
                 train_mean + train_std,
                 alpha=0.15, color='#4A90E2')

ax1.plot(train_sizes_abs, val_mean, 'o-',
         color='#E24B4A', linewidth=2, label='Validation AUC')
ax1.fill_between(train_sizes_abs,
                 val_mean - val_std,
                 val_mean + val_std,
                 alpha=0.15, color='#E24B4A')

ax1.set_xlabel('Training Set Size')
ax1.set_ylabel('AUC Score')
ax1.set_title('Learning Curves', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Chart 2 — CV fold scores
ax2 = axes[1]
fold_scores = cv_results['test_roc_auc']
fold_train = cv_results['train_roc_auc']
folds = range(1, len(fold_scores) + 1)

ax2.plot(folds, fold_train, 'o-',
         color='#4A90E2', linewidth=2,
         label='Train AUC', markersize=8)
ax2.plot(folds, fold_scores, 'o-',
         color='#E24B4A', linewidth=2,
         label='Val AUC', markersize=8)
ax2.fill_between(folds,
                 [fold_scores.mean()] * len(folds),
                 fold_scores,
                 alpha=0.15, color='#E24B4A')

ax2.axhline(y=fold_scores.mean(), color='#E24B4A',
            linestyle='--', alpha=0.7,
            label=f'Mean Val AUC={fold_scores.mean():.4f}')
ax2.set_xlabel('Fold')
ax2.set_ylabel('AUC Score')
ax2.set_title('Cross Validation Stability', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks(folds)

plt.tight_layout()
plt.savefig('models/cv_analysis.png', dpi=150, bbox_inches='tight')
print(f"\nCharts saved to: models/cv_analysis.png")
print(f"\n=== STEP 11 COMPLETE ===")