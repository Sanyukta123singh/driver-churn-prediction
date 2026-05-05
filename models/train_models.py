import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, precision_score,
                             recall_score, f1_score,
                             average_precision_score,
                             confusion_matrix, roc_curve,
                             precision_recall_curve)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 7 — MODEL TRAINING & COMPARISON
# ============================================================
# Models to compare:
# 1. Logistic Regression (baseline — interpretable)
# 2. Random Forest (ensemble — handles non-linearity)
# 3. XGBoost (gradient boosting — best for tabular data)
#
# All trained with:
# - Final 26 features
# - SMOTE for class imbalance
# - Stratified K-Fold cross validation
# - Consistent evaluation metrics
# ============================================================

print("Loading data...")
df = pd.read_csv('data/feature_matrix.csv')
final_features = pd.read_csv(
    'data/final_features.csv', header=None
)[0].tolist()

print(f"Final features: {len(final_features)}")
print(f"Features: {final_features}")

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

print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")
print(f"Train churn rate: {y_train.mean():.1%}")
print(f"Test churn rate: {y_test.mean():.1%}")

# Apply SMOTE to training data
print(f"\nApplying SMOTE...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {pd.Series(y_train_smote).value_counts().to_dict()}")

# Scale for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# EVALUATION FUNCTION
# ============================================================

def full_evaluation(model, X_tr, X_te, y_tr, y_te,
                    model_name, is_scaled=False):
    if is_scaled:
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)[:, 1]
        preds = model.predict(X_te)
    else:
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)[:, 1]
        preds = model.predict(X_te)

    auc = roc_auc_score(y_te, probs)
    pr_auc = average_precision_score(y_te, probs)
    precision = precision_score(y_te, preds, zero_division=0)
    recall = recall_score(y_te, preds)
    f1 = f1_score(y_te, preds)
    cm = confusion_matrix(y_te, preds)

    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  AUC:           {auc:.4f}")
    print(f"  PR-AUC:        {pr_auc:.4f}")
    print(f"  Precision:     {precision:.4f}")
    print(f"  Recall:        {recall:.4f}")
    print(f"  F1:            {f1:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TP: {tp:,}  FP: {fp:,}")
    print(f"    FN: {fn:,}  TN: {tn:,}")
    print(f"  Churners caught:  {tp:,}/{tp+fn:,} ({recall:.1%})")
    print(f"  False alarms:     {fp:,}/{fp+tn:,} ({fp/(fp+tn):.1%})")

    return {
        'model_name': model_name,
        'model_object': model,
        'probs': probs,
        'auc': auc,
        'pr_auc': pr_auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'fp': fp,
        'fn': fn, 'tn': tn
    }

results = []

# ============================================================
# MODEL 1 — LOGISTIC REGRESSION
# ============================================================

print(f"\n=== MODEL 1: LOGISTIC REGRESSION ===")

lr = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
    C=1.0
)

r = full_evaluation(
    lr, X_train_scaled, X_test_scaled,
    y_train_smote, y_test,
    "Logistic Regression", is_scaled=True
)
results.append(r)

# Cross validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr_cv = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
cv_scores = cross_val_score(
    lr_cv, X_train_scaled, y_train_smote,
    cv=cv, scoring='roc_auc'
)
print(f"\n  5-Fold CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================
# MODEL 2 — RANDOM FOREST
# ============================================================

print(f"\n=== MODEL 2: RANDOM FOREST ===")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

r = full_evaluation(
    rf, X_train_smote, X_test,
    y_train_smote, y_test,
    "Random Forest"
)
results.append(r)

# Cross validation
rf_cv = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
cv_scores = cross_val_score(
    rf_cv, X_train_smote, y_train_smote,
    cv=cv, scoring='roc_auc'
)
print(f"\n  5-Fold CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================
# MODEL 3 — XGBOOST
# ============================================================

print(f"\n=== MODEL 3: XGBOOST ===")

pos_weight = (y_train_smote == 0).sum() / (y_train_smote == 1).sum()

xgb = XGBClassifier(
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

r = full_evaluation(
    xgb, X_train_smote, X_test,
    y_train_smote, y_test,
    "XGBoost"
)
results.append(r)

# Cross validation
cv_scores = cross_val_score(
    XGBClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        verbosity=0
    ),
    X_train_smote, y_train_smote,
    cv=cv, scoring='roc_auc'
)
print(f"\n  5-Fold CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================
# COMPARISON TABLE
# ============================================================

print(f"\n{'='*60}")
print(f"FINAL MODEL COMPARISON")
print(f"{'='*60}")

comparison_df = pd.DataFrame([{
    'Model': r['model_name'],
    'AUC': r['auc'],
    'PR-AUC': r['pr_auc'],
    'Precision': r['precision'],
    'Recall': r['recall'],
    'F1': r['f1'],
    'Churners Caught': f"{r['tp']}/{r['tp']+r['fn']}",
    'False Alarms': r['fp']
} for r in results])

print(comparison_df.to_string(index=False))

# ============================================================
# VISUALISATIONS
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Model Comparison — Driver Churn Prediction',
             fontsize=14, fontweight='bold')

colors = ['#4A90E2', '#639922', '#E24B4A']
model_names = [r['model_name'] for r in results]

# Chart 1 — ROC Curves
ax1 = axes[0, 0]
for r, color in zip(results, colors):
    fpr, tpr, _ = roc_curve(y_test, r['probs'])
    ax1.plot(fpr, tpr, label=f"{r['model_name']} (AUC={r['auc']:.4f})",
             color=color, linewidth=2)
ax1.plot([0, 1], [0, 1], 'k--', label='Random baseline')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curves')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Chart 2 — Precision Recall Curves
ax2 = axes[0, 1]
for r, color in zip(results, colors):
    prec, rec, _ = precision_recall_curve(y_test, r['probs'])
    ax2.plot(rec, prec,
             label=f"{r['model_name']} (PR-AUC={r['pr_auc']:.4f})",
             color=color, linewidth=2)
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curves')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Chart 3 — Metrics comparison
ax3 = axes[1, 0]
metrics = ['auc', 'pr_auc', 'precision', 'recall', 'f1']
metric_labels = ['AUC', 'PR-AUC', 'Precision', 'Recall', 'F1']
x = np.arange(len(metrics))
width = 0.25

for i, (r, color) in enumerate(zip(results, colors)):
    values = [r[m] for m in metrics]
    bars = ax3.bar(x + i*width, values, width,
                   label=r['model_name'], color=color, alpha=0.8)

ax3.set_xlabel('Metric')
ax3.set_ylabel('Score')
ax3.set_title('Metrics Comparison')
ax3.set_xticks(x + width)
ax3.set_xticklabels(metric_labels)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim(0.5, 1.0)

# Chart 4 — Confusion matrices
ax4 = axes[1, 1]
best_model_result = max(results, key=lambda x: x['recall'])
cm = confusion_matrix(y_test, 
    best_model_result['model_object'].predict(
        X_test_scaled if best_model_result['model_name'] == 'Logistic Regression' 
        else X_test
    )
)
sns_import = __import__('seaborn')
sns_import.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                   xticklabels=['Predicted: Stay', 'Predicted: Churn'],
                   yticklabels=['Actual: Stay', 'Actual: Churn'],
                   ax=ax4)
ax4.set_title(f'Confusion Matrix — {best_model_result["model_name"]}')

plt.tight_layout()
plt.savefig('models/model_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nCharts saved to: models/model_comparison.png")

# ============================================================
# SAVE BEST MODEL
# ============================================================

best_model = max(results, key=lambda x: x['recall'])
print(f"\n=== BEST MODEL: {best_model['model_name']} ===")
print(f"Selected based on highest Recall: {best_model['recall']:.4f}")
print(f"Because: missing a churner costs more than a false alarm")

joblib.dump(best_model['model_object'], 'models/best_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(final_features, 'models/final_features.pkl')

print(f"\nSaved:")
print(f"  models/best_model.pkl")
print(f"  models/scaler.pkl")
print(f"  models/final_features.pkl")
print(f"\n=== STEP 7 COMPLETE ===")