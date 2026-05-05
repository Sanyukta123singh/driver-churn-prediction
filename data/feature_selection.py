import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 6 — FEATURE SELECTION
# ============================================================
# Why feature selection?
# 1. Remove redundant/correlated features (multicollinearity)
# 2. Improve model interpretability
# 3. Reduce overfitting risk
# 4. Speed up training in production
#
# Methods we'll use:
# 1. Correlation analysis — find highly correlated features
# 2. XGBoost feature importance — which features matter most
# 3. Recursive Feature Elimination (RFE) — systematically remove weak features
# 4. Final feature set selection
# ============================================================

print("Loading feature matrix...")
df = pd.read_csv('data/feature_matrix.csv')

FEATURES = [col for col in df.columns
            if col not in ['driver_id', 'will_churn', 'is_noise_churn']]

X = df[FEATURES]
y = df['will_churn']

# Time based split
split_idx = int(len(df) * 0.8)
train_ids = df['driver_id'].iloc[:split_idx]
test_ids = df['driver_id'].iloc[split_idx:]

X_train = X[df['driver_id'].isin(train_ids)]
X_test = X[df['driver_id'].isin(test_ids)]
y_train = y[df['driver_id'].isin(train_ids)]
y_test = y[df['driver_id'].isin(test_ids)]

print(f"Training set: {len(X_train):,} drivers")
print(f"Test set: {len(X_test):,} drivers")
print(f"Total features: {len(FEATURES)}")

# ============================================================
# METHOD 1 — CORRELATION ANALYSIS
# ============================================================

print(f"\n=== METHOD 1: CORRELATION ANALYSIS ===")

corr_matrix = X_train.corr().abs()

# Find highly correlated pairs (>0.85)
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

high_corr_pairs = []
for col in upper_triangle.columns:
    for idx in upper_triangle.index:
        if upper_triangle.loc[idx, col] > 0.85:
            high_corr_pairs.append({
                'feature_1': idx,
                'feature_2': col,
                'correlation': upper_triangle.loc[idx, col]
            })

high_corr_df = pd.DataFrame(high_corr_pairs).sort_values(
    'correlation', ascending=False
)

print(f"Highly correlated feature pairs (>0.85):")
if len(high_corr_df) > 0:
    print(high_corr_df.to_string(index=False))
else:
    print("No highly correlated pairs found")

# Features to drop based on correlation
# Keep the more interpretable one of each pair
features_to_drop_corr = set()
for _, row in high_corr_df.iterrows():
    # Drop the second feature in each correlated pair
    features_to_drop_corr.add(row['feature_2'])

print(f"\nFeatures to drop due to high correlation: {len(features_to_drop_corr)}")
print(features_to_drop_corr)

# ============================================================
# METHOD 2 — XGBOOST FEATURE IMPORTANCE
# ============================================================

print(f"\n=== METHOD 2: XGBOOST FEATURE IMPORTANCE ===")

pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

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
xgb.fit(X_train, y_train)

importance_df = pd.DataFrame({
    'feature': FEATURES,
    'importance': xgb.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 20 features by XGBoost importance:")
print(importance_df.head(20).to_string(index=False))

print(f"\nBottom 10 features (weakest signal):")
print(importance_df.tail(10).to_string(index=False))

# Features with zero or near-zero importance
weak_features = importance_df[
    importance_df['importance'] < 0.005
]['feature'].tolist()
print(f"\nWeak features (importance < 0.005): {len(weak_features)}")
print(weak_features)

# ============================================================
# METHOD 3 — RECURSIVE FEATURE ELIMINATION (RFE)
# ============================================================

print(f"\n=== METHOD 3: RECURSIVE FEATURE ELIMINATION ===")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Test different feature counts
feature_counts = [10, 15, 20, 25, 30, 46]
rfe_results = []

print(f"Testing different feature counts...")

for n_features in feature_counts:
    lr = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    rfe = RFE(estimator=lr, n_features_to_select=n_features)
    rfe.fit(X_train_scaled, y_train)

    probs = rfe.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, probs)

    rfe_results.append({
        'n_features': n_features,
        'auc': round(auc, 4)
    })
    print(f"  {n_features} features: AUC = {auc:.4f}")

rfe_df = pd.DataFrame(rfe_results)
best_rfe = rfe_df.loc[rfe_df['auc'].idxmax()]
print(f"\nBest RFE result: {best_rfe['n_features']:.0f} features, AUC = {best_rfe['auc']:.4f}")

# Get the selected features for optimal count
optimal_n = int(best_rfe['n_features'])
lr_final = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
rfe_final = RFE(estimator=lr_final, n_features_to_select=optimal_n)
rfe_final.fit(X_train_scaled, y_train)

selected_features = [f for f, s in zip(FEATURES, rfe_final.support_) if s]
rejected_features = [f for f, s in zip(FEATURES, rfe_final.support_) if not s]

print(f"\nSelected features ({len(selected_features)}):")
for f in selected_features:
    importance = importance_df[
        importance_df['feature'] == f
    ]['importance'].values[0]
    print(f"  {f}: {importance:.4f}")

print(f"\nRejected features ({len(rejected_features)}):")
print(rejected_features)

# ============================================================
# FINAL FEATURE SET
# ============================================================

print(f"\n=== FINAL FEATURE SET SELECTION ===")

# Combine insights from all three methods
# Use XGBoost importance + RFE agreement
final_features = [f for f in selected_features
                  if f not in features_to_drop_corr]

# Always keep second_last features (interview question answer)
must_keep = ['second_last_resolved', 'second_last_unresolved',
             'second_last_risk', 'consecutive_unresolved_end']
for f in must_keep:
    if f not in final_features and f in FEATURES:
        final_features.append(f)

print(f"Final feature count: {len(final_features)}")
print(f"Features kept: {final_features}")

# Validate final feature set
xgb_final = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=pos_weight,
    random_state=42,
    verbosity=0
)
xgb_final.fit(X_train[final_features], y_train)
final_probs = xgb_final.predict_proba(X_test[final_features])[:, 1]
final_auc = roc_auc_score(y_test, final_probs)

all_feature_probs = xgb.predict_proba(X_test)[:, 1]
all_feature_auc = roc_auc_score(y_test, all_feature_probs)

print(f"\nAUC with all 46 features:   {all_feature_auc:.4f}")
print(f"AUC with final feature set: {final_auc:.4f}")
print(f"AUC difference:             {final_auc - all_feature_auc:.4f}")

# Save final feature list
pd.Series(final_features).to_csv(
    'data/final_features.csv', index=False, header=False
)
print(f"\nFinal features saved to: data/final_features.csv")

# ============================================================
# VISUALISATION
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Feature Selection Analysis', fontsize=14, fontweight='bold')

# Chart 1 — Top 20 feature importance
ax1 = axes[0]
top20 = importance_df.head(20)
colors = ['#E24B4A' if f in must_keep else
          '#EF9F27' if f in final_features else
          '#4A90E2' for f in top20['feature']]
ax1.barh(range(len(top20)), top20['importance'].values, color=colors)
ax1.set_yticks(range(len(top20)))
ax1.set_yticklabels(top20['feature'].values, fontsize=8)
ax1.set_xlabel('XGBoost Feature Importance')
ax1.set_title('Top 20 Features by Importance')
ax1.invert_yaxis()
ax1.grid(True, alpha=0.3, axis='x')

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#E24B4A', label='Must keep (interview answer)'),
    Patch(facecolor='#EF9F27', label='Selected in final set'),
    Patch(facecolor='#4A90E2', label='Other features')
]
ax1.legend(handles=legend_elements, loc='lower right', fontsize=8)

# Chart 2 — RFE AUC vs feature count
ax2 = axes[1]
ax2.plot(rfe_df['n_features'], rfe_df['auc'],
         marker='o', linewidth=2, color='#4A90E2', markersize=8)
ax2.axvline(x=optimal_n, color='#E24B4A', linestyle='--',
            label=f'Optimal: {optimal_n} features')
ax2.set_xlabel('Number of Features')
ax2.set_ylabel('AUC Score')
ax2.set_title('RFE: AUC vs Feature Count')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data/feature_selection_charts.png', dpi=150, bbox_inches='tight')
print(f"\nCharts saved to: data/feature_selection_charts.png")
print(f"\n=== FEATURE SELECTION COMPLETE ===")