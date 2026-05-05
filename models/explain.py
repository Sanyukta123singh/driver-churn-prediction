import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 10 — SHAP EXPLAINABILITY
# ============================================================
# Why explainability matters:
# 1. Ops team won't trust a black box
# 2. Product needs to know WHAT to fix
# 3. Regulators may require explanation
# 4. Debugging model failures
#
# SHAP = SHapley Additive exPlanations
# Based on game theory — fairly attributes
# each feature's contribution to prediction
#
# Three levels of explanation:
# 1. Global — what drives churn overall?
# 2. Local — why did THIS driver get flagged?
# 3. Interaction — which features work together?
# ============================================================

print("Loading data and model...")
df = pd.read_csv('data/feature_matrix.csv')
final_features = joblib.load('models/final_features.pkl')
model = joblib.load('models/best_model.pkl')
threshold = joblib.load('models/optimal_threshold.pkl')

X = df[final_features]
y = df['will_churn']

# Time based split
split_idx = int(len(df) * 0.8)
test_ids = df['driver_id'].iloc[split_idx:]
X_test = X[df['driver_id'].isin(test_ids)]
y_test = y[df['driver_id'].isin(test_ids)]
df_test = df[df['driver_id'].isin(test_ids)].copy()

print(f"Test set: {len(X_test):,} drivers")

# ============================================================
# COMPUTE SHAP VALUES
# ============================================================

print(f"\nComputing SHAP values (this takes 1-2 minutes)...")

# Use TreeExplainer — optimized for XGBoost
explainer = shap.TreeExplainer(model)

# Use sample for speed (1000 drivers)
sample_idx = np.random.choice(len(X_test), 1000, replace=False)
X_sample = X_test.iloc[sample_idx]
y_sample = y_test.iloc[sample_idx]

shap_values = explainer.shap_values(X_sample)

print(f"SHAP values computed for {len(X_sample):,} drivers")
print(f"Shape: {shap_values.shape}")

# ============================================================
# GLOBAL EXPLANATION 1 — FEATURE IMPORTANCE
# ============================================================

print(f"\n=== GLOBAL EXPLANATION 1: FEATURE IMPORTANCE ===")

# Mean absolute SHAP value per feature
mean_shap = pd.DataFrame({
    'feature': final_features,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)

print(f"\nTop 15 features by SHAP importance:")
print(mean_shap.head(15).to_string(index=False))

print(f"\nBottom 5 features (weakest signal):")
print(mean_shap.tail(5).to_string(index=False))

# ============================================================
# GLOBAL EXPLANATION 2 — DIRECTION OF IMPACT
# ============================================================

print(f"\n=== GLOBAL EXPLANATION 2: DIRECTION OF IMPACT ===")
print(f"How does each feature affect churn probability?")

top10_features = mean_shap.head(10)['feature'].tolist()

for feature in top10_features:
    feat_idx = final_features.index(feature)
    feat_values = X_sample[feature].values
    feat_shap = shap_values[:, feat_idx]

    # Correlation between feature value and SHAP value
    correlation = np.corrcoef(feat_values, feat_shap)[0, 1]
    direction = "↑ Higher value → MORE churn" if correlation > 0 \
        else "↓ Higher value → LESS churn"

    print(f"  {feature:<30} {direction}")

# ============================================================
# LOCAL EXPLANATION — WHY THIS DRIVER?
# ============================================================

print(f"\n=== LOCAL EXPLANATION: WHY FLAGGED? ===")

# Get predictions
probs = model.predict_proba(X_test)[:, 1]
df_test['churn_probability'] = probs
df_test['predicted_churn'] = (probs >= threshold).astype(int)

# Find a high risk driver
high_risk = df_test[
    df_test['churn_probability'] > 0.8
].iloc[0]
driver_id = high_risk['driver_id']

print(f"\nDriver ID: {driver_id:.0f}")
print(f"Churn probability: {high_risk['churn_probability']:.3f}")
print(f"Actual churn: {'YES' if high_risk['will_churn'] == 1 else 'NO'}")

# Get SHAP values for this specific driver
driver_idx = list(X_test.index).index(
    df_test[df_test['driver_id'] == driver_id].index[0]
)

# Find in sample
sample_probs = model.predict_proba(X_sample)[:, 1]
high_risk_sample_idx = np.argmax(sample_probs)

driver_shap = shap_values[high_risk_sample_idx]
driver_features_values = X_sample.iloc[high_risk_sample_idx]

# Top contributing features for this driver
driver_explanation = pd.DataFrame({
    'feature': final_features,
    'shap_value': driver_shap,
    'feature_value': driver_features_values.values
}).sort_values('shap_value', ascending=False)

print(f"\nTop reasons this driver is HIGH risk:")
top_reasons = driver_explanation.head(5)
for _, row in top_reasons.iterrows():
    impact = "INCREASES" if row['shap_value'] > 0 else "DECREASES"
    print(f"  {row['feature']:<30} = {row['feature_value']:.2f} "
          f"→ {impact} churn risk by {abs(row['shap_value']):.3f}")

print(f"\nTop reasons protecting this driver:")
bottom_reasons = driver_explanation.tail(3)
for _, row in bottom_reasons.iterrows():
    print(f"  {row['feature']:<30} = {row['feature_value']:.2f} "
          f"→ reduces churn risk by {abs(row['shap_value']):.3f}")

# ============================================================
# BUSINESS FRIENDLY EXPLANATION
# ============================================================

print(f"\n=== BUSINESS FRIENDLY EXPLANATION ===")
print(f"(How to explain to ops team / product manager)")
print(f"\nFor Driver {driver_id:.0f} flagged at {high_risk['churn_probability']:.0%} risk:")

top3 = driver_explanation.head(3)
explanations = []
for _, row in top3.iterrows():
    feature = row['feature']
    value = row['feature_value']

    if feature == 'unresolved_high_risk':
        explanations.append(
            f"Has {value:.0f} unresolved high-risk tickets (fare disputes/defective trips)"
        )
    elif feature == 'unresolved_count':
        explanations.append(
            f"Has {value:.0f} total unresolved tickets"
        )
    elif feature == 'consecutive_unresolved_end':
        explanations.append(
            f"Last {value:.0f} tickets in a row were unresolved"
        )
    elif feature == 'total_tickets':
        explanations.append(
            f"Filed {value:.0f} total support tickets"
        )
    elif feature == 'weighted_unresolved':
        explanations.append(
            f"Recent tickets are predominantly unresolved (weighted score: {value:.2f})"
        )
    else:
        explanations.append(f"{feature} = {value:.2f}")

for i, exp in enumerate(explanations, 1):
    print(f"  Reason {i}: {exp}")

# ============================================================
# VISUALISATIONS
# ============================================================

print(f"\nGenerating SHAP visualisations...")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('SHAP Explainability — Driver Churn Model',
             fontsize=14, fontweight='bold')

# Chart 1 — Global feature importance
ax1 = axes[0]
top15 = mean_shap.head(15)
colors = ['#E24B4A' if f in [
    'unresolved_high_risk', 'unresolved_count',
    'consecutive_unresolved_end'
] else '#4A90E2' for f in top15['feature']]

ax1.barh(range(len(top15)), top15['mean_abs_shap'].values,
         color=colors)
ax1.set_yticks(range(len(top15)))
ax1.set_yticklabels(top15['feature'].values, fontsize=9)
ax1.set_xlabel('Mean |SHAP Value|')
ax1.set_title('Global Feature Importance (SHAP)', fontweight='bold')
ax1.invert_yaxis()
ax1.grid(True, alpha=0.3, axis='x')

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#E24B4A', label='Top churn drivers'),
    Patch(facecolor='#4A90E2', label='Other features')
]
ax1.legend(handles=legend_elements, loc='lower right')

# Chart 2 — Local explanation waterfall style
ax2 = axes[1]
top_local = driver_explanation.head(10).sort_values('shap_value')
colors_local = ['#E24B4A' if v > 0 else '#4A90E2'
                for v in top_local['shap_value']]

ax2.barh(range(len(top_local)),
         top_local['shap_value'].values,
         color=colors_local)
ax2.set_yticks(range(len(top_local)))
ax2.set_yticklabels(top_local['feature'].values, fontsize=9)
ax2.set_xlabel('SHAP Value (impact on churn probability)')
ax2.set_title('Local Explanation — Why This Driver?',
              fontweight='bold')
ax2.axvline(x=0, color='black', linewidth=0.8)
ax2.grid(True, alpha=0.3, axis='x')

from matplotlib.patches import Patch
legend_local = [
    Patch(facecolor='#E24B4A', label='Increases churn risk'),
    Patch(facecolor='#4A90E2', label='Decreases churn risk')
]
ax2.legend(handles=legend_local, loc='lower right')

plt.tight_layout()
plt.savefig('models/shap_explanation.png', dpi=150,
            bbox_inches='tight')
print(f"Charts saved to: models/shap_explanation.png")
print(f"\n=== STEP 10 COMPLETE ===")