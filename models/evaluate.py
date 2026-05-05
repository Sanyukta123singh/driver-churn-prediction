import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_auc_score, precision_score,
                             recall_score, f1_score,
                             average_precision_score,
                             confusion_matrix, roc_curve,
                             precision_recall_curve,
                             brier_score_loss)
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 9 — PROPER MODEL EVALUATION
# ============================================================
# What we evaluate:
# 1. Confusion matrix — what types of errors are we making?
# 2. ROC & PR curves — model discrimination ability
# 3. Calibration curve — are probabilities trustworthy?
# 4. Business impact — how much value does this create?
# 5. Performance by segment — where does model struggle?
# 6. Noise churn analysis — unavoidable error ceiling
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
train_ids = df['driver_id'].iloc[:split_idx]
test_ids = df['driver_id'].iloc[split_idx:]

X_test = X[df['driver_id'].isin(test_ids)]
y_test = y[df['driver_id'].isin(test_ids)]
df_test = df[df['driver_id'].isin(test_ids)].copy()

print(f"Test set: {len(X_test):,} drivers")
print(f"Optimal threshold: {threshold}")

# Get predictions
probs = model.predict_proba(X_test)[:, 1]
preds = (probs >= threshold).astype(int)
preds_default = (probs >= 0.5).astype(int)

df_test['churn_probability'] = probs
df_test['predicted_churn'] = preds

# ============================================================
# EVALUATION 1 — CONFUSION MATRIX DEEP DIVE
# ============================================================

print(f"\n=== EVALUATION 1: CONFUSION MATRIX ===")

cm = confusion_matrix(y_test, preds)
tn, fp, fn, tp = cm.ravel()

total = len(y_test)
actual_churners = tp + fn
actual_non_churners = tn + fp

print(f"\nAt threshold = {threshold}:")
print(f"  True Positives  (caught churners):     {tp:,} ({tp/actual_churners:.1%} of churners)")
print(f"  False Negatives (missed churners):     {fn:,} ({fn/actual_churners:.1%} of churners)")
print(f"  True Negatives  (correct non-churn):  {tn:,} ({tn/actual_non_churners:.1%} of non-churners)")
print(f"  False Positives (false alarms):        {fp:,} ({fp/actual_non_churners:.1%} of non-churners)")

print(f"\nBusiness interpretation:")
print(f"  Out of {actual_churners:,} drivers about to churn:")
print(f"  → We catch {tp:,} ({tp/actual_churners:.1%}) and can intervene")
print(f"  → We miss {fn:,} ({fn/actual_churners:.1%}) — these drivers will churn silently")
print(f"  Out of {actual_non_churners:,} drivers who won't churn:")
print(f"  → We correctly ignore {tn:,} ({tn/actual_non_churners:.1%})")
print(f"  → We falsely flag {fp:,} ({fp/actual_non_churners:.1%}) — unnecessary messages")

# ============================================================
# EVALUATION 2 — CORE METRICS
# ============================================================

print(f"\n=== EVALUATION 2: CORE METRICS ===")

auc = roc_auc_score(y_test, probs)
pr_auc = average_precision_score(y_test, probs)
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)
brier = brier_score_loss(y_test, probs)

print(f"  AUC:        {auc:.4f}  (discrimination ability)")
print(f"  PR-AUC:     {pr_auc:.4f}  (imbalanced class performance)")
print(f"  Precision:  {precision:.4f}  (accuracy when flagging churn)")
print(f"  Recall:     {recall:.4f}  (churners we catch)")
print(f"  F1:         {f1:.4f}  (balance of precision & recall)")
print(f"  Brier:      {brier:.4f}  (probability calibration — lower is better)")

# ============================================================
# EVALUATION 3 — CALIBRATION ANALYSIS
# ============================================================

print(f"\n=== EVALUATION 3: CALIBRATION ANALYSIS ===")
print(f"Are predicted probabilities trustworthy?")
print(f"If model says 70% churn risk — do 70% actually churn?")

fraction_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)

print(f"\nCalibration bins:")
print(f"{'Predicted Prob':>15} {'Actual Rate':>12} {'Difference':>12}")
print(f"{'-'*40}")
for pred, actual in zip(mean_pred, fraction_pos):
    diff = actual - pred
    print(f"{pred:>15.3f} {actual:>12.3f} {diff:>+12.3f}")

avg_calibration_error = np.mean(np.abs(fraction_pos - mean_pred))
print(f"\nAverage calibration error: {avg_calibration_error:.4f}")
if avg_calibration_error < 0.05:
    print(f"✓ Well calibrated — probabilities are trustworthy")
elif avg_calibration_error < 0.10:
    print(f"~ Moderately calibrated — use with some caution")
else:
    print(f"✗ Poorly calibrated — probabilities need adjustment")

# ============================================================
# EVALUATION 4 — BUSINESS IMPACT
# ============================================================

print(f"\n=== EVALUATION 4: BUSINESS IMPACT ===")

# Assumptions
monthly_value_per_driver = 10000  # INR
intervention_cost = 50            # INR per WhatsApp + incentive
intervention_success_rate = 0.40  # 40% of caught churners retained

drivers_retained = tp * intervention_success_rate
revenue_saved = drivers_retained * monthly_value_per_driver
intervention_cost_total = (tp + fp) * intervention_cost
net_value = revenue_saved - intervention_cost_total

print(f"\nAssumptions:")
print(f"  Monthly value per driver:    ₹{monthly_value_per_driver:,}")
print(f"  Cost per intervention:       ₹{intervention_cost:,}")
print(f"  Intervention success rate:   {intervention_success_rate:.0%}")

print(f"\nAt threshold = {threshold}:")
print(f"  Drivers flagged:             {tp+fp:,}")
print(f"  Churners caught:             {tp:,}")
print(f"  Drivers retained (est):      {drivers_retained:,.0f}")
print(f"  Revenue saved:               ₹{revenue_saved:,.0f}/month")
print(f"  Intervention cost:           ₹{intervention_cost_total:,.0f}/month")
print(f"  Net value:                   ₹{net_value:,.0f}/month")

# Compare with no model
no_model_value = 0
print(f"\n  Without model:               ₹0/month")
print(f"  With model:                  ₹{net_value:,.0f}/month")
print(f"  Model ROI:                   {net_value/intervention_cost_total:.1f}x")

# ============================================================
# EVALUATION 5 — PERFORMANCE BY SEGMENT
# ============================================================

print(f"\n=== EVALUATION 5: PERFORMANCE BY SEGMENT ===")

# By ticket volume
df_test['ticket_segment'] = pd.cut(
    df_test['total_tickets'],
    bins=[0, 1, 3, 6, 50],
    labels=['1 ticket', '2-3 tickets', '4-6 tickets', '7+ tickets']
)

print(f"\nPerformance by ticket volume:")
for segment in ['1 ticket', '2-3 tickets', '4-6 tickets', '7+ tickets']:
    seg_data = df_test[df_test['ticket_segment'] == segment]
    if len(seg_data) > 0 and seg_data['will_churn'].sum() > 0:
        seg_recall = recall_score(
            seg_data['will_churn'],
            seg_data['predicted_churn']
        )
        seg_precision = precision_score(
            seg_data['will_churn'],
            seg_data['predicted_churn'],
            zero_division=0
        )
        churn_rate = seg_data['will_churn'].mean()
        print(f"  {segment:<15} "
              f"n={len(seg_data):>5,} "
              f"churn={churn_rate:.1%} "
              f"recall={seg_recall:.3f} "
              f"precision={seg_precision:.3f}")

# By outlier status
print(f"\nPerformance by outlier status:")
for is_outlier in [0, 1]:
    seg_data = df_test[df_test['is_outlier_driver'] == is_outlier]
    if len(seg_data) > 0 and seg_data['will_churn'].sum() > 0:
        seg_recall = recall_score(
            seg_data['will_churn'],
            seg_data['predicted_churn']
        )
        label = "Outlier drivers" if is_outlier else "Normal drivers"
        churn_rate = seg_data['will_churn'].mean()
        print(f"  {label:<20} "
              f"n={len(seg_data):>5,} "
              f"churn={churn_rate:.1%} "
              f"recall={seg_recall:.3f}")

# ============================================================
# EVALUATION 6 — NOISE CHURN ANALYSIS
# ============================================================

print(f"\n=== EVALUATION 6: NOISE CHURN ANALYSIS ===")
print(f"How much of our error is unavoidable?")

noise_churners = df_test[df_test['is_noise_churn'] == 1]
signal_churners = df_test[
    (df_test['will_churn'] == 1) &
    (df_test['is_noise_churn'] == 0)
]

if len(noise_churners) > 0:
    noise_recall = recall_score(
        noise_churners['will_churn'],
        noise_churners['predicted_churn']
    )
    print(f"\nNoise churners (churned for non-support reasons):")
    print(f"  Count: {len(noise_churners):,}")
    print(f"  Recall: {noise_recall:.3f}")
    print(f"  → These are impossible to predict from support data")

if len(signal_churners) > 0:
    signal_recall = recall_score(
        signal_churners['will_churn'],
        signal_churners['predicted_churn']
    )
    print(f"\nSignal churners (churned due to support issues):")
    print(f"  Count: {len(signal_churners):,}")
    print(f"  Recall: {signal_recall:.3f}")
    print(f"  → These are the actionable churners our model targets")

print(f"\nKey insight: Our true performance ceiling is {signal_recall:.1%} recall")
print(f"on support-driven churners — not the overall {recall:.1%}")

# ============================================================
# VISUALISATIONS
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Model Evaluation — Driver Churn Prediction',
             fontsize=14, fontweight='bold')

# Chart 1 — Confusion Matrix
ax1 = axes[0, 0]
cm_display = np.array([[tn, fp], [fn, tp]])
sns.heatmap(cm_display, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted: Stay', 'Predicted: Churn'],
            yticklabels=['Actual: Stay', 'Actual: Churn'],
            ax=ax1)
ax1.set_title(f'Confusion Matrix (threshold={threshold})',
              fontweight='bold')

# Chart 2 — ROC Curve
ax2 = axes[0, 1]
fpr, tpr, _ = roc_curve(y_test, probs)
ax2.plot(fpr, tpr, color='#E24B4A', linewidth=2,
         label=f'XGBoost (AUC={auc:.4f})')
ax2.plot([0, 1], [0, 1], 'k--', label='Random baseline')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Chart 3 — PR Curve
ax3 = axes[0, 2]
prec_curve, rec_curve, thresholds_pr = precision_recall_curve(y_test, probs)
ax3.plot(rec_curve, prec_curve, color='#4A90E2', linewidth=2,
         label=f'XGBoost (PR-AUC={pr_auc:.4f})')
ax3.axvline(x=recall, color='#E24B4A', linestyle='--',
            label=f'Current recall={recall:.3f}')
ax3.axhline(y=precision, color='#639922', linestyle='--',
            label=f'Current precision={precision:.3f}')
ax3.set_xlabel('Recall')
ax3.set_ylabel('Precision')
ax3.set_title('Precision-Recall Curve', fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Chart 4 — Calibration Curve
ax4 = axes[1, 0]
ax4.plot(mean_pred, fraction_pos, 's-', color='#E24B4A',
         linewidth=2, label='XGBoost')
ax4.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
ax4.set_xlabel('Mean Predicted Probability')
ax4.set_ylabel('Fraction of Positives')
ax4.set_title('Calibration Curve', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Chart 5 — Threshold vs Metrics
ax5 = axes[1, 1]
thresholds_range = np.arange(0.1, 0.9, 0.05)
precisions, recalls, f1s = [], [], []
for t in thresholds_range:
    p = (probs >= t).astype(int)
    precisions.append(precision_score(y_test, p, zero_division=0))
    recalls.append(recall_score(y_test, p))
    f1s.append(f1_score(y_test, p, zero_division=0))

ax5.plot(thresholds_range, precisions, color='#4A90E2',
         linewidth=2, label='Precision')
ax5.plot(thresholds_range, recalls, color='#E24B4A',
         linewidth=2, label='Recall')
ax5.plot(thresholds_range, f1s, color='#639922',
         linewidth=2, label='F1')
ax5.axvline(x=threshold, color='black', linestyle='--',
            label=f'Chosen threshold={threshold}')
ax5.set_xlabel('Threshold')
ax5.set_ylabel('Score')
ax5.set_title('Threshold vs Metrics', fontweight='bold')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# Chart 6 — Business Impact
ax6 = axes[1, 2]
thresholds_biz = np.arange(0.1, 0.9, 0.05)
net_values = []
for t in thresholds_biz:
    p = (probs >= t).astype(int)
    cm_t = confusion_matrix(y_test, p)
    tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
    retained = tp_t * intervention_success_rate
    revenue = retained * monthly_value_per_driver
    cost = (tp_t + fp_t) * intervention_cost
    net_values.append(revenue - cost)

ax6.plot(thresholds_biz, [v/1000 for v in net_values],
         color='#639922', linewidth=2)
ax6.axvline(x=threshold, color='#E24B4A', linestyle='--',
            label=f'Chosen threshold={threshold}')
ax6.set_xlabel('Threshold')
ax6.set_ylabel('Net Value (₹ thousands/month)')
ax6.set_title('Business Impact by Threshold', fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('models/evaluation_charts.png', dpi=150, bbox_inches='tight')
print(f"\nCharts saved to: models/evaluation_charts.png")
print(f"\n=== STEP 9 COMPLETE ===")