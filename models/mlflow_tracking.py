import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import mlflow.sklearn
from sklearn.metrics import (roc_auc_score, precision_score,
                             recall_score, f1_score,
                             average_precision_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 12 — MLFLOW EXPERIMENT TRACKING
# ============================================================
# Why MLflow?
# 1. Track every experiment — parameters, metrics, artifacts
# 2. Compare runs side by side
# 3. Reproduce any result from the past
# 4. Register best model for production
# 5. Standard tool at every ML team (Uber, Meta, Airbnb)
#
# What we track:
# - Parameters: model hyperparameters
# - Metrics: AUC, PR-AUC, Recall, Precision, F1
# - Artifacts: model files, feature lists
# - Tags: experiment metadata
# ============================================================

print("Loading data...")
df = pd.read_csv('data/feature_matrix.csv')
final_features = joblib.load('models/final_features.pkl')
threshold = joblib.load('models/optimal_threshold.pkl')

X = df[final_features]
y = df['will_churn']

split_idx = int(len(df) * 0.8)
train_ids = df['driver_id'].iloc[:split_idx]
test_ids = df['driver_id'].iloc[split_idx:]

X_train = X[df['driver_id'].isin(train_ids)]
X_test = X[df['driver_id'].isin(test_ids)]
y_train = y[df['driver_id'].isin(train_ids)]
y_test = y[df['driver_id'].isin(test_ids)]

# Apply SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
pos_weight = (y_train_smote==0).sum() / (y_train_smote==1).sum()

# Scale for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

print(f"Data ready: {len(X_train):,} train | {len(X_test):,} test")

# ============================================================
# SETUP MLFLOW
# ============================================================

mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("driver_churn_prediction")

print(f"\nMLflow experiment: driver_churn_prediction")
print(f"Tracking URI: ./mlruns")

# ============================================================
# HELPER FUNCTION
# ============================================================

def log_model_run(model, X_tr, X_te, y_tr, y_te,
                  model_name, params, threshold,
                  is_scaled=False):
    with mlflow.start_run(run_name=model_name):

        # Log tags
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("dataset", "driver_churn_50k")
        mlflow.set_tag("features", f"{len(final_features)} features")
        mlflow.set_tag("imbalance_handling", "SMOTE")
        mlflow.set_tag("developer", "Sanyukta Kumari")

        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("n_features", len(final_features))
        mlflow.log_param("train_size", len(X_tr))
        mlflow.log_param("test_size", len(X_te))
        mlflow.log_param("smote_applied", True)

        # Train model
        model.fit(X_tr, y_tr)

        # Get predictions
        if is_scaled:
            probs = model.predict_proba(X_te)[:, 1]
        else:
            probs = model.predict_proba(X_te)[:, 1]

        preds = (probs >= threshold).astype(int)

        # Calculate metrics
        auc = roc_auc_score(y_te, probs)
        pr_auc = average_precision_score(y_te, probs)
        precision = precision_score(y_te, preds, zero_division=0)
        recall = recall_score(y_te, preds)
        f1 = f1_score(y_te, preds)

        # Log metrics
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("threshold", threshold)

        # Log feature list as artifact
        features_df = pd.DataFrame({'feature': final_features})
        features_df.to_csv('/tmp/features.csv', index=False)
        mlflow.log_artifact('/tmp/features.csv')

        # Log model
        if 'XGB' in model_name:
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")

        run_id = mlflow.active_run().info.run_id

        print(f"\n  {model_name}:")
        print(f"    Run ID:    {run_id[:8]}...")
        print(f"    AUC:       {auc:.4f}")
        print(f"    PR-AUC:    {pr_auc:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall:    {recall:.4f}")
        print(f"    F1:        {f1:.4f}")

        return run_id, auc, recall, f1

# ============================================================
# RUN 1 — LOGISTIC REGRESSION
# ============================================================

print(f"\n=== LOGGING RUN 1: LOGISTIC REGRESSION ===")

lr_params = {
    'model': 'LogisticRegression',
    'C': 1.0,
    'class_weight': 'balanced',
    'max_iter': 1000,
    'solver': 'lbfgs'
}

lr = LogisticRegression(
    C=1.0,
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)

lr_run_id, lr_auc, lr_recall, lr_f1 = log_model_run(
    lr, X_train_scaled, X_test_scaled,
    y_train_smote, y_test,
    "Logistic_Regression", lr_params, threshold
)

# ============================================================
# RUN 2 — RANDOM FOREST
# ============================================================

print(f"\n=== LOGGING RUN 2: RANDOM FOREST ===")

rf_params = {
    'model': 'RandomForest',
    'n_estimators': 200,
    'max_depth': 10,
    'min_samples_leaf': 5,
    'class_weight': 'balanced'
}

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_run_id, rf_auc, rf_recall, rf_f1 = log_model_run(
    rf, X_train_smote, X_test,
    y_train_smote, y_test,
    "Random_Forest", rf_params, threshold
)

# ============================================================
# RUN 3 — XGBOOST BASELINE
# ============================================================

print(f"\n=== LOGGING RUN 3: XGBOOST BASELINE ===")

xgb_params = {
    'model': 'XGBoost',
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': round(pos_weight, 2)
}

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

xgb_run_id, xgb_auc, xgb_recall, xgb_f1 = log_model_run(
    xgb, X_train_smote, X_test,
    y_train_smote, y_test,
    "XGBoost_Baseline", xgb_params, threshold
)

# ============================================================
# RUN 4 — XGBOOST WITH DIFFERENT THRESHOLDS
# ============================================================

print(f"\n=== LOGGING RUN 4: XGBOOST HIGH RECALL ===")

xgb_params_hr = {
    'model': 'XGBoost',
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': round(pos_weight, 2),
    'threshold_strategy': 'high_recall'
}

xgb_hr = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=pos_weight,
    random_state=42,
    verbosity=0
)

xgb_hr_run_id, xgb_hr_auc, xgb_hr_recall, xgb_hr_f1 = log_model_run(
    xgb_hr, X_train_smote, X_test,
    y_train_smote, y_test,
    "XGBoost_HighRecall", xgb_params_hr, 0.20
)

# ============================================================
# COMPARE ALL RUNS
# ============================================================

print(f"\n=== ALL RUNS COMPARISON ===")

runs_data = [
    ("Logistic Regression", lr_run_id[:8], lr_auc, lr_recall, lr_f1),
    ("Random Forest", rf_run_id[:8], rf_auc, rf_recall, rf_f1),
    ("XGBoost Baseline", xgb_run_id[:8], xgb_auc, xgb_recall, xgb_f1),
    ("XGBoost HighRecall", xgb_hr_run_id[:8], xgb_hr_auc, xgb_hr_recall, xgb_hr_f1),
]

print(f"\n{'Model':<22} {'Run ID':>10} {'AUC':>8} {'Recall':>8} {'F1':>8}")
print(f"{'-'*60}")
for name, run_id, auc, recall, f1 in runs_data:
    print(f"{name:<22} {run_id:>10} {auc:>8.4f} {recall:>8.4f} {f1:>8.4f}")

# ============================================================
# REGISTER BEST MODEL
# ============================================================

print(f"\n=== REGISTERING BEST MODEL ===")

best_run = max(runs_data, key=lambda x: x[3])
print(f"Best model by recall: {best_run[0]}")
print(f"Run ID: {best_run[1]}")
print(f"Recall: {best_run[3]:.4f}")

print(f"\nMLflow tracking complete!")
print(f"All runs saved to: ./mlruns")
print(f"\nTo view MLflow UI run:")
print(f"  mlflow ui")
print(f"  Then open: http://localhost:5000")
print(f"\n=== STEP 12 COMPLETE ===")
print(f"\nSummary of all experiments:")
print(f"  Total runs logged: {len(runs_data)}")
print(f"  Best model: {best_run[0]}")
print(f"  Best recall: {best_run[3]:.4f}")
print(f"  Best AUC: {best_run[2]:.4f}")