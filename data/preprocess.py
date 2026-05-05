import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 3 — DATA CLEANING & PREPROCESSING
# ============================================================
# Problems to solve:
# 1. Missing values — imputation strategies
# 2. Outliers — IQR capping
# 3. Categorical encoding — ticket type, resolution status
# 4. Temporal features — extract month, day of week
# 5. Save clean data for feature engineering
# ============================================================

print("Loading raw data...")
df = pd.read_csv('data/raw_ticket_data.csv')
df['ticket_date'] = pd.to_datetime(df['ticket_date'])

print(f"Raw data shape: {df.shape}")
print(f"Missing values before cleaning:")
print(df.isnull().sum())

# ============================================================
# CLEANING 1 — MISSING VALUE IMPUTATION
# ============================================================

print(f"\n=== CLEANING 1: MISSING VALUE IMPUTATION ===")

# Strategy for ticket_type:
# Missing ticket type is informative — agent didn't capture it
# Could be rushed interaction = potentially frustrated driver
# Impute with 'unknown' as its own category
df['ticket_type'] = df['ticket_type'].fillna('unknown')
df['ticket_type_missing'] = (df['ticket_type'] == 'unknown').astype(int)
print(f"ticket_type: filled with 'unknown', created missing indicator flag")

# Strategy for resolution_status:
# Missing resolution = ticket never properly closed
# Treat as 'unresolved' — most conservative assumption
df['resolution_status'] = df['resolution_status'].fillna('unresolved')
df['resolution_missing'] = (df['resolution_status'] == 'unresolved').astype(int)
print(f"resolution_status: filled with 'unresolved' (conservative assumption)")

# Strategy for response_time_hours:
# Missing response time = no response logged
# Impute with median per resolution status group
# Unresolved tickets have different response time distribution
median_response_by_resolution = df.groupby('resolution_status')[
    'response_time_hours'
].transform('median')
df['response_time_hours'] = df['response_time_hours'].fillna(
    median_response_by_resolution
)
df['response_time_missing'] = (df['response_time_hours'].isnull()).astype(int)
print(f"response_time_hours: filled with median per resolution group")

print(f"\nMissing values after imputation:")
print(df.isnull().sum())

# ============================================================
# CLEANING 2 — OUTLIER TREATMENT
# ============================================================

print(f"\n=== CLEANING 2: OUTLIER TREATMENT ===")

# Calculate ticket counts per driver
ticket_counts = df.groupby('driver_id')['ticket_number'].count()

# IQR based capping
Q1 = ticket_counts.quantile(0.25)
Q3 = ticket_counts.quantile(0.75)
IQR = Q3 - Q1
upper_cap = Q3 + 1.5 * IQR

print(f"Ticket count outlier threshold: {upper_cap:.0f}")
print(f"Drivers above threshold: {(ticket_counts > upper_cap).sum():,}")

# Flag outlier drivers
outlier_drivers = ticket_counts[ticket_counts > upper_cap].index
df['is_outlier_driver'] = df['driver_id'].isin(outlier_drivers).astype(int)

outlier_count = df['is_outlier_driver'].sum()
print(f"Tickets from outlier drivers: {outlier_count:,} ({outlier_count/len(df)*100:.1f}%)")
print(f"Note: keeping outliers but flagging them — XGBoost handles outliers well")

# Cap response time at 99th percentile
p99_response = df['response_time_hours'].quantile(0.99)
df['response_time_hours'] = df['response_time_hours'].clip(upper=p99_response)
print(f"Response time capped at 99th percentile: {p99_response:.1f} hours")

# ============================================================
# CLEANING 3 — CATEGORICAL ENCODING
# ============================================================

print(f"\n=== CLEANING 3: CATEGORICAL ENCODING ===")

# Encode ticket type
ticket_type_map = {
    'fare_dispute': 0,
    'defective_trip': 1,
    'demand_issue': 2,
    'payment_delay': 3,
    'app_issue': 4,
    'unknown': 5
}
df['ticket_type_encoded'] = df['ticket_type'].map(ticket_type_map)
print(f"ticket_type encoded: {ticket_type_map}")

# Encode resolution status
resolution_map = {
    'resolved': 0,
    'partial': 1,
    'unresolved': 2
}
df['resolution_encoded'] = df['resolution_status'].map(resolution_map)
print(f"resolution_status encoded: {resolution_map}")

# Risk score per ticket type based on EDA findings
# fare_dispute: 57.6% churn, defective_trip: 54%
ticket_risk_map = {
    'fare_dispute': 3,
    'defective_trip': 3,
    'demand_issue': 2,
    'payment_delay': 1,
    'app_issue': 1,
    'unknown': 2
}
df['ticket_risk_score'] = df['ticket_type'].map(ticket_risk_map)
print(f"ticket_risk_score created based on EDA churn rates")

# ============================================================
# CLEANING 4 — TEMPORAL FEATURES
# ============================================================

print(f"\n=== CLEANING 4: TEMPORAL FEATURES ===")

df['ticket_month'] = df['ticket_date'].dt.month
df['ticket_dayofweek'] = df['ticket_date'].dt.dayofweek
df['ticket_quarter'] = df['ticket_date'].dt.quarter

# Is it Q4? (November spike from EDA)
df['is_q4'] = (df['ticket_quarter'] == 4).astype(int)

print(f"Temporal features created: month, dayofweek, quarter, is_q4")
print(f"Q4 tickets: {df['is_q4'].sum():,} ({df['is_q4'].mean()*100:.1f}%)")

# ============================================================
# SAVE CLEAN DATA
# ============================================================

df.to_csv('data/clean_ticket_data.csv', index=False)

print(f"\n=== CLEANING COMPLETE ===")
print(f"Clean data shape: {df.shape}")
print(f"New columns added: {df.shape[1] - 8}")
print(f"\nNew columns:")
new_cols = [
    'ticket_type_missing', 'resolution_missing',
    'response_time_missing', 'is_outlier_driver',
    'ticket_type_encoded', 'resolution_encoded',
    'ticket_risk_score', 'ticket_month',
    'ticket_dayofweek', 'ticket_quarter', 'is_q4'
]
for col in new_cols:
    print(f"  + {col}")

print(f"\nClean data saved to: data/clean_ticket_data.csv")
print(f"\nDecisions made:")
print(f"  1. Missing ticket_type → 'unknown' category + missing flag")
print(f"  2. Missing resolution → 'unresolved' (conservative) + missing flag")
print(f"  3. Missing response_time → median per resolution group")
print(f"  4. Outlier drivers flagged but kept (XGBoost handles them)")
print(f"  5. Response time capped at 99th percentile")
print(f"  6. Categorical encoding based on EDA risk levels")
print(f"  7. Q4 flag added based on seasonal pattern from EDA")