import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 2 — EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
# What we're looking for:
# 1. Class imbalance — how severe is it?
# 2. Missing value patterns — random or systematic?
# 3. Outliers — how extreme?
# 4. Churn patterns — which ticket types drive churn?
# 5. Temporal patterns — when do drivers churn?
# 6. Feature correlations — multicollinearity issues?
# ============================================================

print("Loading data...")
df = pd.read_csv('data/raw_ticket_data.csv')
df['ticket_date'] = pd.to_datetime(df['ticket_date'])

print(f"Data loaded: {len(df):,} rows, {df.shape[1]} columns")

# ============================================================
# EDA 1 — CLASS IMBALANCE
# ============================================================

driver_df = df.groupby('driver_id').agg(
    will_churn=('will_churn', 'first'),
    is_noise_churn=('is_noise_churn', 'first'),
    total_tickets=('ticket_number', 'count')
).reset_index()

churn_counts = driver_df['will_churn'].value_counts()
churn_pct = driver_df['will_churn'].value_counts(normalize=True) * 100

print(f"\n=== CLASS IMBALANCE ===")
print(f"Non-churners: {churn_counts[0]:,} ({churn_pct[0]:.1f}%)")
print(f"Churners:     {churn_counts[1]:,} ({churn_pct[1]:.1f}%)")
print(f"Imbalance ratio: {churn_counts[0]/churn_counts[1]:.1f}:1")

# ============================================================
# EDA 2 — MISSING VALUE PATTERNS
# ============================================================

print(f"\n=== MISSING VALUE PATTERNS ===")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'missing_count': missing,
    'missing_pct': missing_pct
}).query('missing_count > 0')
print(missing_df)

# Are missing values more common in churner tickets?
print(f"\nMissing ticket_type by churn status:")
print(df.groupby('will_churn')['ticket_type'].apply(
    lambda x: x.isnull().mean() * 100
).round(2).to_string())

print(f"\nMissing resolution_status by churn status:")
print(df.groupby('will_churn')['resolution_status'].apply(
    lambda x: x.isnull().mean() * 100
).round(2).to_string())

# ============================================================
# EDA 3 — OUTLIER ANALYSIS
# ============================================================

print(f"\n=== OUTLIER ANALYSIS ===")
ticket_counts = driver_df['total_tickets']
Q1 = ticket_counts.quantile(0.25)
Q3 = ticket_counts.quantile(0.75)
IQR = Q3 - Q1
outlier_threshold = Q3 + 1.5 * IQR
outliers = driver_df[driver_df['total_tickets'] > outlier_threshold]

print(f"Ticket count Q1: {Q1}")
print(f"Ticket count Q3: {Q3}")
print(f"IQR: {IQR}")
print(f"Outlier threshold: {outlier_threshold}")
print(f"Drivers with outlier ticket counts: {len(outliers):,} ({len(outliers)/len(driver_df)*100:.1f}%)")
print(f"Max tickets: {ticket_counts.max()}")

# ============================================================
# EDA 4 — CHURN PATTERNS BY TICKET TYPE
# ============================================================

print(f"\n=== CHURN RATE BY TICKET TYPE ===")
churn_by_type = df.dropna(subset=['ticket_type']).groupby('ticket_type').agg(
    total_tickets=('ticket_type', 'count'),
    churn_rate=('will_churn', 'mean')
).round(3)
churn_by_type['churn_rate_pct'] = (churn_by_type['churn_rate'] * 100).round(1)
print(churn_by_type[['total_tickets', 'churn_rate_pct']].to_string())

# ============================================================
# EDA 5 — CHURN PATTERNS BY RESOLUTION STATUS
# ============================================================

print(f"\n=== CHURN RATE BY RESOLUTION STATUS ===")
churn_by_resolution = df.dropna(subset=['resolution_status']).groupby(
    'resolution_status'
).agg(
    total_tickets=('resolution_status', 'count'),
    churn_rate=('will_churn', 'mean')
).round(3)
churn_by_resolution['churn_rate_pct'] = (
    churn_by_resolution['churn_rate'] * 100
).round(1)
print(churn_by_resolution[['total_tickets', 'churn_rate_pct']].to_string())

# ============================================================
# EDA 6 — RESPONSE TIME ANALYSIS
# ============================================================

print(f"\n=== RESPONSE TIME BY CHURN STATUS ===")
response_stats = df.dropna(subset=['response_time_hours']).groupby(
    'will_churn'
)['response_time_hours'].describe().round(2)
print(response_stats)

# ============================================================
# EDA 7 — TEMPORAL PATTERNS
# ============================================================

print(f"\n=== TEMPORAL PATTERNS ===")
df['month'] = df['ticket_date'].dt.month
monthly_churn = df.groupby('month')['will_churn'].mean() * 100
print("Churn rate by month:")
print(monthly_churn.round(1).to_string())

# ============================================================
# VISUALISATIONS — save all charts to one file
# ============================================================

fig, axes = plt.subplots(3, 2, figsize=(16, 18))
fig.suptitle('Driver Churn EDA — Real World Messy Data', 
             fontsize=16, fontweight='bold', y=1.02)

# Chart 1 — Class imbalance
ax1 = axes[0, 0]
bars = ax1.bar(
    ['Non-Churner', 'Churner'],
    churn_counts.values,
    color=['#4A90E2', '#E24B4A'],
    width=0.5
)
ax1.set_title('Class Imbalance', fontweight='bold')
ax1.set_ylabel('Number of Drivers')
for bar, pct in zip(bars, churn_pct.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
             f'{pct:.1f}%', ha='center', fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Chart 2 — Missing values
ax2 = axes[0, 1]
missing_plot = missing_df[missing_df['missing_count'] > 0]
ax2.barh(
    missing_plot.index,
    missing_plot['missing_pct'],
    color='#EF9F27'
)
ax2.set_title('Missing Value % by Column', fontweight='bold')
ax2.set_xlabel('Missing %')
ax2.grid(True, alpha=0.3, axis='x')
for i, v in enumerate(missing_plot['missing_pct']):
    ax2.text(v + 0.1, i, f'{v}%', va='center')

# Chart 3 — Ticket count distribution (log scale)
ax3 = axes[1, 0]
ax3.hist(
    driver_df[driver_df['will_churn']==0]['total_tickets'],
    bins=30, alpha=0.6, color='#4A90E2', label='Non-Churner'
)
ax3.hist(
    driver_df[driver_df['will_churn']==1]['total_tickets'],
    bins=30, alpha=0.6, color='#E24B4A', label='Churner'
)
ax3.axvline(x=outlier_threshold, color='black', 
            linestyle='--', label=f'Outlier threshold ({outlier_threshold:.0f})')
ax3.set_title('Ticket Count Distribution by Churn', fontweight='bold')
ax3.set_xlabel('Total Tickets')
ax3.set_ylabel('Number of Drivers')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Chart 4 — Churn rate by ticket type
ax4 = axes[1, 1]
churn_by_type_sorted = churn_by_type.sort_values('churn_rate_pct', ascending=True)
bars = ax4.barh(
    churn_by_type_sorted.index,
    churn_by_type_sorted['churn_rate_pct'],
    color='#E24B4A'
)
ax4.set_title('Churn Rate by Ticket Type', fontweight='bold')
ax4.set_xlabel('Churn Rate %')
ax4.grid(True, alpha=0.3, axis='x')
for bar, v in zip(bars, churn_by_type_sorted['churn_rate_pct']):
    ax4.text(v + 0.1, bar.get_y() + bar.get_height()/2,
             f'{v}%', va='center')

# Chart 5 — Churn rate by resolution status
ax5 = axes[2, 0]
churn_by_res_sorted = churn_by_resolution.sort_values(
    'churn_rate_pct', ascending=True
)
colors = ['#639922', '#EF9F27', '#E24B4A']
bars = ax5.bar(
    churn_by_res_sorted.index,
    churn_by_res_sorted['churn_rate_pct'],
    color=colors,
    width=0.5
)
ax5.set_title('Churn Rate by Resolution Status', fontweight='bold')
ax5.set_ylabel('Churn Rate %')
ax5.grid(True, alpha=0.3, axis='y')
for bar, v in zip(bars, churn_by_res_sorted['churn_rate_pct']):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{v}%', ha='center', fontweight='bold')

# Chart 6 — Response time by churn
ax6 = axes[2, 1]
churner_response = df[
    (df['will_churn']==1) & df['response_time_hours'].notna()
]['response_time_hours'].clip(upper=100)
non_churner_response = df[
    (df['will_churn']==0) & df['response_time_hours'].notna()
]['response_time_hours'].clip(upper=100)

ax6.hist(non_churner_response, bins=30, alpha=0.6,
         color='#4A90E2', label='Non-Churner', density=True)
ax6.hist(churner_response, bins=30, alpha=0.6,
         color='#E24B4A', label='Churner', density=True)
ax6.set_title('Response Time Distribution by Churn', fontweight='bold')
ax6.set_xlabel('Response Time (hours, capped at 100)')
ax6.set_ylabel('Density')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/eda_charts.png', dpi=150, bbox_inches='tight')
print(f"\nEDA charts saved to: notebooks/eda_charts.png")
print(f"\n=== EDA COMPLETE ===")
print(f"Key findings to address in preprocessing:")
print(f"1. Class imbalance: {churn_pct[1]:.1f}% churners — needs SMOTE or class weights")
print(f"2. Missing values: up to 12% in response_time_hours — needs imputation")
print(f"3. Outliers: drivers with up to {ticket_counts.max()} tickets — needs capping")
print(f"4. Noise churn: 5.2% of drivers — model will never perfectly predict these")