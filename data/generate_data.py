import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1 — REALISTIC MESSY DRIVER CHURN DATA GENERATION
# ============================================================
# What makes this realistic:
# 1. Class imbalance — only 15% churners (real world ratio)
# 2. Missing values — not all fields always captured
# 3. Outliers — some drivers have extreme ticket counts
# 4. Noisy labels — some drivers churn for non-support reasons
# 5. Multicollinearity — some features naturally correlated
# 6. Temporal structure — tickets happen over time
# ============================================================

random.seed(42)
np.random.seed(42)

NUM_DRIVERS = 50000
CHURN_RATE = 0.15  # realistic 15% churn rate

ticket_types = [
    'fare_dispute',
    'defective_trip',
    'demand_issue',
    'payment_delay',
    'app_issue'
]

resolution_statuses = ['resolved', 'unresolved', 'partial']

print("Generating realistic messy driver churn data...")
print(f"Total drivers: {NUM_DRIVERS:,}")
print(f"Target churn rate: {CHURN_RATE:.0%}")
print(f"This will take a moment...\n")

records = []

for driver_id in range(1, NUM_DRIVERS + 1):

    # --- CLASS IMBALANCE ---
    # Only 15% of drivers churn
    will_churn = random.random() < CHURN_RATE

    # --- NOISY LABELS ---
    # 5% of drivers churn for non-support reasons
    # (moved city, got another job, health issues)
    # These are impossible to predict from support data
    noise_churn = random.random() < 0.05
    if noise_churn:
        will_churn = True  # churn regardless of support quality

    # --- TICKET COUNT ---
    # Churners have more tickets on average
    # But with high variance — some churners have few tickets
    # Some non-churners have many tickets but stay
    if will_churn and not noise_churn:
        # Support-driven churners have more unresolved tickets
        num_tickets = max(1, int(np.random.negative_binomial(4, 0.4)))
    elif noise_churn:
        # Noise churners have normal ticket patterns
        num_tickets = max(1, int(np.random.negative_binomial(2, 0.5)))
    else:
        num_tickets = max(1, int(np.random.negative_binomial(2, 0.6)))

    # --- OUTLIERS ---
    # 1% of drivers have extreme ticket counts (super complainers)
    if random.random() < 0.01:
        num_tickets = random.randint(20, 50)

    # Generate ticket sequence
    start_date = datetime(2023, 1, 1) + timedelta(
        days=random.randint(0, 300)
    )
    ticket_date = start_date

    for ticket_num in range(num_tickets):

        # --- MISSING VALUES ---
        # 8% chance ticket type not captured
        if random.random() < 0.08:
            ticket_type = None
        elif will_churn and not noise_churn:
            ticket_type = random.choices(
                ticket_types,
                weights=[0.35, 0.30, 0.20, 0.10, 0.05]
            )[0]
        else:
            ticket_type = random.choices(
                ticket_types,
                weights=[0.15, 0.15, 0.25, 0.25, 0.20]
            )[0]

        # 10% chance resolution status not captured
        if random.random() < 0.10:
            resolution = None
        elif will_churn and not noise_churn:
            resolution = random.choices(
                resolution_statuses,
                weights=[0.20, 0.60, 0.20]
            )[0]
        else:
            resolution = random.choices(
                resolution_statuses,
                weights=[0.70, 0.10, 0.20]
            )[0]

        # --- TEMPORAL STRUCTURE ---
        # Gap between tickets gets shorter as frustration builds
        if will_churn and ticket_num > 2:
            gap_days = max(1, int(random.gauss(3, 2)))
        else:
            gap_days = max(1, int(random.gauss(10, 5)))

        # --- MULTICOLLINEARITY ---
        # Response time correlated with resolution status
        # Unresolved tickets tend to have longer response times
        if resolution == 'unresolved':
            response_time_hours = random.gauss(48, 12)
        elif resolution == 'partial':
            response_time_hours = random.gauss(24, 8)
        else:
            response_time_hours = random.gauss(6, 3)

        # Add some missing response times
        if random.random() < 0.12:
            response_time_hours = None

        response_time_hours = max(1, response_time_hours) \
            if response_time_hours else None

        ticket_date = ticket_date + timedelta(days=gap_days)

        records.append({
            'driver_id': driver_id,
            'ticket_number': ticket_num + 1,
            'ticket_date': ticket_date,
            'ticket_type': ticket_type,
            'resolution_status': resolution,
            'response_time_hours': round(response_time_hours, 1)
                if response_time_hours else None,
            'will_churn': int(will_churn),
            'is_noise_churn': int(noise_churn)
        })

# Create dataframe
df_raw = pd.DataFrame(records)

# Save raw data
df_raw.to_csv('data/raw_ticket_data.csv', index=False)

# Print summary
print(f"=== DATA GENERATION COMPLETE ===")
print(f"Total ticket records: {len(df_raw):,}")
print(f"Total drivers: {df_raw['driver_id'].nunique():,}")
churn_rate = df_raw.groupby('driver_id')['will_churn'].first().mean()
print(f"Actual churn rate: {churn_rate:.1%}")
noise_rate = df_raw.groupby('driver_id')['is_noise_churn'].first().mean()
print(f"Noise churn rate: {noise_rate:.1%}")
print(f"\n=== MISSING VALUES ===")
print(df_raw.isnull().sum())
print(f"\n=== TICKET COUNT DISTRIBUTION ===")
ticket_counts = df_raw.groupby('driver_id').size()
print(f"Min tickets per driver: {ticket_counts.min()}")
print(f"Max tickets per driver: {ticket_counts.max()}")
print(f"Avg tickets per driver: {ticket_counts.mean():.1f}")
print(f"Median tickets per driver: {ticket_counts.median():.1f}")
print(f"\n=== SAMPLE DATA ===")
print(df_raw.head(10).to_string())
print(f"\nRaw data saved to: data/raw_ticket_data.csv")