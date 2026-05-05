import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 4 — FEATURE ENGINEERING
# ============================================================
# Transforming ticket-level data → one row per driver
# Feature families:
# 1. Sequence features — full ticket history
# 2. Recency weighted features — recent tickets matter more
# 3. Survival analysis features — journey length, escalation
# 4. Resolution quality features — how well issues resolved
# 5. Temporal features — seasonal patterns
# 6. Interaction features — combinations XGBoost can enhance
# ============================================================

print("Loading clean data...")
df = pd.read_csv('data/clean_ticket_data.csv')
df['ticket_date'] = pd.to_datetime(df['ticket_date'])

print(f"Clean data shape: {df.shape}")
print(f"Building features for {df['driver_id'].nunique():,} drivers...")

reference_date = df['ticket_date'].max()

driver_features = []

for driver_id, group in df.groupby('driver_id'):

    group = group.sort_values('ticket_date')
    tickets = group.to_dict('records')
    num_tickets = len(tickets)
    will_churn = tickets[0]['will_churn']
    is_noise_churn = tickets[0]['is_noise_churn']

    # --------------------------------------------------------
    # FEATURE FAMILY 1 — SEQUENCE FEATURES
    # --------------------------------------------------------

    # Basic counts
    total_tickets = num_tickets
    fare_disputes = sum(1 for t in tickets if t['ticket_type'] == 'fare_dispute')
    defective_trips = sum(1 for t in tickets if t['ticket_type'] == 'defective_trip')
    demand_issues = sum(1 for t in tickets if t['ticket_type'] == 'demand_issue')
    payment_delays = sum(1 for t in tickets if t['ticket_type'] == 'payment_delay')
    app_issues = sum(1 for t in tickets if t['ticket_type'] == 'app_issue')
    unknown_tickets = sum(1 for t in tickets if t['ticket_type'] == 'unknown')

    # High risk ticket count (fare_dispute + defective_trip from EDA)
    high_risk_tickets = fare_disputes + defective_trips

    # Resolution counts
    unresolved_count = sum(
        1 for t in tickets if t['resolution_status'] == 'unresolved'
    )
    resolved_count = sum(
        1 for t in tickets if t['resolution_status'] == 'resolved'
    )
    partial_count = sum(
        1 for t in tickets if t['resolution_status'] == 'partial'
    )
    unresolved_rate = unresolved_count / num_tickets
    resolved_rate = resolved_count / num_tickets

    # Repeat issues (same problem recurring = frustration)
    ticket_type_list = [t['ticket_type'] for t in tickets]
    repeat_issues = num_tickets - len(set(ticket_type_list))
    has_repeat_issues = int(repeat_issues > 0)

    # Outlier driver flag
    is_outlier_driver = tickets[0]['is_outlier_driver']

    # --------------------------------------------------------
    # FEATURE FAMILY 2 — LAST & SECOND LAST TICKET
    # (directly addresses interview question)
    # --------------------------------------------------------

    last_ticket_type = tickets[-1]['ticket_type']
    last_ticket_resolved = int(
        tickets[-1]['resolution_status'] == 'resolved'
    )
    last_ticket_unresolved = int(
        tickets[-1]['resolution_status'] == 'unresolved'
    )
    last_ticket_risk = tickets[-1]['ticket_risk_score']
    last_response_time = tickets[-1]['response_time_hours']

    # Second last ticket (what interviewer asked about!)
    if num_tickets >= 2:
        second_last_resolved = int(
            tickets[-2]['resolution_status'] == 'resolved'
        )
        second_last_unresolved = int(
            tickets[-2]['resolution_status'] == 'unresolved'
        )
        second_last_risk = tickets[-2]['ticket_risk_score']
    else:
        second_last_resolved = 1
        second_last_unresolved = 0
        second_last_risk = 0

    # Consecutive unresolved at end (most dangerous pattern)
    consecutive_unresolved_end = 0
    for ticket in reversed(tickets):
        if ticket['resolution_status'] == 'unresolved':
            consecutive_unresolved_end += 1
        else:
            break

    # --------------------------------------------------------
    # FEATURE FAMILY 3 — RECENCY WEIGHTED FEATURES
    # --------------------------------------------------------
    # Weights: last=1.0, second=0.7, third=0.5, fourth=0.3...
    weights = [1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.1, 0.1,
               0.05, 0.05]

    weighted_unresolved = 0
    weighted_high_risk = 0
    weighted_risk_score = 0
    weighted_response_time = 0
    total_weight = 0

    for i, ticket in enumerate(reversed(tickets)):
        w = weights[i] if i < len(weights) else 0.05
        total_weight += w

        if ticket['resolution_status'] == 'unresolved':
            weighted_unresolved += w
        if ticket['ticket_type'] in ['fare_dispute', 'defective_trip']:
            weighted_high_risk += w
        weighted_risk_score += w * ticket['ticket_risk_score']
        weighted_response_time += w * ticket['response_time_hours']

    weighted_risk_score = weighted_risk_score / total_weight
    weighted_response_time = weighted_response_time / total_weight

    # --------------------------------------------------------
    # FEATURE FAMILY 4 — SURVIVAL ANALYSIS FEATURES
    # --------------------------------------------------------

    first_date = group['ticket_date'].min()
    last_date = group['ticket_date'].max()

    # Journey length
    journey_length_days = (last_date - first_date).days + 1

    # Days since last ticket
    days_since_last_ticket = (reference_date - last_date).days

    # Average and minimum gap between tickets
    if num_tickets > 1:
        dates = group['ticket_date'].tolist()
        gaps = [(dates[i+1] - dates[i]).days
                for i in range(len(dates)-1)]
        avg_gap_days = np.mean(gaps)
        min_gap_days = min(gaps)
        max_gap_days = max(gaps)
        gap_std = np.std(gaps)

        # Escalation rate — gaps getting shorter over time
        escalation_rate = gaps[0] - gaps[-1]

        # Tickets per week
        tickets_per_week = num_tickets / max(1, journey_length_days / 7)
    else:
        avg_gap_days = 0
        min_gap_days = 0
        max_gap_days = 0
        gap_std = 0
        escalation_rate = 0
        tickets_per_week = num_tickets

    # --------------------------------------------------------
    # FEATURE FAMILY 5 — RESPONSE TIME FEATURES
    # --------------------------------------------------------

    response_times = [t['response_time_hours'] for t in tickets]
    avg_response_time = np.mean(response_times)
    max_response_time = np.max(response_times)
    response_time_trend = (
        response_times[-1] - response_times[0]
        if num_tickets > 1 else 0
    )

    # --------------------------------------------------------
    # FEATURE FAMILY 6 — TEMPORAL FEATURES
    # --------------------------------------------------------

    last_ticket_month = tickets[-1]['ticket_month']
    last_ticket_quarter = tickets[-1]['ticket_quarter']
    is_q4_churn = int(tickets[-1]['ticket_quarter'] == 4)
    q4_tickets = sum(1 for t in tickets if t['is_q4'] == 1)
    q4_ticket_rate = q4_tickets / num_tickets

    # --------------------------------------------------------
    # FEATURE FAMILY 7 — INTERACTION FEATURES
    # --------------------------------------------------------
    # These capture combinations XGBoost finds naturally
    # but making them explicit helps logistic regression too

    # Unresolved high risk combo (most dangerous)
    unresolved_high_risk = unresolved_count * high_risk_tickets

    # Many tickets + high unresolved rate
    volume_x_unresolved = total_tickets * unresolved_rate

    # Long response time + unresolved
    response_x_unresolved = avg_response_time * unresolved_rate

    driver_features.append({
        'driver_id': driver_id,

        # Sequence features
        'total_tickets': total_tickets,
        'fare_disputes': fare_disputes,
        'defective_trips': defective_trips,
        'demand_issues': demand_issues,
        'payment_delays': payment_delays,
        'app_issues': app_issues,
        'unknown_tickets': unknown_tickets,
        'high_risk_tickets': high_risk_tickets,
        'unresolved_count': unresolved_count,
        'resolved_count': resolved_count,
        'partial_count': partial_count,
        'unresolved_rate': unresolved_rate,
        'resolved_rate': resolved_rate,
        'repeat_issues': repeat_issues,
        'has_repeat_issues': has_repeat_issues,
        'is_outlier_driver': is_outlier_driver,

        # Last & second last ticket
        'last_ticket_resolved': last_ticket_resolved,
        'last_ticket_unresolved': last_ticket_unresolved,
        'last_ticket_risk': last_ticket_risk,
        'last_response_time': last_response_time,
        'second_last_resolved': second_last_resolved,
        'second_last_unresolved': second_last_unresolved,
        'second_last_risk': second_last_risk,
        'consecutive_unresolved_end': consecutive_unresolved_end,

        # Recency weighted
        'weighted_unresolved': weighted_unresolved,
        'weighted_high_risk': weighted_high_risk,
        'weighted_risk_score': weighted_risk_score,
        'weighted_response_time': weighted_response_time,

        # Survival analysis
        'journey_length_days': journey_length_days,
        'days_since_last_ticket': days_since_last_ticket,
        'avg_gap_days': avg_gap_days,
        'min_gap_days': min_gap_days,
        'max_gap_days': max_gap_days,
        'gap_std': gap_std,
        'escalation_rate': escalation_rate,
        'tickets_per_week': tickets_per_week,

        # Response time
        'avg_response_time': avg_response_time,
        'max_response_time': max_response_time,
        'response_time_trend': response_time_trend,

        # Temporal
        'last_ticket_month': last_ticket_month,
        'last_ticket_quarter': last_ticket_quarter,
        'is_q4_churn': is_q4_churn,
        'q4_ticket_rate': q4_ticket_rate,

        # Interaction features
        'unresolved_high_risk': unresolved_high_risk,
        'volume_x_unresolved': volume_x_unresolved,
        'response_x_unresolved': response_x_unresolved,

        # Target
        'will_churn': will_churn,
        'is_noise_churn': is_noise_churn
    })

df_features = pd.DataFrame(driver_features)

# Save
df_features.to_csv('data/feature_matrix.csv', index=False)

print(f"\n=== FEATURE ENGINEERING COMPLETE ===")
print(f"Feature matrix shape: {df_features.shape}")
print(f"Total features created: {df_features.shape[1] - 3}")
print(f"\nFeature families:")
print(f"  Sequence features:        16")
print(f"  Last/second last ticket:   8")
print(f"  Recency weighted:          4")
print(f"  Survival analysis:         8")
print(f"  Response time:             3")
print(f"  Temporal:                  4")
print(f"  Interaction features:      3")
print(f"  Total:                    46")
print(f"\nClass distribution:")
print(df_features['will_churn'].value_counts())
print(f"\nChurn rate: {df_features['will_churn'].mean():.1%}")
print(f"\nFeature matrix saved to: data/feature_matrix.csv")
print(f"\nSample features:")
print(df_features.head(3).to_string())