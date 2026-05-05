from api.agent import run_agent

# Test 4 drivers — one for each risk level
drivers = [
    {
        "driver_id": 1001,
        "total_tickets": 15,
        "fare_disputes": 6,
        "defective_trips": 5,
        "demand_issues": 2,
        "payment_delays": 1,
        "app_issues": 1,
        "unknown_tickets": 0,
        "unresolved_count": 13,
        "resolved_count": 1,
        "partial_count": 1,
        "unresolved_rate": 0.87,
        "resolved_rate": 0.07,
        "last_ticket_unresolved": 1,
        "consecutive_unresolved_end": 5,
        "weighted_unresolved": 3.5,
        "weighted_high_risk": 3.2,
        "days_since_last_ticket": 1,
        "avg_gap_days": 2.1,
        "gap_std": 0.8,
        "escalation_rate": 4.5,
        "tickets_per_week": 2.8,
        "unresolved_high_risk": 78.0,
        "response_x_unresolved": 45.0,
        "second_last_resolved": 0,
        "second_last_unresolved": 1,
        "second_last_risk": 3
    },
    {
        "driver_id": 1002,
        "total_tickets": 7,
        "fare_disputes": 3,
        "defective_trips": 2,
        "demand_issues": 1,
        "payment_delays": 1,
        "app_issues": 0,
        "unknown_tickets": 0,
        "unresolved_count": 5,
        "resolved_count": 1,
        "partial_count": 1,
        "unresolved_rate": 0.71,
        "resolved_rate": 0.14,
        "last_ticket_unresolved": 1,
        "consecutive_unresolved_end": 2,
        "weighted_unresolved": 2.1,
        "weighted_high_risk": 1.8,
        "days_since_last_ticket": 3,
        "avg_gap_days": 5.2,
        "gap_std": 1.5,
        "escalation_rate": 2.1,
        "tickets_per_week": 1.6,
        "unresolved_high_risk": 25.0,
        "response_x_unresolved": 18.0,
        "second_last_resolved": 0,
        "second_last_unresolved": 1,
        "second_last_risk": 3
    },
    {
        "driver_id": 1003,
        "total_tickets": 3,
        "fare_disputes": 1,
        "defective_trips": 0,
        "demand_issues": 1,
        "payment_delays": 1,
        "app_issues": 0,
        "unknown_tickets": 0,
        "unresolved_count": 1,
        "resolved_count": 2,
        "partial_count": 0,
        "unresolved_rate": 0.33,
        "resolved_rate": 0.67,
        "last_ticket_unresolved": 0,
        "consecutive_unresolved_end": 0,
        "weighted_unresolved": 0.3,
        "weighted_high_risk": 0.2,
        "days_since_last_ticket": 10,
        "avg_gap_days": 12.0,
        "gap_std": 3.0,
        "escalation_rate": 0.5,
        "tickets_per_week": 0.4,
        "unresolved_high_risk": 1.0,
        "response_x_unresolved": 2.0,
        "second_last_resolved": 1,
        "second_last_unresolved": 0,
        "second_last_risk": 1
    },
    {
        "driver_id": 1004,
        "total_tickets": 1,
        "fare_disputes": 0,
        "defective_trips": 0,
        "demand_issues": 0,
        "payment_delays": 0,
        "app_issues": 1,
        "unknown_tickets": 0,
        "unresolved_count": 0,
        "resolved_count": 1,
        "partial_count": 0,
        "unresolved_rate": 0.0,
        "resolved_rate": 1.0,
        "last_ticket_unresolved": 0,
        "consecutive_unresolved_end": 0,
        "weighted_unresolved": 0.0,
        "weighted_high_risk": 0.0,
        "days_since_last_ticket": 20,
        "avg_gap_days": 0,
        "gap_std": 0,
        "escalation_rate": 0,
        "tickets_per_week": 0.1,
        "unresolved_high_risk": 0.0,
        "response_x_unresolved": 0.0,
        "second_last_resolved": 1,
        "second_last_unresolved": 0,
        "second_last_risk": 0
    }
]

print("=== DRIVER CHURN AGENT RUNNING ===\n")

for driver in drivers:
    print(f"\n{'='*60}")
    print(f"Processing Driver {driver['driver_id']}...")
    result = run_agent(driver)
    print(f"\n--- RESULT ---")
    print(f"Risk Level:     {result['risk_level']} ({result['churn_probability']:.1%})")
    print(f"Intervention:   {result['intervention_type']}")
    print(f"Action:         {result['action_taken']}")
    print(f"Message:        {result['outreach_message']}")
    print(f"Reasoning:      {result['agent_reasoning']}")