import os
import requests
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from typing import TypedDict
import json

load_dotenv()

# ============================================================
# STEP 14 — LANGGRAPH AGENT
# ============================================================
# More sophisticated than Project 1:
# 1. Calls real ML API (not mock model)
# 2. Reads top risk factors for context
# 3. Routes by risk level (4 levels not 3)
# 4. CRITICAL → manager escalation
# 5. HIGH → WhatsApp + incentive
# 6. MEDIUM → WhatsApp nudge
# 7. LOW → no action
# 8. Logs every decision with reasoning
# ============================================================

load_dotenv()

llm = ChatAnthropic(
    model="claude-opus-4-6",
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

API_URL = "http://127.0.0.1:8000/predict"

# ============================================================
# STATE DEFINITION
# ============================================================

class DriverState(TypedDict):
    # Input
    driver_id: int
    total_tickets: int
    fare_disputes: int
    defective_trips: int
    demand_issues: int
    payment_delays: int
    app_issues: int
    unknown_tickets: int
    unresolved_count: int
    resolved_count: int
    partial_count: int
    unresolved_rate: float
    resolved_rate: float
    last_ticket_unresolved: int
    consecutive_unresolved_end: int
    weighted_unresolved: float
    weighted_high_risk: float
    days_since_last_ticket: int
    avg_gap_days: float
    gap_std: float
    escalation_rate: float
    tickets_per_week: float
    unresolved_high_risk: float
    response_x_unresolved: float
    second_last_resolved: int
    second_last_unresolved: int
    second_last_risk: int

    # Filled by agent
    churn_probability: float
    risk_level: str
    predicted_churn: bool
    top_risk_factors: list
    recommended_action: str
    intervention_type: str
    outreach_message: str
    escalation_reason: str
    action_taken: str
    agent_reasoning: str

# ============================================================
# NODE 1 — CALL ML API
# ============================================================

def call_prediction_api(state: DriverState) -> DriverState:
    print(f"\n[Node 1] Calling ML API for driver {state['driver_id']}...")

    payload = {k: v for k, v in state.items()
               if k not in [
                   'churn_probability', 'risk_level',
                   'predicted_churn', 'top_risk_factors',
                   'recommended_action', 'intervention_type',
                   'outreach_message', 'escalation_reason',
                   'action_taken', 'agent_reasoning'
               ]}

    response = requests.post(API_URL, json=payload)
    result = response.json()

    state['churn_probability'] = result['churn_probability']
    state['risk_level'] = result['risk_level']
    state['predicted_churn'] = result['predicted_churn']
    state['top_risk_factors'] = result['top_risk_factors']
    state['recommended_action'] = result['recommended_action']

    print(f"[Node 1] Risk: {result['risk_level']} "
          f"({result['churn_probability']:.3f})")
    print(f"[Node 1] Top factors: {result['top_risk_factors']}")

    return state

# ============================================================
# NODE 2 — DECIDE INTERVENTION
# ============================================================

def decide_intervention(state: DriverState) -> DriverState:
    print(f"\n[Node 2] Deciding intervention...")

    risk = state['risk_level']

    if risk == 'CRITICAL':
        state['intervention_type'] = 'escalate'
    elif risk == 'HIGH':
        state['intervention_type'] = 'incentive'
    elif risk == 'MEDIUM':
        state['intervention_type'] = 'nudge'
    else:
        state['intervention_type'] = 'none'

    print(f"[Node 2] Intervention: {state['intervention_type']}")
    return state

# ============================================================
# NODE 3 — GENERATE MESSAGE (Claude)
# ============================================================

def generate_message(state: DriverState) -> DriverState:
    print(f"\n[Node 3] Generating personalized message...")

    if state['intervention_type'] == 'none':
        state['outreach_message'] = "No message needed."
        state['agent_reasoning'] = "Driver is LOW risk — no intervention required."
        return state

    if state['intervention_type'] == 'escalate':
        state['outreach_message'] = "ESCALATE — do not send automated message."
        state['agent_reasoning'] = (
            "CRITICAL risk driver needs personal attention from ops team. "
            "Automated message may feel dismissive given severity."
        )
        return state

    risk_factors_text = "\n".join(
        f"- {factor}" for factor in state['top_risk_factors']
    )

    prompt = f"""You are a driver engagement specialist at a ride-hailing company.

Driver ID: {state['driver_id']}
Churn Risk: {state['risk_level']} ({state['churn_probability']:.1%})
Intervention Type: {state['intervention_type']}

Key risk factors for this driver:
{risk_factors_text}

Ticket summary:
- Total tickets: {state['total_tickets']}
- Unresolved: {state['unresolved_count']} ({state['unresolved_rate']:.0%} unresolved rate)
- Tickets per week: {state['tickets_per_week']:.1f}

Write a short WhatsApp message (2-3 sentences) to re-engage this driver.
Guidelines:
- If incentive: offer a specific bonus (₹200 for 10 trips this week)
- If nudge: mention high demand in their area this week
- Be warm, specific, and human — not robotic
- Do NOT mention that we know they're at risk of leaving
- Do NOT use placeholder text like [Name]

Respond with ONLY the WhatsApp message, nothing else."""

    response = llm.invoke(prompt)
    state['outreach_message'] = response.content

    state['agent_reasoning'] = (
        f"Generated {state['intervention_type']} message based on "
        f"{state['risk_level']} risk ({state['churn_probability']:.1%}) "
        f"with {state['unresolved_count']} unresolved tickets."
    )

    print(f"[Node 3] Message generated")
    return state

# ============================================================
# NODE 4 — ROUTE ACTION
# ============================================================

def route_action(state: DriverState) -> DriverState:
    print(f"\n[Node 4] Routing action...")

    intervention = state['intervention_type']

    if intervention == 'escalate':
        state['action_taken'] = (
            f"ESCALATED to ops manager — "
            f"Driver {state['driver_id']} at CRITICAL risk "
            f"({state['churn_probability']:.1%}). "
            f"Reason: {state['top_risk_factors'][0]}"
        )
        state['escalation_reason'] = state['top_risk_factors'][0]
        print(f"[Node 4] 🚨 ESCALATED to manager")

    elif intervention == 'incentive':
        state['action_taken'] = (
            f"WhatsApp + ₹200 incentive sent to driver {state['driver_id']}"
        )
        state['escalation_reason'] = ""
        print(f"[Node 4] 💰 Incentive message queued")

    elif intervention == 'nudge':
        state['action_taken'] = (
            f"WhatsApp nudge sent to driver {state['driver_id']}"
        )
        state['escalation_reason'] = ""
        print(f"[Node 4] 📱 Nudge message queued")

    else:
        state['action_taken'] = "No action taken — driver is stable"
        state['escalation_reason'] = ""
        print(f"[Node 4] ✅ No action needed")

    return state

# ============================================================
# BUILD AGENT GRAPH
# ============================================================

graph = StateGraph(DriverState)

graph.add_node("call_prediction_api", call_prediction_api)
graph.add_node("decide_intervention", decide_intervention)
graph.add_node("generate_message", generate_message)
graph.add_node("route_action", route_action)

graph.set_entry_point("call_prediction_api")
graph.add_edge("call_prediction_api", "decide_intervention")
graph.add_edge("decide_intervention", "generate_message")
graph.add_edge("generate_message", "route_action")
graph.add_edge("route_action", END)

agent = graph.compile()

# ============================================================
# RUN AGENT
# ============================================================

def run_agent(driver_data: dict) -> dict:
    initial_state = {
        **driver_data,
        'churn_probability': 0.0,
        'risk_level': '',
        'predicted_churn': False,
        'top_risk_factors': [],
        'recommended_action': '',
        'intervention_type': '',
        'outreach_message': '',
        'escalation_reason': '',
        'action_taken': '',
        'agent_reasoning': ''
    }

    result = agent.invoke(initial_state)

    return {
        'driver_id': result['driver_id'],
        'churn_probability': result['churn_probability'],
        'risk_level': result['risk_level'],
        'intervention_type': result['intervention_type'],
        'outreach_message': result['outreach_message'],
        'action_taken': result['action_taken'],
        'agent_reasoning': result['agent_reasoning'],
        'top_risk_factors': result['top_risk_factors']
    }