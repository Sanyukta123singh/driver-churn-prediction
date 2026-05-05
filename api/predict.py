import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# PREDICTION LOGIC
# ============================================================

BASE_DIR = Path(__file__).parent.parent

model = joblib.load(BASE_DIR / 'models/production_model.pkl')
threshold = joblib.load(BASE_DIR / 'models/production_threshold.pkl')
features = joblib.load(BASE_DIR / 'models/production_features.pkl')

MODEL_VERSION = "v1.0.0"

print(f"Model loaded successfully")
print(f"Features: {len(features)}")
print(f"Threshold: {threshold}")
print(f"Version: {MODEL_VERSION}")


def get_risk_level(probability: float) -> str:
    if probability >= 0.70:
        return "CRITICAL"
    elif probability >= 0.50:
        return "HIGH"
    elif probability >= 0.30:
        return "MEDIUM"
    else:
        return "LOW"


def get_recommended_action(risk_level: str,
                           data: dict) -> str:
    if risk_level == "CRITICAL":
        return "Immediate personal call from ops team + ₹500 bonus offer"
    elif risk_level == "HIGH":
        return "WhatsApp message + ₹200 incentive for completing 10 trips"
    elif risk_level == "MEDIUM":
        return "WhatsApp nudge about high demand in their area"
    else:
        return "No action needed — driver is stable"


def get_top_risk_factors(data: dict) -> list:
    risk_factors = []

    if data.get('unresolved_high_risk', 0) > 5:
        risk_factors.append(
            f"High unresolved risk tickets: {data['unresolved_high_risk']:.0f}"
        )
    if data.get('unresolved_count', 0) > 3:
        risk_factors.append(
            f"Multiple unresolved tickets: {data['unresolved_count']}"
        )
    if data.get('consecutive_unresolved_end', 0) >= 2:
        risk_factors.append(
            f"Last {data['consecutive_unresolved_end']} tickets unresolved consecutively"
        )
    if data.get('tickets_per_week', 0) > 1.5:
        risk_factors.append(
            f"High ticket frequency: {data['tickets_per_week']:.1f}/week"
        )
    if data.get('unresolved_rate', 0) > 0.6:
        risk_factors.append(
            f"High unresolved rate: {data['unresolved_rate']:.0%}"
        )
    if data.get('second_last_unresolved', 0) == 1:
        risk_factors.append(
            "Second last ticket also unresolved — pattern of neglect"
        )
    if data.get('days_since_last_ticket', 30) < 3:
        risk_factors.append(
            f"Very recent ticket: {data['days_since_last_ticket']} days ago"
        )

    if not risk_factors:
        risk_factors.append("No major risk factors identified")

    return risk_factors[:3]


def predict_churn(driver_data: dict) -> dict:
    input_df = pd.DataFrame([{
        feature: driver_data.get(feature, 0)
        for feature in features
    }])

    probability = float(
        model.predict_proba(input_df)[0][1]
    )
    predicted_churn = probability >= threshold
    risk_level = get_risk_level(probability)
    recommended_action = get_recommended_action(
        risk_level, driver_data
    )
    top_risk_factors = get_top_risk_factors(driver_data)

    return {
        "driver_id": driver_data["driver_id"],
        "churn_probability": round(probability, 4),
        "risk_level": risk_level,
        "predicted_churn": predicted_churn,
        "threshold_used": threshold,
        "top_risk_factors": top_risk_factors,
        "recommended_action": recommended_action,
        "model_version": MODEL_VERSION
    }