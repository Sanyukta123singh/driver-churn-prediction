from pydantic import BaseModel, Field
from typing import Optional

# ============================================================
# API SCHEMAS — Input & Output validation
# ============================================================

class DriverTicketData(BaseModel):
    driver_id: int = Field(..., description="Unique driver identifier")

    # Sequence features
    total_tickets: int = Field(..., ge=0, description="Total support tickets filed")
    fare_disputes: int = Field(..., ge=0, description="Number of fare dispute tickets")
    defective_trips: int = Field(..., ge=0, description="Number of defective trip tickets")
    demand_issues: int = Field(..., ge=0, description="Number of demand issue tickets")
    payment_delays: int = Field(..., ge=0, description="Number of payment delay tickets")
    app_issues: int = Field(..., ge=0, description="Number of app issue tickets")
    unknown_tickets: int = Field(..., ge=0, description="Number of unknown type tickets")
    unresolved_count: int = Field(..., ge=0, description="Total unresolved tickets")
    resolved_count: int = Field(..., ge=0, description="Total resolved tickets")
    partial_count: int = Field(..., ge=0, description="Total partially resolved tickets")
    unresolved_rate: float = Field(..., ge=0, le=1, description="Rate of unresolved tickets")
    resolved_rate: float = Field(..., ge=0, le=1, description="Rate of resolved tickets")
    last_ticket_unresolved: int = Field(..., ge=0, le=1, description="Last ticket unresolved flag")
    consecutive_unresolved_end: int = Field(..., ge=0, description="Consecutive unresolved at end")
    weighted_unresolved: float = Field(..., ge=0, description="Recency weighted unresolved score")
    weighted_high_risk: float = Field(..., ge=0, description="Recency weighted high risk score")
    days_since_last_ticket: int = Field(..., ge=0, description="Days since last ticket")
    avg_gap_days: float = Field(..., ge=0, description="Average days between tickets")
    gap_std: float = Field(..., ge=0, description="Std deviation of gap between tickets")
    escalation_rate: float = Field(..., description="Rate of escalation in ticket frequency")
    tickets_per_week: float = Field(..., ge=0, description="Average tickets filed per week")
    unresolved_high_risk: float = Field(..., ge=0, description="Unresolved x high risk interaction")
    response_x_unresolved: float = Field(..., ge=0, description="Response time x unresolved interaction")
    second_last_resolved: int = Field(..., ge=0, le=1, description="Second last ticket resolved flag")
    second_last_unresolved: int = Field(..., ge=0, le=1, description="Second last ticket unresolved flag")
    second_last_risk: int = Field(..., ge=0, description="Second last ticket risk score")

    class Config:
        json_schema_extra = {
            "example": {
                "driver_id": 12345,
                "total_tickets": 8,
                "fare_disputes": 3,
                "defective_trips": 2,
                "demand_issues": 1,
                "payment_delays": 1,
                "app_issues": 1,
                "unknown_tickets": 0,
                "unresolved_count": 6,
                "resolved_count": 1,
                "partial_count": 1,
                "unresolved_rate": 0.75,
                "resolved_rate": 0.125,
                "last_ticket_unresolved": 1,
                "consecutive_unresolved_end": 3,
                "weighted_unresolved": 2.1,
                "weighted_high_risk": 1.8,
                "days_since_last_ticket": 2,
                "avg_gap_days": 4.5,
                "gap_std": 1.2,
                "escalation_rate": 3.5,
                "tickets_per_week": 1.8,
                "unresolved_high_risk": 30.0,
                "response_x_unresolved": 25.5,
                "second_last_resolved": 0,
                "second_last_unresolved": 1,
                "second_last_risk": 3
            }
        }


class PredictionResponse(BaseModel):
    driver_id: int
    churn_probability: float
    risk_level: str
    predicted_churn: bool
    threshold_used: float
    top_risk_factors: list
    recommended_action: str
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    feature_count: int
    threshold: float
    model_version: str