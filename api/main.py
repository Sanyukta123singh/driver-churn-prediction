from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import (DriverTicketData,
                         PredictionResponse,
                         HealthResponse)
from api.predict import predict_churn, features, threshold, MODEL_VERSION
import time

# ============================================================
# FASTAPI SERVER — Driver Churn Prediction API
# ============================================================

app = FastAPI(
    title="Driver Churn Prediction API",
    description="""
    Production ML API for predicting driver churn risk.

    Built by Sanyukta Kumari — Senior Product Data Scientist

    ## Features
    - XGBoost model trained on 50K drivers
    - 26 engineered features across 7 families
    - SMOTE for class imbalance handling
    - Optimal threshold of 0.20 for maximum recall
    - SHAP-based risk factor explanation

    ## Endpoints
    - GET /health — check API status
    - POST /predict — predict churn for one driver
    - POST /predict/batch — predict churn for multiple drivers
    """,
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/", response_model=HealthResponse)
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "feature_count": len(features),
        "threshold": threshold,
        "model_version": MODEL_VERSION
    }

@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "feature_count": len(features),
        "threshold": threshold,
        "model_version": MODEL_VERSION
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(data: DriverTicketData):
    try:
        start_time = time.time()
        result = predict_churn(data.dict())
        latency = round((time.time() - start_time) * 1000, 2)
        print(f"Prediction for driver {data.driver_id}: "
              f"{result['risk_level']} "
              f"({result['churn_probability']:.3f}) "
              f"in {latency}ms")
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch")
def predict_batch(drivers: list[DriverTicketData]):
    if len(drivers) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Batch size cannot exceed 1000 drivers"
        )
    try:
        results = []
        high_risk_count = 0
        for driver in drivers:
            result = predict_churn(driver.dict())
            results.append(result)
            if result['predicted_churn']:
                high_risk_count += 1

        return {
            "total_drivers": len(results),
            "high_risk_count": high_risk_count,
            "high_risk_rate": round(
                high_risk_count / len(results), 3
            ),
            "predictions": results
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )