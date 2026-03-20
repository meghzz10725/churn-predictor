from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from adaptive_router import route_prediction
from shap_explainer import get_top_reasons

app = FastAPI(
    title="Churn Predictor API",
    description="Adaptive ML system for customer churn prediction with explainability",
    version="1.0.0"
)

class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def root():
    return {
        "status": "running",
        "message": "Churn Predictor API is live!",
        "endpoints": {
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.post("/predict")
def predict(customer: CustomerData):
    try:
        features = np.array([[
            customer.gender,
            customer.SeniorCitizen,
            customer.Partner,
            customer.Dependents,
            customer.tenure,
            customer.PhoneService,
            customer.MultipleLines,
            customer.InternetService,
            customer.OnlineSecurity,
            customer.OnlineBackup,
            customer.DeviceProtection,
            customer.TechSupport,
            customer.StreamingTV,
            customer.StreamingMovies,
            customer.Contract,
            customer.PaperlessBilling,
            customer.PaymentMethod,
            customer.MonthlyCharges,
            customer.TotalCharges
        ]])

        # Step 1 — adaptive router
        prediction = route_prediction(features)

        # Step 2 — SHAP reasons
        reasons = get_top_reasons(features, top_n=3)

        # Step 3 — LLM insight ONLY for uncertain cases
        # This saves API calls — only runs when model is genuinely confused
        llm_insight = None
        if prediction["needs_llm"]:
            from llm_insights import generate_insight
            llm_insight = generate_insight(
                prediction["churn_probability"],
                reasons,
                prediction["model_used"]
            )

        churn_prob = prediction["churn_probability"]

        return {
            "churn_probability": churn_prob,
            "churn_risk": get_risk_label(churn_prob),
            "model_used": prediction["model_used"],
            "confidence": prediction["confidence"],
            "top_reasons": reasons,
            "needs_llm": prediction["needs_llm"],
            "llm_insight": llm_insight,
            "recommendation": get_recommendation(churn_prob)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_risk_label(prob):
    if prob >= 0.75:
        return "HIGH RISK"
    elif prob >= 0.45:
        return "MEDIUM RISK"
    else:
        return "LOW RISK"


def get_recommendation(prob):
    if prob >= 0.75:
        return "Immediate intervention needed — offer retention discount or dedicated support"
    elif prob >= 0.45:
        return "Monitor closely — consider proactive outreach or upgrade offer"
    else:
        return "Low risk — standard engagement is sufficient"