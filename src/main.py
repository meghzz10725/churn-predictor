from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from adaptive_router import route_prediction
from shap_explainer import get_top_reasons

# ── App setup ───────────────────────────────────────────────
app = FastAPI(
    title="Churn Predictor API",
    description="Adaptive ML system for customer churn prediction with explainability",
    version="1.0.0"
)

# ── Input schema ─────────────────────────────────────────────
# Pydantic validates every incoming request automatically
# If any field is missing or wrong type → auto error response
class CustomerData(BaseModel):
    gender: int                # 0=Female, 1=Male
    SeniorCitizen: int         # 0=No, 1=Yes
    Partner: int               # 0=No, 1=Yes
    Dependents: int            # 0=No, 1=Yes
    tenure: int                # months as customer
    PhoneService: int          # 0=No, 1=Yes
    MultipleLines: int         # 0=No, 1=Yes, 2=No phone
    InternetService: int       # 0=DSL, 1=Fiber, 2=No
    OnlineSecurity: int        # 0=No, 1=Yes, 2=No internet
    OnlineBackup: int          # 0=No, 1=Yes, 2=No internet
    DeviceProtection: int      # 0=No, 1=Yes, 2=No internet
    TechSupport: int           # 0=No, 1=Yes, 2=No internet
    StreamingTV: int           # 0=No, 1=Yes, 2=No internet
    StreamingMovies: int       # 0=No, 1=Yes, 2=No internet
    Contract: int              # 0=Month-to-month, 1=One year, 2=Two year
    PaperlessBilling: int      # 0=No, 1=Yes
    PaymentMethod: int         # 0=Bank transfer, 1=Credit card, 2=Electronic, 3=Mailed
    MonthlyCharges: float      # monthly bill amount
    TotalCharges: float        # total amount paid

# ── Health check endpoint ────────────────────────────────────
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

# ── Main predict endpoint ─────────────────────────────────────
@app.post("/predict")
def predict(customer: CustomerData):
    try:
        # Step 1 — convert input to numpy array
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

        # Step 2 — adaptive router decides which model
        prediction = route_prediction(features)

        # Step 3 — SHAP explains the prediction
        reasons = get_top_reasons(features, top_n=3)

        # Step 4 — build final response
        churn_prob = prediction["churn_probability"]

        return {
            "churn_probability": churn_prob,
            "churn_risk": get_risk_label(churn_prob),
            "model_used": prediction["model_used"],
            "confidence": prediction["confidence"],
            "top_reasons": reasons,
            "needs_llm": prediction["needs_llm"],
            "recommendation": get_recommendation(churn_prob)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_risk_label(prob):
    """Convert probability to human readable risk level"""
    if prob >= 0.75:
        return "HIGH RISK"
    elif prob >= 0.45:
        return "MEDIUM RISK"
    else:
        return "LOW RISK"


def get_recommendation(prob):
    """Basic recommendation based on churn probability"""
    if prob >= 0.75:
        return "Immediate intervention needed — offer retention discount or dedicated support"
    elif prob >= 0.45:
        return "Monitor closely — consider proactive outreach or upgrade offer"
    else:
        return "Low risk — standard engagement is sufficient"