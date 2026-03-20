import joblib
import numpy as np

# Load both models and scaler once when file is imported
# This way they stay in memory — faster API responses
log_model = joblib.load('../models/logistic_model.pkl')
xgb_model = joblib.load('../models/xgb_model.pkl')
scaler = joblib.load('../models/scaler.pkl')

def route_prediction(features_array):
    """
    Decides which model handles this prediction.
    
    Logic:
    - Logistic Regression → simple/clear cases (confidence > 85%)
    - XGBoost → complex cases (confidence 55-85%)
    - LLM escalation flag → very uncertain cases (confidence < 55%)
    
    This is your EDGE — most projects just use one model blindly.
    A real production system routes based on complexity!
    """

    # Step 1 — scale features for logistic regression
    features_scaled = scaler.transform(features_array)

    # Step 2 — get logistic regression probability first
    # It's fast and cheap — use it as a first filter
    log_prob = log_model.predict_proba(features_scaled)[0][1]

    # Step 3 — check confidence
    # How far is the probability from 0.5 (total uncertainty)?
    # 0.0 = completely uncertain, 1.0 = completely certain
    log_confidence = abs(log_prob - 0.5) * 2

    # High confidence — logistic regression is enough
    # No need to run expensive XGBoost
    if log_confidence >= 0.70:
        return {
            "churn_probability": round(float(log_prob), 4),
            "model_used": "logistic_regression",
            "confidence": "high",
            "needs_llm": False
        }

    # Medium confidence — use XGBoost for deeper analysis
    xgb_prob = xgb_model.predict_proba(features_array)[0][1]
    xgb_confidence = abs(xgb_prob - 0.5) * 2

    if xgb_confidence >= 0.10:
        return {
            "churn_probability": round(float(xgb_prob), 4),
            "model_used": "xgboost",
            "confidence": "medium",
            "needs_llm": False
        }

    # Very uncertain — flag for LLM reasoning
    # System is genuinely unsure — needs deeper reasoning
    return {
        "churn_probability": round(float(xgb_prob), 4),
        "model_used": "xgboost_with_llm",
        "confidence": "low",
        "needs_llm": True
    }