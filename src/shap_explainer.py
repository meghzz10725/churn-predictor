import shap
import joblib
import numpy as np
import pandas as pd

# Load models and feature names
xgb_model = joblib.load('../models/xgb_model.pkl')
feature_names = joblib.load('../models/feature_names.pkl')

# Create SHAP explainer once at startup
# TreeExplainer is specifically designed for tree based models like XGBoost
# It's fast and exact — not an approximation
explainer = shap.TreeExplainer(xgb_model)

def get_top_reasons(features_array, top_n=3):
    """
    Takes customer features → returns top N reasons for prediction
    in plain English that a business person can understand.
    
    Example output:
    [
      "Month-to-month contract increases churn risk",
      "High monthly charges of $89 increases churn risk",  
      "Low tenure of 3 months increases churn risk"
    ]
    """

    # Step 1 — calculate SHAP values
    # SHAP tells us how much each feature PUSHED the prediction
    # Positive SHAP = pushed towards churn
    # Negative SHAP = pushed away from churn
    features_df = pd.DataFrame(features_array, columns=feature_names)
    shap_values = explainer.shap_values(features_df)

    # Step 2 — get absolute SHAP values
    # We want the features that had the BIGGEST impact
    # regardless of direction first
    shap_abs = np.abs(shap_values[0])

    # Step 3 — get top N feature indices sorted by impact
    top_indices = np.argsort(shap_abs)[::-1][:top_n]

    # Step 4 — build human readable reasons
    reasons = []
    for idx in top_indices:
        feature = feature_names[idx]
        shap_val = shap_values[0][idx]
        actual_val = features_array[0][idx]

        # Determine direction
        direction = "increases" if shap_val > 0 else "decreases"

        # Make feature names human readable
        reason = build_reason(feature, actual_val, direction)
        reasons.append(reason)

    return reasons


def build_reason(feature, value, direction):
    """
    Converts raw feature names into human readable sentences.
    This is what impresses interviewers — you thought about UX!
    """

    # Map technical feature names to plain English
    feature_map = {
        'tenure': f"Tenure of {int(value)} months {direction} churn risk",
        'MonthlyCharges': f"Monthly charges of ${value:.0f} {direction} churn risk",
        'TotalCharges': f"Total charges of ${value:.0f} {direction} churn risk",
        'Contract': f"{'Month-to-month' if value == 0 else 'Long-term'} contract {direction} churn risk",
        'InternetService': f"Internet service type {direction} churn risk",
        'TechSupport': f"{'No tech support' if value == 0 else 'Tech support'} {direction} churn risk",
        'OnlineSecurity': f"{'No online security' if value == 0 else 'Online security'} {direction} churn risk",
        'PaymentMethod': f"Payment method {direction} churn risk",
        'PaperlessBilling': f"{'Paperless billing' if value == 1 else 'Paper billing'} {direction} churn risk",
        'SeniorCitizen': f"{'Senior citizen status' if value == 1 else 'Non-senior status'} {direction} churn risk",
    }

    # Return mapped reason or generic fallback
    return feature_map.get(
        feature,
        f"{feature} (value: {value:.2f}) {direction} churn risk"
    )