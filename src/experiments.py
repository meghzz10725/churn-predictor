import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

# ── Setup ────────────────────────────────────────────────────
mlflow.set_experiment("churn-predictor")

# ── Load and preprocess data ─────────────────────────────────
df = pd.read_csv('../data/telco_churn_clean.csv')
df.drop('customerID', axis=1, inplace=True)
df['Churn'] = (df['Churn'] == 'Yes').astype(int)

le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data loaded! Starting experiments...\n")

# ══════════════════════════════════════════════════════════════
# EXPERIMENTS 1-3 — Logistic Regression with different settings
# ══════════════════════════════════════════════════════════════

lr_configs = [
    {"C": 0.01, "name": "LR_high_regularization"},
    {"C": 1.0,  "name": "LR_default"},
    {"C": 100,  "name": "LR_low_regularization"},
]

# C = inverse of regularization strength
# Low C = strong regularization = simpler model = might underfit
# High C = weak regularization = complex model = might overfit
# Default C=1.0 is usually the sweet spot

for config in lr_configs:
    with mlflow.start_run(run_name=config["name"]):
        model = LogisticRegression(
            C=config["C"],
            max_iter=3000,
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_train_scaled, y_train)

        pred = model.predict(X_test_scaled)
        prob = model.predict_proba(X_test_scaled)[:, 1]

        auc       = roc_auc_score(y_test, prob)
        precision = precision_score(y_test, pred)
        recall    = recall_score(y_test, pred)
        f1        = f1_score(y_test, pred)

        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("C", config["C"])
        mlflow.log_param("class_weight", "balanced")

        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        print(f"{config['name']}: AUC={auc:.4f} F1={f1:.4f}")

# ══════════════════════════════════════════════════════════════
# EXPERIMENTS 4-8 — XGBoost with different depths and trees
# ══════════════════════════════════════════════════════════════

xgb_configs = [
    {"max_depth": 2, "n_estimators": 100, "lr": 0.1,  "name": "XGB_shallow_fast"},
    {"max_depth": 4, "n_estimators": 200, "lr": 0.05, "name": "XGB_medium"},
    {"max_depth": 4, "n_estimators": 500, "lr": 0.01, "name": "XGB_medium_slow_learner"},
    {"max_depth": 6, "n_estimators": 200, "lr": 0.05, "name": "XGB_deep"},
    {"max_depth": 6, "n_estimators": 500, "lr": 0.01, "name": "XGB_deep_slow_learner"},
]

# max_depth — how deep each tree grows
# Shallow (2) = simple, fast, might miss patterns
# Deep (6) = complex, slow, might overfit

# learning_rate — how much each tree corrects previous mistakes
# High (0.1) = learns fast but might overshoot
# Low (0.01) = learns slowly but more precisely

# n_estimators — how many trees
# More trees + slow learning = usually better but slower

for config in xgb_configs:
    with mlflow.start_run(run_name=config["name"]):
        model = XGBClassifier(
            max_depth=config["max_depth"],
            n_estimators=config["n_estimators"],
            learning_rate=config["lr"],
            scale_pos_weight=3,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]

        auc       = roc_auc_score(y_test, prob)
        precision = precision_score(y_test, pred)
        recall    = recall_score(y_test, pred)
        f1        = f1_score(y_test, pred)

        mlflow.log_param("model_type", "xgboost")
        mlflow.log_param("max_depth", config["max_depth"])
        mlflow.log_param("n_estimators", config["n_estimators"])
        mlflow.log_param("learning_rate", config["lr"])
        mlflow.log_param("scale_pos_weight", 3)

        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        print(f"{config['name']}: AUC={auc:.4f} F1={f1:.4f}")

print("\nAll 8 experiments logged to MLflow!")
print("Go check http://127.0.0.1:5000 to compare them!")