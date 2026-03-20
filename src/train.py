import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
# Tell MLflow where to store runs
mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("churn-predictor")

# ── MLflow setup ─────────────────────────────────────────────
# This creates a local folder called "mlruns" that stores all experiments
mlflow.set_experiment("churn-predictor")

# ── 1. Load clean data ───────────────────────────────────────
df = pd.read_csv('../data/telco_churn_clean.csv')

# ── 2. Preprocessing ─────────────────────────────────────────
df.drop('customerID', axis=1, inplace=True)
df['Churn'] = (df['Churn'] == 'Yes').astype(int)

le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

print("Preprocessing done!")

# ── 3. Split ──────────────────────────────────────────────────
X = df.drop('Churn', axis=1)
y = df['Churn']

feature_names = list(X.columns)
joblib.dump(feature_names, '../models/feature_names.pkl')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 4. Scale for Logistic Regression ─────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, '../models/scaler.pkl')

# ══════════════════════════════════════════════════════════════
# MLflow Run 1 — Logistic Regression
# Everything inside "with mlflow.start_run()" gets tracked
# ══════════════════════════════════════════════════════════════
print("\nTraining Logistic Regression...")

with mlflow.start_run(run_name="logistic_regression"):

    # Train model
    log_model = LogisticRegression(
        max_iter=3000,
        class_weight='balanced',
        random_state=42,
        solver='lbfgs'
    )
    log_model.fit(X_train_scaled, y_train)

    # Evaluate
    log_pred = log_model.predict(X_test_scaled)
    log_prob = log_model.predict_proba(X_test_scaled)[:, 1]

    auc = roc_auc_score(y_test, log_prob)
    precision = precision_score(y_test, log_pred)
    recall = recall_score(y_test, log_pred)
    f1 = f1_score(y_test, log_pred)

    # ── LOG PARAMETERS ──────────────────────────────────────
    # These are the settings you used
    mlflow.log_param("model_type", "logistic_regression")
    mlflow.log_param("max_iter", 3000)
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("solver", "lbfgs")
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))

    # ── LOG METRICS ─────────────────────────────────────────
    # These are the results you got
    mlflow.log_metric("roc_auc", auc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # ── LOG MODEL ───────────────────────────────────────────
    # Saves the actual model inside MLflow
    mlflow.sklearn.log_model(log_model, "logistic_model")

    print(f"\nLogistic Regression Results:")
    print(classification_report(y_test, log_pred))
    print(f"ROC-AUC: {auc:.4f}")
    print(f"MLflow run logged!")

# Save model file as before
joblib.dump(log_model, '../models/logistic_model.pkl')
print("logistic_model.pkl saved!")

# ══════════════════════════════════════════════════════════════
# MLflow Run 2 — XGBoost with tuning
# ══════════════════════════════════════════════════════════════
print("\nTuning XGBoost...")

param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5],
    'scale_pos_weight': [2.5, 3, 3.5],
}

xgb_base = XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)

xgb_search = RandomizedSearchCV(
    xgb_base,
    param_distributions=param_grid,
    n_iter=30,
    scoring='roc_auc',
    cv=5,
    random_state=42,
    n_jobs=1,
    verbose=1
)

xgb_search.fit(X_train, y_train)
xgb_model = xgb_search.best_estimator_

with mlflow.start_run(run_name="xgboost_tuned"):

    xgb_pred = xgb_model.predict(X_test)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, xgb_prob)
    precision = precision_score(y_test, xgb_pred)
    recall = recall_score(y_test, xgb_pred)
    f1 = f1_score(y_test, xgb_pred)

    # ── LOG PARAMETERS ──────────────────────────────────────
    mlflow.log_param("model_type", "xgboost")
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))

    # Log best params from tuning
    for param, value in xgb_search.best_params_.items():
        mlflow.log_param(param, value)

    # ── LOG METRICS ─────────────────────────────────────────
    mlflow.log_metric("roc_auc", auc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("cv_best_score", xgb_search.best_score_)

    # ── LOG MODEL ───────────────────────────────────────────
    mlflow.xgboost.log_model(xgb_model, "xgboost_model")

    print(f"\nXGBoost Results:")
    print(classification_report(y_test, xgb_pred))
    print(f"ROC-AUC: {auc:.4f}")
    print(f"MLflow run logged!")

joblib.dump(xgb_model, '../models/xgb_model.pkl')
print("xgb_model.pkl saved!")

print("\n── All done ──")
print("logistic_model.pkl ✅")
print("xgb_model.pkl ✅")
print("feature_names.pkl ✅")
print("scaler.pkl ✅")
print("MLflow runs logged ✅")