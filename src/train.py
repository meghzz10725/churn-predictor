import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

# ── 1. Load clean data ──────────────────────────────────────
df = pd.read_csv('../data/telco_churn_clean.csv')

# ── 2. Preprocessing ────────────────────────────────────────
df.drop('customerID', axis=1, inplace=True)
df['Churn'] = (df['Churn'] == 'Yes').astype(int)

le = LabelEncoder()
for col in df.select_dtypes(include='str').columns:
    df[col] = le.fit_transform(df[col])

print("Preprocessing done!")
print(f"Features: {df.shape[1] - 1}")

# ── 3. Split ─────────────────────────────────────────────────
X = df.drop('Churn', axis=1)
y = df['Churn']

feature_names = list(X.columns)
joblib.dump(feature_names, '../models/feature_names.pkl')
print("feature_names.pkl saved!")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {X_train.shape[0]}")
print(f"Test size: {X_test.shape[0]}")

# ── 4. Scale features for Logistic Regression ───────────────
# Logistic regression needs scaled features to converge properly
# XGBoost doesn't need scaling — tree based models are scale invariant
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler — needed later in API
joblib.dump(scaler, '../models/scaler.pkl')
print("scaler.pkl saved!")

# ── 5. Model 1 — Logistic Regression ────────────────────────
print("\nTraining Logistic Regression...")

log_model = LogisticRegression(
    max_iter=3000,
    class_weight='balanced',
    random_state=42,
    solver='lbfgs'        # back to lbfgs but with scaled data now
)
log_model.fit(X_train_scaled, y_train)

log_pred = log_model.predict(X_test_scaled)
log_prob = log_model.predict_proba(X_test_scaled)[:, 1]

print("\nLogistic Regression Results:")
print(classification_report(y_test, log_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, log_prob):.4f}")

joblib.dump(log_model, '../models/logistic_model.pkl')
print("logistic_model.pkl saved!")

# ── 6. Model 2 — XGBoost tuned ──────────────────────────────
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
    n_jobs=-1,
    verbose=1
)

xgb_search.fit(X_train, y_train)

print(f"\nBest parameters:")
for k, v in xgb_search.best_params_.items():
    print(f"  {k}: {v}")

xgb_model = xgb_search.best_estimator_

xgb_pred = xgb_model.predict(X_test)
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

print("\nTuned XGBoost Results:")
print(classification_report(y_test, xgb_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, xgb_prob):.4f}")

joblib.dump(xgb_model, '../models/xgb_model.pkl')
print("xgb_model.pkl saved!")

print("\n── All models saved ──")
print("logistic_model.pkl ✅")
print("xgb_model.pkl ✅")
print("feature_names.pkl ✅")
print("scaler.pkl ✅")