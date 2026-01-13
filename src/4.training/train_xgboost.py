# src/training/train_xgboost.py

import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
import yaml

# --- Load Config ---
with open("src/config/settings.yaml") as f:
    settings = yaml.safe_load(f)

with open("src/config/model_params.yaml") as f:
    model_params = yaml.safe_load(f).get("xgboost", {})


def train_xgboost(features_path=None, model_output_path=None):
    """
    Train an XGBoost model for fraud detection with train/val/test split.
    Automatically encodes categorical features and handles class imbalance.
    """
    features_path = features_path or settings["data"]["processed_path"]
    model_output_path = model_output_path or settings["model"]["artifact_path"]

    # --- Load feature data ---
    df = pd.read_csv(features_path)

    if "isFraud" not in df.columns:
        raise KeyError("Target column 'isFraud' not found in dataset")

    X = df.drop(columns="isFraud")
    y = df["isFraud"]

    # --- Identify categorical columns ---
    cat_cols = X.select_dtypes(include=["object"]).columns
    num_cols = X.select_dtypes(exclude=["object"]).columns
    print(f"[INFO] Categorical columns: {len(cat_cols)}, Numeric columns: {len(num_cols)}")

    # --- Label encode categorical features ---
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # --- Train/Validation/Test Split (70/15/15) ---
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    # --- Handle Class Imbalance ---
    fraud = y_train.sum()
    non_fraud = len(y_train) - fraud
    scale_pos_weight = non_fraud / (fraud + 1e-6)  # avoid division by zero

    # --- Merge with model_params ---
    model_params.update({
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "logloss",
        "random_state": 42
    })

    # --- Train XGBoost ---
    model = XGBClassifier(**model_params)
    model.fit(X_train, y_train)

    # --- Save Model ---
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"[INFO] Model saved at {model_output_path}")

    # --- Evaluate ---
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("[METRICS] Classification Report:")
    print(classification_report(y_test, y_pred))
    print("[METRICS] Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"[METRICS] ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"[METRICS] PR-AUC: {average_precision_score(y_test, y_proba):.4f}")

    # --- Save predictions ---
    results = X_test.copy()
    results["is_fraud"] = y_test.values
    results["risk_score"] = y_proba
    results_path = os.path.join(os.path.dirname(model_output_path), "results.csv")
    results.to_csv(results_path, index=False)
    print(f"[INFO] Predictions saved to {results_path}")

    return model, results


if __name__ == "__main__":
    train_xgboost()

