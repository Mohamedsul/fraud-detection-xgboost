
import pandas as pd
import joblib
import yaml

from decisioning.threshold_policy import apply_threshold
from features.feature_pipeline import build_features
from features.encoding import encode_categoricals  # must match training

# ----------------------------
# Load config
# ----------------------------
with open("src/config/settings.yaml") as f:
    settings = yaml.safe_load(f)

MODEL_PATH = settings["model"]["artifact_path"]

model = joblib.load(MODEL_PATH)


def score_transactions(
    df_raw: pd.DataFrame,
    reference_df: pd.DataFrame
) -> pd.DataFrame:
    """
    End-to-end fraud inference pipeline
    """

    # 1. Feature engineering
    df_features = build_features(df_raw)

    # 2. Encode categoricals (training-safe)
    df_features = encode_categoricals(
        df_features,
        reference_df=reference_df
    )

    # 3. Score
    df_features["score"] = model.predict_proba(df_features)[:, 1]

    # 4. Apply business threshold
    df_result = apply_threshold(
        df_features,
        score_col="score"
    )

    return df_result
