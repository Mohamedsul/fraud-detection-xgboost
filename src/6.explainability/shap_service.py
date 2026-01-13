
import os
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import yaml

# --- Load configuration ---
CONFIG_PATH = "src/config/settings.yaml"
MODEL_PATH_DEFAULT = "src/config/model.pkl"

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        settings = yaml.safe_load(f)
        MODEL_PATH = settings.get("model", {}).get("artifact_path", MODEL_PATH_DEFAULT)
else:
    MODEL_PATH = MODEL_PATH_DEFAULT


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label encode all object/categorical columns.
    """
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def explain_shap(
    df: pd.DataFrame,
    model_path: str = None,
    target_col: str = None,
    sample_size: int = 2000
) -> tuple[np.ndarray, shap.Explainer, pd.DataFrame]:
    """
    Compute SHAP values for a batch of transactions.
    
    Args:
        df: DataFrame of features (can include target column)
        model_path: Path to trained model
        target_col: Optional target column to drop (e.g., 'isFraud')
        sample_size: Number of rows to sample for SHAP computation
    
    Returns:
        shap_values: Array of SHAP values
        explainer: SHAP explainer object
        df_sample: Sampled and encoded DataFrame used for SHAP
    """
    model_path = model_path or MODEL_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = joblib.load(model_path)
    df_proc = df.copy()

    if target_col and target_col in df_proc.columns:
        df_proc = df_proc.drop(columns=[target_col])

    df_proc = encode_categoricals(df_proc)

    if sample_size and sample_size < len(df_proc):
        df_sample = df_proc.sample(sample_size, random_state=42)
    else:
        df_sample = df_proc.copy()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_sample)

    return shap_values, explainer, df_sample


def plot_shap_summary(shap_values, df_sample: pd.DataFrame, plot_type: str = "dot"):
    """
    Plot global SHAP summary.
    
    Args:
        shap_values: Array of SHAP values
        df_sample: DataFrame used for SHAP
        plot_type: "dot" or "bar"
    """
    shap.summary_plot(shap_values, df_sample, plot_type=plot_type)


def plot_shap_force(
    shap_values,
    explainer: shap.Explainer,
    df_sample: pd.DataFrame,
    row_idx: int
):
    """
    Plot SHAP force plot for a single transaction.
    
    Args:
        shap_values: Array of SHAP values
        explainer: SHAP explainer object
        df_sample: Sampled DataFrame
        row_idx: Index in df_sample to explain
    """
    shap.force_plot(
        explainer.expected_value,
        shap_values[row_idx],
        df_sample.iloc[row_idx],
        matplotlib=True
    )


def plot_dependence(
    shap_values,
    df_sample: pd.DataFrame,
    feature: str
):
    """
    Plot SHAP dependence plot for a specific feature.
    """
    shap.dependence_plot(feature, shap_values, df_sample)


if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("../data/features_train.csv")
    shap_vals, explainer, df_sample = explain_shap(df, target_col="isFraud", sample_size=2000)

    # Global summary plots
    plot_shap_summary(shap_vals, df_sample, plot_type="dot")
    plot_shap_summary(shap_vals, df_sample, plot_type="bar")

    # Explain first fraud transaction
    fraud_idx = df_sample.index[df.loc[df_sample.index, "isFraud"] == 1][0]
    plot_shap_force(shap_vals, explainer, df_sample, df_sample.index.get_loc(fraud_idx))

    # Explain first legit transaction
    legit_idx = df_sample.index[df.loc[df_sample.index, "isFraud"] == 0][0]
    plot_shap_force(shap_vals, explainer, df_sample, df_sample.index.get_loc(legit_idx))

    # Feature dependence example
    plot_dependence(shap_vals, df_sample, feature="TransactionAmt")
