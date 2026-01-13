import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.preprocessing import LabelEncoder
import joblib

# -----------------------------
# Categorical Encoding
# -----------------------------
def encode_reference_current(reference_df, current_df):
    reference_df = reference_df.copy()
    current_df = current_df.copy()

    cat_cols = reference_df.select_dtypes(include="object").columns

    for col in cat_cols:
        le = LabelEncoder()

        reference_df[col] = reference_df[col].astype(str)
        current_df[col] = current_df[col].astype(str)

        le.fit(reference_df[col])

        current_df[col] = current_df[col].apply(
            lambda x: x if x in le.classes_ else "UNKNOWN"
        )

        le.classes_ = np.append(le.classes_, "UNKNOWN")

        reference_df[col] = le.transform(reference_df[col])
        current_df[col] = le.transform(current_df[col])

    return reference_df, current_df


# -----------------------------
# Population Stability Index
# -----------------------------
def calculate_psi(expected, actual, bins=10):
    expected = np.asarray(expected)
    actual = np.asarray(actual)

    breakpoints = np.percentile(
        expected, np.linspace(0, 100, bins + 1)
    )

    expected_dist = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_dist = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    psi = np.sum(
        (actual_dist - expected_dist) *
        np.log((actual_dist + 1e-6) / (expected_dist + 1e-6))
    )

    return psi


# -----------------------------
# Feature Drift (PSI + KS)
# -----------------------------
def compute_feature_drift(reference_df, current_df, features):
    psi_results = {}
    ks_results = {}

    for col in features:
        psi_results[col] = calculate_psi(
            reference_df[col], current_df[col]
        )

        ks_stat, p_value = ks_2samp(
            reference_df[col], current_df[col]
        )

        ks_results[col] = {
            "ks_statistic": ks_stat,
            "p_value": p_value
        }

    psi_df = pd.DataFrame.from_dict(
        psi_results, orient="index", columns=["PSI"]
    )

    ks_df = pd.DataFrame(ks_results).T

    return psi_df, ks_df


# -----------------------------
# Score Drift
# -----------------------------
def compute_score_drift(model_path, reference_df, current_df):
    model = joblib.load(model_path)

    ref_scores = model.predict_proba(reference_df)[:, 1]
    cur_scores = model.predict_proba(current_df)[:, 1]

    ks_stat, p_value = ks_2samp(ref_scores, cur_scores)

    return ref_scores, cur_scores, ks_stat, p_value


# -----------------------------
# Visualization
# -----------------------------
def plot_psi(psi_df):
    psi_df.sort_values("PSI").plot(
        kind="barh",
        figsize=(8, 4),
        title="Population Stability Index (PSI)"
    )
    plt.show()


def plot_score_distribution(ref_scores, cur_scores):
    plt.figure(figsize=(8, 4))
    plt.hist(ref_scores, bins=50, alpha=0.6, label="Reference", density=True)
    plt.hist(cur_scores, bins=50, alpha=0.6, label="Current", density=True)
    plt.title("Fraud Score Distribution Drift")
    plt.legend()
    plt.show()
