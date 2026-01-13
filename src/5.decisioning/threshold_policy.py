import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os


def compute_cost(y_true, y_proba, threshold, cost_fn=1000, cost_fp=10):
    """
    Compute total business cost for a given threshold.
    
    Args:
        y_true: Ground truth labels (0/1)
        y_proba: Model predicted scores
        threshold: Fraud probability threshold
        cost_fn: Cost of missing a fraud
        cost_fp: Cost of flagging a legit transaction
    
    Returns:
        dict: threshold, tn, fp, fn, tp, total_cost
    """
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = fn * cost_fn + fp * cost_fp

    return {
        "threshold": threshold,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "total_cost": total_cost
    }


def threshold_optimization(results_csv_path, cost_fn=1000, cost_fp=10, plot=True):
    """
    Optimize fraud probability threshold based on business costs.
    
    Args:
        results_csv_path: Path to CSV with 'is_fraud' and 'risk_score'
        cost_fn: Cost of missing fraud
        cost_fp: Cost of false positives
        plot: Whether to plot threshold vs total cost
    
    Returns:
        optimal_threshold: Threshold that minimizes total cost
        results_df: DataFrame with all thresholds and costs
    """
    if not os.path.exists(results_csv_path):
        raise FileNotFoundError(f"{results_csv_path} does not exist")

    df = pd.read_csv(results_csv_path)

    if "is_fraud" not in df.columns or "risk_score" not in df.columns:
        raise KeyError("CSV must contain 'is_fraud' and 'risk_score' columns")

    y_true = df["is_fraud"].values
    y_proba = df["risk_score"].values

    thresholds = np.linspace(0.01, 0.99, 99)
    results = [compute_cost(y_true, y_proba, t, cost_fn, cost_fp) for t in thresholds]

    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df["total_cost"].idxmin()]
    optimal_threshold = best_row["threshold"]

    # Optional plotting
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(results_df["threshold"], results_df["total_cost"], label="Total Cost")
        plt.axvline(optimal_threshold, linestyle="--", color="red", label="Optimal Threshold")
        plt.xlabel("Fraud Probability Threshold")
        plt.ylabel("Total Business Cost")
        plt.title("Threshold vs Business Cost")
        plt.legend()
        plt.show()

    # Confusion matrix at optimal threshold
    y_pred_opt = (y_proba >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_opt)

    print(f"[INFO] Optimal Threshold: {optimal_threshold:.4f}")
    print(f"[INFO] Total Cost at Optimal Threshold: {best_row['total_cost']}")
    print("[INFO] Confusion Matrix at Optimal Threshold:")
    print(cm)

    return optimal_threshold, results_df


if __name__ == "__main__":
    threshold_optimization("../data/results.csv")
