# src/main.py
"""
Main orchestration entrypoint for the Fraud Detection System

Pipeline:
1. Load raw data
2. Schema validation
3. Feature engineering
4. Model training OR inference
5. Threshold decisioning
6. Monitoring hooks
"""

import argparse
import pandas as pd
import yaml

# Ingestion
from ingestion.schema_validation import validate_schema

# Features
from features.feature_pipeline import build_features

# Training
from training.train_xgboost import train_model

# Serving
from serving.inference import score_transactions

# Monitoring
from monitoring.data_drift import monitor_feature_drift


# -------------------------------------------------
# Load Config
# -------------------------------------------------
with open("src/config/settings.yaml") as f:
    settings = yaml.safe_load(f)


# -------------------------------------------------
# CLI Interface
# -------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Fraud Detection Pipeline")

    parser.add_argument(
        "--mode",
        choices=["train", "inference"],
        required=True,
        help="Run mode: train or inference"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV data"
    )

    parser.add_argument(
        "--output",
        default="data/output.csv",
        help="Output file path"
    )

    return parser.parse_args()


# -------------------------------------------------
# Train Pipeline
# -------------------------------------------------
def run_training(input_path):
    print("🔹 Loading training data...")
    df_raw = pd.read_csv(input_path)

    print("🔹 Validating schema...")
    validate_schema(df_raw)

    print("🔹 Building features...")
    df_features = build_features(df_raw)

    print("🔹 Training model...")
    model = train_model()

    print("✅ Training completed successfully")


# -------------------------------------------------
# Inference Pipeline
# -------------------------------------------------
def run_inference(input_path, output_path):
    print("🔹 Loading new transactions...")
    df_raw = pd.read_csv(input_path)

    print("🔹 Validating schema...")
    validate_schema(df_raw)

    print("🔹 Loading reference (training) features...")
    reference_df = pd.read_csv(settings["data"]["reference_features_path"])

    print("🔹 Scoring transactions...")
    results = score_transactions(
        df_raw=df_raw,
        reference_df=reference_df
    )

    print("🔹 Saving results...")
    results.to_csv(output_path, index=False)

    print("✅ Inference completed successfully")


# -------------------------------------------------
# Entry Point
# -------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        run_training(args.input)

    elif args.mode == "inference":
        run_inference(args.input, args.output)
