import pandas as pd
import json

# Load required schema columns
with open("schema.json") as f:
    REQUIRED_COLUMNS = json.load(f)["columns"]

def validate_schema(df: pd.DataFrame) -> None:
    """
    Validate that required columns exist and basic sanity checks pass.
    """
    # Check missing columns
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check columns with >50% missing values
    high_missing = df.isnull().mean()[df.isnull().mean() > 0.5]
    if not high_missing.empty:
        raise ValueError(f"Columns with >50% missing values: {list(high_missing.index)}")

    # Check negative transaction amounts
    if (df["TransactionAmt"] < 0).any():
        raise ValueError("Negative transaction amounts detected")

    print("Schema validation passed ✅")
