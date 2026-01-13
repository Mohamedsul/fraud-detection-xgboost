import pandas as pd
import numpy as np
from src.categorical_encoding import CategoricalEncoder

WINDOW_1H = 3600  # 1 hour in seconds

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features for fraud detection:
        - Amount normalization & z-score
        - Velocity / rolling features
        - Device and identity features
        - Missing value indicators
        - Categorical encoding
    """
    df = df.copy()

    # --- Time conversion ---
    df['TransactionDT'] = pd.to_datetime(df['TransactionDT'], unit='s')
    df['hour'] = df['TransactionDT'].dt.hour
    df['day_of_week'] = df['TransactionDT'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # --- Velocity Features ---
    # Time since last transaction per card
    df['time_since_last_card_tx'] = df.groupby('card1')['TransactionDT'].diff().dt.total_seconds().fillna(-1)

    # Transactions count within 1h window
    df['card_tx_count_1h'] = (
        df.groupby('card1')['TransactionDT']
          .transform(lambda x: (x.diff().dt.total_seconds() <= WINDOW_1H).cumsum())
    )

    # --- Rolling / Aggregation Features ---
    # Rolling mean & std per card
    df['card_avg_amount'] = (
        df.groupby('card1')['TransactionAmt']
          .expanding()
          .mean()
          .shift(1)
          .reset_index(level=0, drop=True)
    )
    df['card_std_amount'] = (
        df.groupby('card1')['TransactionAmt']
          .expanding()
          .std()
          .shift(1)
          .reset_index(level=0, drop=True)
    )

    # Amount deviation features
    df['amount_ratio'] = df['TransactionAmt'] / (df['card_avg_amount'] + 1e-6)
    df['amount_zscore'] = (df['TransactionAmt'] - df['card_avg_amount']) / (df['card_std_amount'] + 1e-6)

    # --- Identity & Device Features ---
    df['has_identity'] = (~df['DeviceType'].isna()).astype(int)
    df['DeviceType'] = df['DeviceType'].fillna('missing')

    # --- Email Domain Features ---
    for col in ['P_emaildomain', 'R_emaildomain']:
        if col in df.columns:
            df[col] = df[col].fillna('missing')

    # --- Missing Indicators for id_* columns ---
    identity_cols = [col for col in df.columns if col.startswith('id_')]
    for col in identity_cols:
        df[f'{col}_missing'] = df[col].isna().astype(int)

    # --- Categorical Encoding ---
    cat_cols = ['ProductCD', 'card4', 'card6', 'DeviceType', 'P_emaildomain', 'R_emaildomain']
    encoder = CategoricalEncoder()
    df[cat_cols] = encoder.fit_transform(df[cat_cols])

    # --- Drop Columns not needed for modeling ---
    DROP_COLS = ['TransactionID', 'TransactionDT']
    drop_cols = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop_cols)

    return df
