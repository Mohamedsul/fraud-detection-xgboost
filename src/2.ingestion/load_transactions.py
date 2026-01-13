import pandas as pd
from pathlib import Path

def load_transactions(path:str) -> pd.DataFrame:
    """
    Loads transaction data from disk file into a pandas DataFrame.

    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f'Data file not found at {path}')
    
    if file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
    elif file_path.suffix in ['.aparquet', '.pq']:
        df = pd.read_parquet(file_path)
    else:
        raise ValueError('Unsupported file format. Please provide a CSV or Parquet file.')
    
    return df


