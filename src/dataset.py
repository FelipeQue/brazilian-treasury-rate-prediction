import pandas as pd

from src.config import RAW_FILE

def load_raw_data() -> pd.DataFrame:
    """Lê o CSV bruto em data/raw/ e devolve um DataFrame."""
    return pd.read_csv(RAW_FILE)
