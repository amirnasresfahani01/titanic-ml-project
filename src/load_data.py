import pandas as pd
from src.config import DATA_PATH


def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.lower()
    return df