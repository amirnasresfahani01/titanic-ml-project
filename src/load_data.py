import pandas as pd


def load_data():
    df = pd.read_csv("../data/titanic.csv")
    df.columns = df.columns.str.lower()
    return df