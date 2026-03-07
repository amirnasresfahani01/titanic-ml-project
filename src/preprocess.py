import pandas as pd


def select_features(df: pd.DataFrame):
    features = df[["pclass", "sex", "age"]].copy()
    target = df["survived"].copy()

    features["sex"] = features["sex"].map({
        "male": 0,
        "female": 1
    })

    features["age"] = features["age"].fillna(features["age"].median())

    return features, target