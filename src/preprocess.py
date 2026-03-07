import pandas as pd


BASE_FEATURES = ["pclass", "sex", "age"]


def preprocess_features(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()

    features["sex"] = features["sex"].map({
        "male": 0,
        "female": 1
    })

    features["age"] = features["age"].fillna(features["age"].median())

    return features


def select_features(df: pd.DataFrame):
    features = df[BASE_FEATURES].copy()
    target = df["survived"].copy()

    features = preprocess_features(features)

    return features, target