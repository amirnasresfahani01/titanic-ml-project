import pandas as pd


def add_family_size(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    return df


def add_is_alone(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_alone"] = (df["family_size"] == 1).astype(int)
    return df


def add_title(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["title"] = df["name"].str.extract(r" ([A-Za-z]+)\.", expand=False)

    df["title"] = df["title"].replace(
        ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"],
        "Rare"
    )

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_family_size(df)
    df = add_is_alone(df)
    df = add_title(df)
    return df