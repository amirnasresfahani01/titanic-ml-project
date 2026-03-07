from pathlib import Path
import pickle
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "titanic_model.pkl"


def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


def predict_passenger(data: pd.DataFrame):
    model = load_model()

    features = data[["pclass", "sex", "age"]].copy()

    features["sex"] = features["sex"].map({
        "male": 0,
        "female": 1
    })

    features["age"] = features["age"].fillna(features["age"].median())

    prediction = model.predict(features)

    return prediction