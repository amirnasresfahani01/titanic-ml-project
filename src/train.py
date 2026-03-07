import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.config import MODEL_PATH, TEST_SIZE, RANDOM_STATE


def split_data(features, target):

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test


def build_model():
    model = LogisticRegression(max_iter=1000)
    return model


def save_model(model):

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)