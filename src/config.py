from pathlib import Path

# project root
BASE_DIR = Path(__file__).resolve().parent.parent

# data path
DATA_PATH = BASE_DIR / "data" / "titanic.csv"

# model path
MODEL_PATH = BASE_DIR / "models" / "titanic_model.pkl"

# model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42