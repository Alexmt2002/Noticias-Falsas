import os

BASE_DIR = os.path.dirname(__file__)

MODEL_DIR = os.path.join(BASE_DIR, "model")

DATA_DIR = os.path.join(
    BASE_DIR,
    "..",
    "data"
)

DATASET_PATH = os.path.join(
    DATA_DIR,
    "dataset_limpio.csv"
)

MODEL_NAME = "all-MiniLM-L6-v2"