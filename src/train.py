from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

try:
    from src.model import build_baseline_model
    from src.preprocess import CLEAN_DATASET_PATH, load_dataset, save_clean_dataset
except ImportError:  # pragma: no cover
    from model import build_baseline_model
    from preprocess import CLEAN_DATASET_PATH, load_dataset, save_clean_dataset


def train_baseline_model(dataset_path: str | Path = CLEAN_DATASET_PATH):
    df = load_dataset(dataset_path)
    df = df.copy()

    target_col = "Rainfall"
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' was not found in the dataset.")

    feature_columns = [col for col in df.columns if col != target_col]
    X = df[feature_columns]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_baseline_model()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    print(f"Baseline MSE: {mse:.4f}")
    return model, mse


if __name__ == "__main__":
    train_baseline_model()
