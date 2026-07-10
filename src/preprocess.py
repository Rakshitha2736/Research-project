from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATASET_PATH = DATA_DIR / "india_weather_rainfall_data.xlsx"
CLEAN_DATASET_PATH = DATA_DIR / "clean_dataset.csv"


def load_dataset(path: str | Path = DATASET_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Please place your Excel file there and re-run."
        )
    return pd.read_excel(path)


def inspect_dataset(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numerical_features": df.select_dtypes(include=["number"]).columns.tolist(),
        "categorical_features": df.select_dtypes(exclude=["number"]).columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date_of_record" in df.columns:
        df["date_of_record"] = pd.to_datetime(df["date_of_record"], errors="coerce")
    if "date_of_record" in df.columns:
        df = df.sort_values("date_of_record").reset_index(drop=True)
    return df


def save_clean_dataset(df: pd.DataFrame, output_path: str | Path = CLEAN_DATASET_PATH) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path
