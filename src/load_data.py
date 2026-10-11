
from pathlib import Path
import pandas as pd

FILE = "detect_dataset.csv"


def load_raw(data_dir="data"):
    path = Path(data_dir) / FILE

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)

if __name__ == "__main__":
    df = load_raw()
    df = df.dropna(axis=1, how="all")
    df = df.rename(columns={"Output (S)": "fault"})

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\nDataset shape:", df.shape)
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nFirst five rows:")
    print(df.head().to_string())