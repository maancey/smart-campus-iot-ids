from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAINING_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "UNSW_NB15_training-set.csv"
)

TESTING_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "UNSW_NB15_testing-set.csv"
)


def inspect_dataset(path: Path, name: str) -> None:
    print("=" * 70)
    print(f"{name}")
    print("=" * 70)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    print(f"File: {path}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nMissing values:")
    missing = df.isna().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("  None")
    else:
        print(missing.to_string())

    print("\nDuplicate rows:")
    print(f"  {df.duplicated().sum()}")

    if "label" in df.columns:
        print("\nBinary label distribution:")
        print(df["label"].value_counts(dropna=False).to_string())

    if "attack_cat" in df.columns:
        print("\nAttack category distribution:")
        print(df["attack_cat"].value_counts(dropna=False).to_string())

    print("\nCategorical columns:")
    categorical = df.select_dtypes(include=["object", "category"]).columns

    if len(categorical) == 0:
        print("  None")
    else:
        for column in categorical:
            print(f"  - {column}")

    print()


def main() -> None:
    inspect_dataset(TRAINING_FILE, "TRAINING DATASET")
    inspect_dataset(TESTING_FILE, "TESTING DATASET")


if __name__ == "__main__":
    main()