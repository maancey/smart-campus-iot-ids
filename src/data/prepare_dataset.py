from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_FILE = RAW_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = RAW_DIR / "UNSW_NB15_testing-set.csv"

TRAIN_OUTPUT = PROCESSED_DIR / "train_selected.csv"
TEST_OUTPUT = PROCESSED_DIR / "test_selected.csv"

SELECTED_CATEGORIES = [
    "Normal",
    "Exploits",
    "Reconnaissance",
    "Backdoor",
    "DoS",
]


def prepare_dataset(input_file: Path, output_file: Path) -> None:
    df = pd.read_csv(input_file)

    # Remove rows without a valid attack category or binary label.
    df = df.dropna(subset=["attack_cat", "label"]).copy()

    # Retain only the predefined experimental population.
    df = df[df["attack_cat"].isin(SELECTED_CATEGORIES)].copy()

    # Convert the target explicitly to integer.
    df["label"] = df["label"].astype(int)

    # Save the curated dataset.
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print("=" * 70)
    print(f"INPUT : {input_file.name}")
    print(f"OUTPUT: {output_file}")
    print("=" * 70)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nAttack-category distribution:")
    print(df["attack_cat"].value_counts().to_string())

    print("\nBinary-label distribution:")
    print(df["label"].value_counts().sort_index().to_string())

    print()


def main() -> None:
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")

    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Testing file not found: {TEST_FILE}")

    prepare_dataset(TRAIN_FILE, TRAIN_OUTPUT)
    prepare_dataset(TEST_FILE, TEST_OUTPUT)


if __name__ == "__main__":
    main()