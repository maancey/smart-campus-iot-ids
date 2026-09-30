from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_FILE = PROCESSED_DIR / "X_train_preprocessed.csv"

CORRELATION_OUTPUT = PROCESSED_DIR / "training_correlation_matrix.csv"


def main() -> None:
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training data not found: {TRAIN_FILE}")

    X_train = pd.read_csv(TRAIN_FILE)

    print("=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    print(f"Training shape: {X_train.shape}")

    correlation = X_train.corr(method="pearson")

    correlation.to_csv(CORRELATION_OUTPUT)

    threshold = 0.90

    upper = correlation.where(
        ~pd.DataFrame(
            __import__("numpy").tril(
                __import__("numpy").ones(correlation.shape),
                k=0,
            ),
            index=correlation.index,
            columns=correlation.columns,
        ).astype(bool)
    )

    high_correlation_pairs = []

    for column in upper.columns:
        for row in upper.index:
            value = upper.loc[row, column]

            if pd.notna(value) and abs(value) > threshold:
                high_correlation_pairs.append(
                    (row, column, value)
                )

    print(f"\nCorrelation threshold: |r| > {threshold}")
    print(
        f"Highly correlated feature pairs: "
        f"{len(high_correlation_pairs)}"
    )

    if high_correlation_pairs:
        print("\nFirst 30 highly correlated pairs:")

        for feature_a, feature_b, value in high_correlation_pairs[:30]:
            print(
                f"{feature_a} <-> {feature_b}: "
                f"r={value:.4f}"
            )

    print(f"\nSaved correlation matrix:")
    print(CORRELATION_OUTPUT)


if __name__ == "__main__":
    main()