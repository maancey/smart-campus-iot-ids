from pathlib import Path

import pandas as pd
from sklearn.feature_selection import mutual_info_classif


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FEATURE_FILE = PROCESSED_DIR / "X_train_corr_reduced.csv"
TARGET_FILE = PROCESSED_DIR / "y_train.csv"

OUTPUT_FILE = PROCESSED_DIR / "mutual_information_ranking.csv"


def main() -> None:
    print("=" * 70)
    print("MUTUAL INFORMATION FEATURE RANKING")
    print("=" * 70)

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Reduced training features not found: {FEATURE_FILE}"
        )

    if not TARGET_FILE.exists():
        raise FileNotFoundError(
            f"Training target not found: {TARGET_FILE}"
        )

    X_train = pd.read_csv(FEATURE_FILE)
    y_train = pd.read_csv(TARGET_FILE).squeeze("columns")

    print(f"Training shape: {X_train.shape}")

    # Mutual information is calculated using training data only.
    mi_scores = mutual_info_classif(
        X_train,
        y_train,
        discrete_features="auto",
        random_state=42,
    )

    ranking = pd.DataFrame(
        {
            "feature": X_train.columns,
            "mutual_information": mi_scores,
        }
    )

    ranking = ranking.sort_values(
        by="mutual_information",
        ascending=False,
    ).reset_index(drop=True)

    ranking["rank"] = ranking.index + 1

    ranking = ranking[
        [
            "rank",
            "feature",
            "mutual_information",
        ]
    ]

    ranking.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nTop 30 features:")
    print(ranking.head(30).to_string(index=False))

    print("\nMutual-information statistics:")
    print(
        ranking["mutual_information"].describe().to_string()
    )

    print(f"\nSaved ranking:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()