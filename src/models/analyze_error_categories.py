from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

TEST_FEATURES_FILE = PROCESSED_DIR / "X_test_preprocessed.csv"
TEST_SELECTED_FILE = PROCESSED_DIR / "test_selected.csv"
FEATURES_FILE = MODEL_DIR / "final_selected_features.csv"
MODEL_FILE = MODEL_DIR / "ablation_d_selected_pruned.joblib"

OUTPUT_FILE = RESULTS_DIR / "error_category_analysis.csv"


def main() -> None:
    print("=" * 70)
    print("FINAL MODEL ERROR ANALYSIS BY ATTACK CATEGORY")
    print("=" * 70)

    X_test = pd.read_csv(TEST_FEATURES_FILE)
    test_df = pd.read_csv(TEST_SELECTED_FILE)

    feature_df = pd.read_csv(FEATURES_FILE)
    selected_features = feature_df["feature"].tolist()

    model = joblib.load(MODEL_FILE)

    # Verify row alignment before prediction.
    if len(X_test) != len(test_df):
        raise ValueError("Test feature and audit rows do not match.")

    print(f"Test rows: {len(X_test):,}")
    print(f"Selected features: {len(selected_features)}")

    X_selected = X_test[selected_features]

    predictions = model.predict(X_selected)

    # Verify prediction length.
    if len(predictions) != len(test_df):
        raise ValueError("Prediction count does not match test rows.")

    analysis = test_df[["attack_cat", "label"]].copy()
    analysis["prediction"] = predictions

    # ---------------------------------------------------------------
    # Category-level results
    # ---------------------------------------------------------------

    rows = []

    for category, group in analysis.groupby("attack_cat", sort=True):
        total = len(group)

        if category == "Normal":
            correctly_normal = ((group["label"] == 0) &
                                (group["prediction"] == 0)).sum()
            false_positive = ((group["label"] == 0) &
                              (group["prediction"] == 1)).sum()

            rows.append({
                "attack_category": category,
                "total_samples": total,
                "correctly_detected": correctly_normal,
                "missed_as_normal": 0,
                "false_positives": false_positive,
                "detection_rate": correctly_normal / total,
            })

        else:
            detected = ((group["label"] == 1) &
                        (group["prediction"] == 1)).sum()
            missed = ((group["label"] == 1) &
                      (group["prediction"] == 0)).sum()

            rows.append({
                "attack_category": category,
                "total_samples": total,
                "correctly_detected": detected,
                "missed_as_normal": missed,
                "false_positives": 0,
                "detection_rate": detected / total,
            })

    results = pd.DataFrame(rows)

    print("\n========================================")
    print("CATEGORY-LEVEL RESULTS")
    print("========================================")

    print(
        results.to_string(
            index=False,
            formatters={
                "detection_rate": "{:.4f}".format,
            },
        )
    )

    # ---------------------------------------------------------------
    # Confusion table
    # ---------------------------------------------------------------

    print("\n========================================")
    print("CATEGORY × PREDICTION")
    print("========================================")

    table = pd.crosstab(
        analysis["attack_cat"],
        analysis["prediction"],
        margins=True,
    )

    print(table)

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nSaved:")
    print(f"  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()