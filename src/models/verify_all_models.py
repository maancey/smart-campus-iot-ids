from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results")

RESULT_FILES = [
    "final_test_results.csv",
    "random_forest_100_results.csv",
    "linear_svm_results.csv",
    "compact_nn_2x64_results.csv",
]


def main():

    rows = []

    for filename in RESULT_FILES:

        path = RESULTS_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing result file: {path}"
            )

        df = pd.read_csv(path)

        if len(df) != 1:
            raise ValueError(
                f"{filename} should contain exactly one result row."
            )

        rows.append(df.iloc[0].to_dict())

    results = pd.DataFrame(rows)

    # Normalize feature-count field.
    # The proposed tree uses "k_features",
    # while the baselines use "features".
    if "features" not in results.columns:
        results["features"] = pd.NA

    if "k_features" in results.columns:
        results["features"] = results["features"].fillna(
            results["k_features"]
        )

    results["features"] = pd.to_numeric(
        results["features"]
    )

    print("\n========================================")
    print("CONSOLIDATED MODEL VERIFICATION")
    print("========================================")

    print("\nModels:")
    print(
        results[
            [
                "model",
                "test_samples",
                "features",
                "accuracy",
                "precision",
                "recall",
                "f1",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------
    # 1. Same test size
    # --------------------------------------------------

    test_sizes = results["test_samples"].unique()

    assert len(test_sizes) == 1, (
        "ERROR: Models used different test-set sizes."
    )

    assert test_sizes[0] == 56300, (
        "ERROR: Expected 56,300 test samples."
    )

    print("\n[PASS] All models use 56,300 test samples.")

    # --------------------------------------------------
    # 2. Same feature count
    # --------------------------------------------------

    feature_counts = results["features"].unique()

    assert len(feature_counts) == 1, (
        "ERROR: Models used different feature counts."
    )

    assert feature_counts[0] == 40, (
        "ERROR: Expected 40 features."
    )

    print("[PASS] All models use the same 40 features.")

    # --------------------------------------------------
    # 3. Confusion matrix totals
    # --------------------------------------------------

    for _, row in results.iterrows():

        total = (
            row["true_negative"]
            + row["false_positive"]
            + row["false_negative"]
            + row["true_positive"]
        )

        assert total == row["test_samples"], (
            f"ERROR: Confusion matrix total does not match "
            f"test size for {row['model']}."
        )

    print(
        "[PASS] Every confusion matrix sums to 56,300."
    )

    # --------------------------------------------------
    # 4. Accuracy verification
    # --------------------------------------------------

    for _, row in results.iterrows():

        tn = row["true_negative"]
        fp = row["false_positive"]
        fn = row["false_negative"]
        tp = row["true_positive"]

        calculated_accuracy = (
            (tn + tp)
            / (tn + fp + fn + tp)
        )

        assert abs(
            calculated_accuracy - row["accuracy"]
        ) < 1e-6, (
            f"ERROR: Accuracy mismatch for {row['model']}."
        )

    print(
        "[PASS] Accuracy matches confusion matrix "
        "for all models."
    )

    # --------------------------------------------------
    # 5. Precision / recall / F1 verification
    # --------------------------------------------------

    for _, row in results.iterrows():

        tn = row["true_negative"]
        fp = row["false_positive"]
        fn = row["false_negative"]
        tp = row["true_positive"]

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

        assert abs(
            precision - row["precision"]
        ) < 1e-6, (
            f"ERROR: Precision mismatch for {row['model']}."
        )

        assert abs(
            recall - row["recall"]
        ) < 1e-6, (
            f"ERROR: Recall mismatch for {row['model']}."
        )

        assert abs(
            f1 - row["f1"]
        ) < 1e-6, (
            f"ERROR: F1 mismatch for {row['model']}."
        )

    print(
        "[PASS] Precision, recall, and F1 match "
        "the confusion matrices."
    )

    # --------------------------------------------------
    # 6. Save consolidated verified table
    # --------------------------------------------------

    comparison = results[
        [
            "model",
            "features",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "true_negative",
            "false_positive",
            "false_negative",
            "true_positive",
            "training_time_seconds",
            "mean_test_latency_ms",
            "serialized_model_size_kb",
        ]
    ].copy()

    comparison.to_csv(
        RESULTS_DIR / "model_comparison_verified.csv",
        index=False,
    )

    print("\n========================================")
    print("VERIFIED COMPARISON")
    print("========================================")

    print(
        comparison.to_string(index=False)
    )

    print("\n[PASS] All model results verified.")

    print(
        "\nSaved:"
        "\nresults\\model_comparison_verified.csv"
    )


if __name__ == "__main__":
    main()