from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results")

FILES = {
    "A - Full + Unpruned":
        "ablation_a_full_unpruned_results.csv",

    "B - Correlation Reduced + Unpruned":
        "ablation_b_corr_unpruned_results.csv",

    "C - Full + Pruned":
        "ablation_c_full_pruned_results.csv",

    "D - 40 Features + Pruned":
        "final_test_results.csv",
}


def main():

    rows = []

    for label, filename in FILES.items():

        path = RESULTS_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing result file: {path}"
            )

        df = pd.read_csv(path)

        if len(df) != 1:
            raise ValueError(
                f"{filename} must contain exactly one row."
            )

        row = df.iloc[0].to_dict()
        row["configuration"] = label

        # Normalize feature-count field.
        if pd.isna(row.get("features", None)):
            row["features"] = row.get("k_features")

        rows.append(row)

    results = pd.DataFrame(rows)

    # --------------------------------------------------
    # Required columns
    # --------------------------------------------------

    columns = [
        "configuration",
        "features",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "tree_depth",
        "tree_nodes",
        "tree_leaves",
        "mean_test_latency_ms",
        "serialized_model_size_kb",
    ]

    comparison = results[columns].copy()

    # --------------------------------------------------
    # Verify feature counts
    # --------------------------------------------------

    expected_features = {
        "A - Full + Unpruned": 194,
        "B - Correlation Reduced + Unpruned": 185,
        "C - Full + Pruned": 194,
        "D - 40 Features + Pruned": 40,
    }

    for _, row in comparison.iterrows():

        expected = expected_features[
            row["configuration"]
        ]

        assert int(row["features"]) == expected, (
            f"Feature-count mismatch for "
            f"{row['configuration']}"
        )

    print(
        "[PASS] Feature counts match "
        "the experimental configurations."
    )

    # --------------------------------------------------
    # Verify test-set size
    # --------------------------------------------------

    for _, row in results.iterrows():

        assert int(row["test_samples"]) == 56300, (
            f"Wrong test size for "
            f"{row['configuration']}"
        )

    print(
        "[PASS] All ablations use 56,300 test samples."
    )

    # --------------------------------------------------
    # Verify confusion matrices and metrics
    # --------------------------------------------------

    for _, row in results.iterrows():

        tn = row["true_negative"]
        fp = row["false_positive"]
        fn = row["false_negative"]
        tp = row["true_positive"]

        total = tn + fp + fn + tp

        assert total == 56300, (
            f"Confusion matrix error for "
            f"{row['configuration']}"
        )

        accuracy = (tn + tp) / total
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

        assert abs(
            accuracy - row["accuracy"]
        ) < 1e-6

        assert abs(
            precision - row["precision"]
        ) < 1e-6

        assert abs(
            recall - row["recall"]
        ) < 1e-6

        assert abs(
            f1 - row["f1"]
        ) < 1e-6

    print(
        "[PASS] All ablation metrics match "
        "their confusion matrices."
    )

    # --------------------------------------------------
    # Calculate change relative to A
    # --------------------------------------------------

    baseline = comparison[
        comparison["configuration"]
        == "A - Full + Unpruned"
    ].iloc[0]

    comparison["f1_change_vs_A"] = (
        comparison["f1"]
        - baseline["f1"]
    )

    comparison["node_reduction_vs_A_pct"] = (
        1
        - comparison["tree_nodes"]
        / baseline["tree_nodes"]
    ) * 100

    comparison["size_reduction_vs_A_pct"] = (
        1
        - comparison["serialized_model_size_kb"]
        / baseline["serialized_model_size_kb"]
    ) * 100

    comparison["feature_reduction_vs_A_pct"] = (
        1
        - comparison["features"]
        / baseline["features"]
    ) * 100

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    output = (
        RESULTS_DIR
        / "ablation_comparison_verified.csv"
    )

    comparison.to_csv(
        output,
        index=False,
    )

    # --------------------------------------------------
    # Display
    # --------------------------------------------------

    print("\n========================================")
    print("VERIFIED ABLATION COMPARISON")
    print("========================================")

    print(
        comparison.to_string(index=False)
    )

    print(
        "\n[PASS] All four ablation configurations verified."
    )

    print(f"\nSaved:\n{output}")


if __name__ == "__main__":
    main()