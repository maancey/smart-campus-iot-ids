from pathlib import Path

import joblib
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

TEST_X_FILE = PROCESSED_DIR / "X_test_preprocessed.csv"
TEST_Y_FILE = PROCESSED_DIR / "y_test.csv"

FEATURES_FILE = MODEL_DIR / "final_selected_features.csv"
MODEL_FILE = MODEL_DIR / "ablation_d_selected_pruned.joblib"

OUTPUT_FILE = RESULTS_DIR / "feature_importance_analysis.csv"

K = 40
N_REPEATS = 10
RANDOM_STATE = 42


def main() -> None:

    print("=" * 70)
    print("FINAL MODEL FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------

    X_test = pd.read_csv(TEST_X_FILE)
    y_test = pd.read_csv(TEST_Y_FILE).iloc[:, 0]

    feature_df = pd.read_csv(FEATURES_FILE)

    if "feature" not in feature_df.columns:
        raise ValueError(
            "final_selected_features.csv must contain a 'feature' column."
        )

    if "rank" in feature_df.columns:
        feature_df = feature_df.sort_values("rank")

    selected_features = feature_df["feature"].tolist()

    if len(selected_features) != K:
        raise ValueError(
            f"Expected exactly {K} selected features, "
            f"but found {len(selected_features)}."
        )

    missing_features = [
        feature for feature in selected_features
        if feature not in X_test.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing selected features in test data: {missing_features}"
        )

    X_test = X_test[selected_features]

    model = joblib.load(MODEL_FILE)

    print(f"Test rows: {len(X_test):,}")
    print(f"Selected features: {len(selected_features)}")

    # ---------------------------------------------------------------
    # Baseline F1
    # ---------------------------------------------------------------

    baseline_prediction = model.predict(X_test)

    baseline_f1 = f1_score(
        y_test,
        baseline_prediction,
        zero_division=0,
    )

    print(f"Baseline test F1: {baseline_f1:.6f}")

    # ---------------------------------------------------------------
    # 1. Tree impurity importance
    # ---------------------------------------------------------------

    impurity_importance = pd.DataFrame({
        "feature": selected_features,
        "impurity_importance": model.feature_importances_,
    })

    # ---------------------------------------------------------------
    # 2. Permutation importance using held-out test set
    # ---------------------------------------------------------------

    print("\nCalculating permutation importance...")
    print(f"Repeats per feature: {N_REPEATS}")

    permutation = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="f1",
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    permutation_results = pd.DataFrame({
        "feature": selected_features,
        "permutation_importance_mean": permutation.importances_mean,
        "permutation_importance_std": permutation.importances_std,
    })

    # ---------------------------------------------------------------
    # Combine results
    # ---------------------------------------------------------------

    results = impurity_importance.merge(
        permutation_results,
        on="feature",
    )

    results["abs_permutation_importance_mean"] = (
        results["permutation_importance_mean"].abs()
    )

    results = results.sort_values(
        by="permutation_importance_mean",
        ascending=False,
    ).reset_index(drop=True)

    results.insert(
        0,
        "permutation_rank",
        range(1, len(results) + 1),
    )

    # Add impurity rank separately.
    impurity_rank = (
        results["impurity_importance"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    results.insert(
        1,
        "impurity_rank",
        impurity_rank,
    )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------

    print("\n========================================")
    print("TOP FEATURE IMPORTANCE RESULTS")
    print("========================================")

    print(
        results[
            [
                "permutation_rank",
                "feature",
                "permutation_importance_mean",
                "permutation_importance_std",
                "impurity_importance",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    print("\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()