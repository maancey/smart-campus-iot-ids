from pathlib import Path
import time

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# =========================================================
# Experiment configuration
# =========================================================

RANDOM_STATE = 42
N_ESTIMATORS = 100
K = 40


# =========================================================
# Input data
# =========================================================

TRAIN_X = Path(
    "data/processed/X_train_preprocessed.csv"
)

TRAIN_Y = Path(
    "data/processed/y_train.csv"
)

TEST_X = Path(
    "data/processed/X_test_preprocessed.csv"
)

TEST_Y = Path(
    "data/processed/y_test.csv"
)


# =========================================================
# Final locked feature set
#
# This is the exact 40-feature set used by the proposed
# final decision-tree model.
# =========================================================

FEATURES = Path(
    "models/final_selected_features.csv"
)


# =========================================================
# Output paths
# =========================================================

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_PATH = (
    MODEL_DIR /
    "random_forest_100.joblib"
)

RESULTS_PATH = (
    RESULTS_DIR /
    "random_forest_100_results.csv"
)


def main():

    # -----------------------------------------------------
    # Create output directories.
    # -----------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Load data.
    # -----------------------------------------------------

    print("Loading data...")

    X_train = pd.read_csv(
        TRAIN_X
    )

    y_train = pd.read_csv(
        TRAIN_Y
    ).iloc[:, 0]

    X_test = pd.read_csv(
        TEST_X
    )

    y_test = pd.read_csv(
        TEST_Y
    ).iloc[:, 0]

    print(
        f"Full training data: "
        f"{X_train.shape}"
    )

    print(
        f"Full test data: "
        f"{X_test.shape}"
    )

    # -----------------------------------------------------
    # Load exactly the same final locked features used by
    # the proposed decision-tree model.
    # -----------------------------------------------------

    feature_table = pd.read_csv(
        FEATURES
    )

    selected_features = (
        feature_table
        .sort_values("rank")
        ["feature"]
        .tolist()
    )

    # -----------------------------------------------------
    # Verify feature count.
    # -----------------------------------------------------

    if len(selected_features) != K:
        raise ValueError(
            f"Expected {K} selected features, "
            f"but found {len(selected_features)}."
        )

    # -----------------------------------------------------
    # Verify that every selected feature exists in both
    # training and test data.
    # -----------------------------------------------------

    missing_train_features = [
        feature
        for feature in selected_features
        if feature not in X_train.columns
    ]

    missing_test_features = [
        feature
        for feature in selected_features
        if feature not in X_test.columns
    ]

    if missing_train_features:
        raise ValueError(
            "Selected features missing from "
            f"training data: {missing_train_features}"
        )

    if missing_test_features:
        raise ValueError(
            "Selected features missing from "
            f"test data: {missing_test_features}"
        )

    print(
        f"\nLocked feature count: "
        f"{len(selected_features)}"
    )

    # -----------------------------------------------------
    # Apply the identical 40-feature representation to
    # both training and test data.
    # -----------------------------------------------------

    X_train = X_train[
        selected_features
    ]

    X_test = X_test[
        selected_features
    ]

    print(
        f"Selected training data: "
        f"{X_train.shape}"
    )

    print(
        f"Selected test data: "
        f"{X_test.shape}"
    )

    # -----------------------------------------------------
    # Random Forest baseline
    #
    # 100 trees, fixed random state for reproducibility.
    # Other parameters remain at sklearn defaults.
    # -----------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print(
        "\nTraining Random Forest..."
    )

    train_start = time.perf_counter()

    model.fit(
        X_train,
        y_train
    )

    training_time = (
        time.perf_counter()
        - train_start
    )

    print(
        f"Training time: "
        f"{training_time:.4f} seconds"
    )

    # -----------------------------------------------------
    # Test prediction
    # -----------------------------------------------------

    print(
        "\nEvaluating on untouched test set..."
    )

    test_start = time.perf_counter()

    y_pred = model.predict(
        X_test
    )

    test_time = (
        time.perf_counter()
        - test_start
    )

    # -----------------------------------------------------
    # Classification metrics
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    tn, fp, fn, tp = cm.ravel()

    # -----------------------------------------------------
    # Mean host-side prediction latency.
    #
    # IMPORTANT:
    # This is Python/scikit-learn host inference time,
    # not ESP32 embedded latency.
    # -----------------------------------------------------

    latency_ms = (
        test_time
        / len(X_test)
        * 1000
    )

    # -----------------------------------------------------
    # Save serialized model.
    #
    # IMPORTANT:
    # This is serialized Joblib size, not ESP32
    # firmware flash usage.
    # -----------------------------------------------------

    joblib.dump(
        model,
        MODEL_PATH
    )

    model_size_bytes = (
        MODEL_PATH.stat().st_size
    )

    model_size_kb = (
        model_size_bytes / 1024
    )

    # -----------------------------------------------------
    # Save results.
    # -----------------------------------------------------

    results = {
        "model": "Random Forest",
        "n_estimators": N_ESTIMATORS,
        "random_state": RANDOM_STATE,

        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": len(selected_features),

        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,

        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),

        "training_time_seconds": (
            training_time
        ),

        "test_prediction_time_seconds": (
            test_time
        ),

        "mean_test_latency_ms": (
            latency_ms
        ),

        "serialized_model_size_bytes": (
            model_size_bytes
        ),

        "serialized_model_size_kb": (
            model_size_kb
        ),
    }

    pd.DataFrame(
        [results]
    ).to_csv(
        RESULTS_PATH,
        index=False,
    )

    # -----------------------------------------------------
    # Report
    # -----------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "RANDOM FOREST RESULTS"
    )

    print(
        "========================================"
    )

    print(
        f"Accuracy : {accuracy:.6f}"
    )

    print(
        f"Precision: {precision:.6f}"
    )

    print(
        f"Recall   : {recall:.6f}"
    )

    print(
        f"F1       : {f1:.6f}"
    )

    print(
        "\nConfusion matrix:"
    )

    print(
        cm
    )

    print(
        "\nTiming:"
    )

    print(
        f"Training: "
        f"{training_time:.4f} s"
    )

    print(
        f"Test prediction: "
        f"{test_time:.6f} s"
    )

    print(
        f"Mean latency: "
        f"{latency_ms:.6f} ms/sample"
    )

    print(
        "\nSerialized model:"
    )

    print(
        f"Size: "
        f"{model_size_kb:.2f} KB"
    )

    print(
        "\nSaved:"
    )

    print(
        MODEL_PATH
    )

    print(
        RESULTS_PATH
    )


if __name__ == "__main__":
    main()