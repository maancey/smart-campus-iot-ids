from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
K = 40
CCP_ALPHA = 0.0000413333

TRAIN_X = Path("data/processed/X_train_preprocessed.csv")
TRAIN_Y = Path("data/processed/y_train.csv")

TEST_X = Path("data/processed/X_test_preprocessed.csv")
TEST_Y = Path("data/processed/y_test.csv")

MI_RANKING = Path(
    "data/processed/leakage_safe_mutual_information_ranking.csv"
)

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_PATH = MODEL_DIR / "final_pruned_decision_tree.joblib"
FEATURE_PATH = MODEL_DIR / "final_selected_features.csv"
RESULTS_PATH = RESULTS_DIR / "final_test_results.csv"


def main():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training and test data...")

    X_train = pd.read_csv(TRAIN_X)
    y_train = pd.read_csv(TRAIN_Y).iloc[:, 0]

    X_test = pd.read_csv(TEST_X)
    y_test = pd.read_csv(TEST_Y).iloc[:, 0]

    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")

    # ---------------------------------------------------------
    # Load LOCKED feature ranking.
    # ---------------------------------------------------------

    ranking = pd.read_csv(MI_RANKING)

    selected_features = (
        ranking
        .sort_values("rank")
        .head(K)["feature"]
        .tolist()
    )

    print(f"\nSelected features: {len(selected_features)}")

    # ---------------------------------------------------------
    # Apply exactly the same features to train and test.
    # ---------------------------------------------------------

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    print(
        f"Selected training shape: "
        f"{X_train_selected.shape}"
    )

    print(
        f"Selected test shape: "
        f"{X_test_selected.shape}"
    )

    # Save exact feature list.
    pd.DataFrame({
        "rank": range(1, len(selected_features) + 1),
        "feature": selected_features
    }).to_csv(
        FEATURE_PATH,
        index=False
    )

    # ---------------------------------------------------------
    # Train FINAL model on ALL training data.
    # ---------------------------------------------------------

    print("\nTraining final pruned decision tree...")

    model = DecisionTreeClassifier(
        criterion="gini",
        ccp_alpha=CCP_ALPHA,
        random_state=RANDOM_STATE
    )

    training_start = time.perf_counter()

    model.fit(
        X_train_selected,
        y_train
    )

    training_time = time.perf_counter() - training_start

    print(
        f"Training time: "
        f"{training_time:.4f} seconds"
    )

    print(
        f"Tree depth: {model.get_depth()}"
    )

    print(
        f"Tree nodes: {model.tree_.node_count}"
    )

    print(
        f"Tree leaves: {model.get_n_leaves()}"
    )

    # ---------------------------------------------------------
    # Final TEST evaluation.
    #
    # This is the first time the test labels are used.
    # ---------------------------------------------------------

    print("\nEvaluating on untouched test set...")

    test_start = time.perf_counter()

    predictions = model.predict(
        X_test_selected
    )

    total_test_time = (
        time.perf_counter() - test_start
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    # ---------------------------------------------------------
    # Per-sample latency.
    # ---------------------------------------------------------

    latency_ms = (
        total_test_time
        / len(X_test_selected)
        * 1000
    )

    # ---------------------------------------------------------
    # Model file size.
    #
    # IMPORTANT:
    # This is the serialized sklearn model size, NOT ESP32
    # firmware flash usage.
    # ---------------------------------------------------------

    joblib.dump(
        model,
        MODEL_PATH
    )

    model_size_bytes = MODEL_PATH.stat().st_size
    model_size_kb = model_size_bytes / 1024

    # ---------------------------------------------------------
    # Save results.
    # ---------------------------------------------------------

    results = {
        "model": "DecisionTreeClassifier",
        "criterion": "gini",
        "k_features": K,
        "ccp_alpha": CCP_ALPHA,
        "random_state": RANDOM_STATE,

        "train_samples": len(X_train_selected),
        "test_samples": len(X_test_selected),

        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,

        "tree_depth": model.get_depth(),
        "tree_nodes": model.tree_.node_count,
        "tree_leaves": model.get_n_leaves(),

        "training_time_seconds": training_time,
        "test_prediction_time_seconds": total_test_time,
        "mean_test_latency_ms": latency_ms,

        "serialized_model_size_bytes": model_size_bytes,
        "serialized_model_size_kb": model_size_kb,

        "true_negative": int(cm[0, 0]),
        "false_positive": int(cm[0, 1]),
        "false_negative": int(cm[1, 0]),
        "true_positive": int(cm[1, 1]),
    }

    results_df = pd.DataFrame([results])

    results_df.to_csv(
        RESULTS_PATH,
        index=False
    )

    # ---------------------------------------------------------
    # Print final report.
    # ---------------------------------------------------------

    print("\n========================================")
    print("FINAL TEST RESULTS")
    print("========================================")

    print(f"Accuracy : {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall   : {recall:.6f}")
    print(f"F1       : {f1:.6f}")

    print("\nConfusion matrix:")
    print(cm)

    print("\nTree structure:")
    print(f"Depth : {model.get_depth()}")
    print(f"Nodes : {model.tree_.node_count}")
    print(f"Leaves: {model.get_n_leaves()}")

    print("\nTiming:")
    print(
        f"Total test prediction time: "
        f"{total_test_time:.6f} s"
    )

    print(
        f"Mean latency per sample: "
        f"{latency_ms:.6f} ms"
    )

    print("\nSerialized model:")
    print(
        f"Size: {model_size_kb:.2f} KB"
    )

    print("\nSaved:")
    print(MODEL_PATH)
    print(FEATURE_PATH)
    print(RESULTS_PATH)


if __name__ == "__main__":
    main()