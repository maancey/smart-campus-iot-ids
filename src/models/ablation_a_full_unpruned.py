from pathlib import Path
import time

import joblib
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


TRAIN_X = Path("data/processed/X_train_preprocessed.csv")
TRAIN_Y = Path("data/processed/y_train.csv")

TEST_X = Path("data/processed/X_test_preprocessed.csv")
TEST_Y = Path("data/processed/y_test.csv")

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_PATH = MODEL_DIR / "ablation_a_full_unpruned.joblib"
RESULTS_PATH = RESULTS_DIR / "ablation_a_full_unpruned_results.csv"


def main():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")

    X_train = pd.read_csv(TRAIN_X)
    y_train = pd.read_csv(TRAIN_Y).iloc[:, 0]

    X_test = pd.read_csv(TEST_X)
    y_test = pd.read_csv(TEST_Y).iloc[:, 0]

    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    print(f"Features: {X_train.shape[1]}")

    # --------------------------------------------------
    # Ablation A:
    # Full feature set + no pruning
    # --------------------------------------------------

    model = DecisionTreeClassifier(
        criterion="gini",
        random_state=42,
    )

    print("\nTraining full unpruned decision tree...")

    train_start = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - train_start

    print(
        f"Training time: "
        f"{training_time:.4f} seconds"
    )

    # --------------------------------------------------
    # Test evaluation
    # --------------------------------------------------

    print("\nEvaluating on test set...")

    test_start = time.perf_counter()

    y_pred = model.predict(X_test)

    test_time = time.perf_counter() - test_start

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

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

    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    latency_ms = (
        test_time / len(X_test) * 1000
    )

    # --------------------------------------------------
    # Model structure
    # --------------------------------------------------

    depth = model.get_depth()
    nodes = model.tree_.node_count
    leaves = model.get_n_leaves()

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    joblib.dump(model, MODEL_PATH)

    model_size_bytes = MODEL_PATH.stat().st_size
    model_size_kb = model_size_bytes / 1024

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    results = {
        "model": "Ablation A - Full Unpruned DT",

        "features": X_train.shape[1],

        "criterion": "gini",
        "ccp_alpha": 0.0,

        "train_samples": len(X_train),
        "test_samples": len(X_test),

        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,

        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),

        "tree_depth": depth,
        "tree_nodes": nodes,
        "tree_leaves": leaves,

        "training_time_seconds": training_time,
        "test_prediction_time_seconds": test_time,
        "mean_test_latency_ms": latency_ms,

        "serialized_model_size_bytes": model_size_bytes,
        "serialized_model_size_kb": model_size_kb,
    }

    pd.DataFrame([results]).to_csv(
        RESULTS_PATH,
        index=False,
    )

    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    print("\n========================================")
    print("ABLATION A RESULTS")
    print("Full Features + No Pruning")
    print("========================================")

    print(f"Accuracy : {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall   : {recall:.6f}")
    print(f"F1       : {f1:.6f}")

    print("\nConfusion matrix:")
    print(cm)

    print("\nTree structure:")
    print(f"Depth : {depth}")
    print(f"Nodes : {nodes}")
    print(f"Leaves: {leaves}")

    print("\nTiming:")
    print(
        f"Training: {training_time:.4f} s"
    )
    print(
        f"Test prediction: {test_time:.6f} s"
    )
    print(
        f"Mean latency: {latency_ms:.6f} ms/sample"
    )

    print("\nSerialized model:")
    print(
        f"Size: {model_size_kb:.2f} KB"
    )

    print("\nSaved:")
    print(MODEL_PATH)
    print(RESULTS_PATH)


if __name__ == "__main__":
    main()