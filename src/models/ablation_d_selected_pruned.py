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

FEATURES = Path("models/final_selected_features.csv")

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_PATH = MODEL_DIR / "ablation_d_selected_pruned.joblib"
RESULTS_PATH = RESULTS_DIR / "ablation_d_selected_pruned_results.csv"

K = 40
CCP_ALPHA = 0.0000413333


def main():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")

    X_train_full = pd.read_csv(TRAIN_X)
    y_train = pd.read_csv(TRAIN_Y).iloc[:, 0]

    X_test_full = pd.read_csv(TEST_X)
    y_test = pd.read_csv(TEST_Y).iloc[:, 0]

    print(f"Full training data: {X_train_full.shape}")
    print(f"Full test data: {X_test_full.shape}")

    # --------------------------------------------------
    # Load locked final 40-feature set
    # --------------------------------------------------

    feature_df = pd.read_csv(FEATURES)

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

    missing_train = [
        f for f in selected_features
        if f not in X_train_full.columns
    ]

    missing_test = [
        f for f in selected_features
        if f not in X_test_full.columns
    ]

    if missing_train:
        raise ValueError(
            f"Missing selected features in training data: {missing_train}"
        )

    if missing_test:
        raise ValueError(
            f"Missing selected features in test data: {missing_test}"
        )

    X_train = X_train_full[selected_features]
    X_test = X_test_full[selected_features]

    print(f"\nLocked feature count: {len(selected_features)}")
    print(f"Selected training data: {X_train.shape}")
    print(f"Selected test data: {X_test.shape}")

    # --------------------------------------------------
    # Ablation D:
    # Final 40 selected features + cost-complexity pruning
    # --------------------------------------------------

    model = DecisionTreeClassifier(
        criterion="gini",
        ccp_alpha=CCP_ALPHA,
        random_state=42,
    )

    print("\nTraining selected-feature pruned tree...")

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
    # Tree structure
    # --------------------------------------------------

    depth = model.get_depth()
    nodes = model.tree_.node_count
    leaves = model.get_n_leaves()

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    joblib.dump(
        model,
        MODEL_PATH
    )

    model_size_bytes = MODEL_PATH.stat().st_size
    model_size_kb = model_size_bytes / 1024

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    results = {
        "model": "Ablation D - Selected 40 Pruned DT",

        "features": len(selected_features),

        "criterion": "gini",
        "ccp_alpha": CCP_ALPHA,

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
    print("ABLATION D RESULTS")
    print("Selected 40 Features + Pruning")
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