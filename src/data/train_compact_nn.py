from pathlib import Path
import time
import joblib
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42
K = 40

TRAIN_X = Path("data/processed/X_train_preprocessed.csv")
TRAIN_Y = Path("data/processed/y_train.csv")
TEST_X = Path("data/processed/X_test_preprocessed.csv")
TEST_Y = Path("data/processed/y_test.csv")

FEATURES = Path("models/final_selected_features.csv")

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_PATH = MODEL_DIR / "compact_nn_2x64.joblib"
RESULTS_PATH = RESULTS_DIR / "compact_nn_2x64_results.csv"


# ============================================================
# Main
# ============================================================

def main():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")

    X_train = pd.read_csv(TRAIN_X)
    y_train = pd.read_csv(TRAIN_Y).squeeze("columns")

    X_test = pd.read_csv(TEST_X)
    y_test = pd.read_csv(TEST_Y).squeeze("columns")

    print(f"Full training data: {X_train.shape}")
    print(f"Full test data: {X_test.shape}")

    # --------------------------------------------------------
    # Load locked final feature set
    # --------------------------------------------------------

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
        if f not in X_train.columns
    ]

    missing_test = [
        f for f in selected_features
        if f not in X_test.columns
    ]

    if missing_train:
        raise ValueError(
            f"Missing selected features in training data: {missing_train}"
        )

    if missing_test:
        raise ValueError(
            f"Missing selected features in test data: {missing_test}"
        )

    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    print(f"\nLocked feature count: {len(selected_features)}")
    print(f"Selected training data: {X_train.shape}")
    print(f"Selected test data: {X_test.shape}")

    # --------------------------------------------------------
    # Train compact neural network
    # --------------------------------------------------------

    print("\nTraining Compact Neural Network...")

    model = MLPClassifier(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        random_state=RANDOM_STATE,
        max_iter=300,
        early_stopping=False,
    )

    start_train = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start_train

    print(f"Training time: {train_time:.4f} seconds")
    print(f"Training iterations: {model.n_iter_}")

    # --------------------------------------------------------
    # Evaluate on untouched test set
    # --------------------------------------------------------

    print("\nEvaluating on untouched test set...")

    start_predict = time.perf_counter()
    y_pred = model.predict(X_test)
    prediction_time = time.perf_counter() - start_predict

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    cm = confusion_matrix(y_test, y_pred)

    mean_latency_ms = (
        prediction_time / len(X_test)
    ) * 1000

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(model, MODEL_PATH)

    model_size_kb = MODEL_PATH.stat().st_size / 1024

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 40)
    print("COMPACT NEURAL NETWORK RESULTS")
    print("=" * 40)

    print(f"Accuracy : {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall   : {recall:.6f}")
    print(f"F1       : {f1:.6f}")

    print("\nConfusion matrix:")
    print(cm)

    print("\nTiming:")
    print(f"Training: {train_time:.4f} s")
    print(f"Test prediction: {prediction_time:.6f} s")
    print(f"Mean latency: {mean_latency_ms:.6f} ms/sample")

    print("\nSerialized model:")
    print(f"Size: {model_size_kb:.2f} KB")

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results = pd.DataFrame([{
        "model": "Compact NN 2x64",
        "n_features": K,
        "hidden_layers": "64,64",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "train_time_seconds": train_time,
        "test_prediction_time_seconds": prediction_time,
        "mean_latency_ms_per_sample": mean_latency_ms,
        "serialized_model_size_kb": model_size_kb,
        "training_iterations": model.n_iter_,
        "tn": cm[0, 0],
        "fp": cm[0, 1],
        "fn": cm[1, 0],
        "tp": cm[1, 1],
    }])

    results.to_csv(RESULTS_PATH, index=False)

    print("\nSaved:")
    print(MODEL_PATH)
    print(RESULTS_PATH)


if __name__ == "__main__":
    main()