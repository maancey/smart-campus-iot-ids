from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


TEST_X = Path("data/processed/X_test_preprocessed.csv")
TEST_Y = Path("data/processed/y_test.csv")

FEATURES = Path("models/final_selected_features.csv")
MODEL = Path("models/final_pruned_decision_tree.joblib")


def main():

    # Load final model and locked feature list
    model = joblib.load(MODEL)

    features = pd.read_csv(FEATURES)["feature"].tolist()

    # Load untouched test data
    X_test = pd.read_csv(TEST_X)
    y_test = pd.read_csv(TEST_Y).iloc[:, 0]

    X_test = X_test[features]

    # Generate predictions
    y_pred = model.predict(X_test)

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------

    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    print("\n========================================")
    print("CONFUSION MATRIX")
    print("========================================")
    print(cm)

    print(f"\nTN = {tn}")
    print(f"FP = {fp}")
    print(f"FN = {fn}")
    print(f"TP = {tp}")

    # --------------------------------------------------
    # Class-wise metrics
    # --------------------------------------------------

    print("\n========================================")
    print("CLASS-WISE METRICS")
    print("========================================")

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Attack"],
        digits=6,
        zero_division=0,
    )

    print(report)

    # --------------------------------------------------
    # Manual binary calculations
    # --------------------------------------------------

    normal_precision = tn / (tn + fn)
    normal_recall = tn / (tn + fp)
    normal_f1 = (
        2 * normal_precision * normal_recall
        / (normal_precision + normal_recall)
    )

    attack_precision = tp / (tp + fp)
    attack_recall = tp / (tp + fn)
    attack_f1 = (
        2 * attack_precision * attack_recall
        / (attack_precision + attack_recall)
    )

    accuracy = (tn + tp) / (tn + fp + fn + tp)

    macro_f1 = (normal_f1 + attack_f1) / 2

    print("========================================")
    print("MANUAL VERIFICATION")
    print("========================================")

    print(f"Accuracy       : {accuracy:.6f}")
    print(f"Normal F1      : {normal_f1:.6f}")
    print(f"Attack F1      : {attack_f1:.6f}")
    print(f"Macro F1       : {macro_f1:.6f}")

    # --------------------------------------------------
    # Compare with sklearn
    # --------------------------------------------------

    print("\n========================================")
    print("SKLEARN CROSS-CHECK")
    print("========================================")

    print(
        f"Accuracy       : "
        f"{accuracy_score(y_test, y_pred):.6f}"
    )

    print(
        f"Attack precision: "
        f"{precision_score(y_test, y_pred):.6f}"
    )

    print(
        f"Attack recall   : "
        f"{recall_score(y_test, y_pred):.6f}"
    )

    print(
        f"Attack F1       : "
        f"{f1_score(y_test, y_pred):.6f}"
    )


if __name__ == "__main__":
    main()