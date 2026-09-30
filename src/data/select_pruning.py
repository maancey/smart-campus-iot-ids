from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
N_SPLITS = 5
K = 40

# Local refinement around the best region found previously.
ALPHA_MIN = 0.000028
ALPHA_MAX = 0.000060
N_ALPHA_VALUES = 25

INPUT_X = Path("data/processed/X_train_preprocessed.csv")
INPUT_Y = Path("data/processed/y_train.csv")
INPUT_RANKING = Path(
    "data/processed/leakage_safe_mutual_information_ranking.csv"
)

OUTPUT_RESULTS = Path(
    "data/processed/pruning_alpha_refinement_results.csv"
)

OUTPUT_SELECTED = Path(
    "data/processed/selected_pruning_configuration.csv"
)


def main():

    print("Loading data...")

    X = pd.read_csv(INPUT_X)
    y = pd.read_csv(INPUT_Y).iloc[:, 0]

    ranking = pd.read_csv(INPUT_RANKING)

    selected_features = (
        ranking
        .sort_values("rank")
        .head(K)["feature"]
        .tolist()
    )

    X = X[selected_features]

    print(f"Selected features: {len(selected_features)}")
    print(f"Training data: {X.shape}")

    # ---------------------------------------------------------
    # Create fixed 5-fold CV partitions.
    # ---------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    folds = list(cv.split(X, y))

    print(f"CV folds: {N_SPLITS}")

    # ---------------------------------------------------------
    # Dense local alpha grid.
    # ---------------------------------------------------------

    alpha_grid = np.linspace(
        ALPHA_MIN,
        ALPHA_MAX,
        N_ALPHA_VALUES
    )

    print(
        f"\nAlpha refinement range: "
        f"{ALPHA_MIN:.8f} to {ALPHA_MAX:.8f}"
    )

    print(
        f"Alpha values: {N_ALPHA_VALUES}"
    )

    # ---------------------------------------------------------
    # Evaluate every alpha on every fold.
    # ---------------------------------------------------------

    results = []

    for alpha_number, alpha in enumerate(
        alpha_grid,
        start=1
    ):

        fold_f1 = []
        fold_depth = []
        fold_nodes = []
        fold_leaves = []

        for train_idx, val_idx in folds:

            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]

            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            model = DecisionTreeClassifier(
                criterion="gini",
                ccp_alpha=float(alpha),
                random_state=RANDOM_STATE
            )

            model.fit(X_train, y_train)

            predictions = model.predict(X_val)

            f1 = f1_score(
                y_val,
                predictions,
                zero_division=0
            )

            fold_f1.append(f1)
            fold_depth.append(model.get_depth())
            fold_nodes.append(model.tree_.node_count)
            fold_leaves.append(model.get_n_leaves())

        results.append({
            "ccp_alpha": alpha,
            "n_folds": len(fold_f1),
            "mean_validation_f1": np.mean(fold_f1),
            "std_validation_f1": np.std(fold_f1),
            "mean_tree_depth": np.mean(fold_depth),
            "mean_tree_nodes": np.mean(fold_nodes),
            "mean_tree_leaves": np.mean(fold_leaves)
        })

        print(
            f"Evaluated {alpha_number}/"
            f"{N_ALPHA_VALUES}"
        )

    results_df = pd.DataFrame(results)

    # ---------------------------------------------------------
    # Sort by mean F1, then smaller tree.
    # ---------------------------------------------------------

    results_df = results_df.sort_values(
        by=[
            "mean_validation_f1",
            "mean_tree_nodes",
            "ccp_alpha"
        ],
        ascending=[
            False,
            True,
            True
        ]
    ).reset_index(drop=True)

    results_df.to_csv(
        OUTPUT_RESULTS,
        index=False
    )

    best = results_df.iloc[0]

    selected = pd.DataFrame([{
        "k": K,
        "ccp_alpha": best["ccp_alpha"],
        "mean_cv_f1": best["mean_validation_f1"],
        "std_cv_f1": best["std_validation_f1"],
        "mean_tree_depth": best["mean_tree_depth"],
        "mean_tree_nodes": best["mean_tree_nodes"],
        "mean_tree_leaves": best["mean_tree_leaves"],
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "alpha_search": "local_dense_refinement"
    }])

    selected.to_csv(
        OUTPUT_SELECTED,
        index=False
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    print("\nTop 10 refined pruning configurations:")
    print(
        results_df.head(10).to_string(index=False)
    )

    print("\nSelected refined pruning configuration:")

    print(
        f"ccp_alpha = "
        f"{best['ccp_alpha']:.10f}"
    )

    print(
        f"mean 5-fold F1 = "
        f"{best['mean_validation_f1']:.6f}"
    )

    print(
        f"std 5-fold F1 = "
        f"{best['std_validation_f1']:.6f}"
    )

    print(
        f"mean tree depth = "
        f"{best['mean_tree_depth']:.2f}"
    )

    print(
        f"mean tree nodes = "
        f"{best['mean_tree_nodes']:.2f}"
    )

    print(
        f"mean tree leaves = "
        f"{best['mean_tree_leaves']:.2f}"
    )

    print("\nSaved:")
    print(OUTPUT_RESULTS)
    print(OUTPUT_SELECTED)


if __name__ == "__main__":
    main()