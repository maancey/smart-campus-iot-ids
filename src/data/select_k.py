from pathlib import Path

import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
VALIDATION_SIZE = 0.20

K_VALUES = [10, 15, 20, 25, 30, 40, 50]

INPUT_X = Path("data/processed/X_train_preprocessed.csv")
INPUT_Y = Path("data/processed/y_train.csv")

OUTPUT_RESULTS = Path("data/processed/k_selection_results.csv")
OUTPUT_RANKING = Path("data/processed/leakage_safe_mutual_information_ranking.csv")
OUTPUT_REMOVED = Path("data/processed/leakage_safe_correlation_removed_features.csv")


def reduce_correlation(X):
    """
    Remove highly correlated numeric features using training data only.

    Features are removed from groups where |r| > 0.9.
    The representative feature is the one with the lowest
    mean absolute correlation within its correlated group.
    """

    # Original numeric predictors only.
    numeric_features = [
        "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate",
        "sttl", "dttl", "sload", "dload", "sloss", "dloss",
        "sinpkt", "dinpkt", "sjit", "djit", "swin", "stcpb",
        "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean",
        "dmean", "trans_depth", "response_body_len", "ct_srv_src",
        "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
        "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login",
        "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm",
        "ct_srv_dst", "is_sm_ips_ports"
    ]

    available = [f for f in numeric_features if f in X.columns]

    corr = X[available].corr().abs()

    # Build graph of correlated features.
    graph = {f: set() for f in available}

    for i, f1 in enumerate(available):
        for f2 in available[i + 1:]:
            if corr.loc[f1, f2] > 0.90:
                graph[f1].add(f2)
                graph[f2].add(f1)

    # Find connected groups.
    visited = set()
    groups = []

    for feature in available:
        if feature in visited or not graph[feature]:
            continue

        stack = [feature]
        group = set()

        while stack:
            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)
            group.add(current)

            for neighbour in graph[current]:
                if neighbour not in visited:
                    stack.append(neighbour)

        groups.append(sorted(group))

    removed = []
    retained = []

    for group in groups:
        if len(group) <= 1:
            continue

        # Lowest mean absolute correlation within the group.
        mean_corr = {
            feature: corr.loc[feature, group].drop(feature).mean()
            for feature in group
        }

        representative = min(
            group,
            key=lambda feature: (mean_corr[feature], feature)
        )

        for feature in group:
            if feature != representative:
                removed.append(feature)

        retained.append(representative)

    removed = sorted(set(removed))

    X_reduced = X.drop(columns=removed)

    return X_reduced, removed, groups


def main():

    print("Loading training data...")

    X = pd.read_csv(INPUT_X)
    y = pd.read_csv(INPUT_Y).iloc[:, 0]

    print(f"Full training data: {X.shape}")

    # ---------------------------------------------------------
    # IMPORTANT:
    # Split BEFORE feature selection.
    # ---------------------------------------------------------

    X_inner, X_validation, y_inner, y_validation = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print(f"Feature-selection training partition: {X_inner.shape}")
    print(f"Validation partition: {X_validation.shape}")

    # ---------------------------------------------------------
    # Correlation reduction using INNER TRAINING ONLY
    # ---------------------------------------------------------

    X_inner_reduced, removed_features, groups = reduce_correlation(
        X_inner
    )

    # Apply exactly the same retained feature set to validation.
    retained_after_corr = X_inner_reduced.columns.tolist()

    X_validation_reduced = X_validation[
        retained_after_corr
    ].copy()

    print()
    print("Correlation reduction:")
    print(f"Original features: {X_inner.shape[1]}")
    print(f"Removed features: {len(removed_features)}")
    print(f"Remaining features: {X_inner_reduced.shape[1]}")

    if groups:
        print("\nCorrelated groups:")
        for group in groups:
            print("  " + ", ".join(group))

    # ---------------------------------------------------------
    # Mutual information using INNER TRAINING ONLY
    # ---------------------------------------------------------

    print("\nCalculating mutual information...")

    # Explicitly identify one-hot categorical features as discrete.
    discrete_mask = [
        (
            column.startswith("proto_")
            or column.startswith("service_")
            or column.startswith("state_")
        )
        for column in X_inner_reduced.columns
    ]

    mi_scores = mutual_info_classif(
        X_inner_reduced,
        y_inner,
        discrete_features=discrete_mask,
        random_state=RANDOM_STATE
    )

    ranking = pd.DataFrame({
        "feature": X_inner_reduced.columns,
        "mutual_information": mi_scores,
        "is_discrete": discrete_mask
    })

    ranking = ranking.sort_values(
        by=["mutual_information", "feature"],
        ascending=[False, True]
    ).reset_index(drop=True)

    ranking["rank"] = range(1, len(ranking) + 1)

    ranking.to_csv(
        OUTPUT_RANKING,
        index=False
    )

    print("\nTop 20 MI features:")
    print(
        ranking[
            ["rank", "feature", "mutual_information"]
        ].head(20).to_string(index=False)
    )

    # ---------------------------------------------------------
    # Evaluate candidate k values
    # ---------------------------------------------------------

    results = []

    for k in K_VALUES:

        selected_features = ranking.head(k)["feature"].tolist()

        X_train_k = X_inner_reduced[selected_features]
        X_val_k = X_validation_reduced[selected_features]

        model = DecisionTreeClassifier(
            criterion="gini",
            random_state=RANDOM_STATE
        )

        model.fit(X_train_k, y_inner)

        predictions = model.predict(X_val_k)

        accuracy = accuracy_score(
            y_validation,
            predictions
        )

        precision = precision_score(
            y_validation,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_validation,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_validation,
            predictions,
            zero_division=0
        )

        results.append({
            "k": k,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tree_depth": model.get_depth(),
            "tree_nodes": model.tree_.node_count,
            "tree_leaves": model.get_n_leaves()
        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by=["f1", "tree_nodes"],
        ascending=[False, True]
    ).reset_index(drop=True)

    results_df.to_csv(
        OUTPUT_RESULTS,
        index=False
    )

    # ---------------------------------------------------------
    # Save removed features
    # ---------------------------------------------------------

    pd.DataFrame({
        "removed_feature": removed_features
    }).to_csv(
        OUTPUT_REMOVED,
        index=False
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    print("\nK-selection results:")
    print(results_df.to_string(index=False))

    best = results_df.iloc[0]

    print("\nSelected configuration:")
    print(f"k = {int(best['k'])}")
    print(f"validation F1 = {best['f1']:.6f}")
    print(f"tree depth = {int(best['tree_depth'])}")
    print(f"tree nodes = {int(best['tree_nodes'])}")
    print(f"tree leaves = {int(best['tree_leaves'])}")

    print("\nSaved:")
    print(OUTPUT_RESULTS)
    print(OUTPUT_RANKING)
    print(OUTPUT_REMOVED)


if __name__ == "__main__":
    main()