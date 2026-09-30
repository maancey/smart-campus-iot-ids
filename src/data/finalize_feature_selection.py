from pathlib import Path

import pandas as pd
from sklearn.feature_selection import mutual_info_classif


TRAIN_X = Path("data/processed/X_train_preprocessed.csv")
TRAIN_Y = Path("data/processed/y_train.csv")

OUTPUT_FEATURES = Path(
    "data/processed/final_selected_features.csv"
)

OUTPUT_RANKING = Path(
    "data/processed/final_mutual_information_ranking.csv"
)

OUTPUT_TRAIN = Path(
    "data/processed/X_train_final_40.csv"
)

K = 40
CORRELATION_THRESHOLD = 0.90
RANDOM_STATE = 42


def main():

    print("Loading full training data...")

    X = pd.read_csv(TRAIN_X)
    y = pd.read_csv(TRAIN_Y).iloc[:, 0]

    print(f"Training data: {X.shape}")

    # --------------------------------------------------
    # Identify original numeric features.
    # One-hot categorical features are excluded from
    # correlation filtering.
    # --------------------------------------------------

    categorical_prefixes = (
        "proto_",
        "service_",
        "state_",
    )

    numeric_features = [
        col
        for col in X.columns
        if not col.startswith(categorical_prefixes)
    ]

    X_numeric = X[numeric_features]

    print(
        f"Numeric features for correlation filtering: "
        f"{len(numeric_features)}"
    )

    # --------------------------------------------------
    # Correlation reduction
    #
    # Same deterministic rule used previously:
    # within each correlated group, retain the feature
    # having the lowest mean absolute correlation.
    # --------------------------------------------------

    corr = X_numeric.corr().abs()


    # Avoid relying on pandas/numpy internals:
    upper = corr.where(
        __import__("numpy").triu(
            __import__("numpy").ones(corr.shape),
            k=1
        ).astype(bool)
    )

    graph = {
        feature: set()
        for feature in numeric_features
    }

    for i, feature_a in enumerate(numeric_features):

        for feature_b in numeric_features[i + 1:]:

            if corr.loc[feature_a, feature_b] > CORRELATION_THRESHOLD:

                graph[feature_a].add(feature_b)
                graph[feature_b].add(feature_a)

    # --------------------------------------------------
    # Find connected correlated groups.
    # --------------------------------------------------

    visited = set()
    retained_numeric = set()

    for feature in numeric_features:

        if feature in visited:
            continue

        stack = [feature]
        group = []

        while stack:

            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)
            group.append(current)

            stack.extend(
                graph[current] - visited
            )

        if len(group) == 1:

            retained_numeric.add(group[0])

        else:

            mean_abs_corr = {
                f: corr.loc[f, group].drop(
                    labels=[f]
                ).mean()
                for f in group
            }

            representative = sorted(
                group,
                key=lambda f: (
                    mean_abs_corr[f],
                    f
                )
            )[0]

            retained_numeric.add(
                representative
            )

    # Keep all encoded categorical features.
    categorical_features = [
        col
        for col in X.columns
        if col.startswith(categorical_prefixes)
    ]

    correlation_reduced_features = (
        sorted(retained_numeric)
        + categorical_features
    )

    print(
        "After correlation reduction:",
        len(correlation_reduced_features)
    )

    X_reduced = X[
        correlation_reduced_features
    ]

    # --------------------------------------------------
    # Mutual information
    # --------------------------------------------------

    discrete_mask = [
        col.startswith(categorical_prefixes)
        for col in X_reduced.columns
    ]

    print("\nCalculating mutual information...")

    mi = mutual_info_classif(
        X_reduced,
        y,
        discrete_features=discrete_mask,
        random_state=RANDOM_STATE,
    )

    ranking = pd.DataFrame({
        "feature": X_reduced.columns,
        "mutual_information": mi,
    })

    ranking = ranking.sort_values(
        by=[
            "mutual_information",
            "feature",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    ranking["rank"] = (
        ranking.index + 1
    )

    # --------------------------------------------------
    # Select top 40.
    # --------------------------------------------------

    selected_features = (
        ranking
        .head(K)["feature"]
        .tolist()
    )

    X_final = X[
        selected_features
    ]

    print(
        f"\nSelected features: {len(selected_features)}"
    )

    print("\nSelected feature list:")

    for i, feature in enumerate(
        selected_features,
        start=1
    ):
        print(
            f"{i:02d}. {feature}"
        )

    # --------------------------------------------------
    # Save artifacts.
    # --------------------------------------------------

    pd.DataFrame({
        "feature": selected_features,
        "rank": range(1, K + 1),
    }).to_csv(
        OUTPUT_FEATURES,
        index=False,
    )

    ranking.to_csv(
        OUTPUT_RANKING,
        index=False,
    )

    X_final.to_csv(
        OUTPUT_TRAIN,
        index=False,
    )

    print("\nSaved:")
    print(OUTPUT_FEATURES)
    print(OUTPUT_RANKING)
    print(OUTPUT_TRAIN)


if __name__ == "__main__":
    main()