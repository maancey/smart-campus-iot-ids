from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_FEATURES = PROCESSED_DIR / "X_train_preprocessed.csv"
TRAIN_TARGET = PROCESSED_DIR / "y_train.csv"

REDUCED_FEATURES = PROCESSED_DIR / "X_train_corr_reduced.csv"
REMOVED_FEATURES = PROCESSED_DIR / "correlation_removed_features.csv"
CORRELATED_PAIRS = PROCESSED_DIR / "high_correlation_numeric_pairs.csv"

CORRELATION_THRESHOLD = 0.90


# These are the original numeric predictors.
# One-hot encoded categorical variables are deliberately excluded
# from this correlation-reduction stage.
NUMERIC_FEATURES = [
    "dur",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "tcprtt",
    "synack",
    "ackdat",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
]


def find_correlated_pairs(X_numeric: pd.DataFrame) -> list[tuple[str, str, float]]:
    correlation = X_numeric.corr(method="pearson")

    pairs = []

    for i, feature_a in enumerate(correlation.columns):
        for feature_b in correlation.columns[i + 1:]:
            value = correlation.loc[feature_a, feature_b]

            if pd.notna(value) and abs(value) > CORRELATION_THRESHOLD:
                pairs.append(
                    (feature_a, feature_b, float(value))
                )

    return pairs


def build_groups(
    pairs: list[tuple[str, str, float]]
) -> list[set[str]]:
    groups = []

    for feature_a, feature_b, _ in pairs:
        matching_groups = [
            group
            for group in groups
            if feature_a in group or feature_b in group
        ]

        if not matching_groups:
            groups.append({feature_a, feature_b})
            continue

        merged = {feature_a, feature_b}

        for group in matching_groups:
            merged.update(group)
            groups.remove(group)

        groups.append(merged)

    return groups


def select_group_representative(
    group: set[str],
    X_numeric: pd.DataFrame,
) -> str:
    group_features = sorted(group)

    if len(group_features) == 1:
        return group_features[0]

    group_correlation = (
        X_numeric[group_features]
        .corr(method="pearson")
        .abs()
    )

    mean_redundancy = {}

    for feature in group_features:
        other_features = [
            other
            for other in group_features
            if other != feature
        ]

        mean_redundancy[feature] = (
            group_correlation.loc[
                feature,
                other_features,
            ].mean()
        )

    # Lower mean redundancy is preferred.
    # Alphabetical order provides a deterministic tie-break.
    ranked = sorted(
        group_features,
        key=lambda feature: (
            mean_redundancy[feature],
            feature,
        ),
    )

    return ranked[0]
    scores = {}

    for feature in group:
        correlation = X_numeric[feature].corr(y)
        scores[feature] = abs(correlation)

    # Deterministic tie-break:
    # highest absolute label correlation first,
    # then alphabetical feature name.
    ranked = sorted(
        group,
        key=lambda feature: (-scores[feature], feature),
    )

    return ranked[0]


def main() -> None:
    print("=" * 70)
    print("CORRELATION-BASED FEATURE REDUCTION")
    print("=" * 70)

    if not TRAIN_FEATURES.exists():
        raise FileNotFoundError(
            f"Training features not found: {TRAIN_FEATURES}"
        )

    if not TRAIN_TARGET.exists():
        raise FileNotFoundError(
            f"Training target not found: {TRAIN_TARGET}"
        )

    X_train = pd.read_csv(TRAIN_FEATURES)
    y_train = pd.read_csv(TRAIN_TARGET).squeeze("columns")

    missing_features = [
        feature
        for feature in NUMERIC_FEATURES
        if feature not in X_train.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing expected numeric features: {missing_features}"
        )

    X_numeric = X_train[NUMERIC_FEATURES].copy()

    print(f"Total preprocessed features: {X_train.shape[1]}")
    print(f"Numeric features analysed: {X_numeric.shape[1]}")
    print(f"Correlation threshold: |r| > {CORRELATION_THRESHOLD}")

    pairs = find_correlated_pairs(X_numeric)

    print(f"\nHigh-correlation numeric pairs: {len(pairs)}")

    pair_records = []

    for feature_a, feature_b, correlation in pairs:
        pair_records.append(
            {
                "feature_a": feature_a,
                "feature_b": feature_b,
                "pearson_r": correlation,
                "absolute_r": abs(correlation),
            }
        )

        print(
            f"{feature_a} <-> {feature_b}: "
            f"r={correlation:.4f}"
        )

    pd.DataFrame(pair_records).to_csv(
        CORRELATED_PAIRS,
        index=False,
    )

    groups = build_groups(pairs)

    print(f"\nCorrelated feature groups: {len(groups)}")

    features_to_remove = set()

    removal_records = []

    for group in sorted(groups, key=lambda g: sorted(g)[0]):
        representative = select_group_representative(
    group,
    X_numeric,
)
        removed = sorted(group - {representative})

        print("\nGroup:")
        print("  " + ", ".join(sorted(group)))
        print(f"  Retained: {representative}")

        for feature in removed:

            features_to_remove.add(feature)

            removal_records.append(
                {
                    "feature": feature,
                    "retained_feature": representative,
                    "mean_group_redundancy": (
    X_numeric[
        sorted(group)
    ]
    .corr(method="pearson")
    .abs()
    .loc[feature, sorted(group)]
    .drop(feature)
    .mean()
),
                }
            )

            print(
    f"  Removed: {feature}"
)

    X_reduced = X_train.drop(
        columns=sorted(features_to_remove)
    )

    X_reduced.to_csv(
        REDUCED_FEATURES,
        index=False,
    )

    pd.DataFrame(removal_records).to_csv(
        REMOVED_FEATURES,
        index=False,
    )

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"Original features: {X_train.shape[1]}")
    print(f"Removed features : {len(features_to_remove)}")
    print(f"Remaining features: {X_reduced.shape[1]}")

    print("\nSaved:")
    print(f"  {CORRELATED_PAIRS}")
    print(f"  {REMOVED_FEATURES}")
    print(f"  {REDUCED_FEATURES}")


if __name__ == "__main__":
    main()