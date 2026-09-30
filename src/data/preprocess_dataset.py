from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder



PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

TRAIN_FILE = PROCESSED_DIR / "train_selected.csv"
TEST_FILE = PROCESSED_DIR / "test_selected.csv"

TRAIN_OUTPUT = PROCESSED_DIR / "X_train_preprocessed.csv"
TEST_OUTPUT = PROCESSED_DIR / "X_test_preprocessed.csv"

TRAIN_TARGET_OUTPUT = PROCESSED_DIR / "y_train.csv"
TEST_TARGET_OUTPUT = PROCESSED_DIR / "y_test.csv"

PREPROCESSOR_OUTPUT = MODEL_DIR / "preprocessor.joblib"

TARGET_COLUMN = "label"
AUDIT_COLUMN = "attack_cat"
ID_COLUMN = "id"

CATEGORICAL_COLUMNS = [
    "proto",
    "service",
    "state",
]


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


def build_preprocessor(
    X_train: pd.DataFrame,
) -> ColumnTransformer:
    categorical_columns = [
        column
        for column in CATEGORICAL_COLUMNS
        if column in X_train.columns
    ]

    numerical_columns = [
        column
        for column in X_train.columns
        if column not in categorical_columns
    ]

    numeric_pipeline = SimpleImputer(
        strategy="median"
    )

    categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                dtype="float32",
            ),
        ),
    ]
)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numerical_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def main() -> None:
    print("=" * 70)
    print("PREPROCESSING DATASET")
    print("=" * 70)

    train_df = load_data(TRAIN_FILE)
    test_df = load_data(TEST_FILE)

    print(f"Training rows: {len(train_df):,}")
    print(f"Testing rows : {len(test_df):,}")

    # ---------------------------------------------------------------
    # Separate target and audit information.
    # ---------------------------------------------------------------

    y_train = train_df[TARGET_COLUMN].astype(int)
    y_test = test_df[TARGET_COLUMN].astype(int)

    X_train = train_df.drop(
        columns=[TARGET_COLUMN, AUDIT_COLUMN, ID_COLUMN]
    ).copy()

    X_test = test_df.drop(
        columns=[TARGET_COLUMN, AUDIT_COLUMN, ID_COLUMN]
    ).copy()

    # Ensure train and test have identical predictor columns.
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "Training and testing predictor columns do not match."
        )

    print(f"\nOriginal predictor columns: {X_train.shape[1]}")

    # ---------------------------------------------------------------
    # Build and fit preprocessing ONLY on training data.
    # ---------------------------------------------------------------

    preprocessor = build_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)

    # The test set is transformed using the already-fitted
    # training preprocessor. No test information is used for fitting.
    X_test_processed = preprocessor.transform(X_test)

    # ---------------------------------------------------------------
    # Convert to DataFrames with feature names.
    # ---------------------------------------------------------------

    feature_names = preprocessor.get_feature_names_out()

    X_train_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
    )

    X_test_processed = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
    )

    # ---------------------------------------------------------------
    # Save outputs.
    # ---------------------------------------------------------------

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X_train_processed.to_csv(
        TRAIN_OUTPUT,
        index=False,
    )

    X_test_processed.to_csv(
        TEST_OUTPUT,
        index=False,
    )

    y_train.to_csv(
        TRAIN_TARGET_OUTPUT,
        index=False,
        header=True,
    )

    y_test.to_csv(
        TEST_TARGET_OUTPUT,
        index=False,
        header=True,
    )

    joblib.dump(
        preprocessor,
        PREPROCESSOR_OUTPUT,
    )

    # ---------------------------------------------------------------
    # Report results.
    # ---------------------------------------------------------------

    print(f"Preprocessed training shape: {X_train_processed.shape}")
    print(f"Preprocessed testing shape : {X_test_processed.shape}")

    print("\nClass distribution:")
    print("Training:")
    print(y_train.value_counts().sort_index().to_string())

    print("\nTesting:")
    print(y_test.value_counts().sort_index().to_string())

    print("\nGenerated feature count:")
    print(len(feature_names))

    print("\nFirst 20 generated features:")
    for feature in feature_names[:20]:
        print(f"  - {feature}")

    print("\nSaved:")
    print(f"  {TRAIN_OUTPUT}")
    print(f"  {TEST_OUTPUT}")
    print(f"  {TRAIN_TARGET_OUTPUT}")
    print(f"  {TEST_TARGET_OUTPUT}")
    print(f"  {PREPROCESSOR_OUTPUT}")


if __name__ == "__main__":
    main()