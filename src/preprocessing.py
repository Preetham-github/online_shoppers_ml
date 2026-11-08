import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


DATA_PATH = "data/raw/online_shoppers.csv"
TARGET_COLUMN = "Revenue"
RANDOM_STATE = 42

NUMERICAL_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]

CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

EXPECTED_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def load_data():

    df = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing."
        )

    missing_features = [
        column
        for column in EXPECTED_FEATURES
        if column not in df.columns
    ]

    unexpected_features = [
        column
        for column in df.columns
        if column not in EXPECTED_FEATURES
        and column != TARGET_COLUMN
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    if unexpected_features:
        raise ValueError(
            f"Unexpected columns found: {unexpected_features}"
        )

    X = df[EXPECTED_FEATURES].copy()

    y_raw = df[TARGET_COLUMN]

    if y_raw.isna().any():
        raise ValueError(
            "The target column contains missing values."
        )

    if y_raw.dtype == object:
        y_raw = (
            y_raw.astype(str)
            .str.strip()
            .str.lower()
            .map({
                "true": 1,
                "false": 0,
                "1": 1,
                "0": 0,
            })
        )

    y = pd.to_numeric(
        y_raw,
        errors="raise"
    ).astype(int)

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(
            "Revenue must contain only binary values: 0 and 1."
        )

    if y.nunique() != 2:
        raise ValueError(
            "Both target classes must be present."
        )

    for column in NUMERICAL_FEATURES:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    for column in CATEGORICAL_FEATURES:
        X[column] = X[column].astype("object")

    return X, y


def create_preprocessor(dense_output=False):

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
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
                    sparse_output=not dense_output,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        sparse_threshold=0 if dense_output else 0.3,
    )

    return preprocessor


def split_data():

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    print("\nTraining target distribution:")
    print(
        y_train.value_counts(normalize=True).sort_index()
    )

    print("\nTesting target distribution:")
    print(
        y_test.value_counts(normalize=True).sort_index()
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    split_data()