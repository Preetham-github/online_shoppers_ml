
import pandas as pd

DATA_PATH = "data/raw/online_shoppers.csv"


def inspect_dataset():

    df = pd.read_csv(DATA_PATH)

    print("\n" + "=" * 65)
    print("1. DATASET DIMENSIONS")
    print("=" * 65)

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\n" + "=" * 65)
    print("2. DATA TYPES")
    print("=" * 65)

    print(df.dtypes)

    print("\n" + "=" * 65)
    print("3. MISSING VALUES")
    print("=" * 65)

    missing = df.isnull().sum()

    print(missing[missing > 0])

    print(f"\nTotal missing values: {df.isnull().sum().sum()}")

    print("\n" + "=" * 65)
    print("4. DUPLICATE RECORDS")
    print("=" * 65)

    print(f"Duplicate rows: {df.duplicated().sum()}")

    print("\n" + "=" * 65)
    print("5. TARGET DISTRIBUTION")
    print("=" * 65)

    print(df["Revenue"].value_counts())

    print("\nTarget percentage:")

    print(
        (df["Revenue"].value_counts(normalize=True) * 100).round(2)
    )

    print("\n" + "=" * 65)
    print("6. NUMERICAL FEATURE SUMMARY")
    print("=" * 65)

    numerical = df.select_dtypes(include=["number"])

    print(numerical.describe().round(2))

    print("\n" + "=" * 65)
    print("7. CATEGORICAL FEATURE SUMMARY")
    print("=" * 65)

    categorical = df.select_dtypes(
        include=["object", "bool", "category"]
    )

    print(categorical.describe())

    print("\n" + "=" * 65)
    print("8. UNIQUE VALUES IN CATEGORICAL COLUMNS")
    print("=" * 65)

    for column in categorical.columns:

        print(f"\n{column}:")
        print(df[column].value_counts().head(10))


if __name__ == "__main__":
    inspect_dataset()
