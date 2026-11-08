
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_PATH = "data/raw/online_shoppers.csv"
OUTPUT_DIR = Path("reports/eda")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")


def load_data():
    return pd.read_csv(DATA_PATH)


def plot_target_distribution(df):

    plt.figure(figsize=(8, 5))

    ax = sns.countplot(
        data=df,
        x="Revenue",
        hue="Revenue",
        legend=False
    )

    plt.title("Purchase vs Non-Purchase Distribution")
    plt.xlabel("Revenue")
    plt.ylabel("Number of Sessions")

    for container in ax.containers:
        ax.bar_label(container)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "target_distribution.png", dpi=300)
    plt.close()


def plot_numerical_distributions(df):

    numerical_columns = [
        "Administrative",
        "Administrative_Duration",
        "Informational",
        "Informational_Duration",
        "ProductRelated",
        "ProductRelated_Duration",
        "BounceRates",
        "ExitRates",
        "PageValues",
        "SpecialDay"
    ]

    for column in numerical_columns:

        plt.figure(figsize=(8, 5))

        sns.histplot(
            data=df,
            x=column,
            hue="Revenue",
            kde=True,
            bins=30,
            element="step"
        )

        plt.title(f"Distribution of {column}")
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"{column}_distribution.png",
            dpi=300
        )

        plt.close()


def plot_outliers(df):

    numerical_columns = [
        "Administrative_Duration",
        "Informational_Duration",
        "ProductRelated_Duration",
        "BounceRates",
        "ExitRates",
        "PageValues"
    ]

    for column in numerical_columns:

        plt.figure(figsize=(8, 5))

        sns.boxplot(
            data=df,
            x="Revenue",
            y=column,
            hue="Revenue",
            legend=False
        )

        plt.title(f"Outlier Analysis: {column}")
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"{column}_boxplot.png",
            dpi=300
        )

        plt.close()


def plot_categorical_features(df):

    categorical_columns = [
        "Month",
        "VisitorType",
        "Weekend"
    ]

    for column in categorical_columns:

        plt.figure(figsize=(10, 5))

        sns.countplot(
            data=df,
            x=column,
            hue="Revenue"
        )

        plt.title(f"{column} vs Revenue")
        plt.xlabel(column)
        plt.ylabel("Number of Sessions")

        plt.xticks(rotation=30)
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"{column}_revenue.png",
            dpi=300
        )

        plt.close()


def plot_correlation(df):

    correlation_columns = [
        "Administrative",
        "Administrative_Duration",
        "Informational",
        "Informational_Duration",
        "ProductRelated",
        "ProductRelated_Duration",
        "BounceRates",
        "ExitRates",
        "PageValues",
        "SpecialDay"
    ]

    correlation_data = df[correlation_columns].copy()

    correlation_data["Weekend"] = df["Weekend"].astype(int)
    correlation_data["Revenue"] = df["Revenue"].astype(int)

    plt.figure(figsize=(14, 10))

    correlation = correlation_data.corr()

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        center=0
    )

    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "correlation_heatmap.png",
        dpi=300
    )

    plt.close()


def analyze_duplicates(df):

    duplicate_rows = df[df.duplicated(keep=False)]

    print("\nDUPLICATE ANALYSIS")
    print("-" * 50)

    print("Exact duplicate rows:", df.duplicated().sum())

    print("\nTarget distribution among duplicated rows:")
    print(duplicate_rows["Revenue"].value_counts())

    print("\nUnique records after removing exact duplicates:",
          len(df.drop_duplicates()))


def run_eda():

    df = load_data()

    print("Starting Exploratory Data Analysis...")

    plot_target_distribution(df)
    print("Target distribution generated.")

    plot_numerical_distributions(df)
    print("Numerical distributions generated.")

    plot_outliers(df)
    print("Outlier visualizations generated.")

    plot_categorical_features(df)
    print("Categorical visualizations generated.")

    plot_correlation(df)
    print("Correlation heatmap generated.")

    analyze_duplicates(df)

    print("\nEDA completed successfully!")
    print(f"Visualizations saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_eda()
