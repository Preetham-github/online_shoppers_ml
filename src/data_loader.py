
from ucimlrepo import fetch_ucirepo
import pandas as pd
from pathlib import Path

def download_dataset():
    print("Downloading Online Shoppers Purchasing Intention dataset...")

    dataset = fetch_ucirepo(id=468)

    features = dataset.data.features
    target = dataset.data.targets

    df = pd.concat([features, target], axis=1)

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "online_shoppers.csv"

    df.to_csv(output_path, index=False)

    print("\nDataset downloaded successfully!")
    print(f"Saved location: {output_path}")
    print(f"Number of records: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    return df


if __name__ == "__main__":
    df = download_dataset()

    print("\nFirst 5 records:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())
