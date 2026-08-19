from pathlib import Path

import pandas as pd

INPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "telco_clean.csv"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "telco_features.csv"

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def build_features():
    df = pd.read_csv(INPUT_PATH)

    df["services_count"] = df[SERVICE_COLUMNS].apply(
        lambda col: ~col.isin(["No", "No phone service", "No internet service"])
    ).sum(axis=1)

    df["new_customer"] = (df["tenure"] < 6).astype(int)

    df = pd.get_dummies(df, drop_first=True)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Shape after feature engineering: {df.shape}")
    print(f"Saved features to {OUTPUT_PATH}")


if __name__ == "__main__":
    build_features()
