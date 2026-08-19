from pathlib import Path

import pandas as pd

INPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "telco_churn.csv"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "telco_clean.csv"


def clean_data():
    df = pd.read_csv(INPUT_PATH)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    rows_before = len(df)
    df = df.dropna(subset=["TotalCharges"])
    rows_dropped = rows_before - len(df)

    df = df.drop(columns=["customerID"])
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Rows dropped: {rows_dropped}")
    print(f"Saved cleaned dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    clean_data()
