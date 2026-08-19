from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_PATH = DATA_DIR / "telco_features.csv"


def split_and_scale():
    df = pd.read_csv(INPUT_PATH)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    X_train_scaled.to_csv(DATA_DIR / "X_train.csv", index=False)
    X_test_scaled.to_csv(DATA_DIR / "X_test.csv", index=False)
    y_train.to_csv(DATA_DIR / "y_train.csv", index=False)
    y_test.to_csv(DATA_DIR / "y_test.csv", index=False)
    joblib.dump(scaler, DATA_DIR / "scaler.joblib")

    print(f"Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")
    print(f"Train churn rate: {y_train.mean():.4f}, Test churn rate: {y_test.mean():.4f}")
    print(f"Saved train/test sets and scaler to {DATA_DIR}")


if __name__ == "__main__":
    split_and_scale()
