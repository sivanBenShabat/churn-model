from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def train_model():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()

    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "churn_model.joblib")
    print(f"Saved model to {MODELS_DIR / 'churn_model.joblib'}")

    coefs = pd.Series(model.coef_[0], index=X_train.columns).sort_values()

    print("\nTop 5 negative coefficients (reduce churn risk):")
    print(coefs.head(5))

    print("\nTop 5 positive coefficients (increase churn risk):")
    print(coefs.tail(5)[::-1])


if __name__ == "__main__":
    train_model()
